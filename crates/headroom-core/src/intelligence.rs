//! Context-intelligence hooks for the Rust live-zone dispatcher.
//!
//! Rust twin (deliberately smaller) of `headroom/intelligence/`:
//!
//! * [`IntelligenceSettings`] — the same enablement contract as the Python
//!   proxy: `HEADROOM_INTELLIGENCE=off|safe|full` selects a posture and the
//!   per-feature variables (`HEADROOM_TASK_QUERY`, `HEADROOM_INVARIANT_GUARD`,
//!   `HEADROOM_POLICY_RISK_BUDGET`) override it in either direction. Off by
//!   default, so the dispatcher stays byte-identical unless asked.
//! * [`relevance_query`] — the latest user prose of a request (Anthropic
//!   `messages`, OpenAI chat `messages` or Responses `input` items), used
//!   instead of an empty query by the relevance-aware compressors.
//! * [`InvariantSet`] — a basic invariant guard: explicit user entities,
//!   exit/status codes, error lines and test summaries must survive a lossy
//!   rewrite unless the original is recoverable (CCR marker + stored
//!   original). Same hard-veto names as the Python guard.

use std::sync::LazyLock;

use regex::Regex;
use serde_json::Value;

/// Posture selected by `HEADROOM_INTELLIGENCE`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Level {
    Off,
    Safe,
    Full,
}

/// Parse a posture value with the Python aliases (`on`/`1` = safe, …).
pub fn parse_level(raw: Option<&str>) -> Level {
    let Some(raw) = raw else { return Level::Off };
    match raw.trim().to_ascii_lowercase().as_str() {
        "safe" | "on" | "1" | "true" | "yes" | "default" | "standard" => Level::Safe,
        "full" | "max" | "aggressive" | "all" => Level::Full,
        _ => Level::Off,
    }
}

/// Parse a per-feature boolean (`None` = unset/invalid: posture decides).
pub fn parse_bool(raw: Option<&str>) -> Option<bool> {
    let value = raw?.trim().to_ascii_lowercase();
    match value.as_str() {
        "1" | "true" | "yes" | "on" | "enabled" => Some(true),
        "0" | "false" | "no" | "off" | "disabled" => Some(false),
        _ => None,
    }
}

/// Resolved live-zone intelligence switches.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct IntelligenceSettings {
    pub task_query: bool,
    pub invariant_guard: bool,
    pub policy_budget: bool,
}

impl IntelligenceSettings {
    /// Resolve from an env lookup (`std::env::var` in production).
    pub fn resolve(get: impl Fn(&str) -> Option<String>) -> Self {
        let level = parse_level(get("HEADROOM_INTELLIGENCE").as_deref());
        let posture = level != Level::Off; // all three are "safe" features
        let feature = |name: &str| parse_bool(get(name).as_deref()).unwrap_or(posture);
        Self {
            task_query: feature("HEADROOM_TASK_QUERY"),
            invariant_guard: feature("HEADROOM_INVARIANT_GUARD"),
            policy_budget: feature("HEADROOM_POLICY_RISK_BUDGET"),
        }
    }

    /// Resolve from the process environment. Read per request (a few
    /// lookups) so a test or an operator change never sees a stale value.
    pub fn from_env() -> Self {
        Self::resolve(|name| std::env::var(name).ok())
    }

    pub fn any(&self) -> bool {
        self.task_query || self.invariant_guard || self.policy_budget
    }
}

// ─── Relevance query ────────────────────────────────────────────────────

const QUERY_MAX_CHARS: usize = 600;

static SYSTEM_REMINDER_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?s)<system-reminder>.*?</system-reminder>").expect("SYSTEM_REMINDER_RE")
});

fn prose_of(content: &Value) -> String {
    match content {
        Value::String(s) => s.clone(),
        Value::Array(parts) => parts
            .iter()
            .filter_map(|p| {
                let kind = p.get("type").and_then(Value::as_str).unwrap_or("text");
                if matches!(kind, "text" | "input_text") {
                    p.get("text").and_then(Value::as_str).map(str::to_owned)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

/// The latest user prose (system reminders stripped), capped.
///
/// Walks `messages` (or Responses `input` items) from the back and returns
/// the first user turn that carries real text — a tool-result-only user
/// message is skipped, so the query is the instruction the tool output is
/// being gathered for.
pub fn relevance_query(messages: &[Value]) -> String {
    for msg in messages.iter().rev() {
        if msg.get("role").and_then(Value::as_str) != Some("user") {
            continue;
        }
        let Some(content) = msg.get("content") else {
            continue;
        };
        let prose = SYSTEM_REMINDER_RE
            .replace_all(&prose_of(content), " ")
            .to_string();
        let trimmed = prose.trim();
        if trimmed.is_empty() {
            continue;
        }
        return trimmed.chars().take(QUERY_MAX_CHARS).collect();
    }
    String::new()
}

// ─── Explicit entities ──────────────────────────────────────────────────

static QUOTED_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r#"`([^`\n]{2,80})`|"([^"\n]{2,80})""#).expect("QUOTED_RE"));
static PATH_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?:[\w.-]+[/\\])*[\w-]+\.[A-Za-z][A-Za-z0-9]{0,5}\b").expect("PATH_RE")
});
static IDENT_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\b(?:[a-z]+_[a-z0-9_]+|[a-z]+[A-Z][A-Za-z0-9]+|[A-Z][a-z]+[A-Z][A-Za-z0-9]+)\b")
        .expect("IDENT_RE")
});

/// Entities the user named explicitly: quoted/backticked spans, file paths
/// and code identifiers (snake_case / camelCase / PascalCase compounds).
pub fn explicit_entities(query: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut push = |s: &str| {
        let s = s.trim();
        if s.len() >= 2 && !out.iter().any(|e| e == s) {
            out.push(s.to_owned());
        }
    };
    for caps in QUOTED_RE.captures_iter(query) {
        if let Some(m) = caps.get(1).or_else(|| caps.get(2)) {
            push(m.as_str());
        }
    }
    for m in PATH_RE.find_iter(query) {
        let text = m.as_str();
        // "e.g", "i.e" and version-ish fragments are not paths.
        if text.len() > 3 && text.contains(|c: char| c.is_ascii_alphabetic()) {
            push(text);
        }
    }
    for m in IDENT_RE.find_iter(query) {
        push(m.as_str());
    }
    out.truncate(32);
    out
}

// ─── Invariant guard ────────────────────────────────────────────────────

static ERROR_LINE_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?m)^.*(?:\b(?:Error|Exception|Traceback|FAILED|FAIL|panic(?:ked)?|fatal|assert(?:ion)?(?:Error)? failed|AssertionError|Segmentation fault|error\[E\d+\]|error:|ERROR|CRITICAL|undefined reference|cannot find)\b).*$",
    )
    .expect("ERROR_LINE_RE")
});
static EXIT_CODE_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r#"(?i)(?:exit(?:ed)?(?: with)?(?: code| status)?[:= ]+|exit_code["']?\s*[:=]\s*|returned non-zero exit status |status code[:= ]+|Process exited with code )(-?\d{1,3})\b"#,
    )
    .expect("EXIT_CODE_RE")
});
static TEST_SUMMARY_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)\b\d+\s+(?:passed|failed|errors?|skipped|xfailed|xpassed|warnings?|deselected|tests? ran)\b|\bTests?:\s+\d+[^\n]{0,80}|\btest result: (?:ok|FAILED)\.[^\n]{0,120}",
    )
    .expect("TEST_SUMMARY_RE")
});
static ANNOTATION_LINE_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?im)^.*(?:<<ccr:|hash=[0-9a-f]{6,}|\bcompressed\b|\belided\b|\bomitted\b|\bheadroom\b|\bretrieve\b|\(repeated \d+|\bunchanged\b|\bdelta\b|\bsummar(?:y|ized)\b|\bmore (?:items|rows|lines|matches|results)\b|\btruncated\b).*$",
    )
    .expect("ANNOTATION_LINE_RE")
});
static BRACKET_ANNOTATION_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)\[[^\]\n]*(?:compress|omit|elid|retriev|hash=|headroom|repeated|more)[^\]\n]*\]",
    )
    .expect("BRACKET_ANNOTATION_RE")
});

const MAX_SCAN: usize = 400_000;

fn dedupe_push(out: &mut Vec<String>, item: &str, cap: usize) {
    if out.len() < cap && !out.iter().any(|e| e == item) {
        out.push(item.to_owned());
    }
}

fn floor_char_boundary(s: &str, mut i: usize) -> usize {
    i = i.min(s.len());
    while !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

/// Invariants of one original block.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct InvariantSet {
    pub user_entities: Vec<String>,
    pub error_lines: Vec<String>,
    pub exit_codes: Vec<String>,
    pub test_summaries: Vec<String>,
}

fn present(item: &str, candidate: &str, candidate_lower: &str) -> bool {
    if candidate.contains(item) {
        return true;
    }
    let probe = item.trim();
    let end = floor_char_boundary(probe, 80);
    candidate_lower.contains(&probe[..end].to_lowercase())
}

fn exit_codes_in(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    for caps in EXIT_CODE_RE.captures_iter(text) {
        if let Some(code) = caps.get(1) {
            dedupe_push(&mut out, code.as_str(), 64);
        }
    }
    out
}

/// Remove Headroom/compressor bookkeeping before value checks.
pub fn strip_annotations(text: &str) -> String {
    let text = BRACKET_ANNOTATION_RE.replace_all(text, " ");
    ANNOTATION_LINE_RE.replace_all(&text, " ").into_owned()
}

impl InvariantSet {
    /// Extract invariants from `original` (bounded scan).
    pub fn extract(original: &str, entities: &[String]) -> Self {
        let scan: std::borrow::Cow<'_, str> = if original.len() > MAX_SCAN {
            let head = floor_char_boundary(original, MAX_SCAN / 2);
            let tail = floor_char_boundary(original, original.len() - MAX_SCAN / 2);
            format!("{}\n{}", &original[..head], &original[tail..]).into()
        } else {
            original.into()
        };
        let lowered = scan.to_lowercase();
        let mut set = InvariantSet::default();
        for e in entities {
            if e.len() >= 2 && lowered.contains(&e.to_lowercase()) {
                dedupe_push(&mut set.user_entities, e, 32);
            }
        }
        for m in ERROR_LINE_RE.find_iter(&scan) {
            let line = m.as_str().trim();
            let end = floor_char_boundary(line, 160);
            dedupe_push(&mut set.error_lines, &line[..end], 64);
        }
        set.exit_codes = exit_codes_in(&scan);
        for m in TEST_SUMMARY_RE.find_iter(&scan) {
            dedupe_push(&mut set.test_summaries, m.as_str().trim(), 32);
        }
        set
    }

    /// First hard-veto violated by `candidate`, or `None` when it is safe.
    ///
    /// `recoverable` = the exact original is retrievable (verified CCR
    /// marker): omissions are then allowed, value changes never are.
    pub fn violation(&self, candidate: &str, recoverable: bool) -> Option<&'static str> {
        let lower = candidate.to_lowercase();
        if !recoverable
            && self
                .user_entities
                .iter()
                .any(|e| !present(e, candidate, &lower))
        {
            return Some("user_entity_dropped");
        }
        if !self.exit_codes.is_empty() {
            let claimed = exit_codes_in(&strip_annotations(candidate));
            if claimed.iter().any(|c| !self.exit_codes.contains(c)) {
                return Some("exit_code_changed");
            }
            if !recoverable {
                let reported = exit_codes_in(candidate);
                if self.exit_codes.iter().any(|c| !reported.contains(c)) {
                    return Some("exit_code_dropped");
                }
            }
        }
        if !recoverable {
            if self
                .error_lines
                .iter()
                .any(|e| !present(e, candidate, &lower))
            {
                return Some("error_signal_dropped");
            }
            if self
                .test_summaries
                .iter()
                .any(|t| !present(t, candidate, &lower))
            {
                return Some("test_summary_dropped");
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn env<'a>(pairs: &'a [(&'a str, &'a str)]) -> impl Fn(&str) -> Option<String> + 'a {
        move |name| {
            pairs
                .iter()
                .find(|(k, _)| *k == name)
                .map(|(_, v)| (*v).to_owned())
        }
    }

    #[test]
    fn settings_default_off_and_posture_aliases() {
        assert_eq!(
            IntelligenceSettings::resolve(env(&[])),
            IntelligenceSettings::default()
        );
        let on = IntelligenceSettings::resolve(env(&[("HEADROOM_INTELLIGENCE", "on")]));
        assert!(on.task_query && on.invariant_guard && on.policy_budget);
        let full = IntelligenceSettings::resolve(env(&[("HEADROOM_INTELLIGENCE", "FULL")]));
        assert!(full.any());
        let bogus = IntelligenceSettings::resolve(env(&[("HEADROOM_INTELLIGENCE", "maybe")]));
        assert!(!bogus.any());
    }

    #[test]
    fn per_feature_vars_override_posture_both_ways() {
        let s = IntelligenceSettings::resolve(env(&[
            ("HEADROOM_INTELLIGENCE", "safe"),
            ("HEADROOM_POLICY_RISK_BUDGET", "0"),
        ]));
        assert!(s.task_query && s.invariant_guard && !s.policy_budget);
        let s = IntelligenceSettings::resolve(env(&[("HEADROOM_TASK_QUERY", "yes")]));
        assert!(s.task_query && !s.invariant_guard && !s.policy_budget);
    }

    #[test]
    fn relevance_query_skips_tool_result_only_turns_and_reminders() {
        let messages = vec![
            json!({"role": "user", "content": [
                {"type": "text", "text": "<system-reminder>ctx</system-reminder>fix `parse_header` in src/http.rs"}
            ]}),
            json!({"role": "assistant", "content": [{"type": "tool_use", "id": "t", "name": "Bash", "input": {}}]}),
            json!({"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t", "content": "ok"}]}),
        ];
        assert_eq!(
            relevance_query(&messages),
            "fix `parse_header` in src/http.rs"
        );
        let responses = vec![
            json!({"type": "message", "role": "user", "content": [{"type": "input_text", "text": "why is svc-007 failing"}]}),
            json!({"type": "function_call_output", "call_id": "c", "output": "x"}),
        ];
        assert_eq!(relevance_query(&responses), "why is svc-007 failing");
        assert_eq!(relevance_query(&[]), "");
    }

    #[test]
    fn explicit_entities_cover_quotes_paths_and_identifiers() {
        let e = explicit_entities(
            "fix `parse_header` in src/http.rs, see e.g ConfigLoader and \"exact phrase\"",
        );
        for want in [
            "parse_header",
            "src/http.rs",
            "ConfigLoader",
            "exact phrase",
        ] {
            assert!(e.iter().any(|x| x == want), "missing {want}: {e:?}");
        }
        assert!(!e.iter().any(|x| x == "e.g"));
    }

    #[test]
    fn guard_vetoes_unrecoverable_signal_loss_and_value_changes() {
        let original = "running tests\nFAILED tests/test_x.py::test_a - AssertionError\n3 passed, 1 failed\nProcess exited with code 1\n";
        let set = InvariantSet::extract(original, &["test_a".to_owned()]);
        assert_eq!(set.exit_codes, vec!["1"]);
        assert!(!set.error_lines.is_empty() && !set.test_summaries.is_empty());
        assert_eq!(
            set.violation("all good", false),
            Some("user_entity_dropped")
        );
        let kept = "FAILED tests/test_x.py::test_a - AssertionError\n3 passed, 1 failed\nProcess exited with code 1";
        assert_eq!(set.violation(kept, false), None);
        // Omission is fine when the original is recoverable…
        assert_eq!(set.violation("test_a summary <<ccr:abcdef12>>", true), None);
        // …but a changed value never is.
        assert_eq!(
            set.violation("test_a\nProcess exited with code 2\n<<ccr:abcdef12>>", true),
            Some("exit_code_changed")
        );
        let no_errors = "test_a\n3 passed, 1 failed\nProcess exited with code 1";
        assert_eq!(
            set.violation(no_errors, false),
            Some("error_signal_dropped")
        );
    }

    #[test]
    fn annotations_are_not_value_claims() {
        let set = InvariantSet::extract("exit code: 0\n", &[]);
        // A compressor's bookkeeping line is not a changed exit code.
        let cand = "exit code: 0\n[42 items compressed, exit code: 9 omitted]";
        assert_eq!(set.violation(cand, false), None);
    }

    #[test]
    fn extraction_is_bounded_and_utf8_safe() {
        let big = "é".repeat(MAX_SCAN);
        let set = InvariantSet::extract(&big, &[]);
        assert!(set.error_lines.is_empty());
    }
}
