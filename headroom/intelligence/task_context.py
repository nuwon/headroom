"""TaskContext: one shared definition of "what matters right now" (plan §5).

Every relevance-aware decision in the intelligence layer — query-conditioned
compression, invariant extraction, arbiter scoring, indexed CCR retrieval,
budget allocation — reads the same :class:`TaskContext`, built once per
request from the provider-normalized message list.

Nothing here invents intent: ``current_user_goal_text`` is the latest explicit
user text plus structural context (tool names/inputs). Model reasoning text is
never treated as a goal.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from .messages import (
    ToolCall,
    build_tool_call_index,
    is_tool_result_only,
    iter_tool_results,
    message_text,
)

# ---------------------------------------------------------------------------
# Entity extraction
# ---------------------------------------------------------------------------

# Unix and Windows paths with at least one separator and a plausible filename,
# or a bare filename with a known extension.
_PATH_RE = re.compile(
    r"(?:(?<![\w/\\.])(?:[A-Za-z]:[\\/]|\.{1,2}[\\/]|[\\/]|~[\\/])?"
    r"(?:[\w.@+-]+[\\/])+[\w.@+-]+)"
    r"|(?<![\w/\\.-])[\w-]+\.(?:py|pyi|rs|ts|tsx|js|jsx|mjs|cjs|go|java|kt|rb|php|c|h|cc|cpp|hpp|cs|"
    r"swift|scala|sh|bash|zsh|ps1|toml|yaml|yml|json|md|txt|sql|lock|cfg|ini|xml|html|css|vue|svelte)\b"
)
_URL_RE = re.compile(r"\bhttps?://[^\s<>\"'`)\]]+")
_UUID_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)
_HEX_HASH_RE = re.compile(r"\b(?=[0-9a-f]*[a-f])(?=[0-9a-f]*\d)[0-9a-f]{7,64}\b")
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_NUMERIC_ID_RE = re.compile(r"(?<![\w.])#?\d{4,}(?![\w.])")
# Symbols: CamelCase, snake_case with an underscore, dotted.qualified names,
# and call-shaped tokens foo(…).
_SYMBOL_RE = re.compile(
    r"\b(?:[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]*)+"  # CamelCase
    r"|[a-z][a-z0-9]*(?:_[a-z0-9]+)+"  # snake_case
    r"|[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*){1,4}(?=\()"  # a.b.c(
    r"|[A-Za-z_][A-Za-z0-9_]{2,}(?=\())"  # name(
)
_BACKTICK_RE = re.compile(r"`([^`\n]{2,120})`")
_QUOTED_RE = re.compile(r"\"([^\"\n]{2,120})\"|'([^'\n]{3,120})'")
_ERROR_LINE_RE = re.compile(
    r"^.*(?:\b(?:Error|Exception|Traceback|FAILED|FAIL|panic(?:ked)?|fatal|"
    r"assert(?:ion)?(?:Error)? failed|AssertionError|Segmentation fault|"
    r"error\[E\d+\]|error:|ERROR|CRITICAL|undefined reference|cannot find)\b).*$",
    re.MULTILINE,
)
_EXIT_CODE_RE = re.compile(
    r"(?:exit(?:ed)?(?: with)?(?: code| status)?[:= ]+|exit_code[\"']?\s*[:=]\s*|"
    r"returned non-zero exit status |status code[:= ]+|Process exited with code )(-?\d{1,3})\b",
    re.IGNORECASE,
)
_NUMBER_UNIT_RE = re.compile(
    r"(?<![\w.])[-+]?\$?\d+(?:[.,]\d+)*\s?"
    r"(?:%|ms|s|sec|secs|seconds|min|mins|minutes|h|hrs|hours|days|"
    r"[KMGT]i?B|bytes|B|kb|mb|gb|tb|px|em|rem|GHz|MHz|Hz|°C|°F|"
    r"USD|EUR|tokens|items|rows|lines|files|tests|passed|failed|errors|warnings)\b"
)
_FILE_LINE_RE = re.compile(r"(?:[\w./\\-]+\.[A-Za-z0-9]{1,6}):(\d+)(?::(\d+))?")

_STOP_WORDS = frozenset(
    "a an and are as at be been but by can could did do does for from had has have how i if in "
    "into is it its me my no not of on or our please should so than that the their them then "
    "there these this those to too up us was we were what when where which who why will with "
    "would you your just also make sure let lets now get got".split()
)


@dataclass(frozen=True)
class Entities:
    paths: tuple[str, ...] = ()
    symbols: tuple[str, ...] = ()
    identifiers: tuple[str, ...] = ()
    urls: tuple[str, ...] = ()
    hashes: tuple[str, ...] = ()
    quoted: tuple[str, ...] = ()
    error_signals: tuple[str, ...] = ()
    exit_codes: tuple[str, ...] = ()
    numbers_with_units: tuple[str, ...] = ()
    file_locations: tuple[str, ...] = ()

    def exact_terms(self) -> list[str]:
        """Terms that must match exactly (never dropped before prose)."""
        return _dedupe(
            [
                *self.quoted,
                *self.paths,
                *self.identifiers,
                *self.hashes,
                *self.urls,
                *self.symbols,
                *self.file_locations,
            ]
        )


def _dedupe(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        key = item.strip()
        if key and key not in seen:
            seen.add(key)
            out.append(key)
    return out


def _clean_path(p: str) -> str:
    return p.strip().rstrip(".,;:)]}'\"")


def extract_entities(text: str, *, max_items: int = 64) -> Entities:
    """Extract exact-match entities from ``text`` (bounded, deterministic)."""
    if not text:
        return Entities()
    sample = text if len(text) <= 200_000 else text[:100_000] + text[-100_000:]
    urls = _dedupe(_URL_RE.findall(sample))[:max_items]
    url_set = set(urls)
    paths = [
        _clean_path(m.group(0))
        for m in _PATH_RE.finditer(sample)
        if not any(m.group(0) in u for u in url_set)
    ]
    paths = [p for p in _dedupe(paths) if len(p) > 2 and not p.startswith("//")][:max_items]
    uuids = _UUID_RE.findall(sample)
    hashes = _dedupe(h for h in _HEX_HASH_RE.findall(sample))[:max_items]
    emails = _EMAIL_RE.findall(sample)
    numeric_ids = [n.lstrip("#") for n in _NUMERIC_ID_RE.findall(sample)]
    identifiers = _dedupe([*uuids, *emails, *numeric_ids])[:max_items]
    symbols = _dedupe(_SYMBOL_RE.findall(sample))[:max_items]
    quoted: list[str] = []
    for m in _BACKTICK_RE.finditer(sample):
        quoted.append(m.group(1))
    for m in _QUOTED_RE.finditer(sample):
        quoted.append(m.group(1) or m.group(2) or "")
    errors = _dedupe(line.strip()[:240] for line in _ERROR_LINE_RE.findall(sample))[:max_items]
    exit_codes = _dedupe(_EXIT_CODE_RE.findall(sample))[:max_items]
    numbers = _dedupe(m.group(0).strip() for m in _NUMBER_UNIT_RE.finditer(sample))[:max_items]
    file_locations = _dedupe(m.group(0) for m in _FILE_LINE_RE.finditer(sample))[:max_items]
    return Entities(
        paths=tuple(paths),
        symbols=tuple(symbols),
        identifiers=tuple(identifiers),
        urls=tuple(urls),
        hashes=tuple(hashes),
        quoted=tuple(_dedupe(quoted)[:max_items]),
        error_signals=tuple(errors),
        exit_codes=tuple(exit_codes),
        numbers_with_units=tuple(numbers),
        file_locations=tuple(file_locations),
    )


# ---------------------------------------------------------------------------
# Partial / truncated input provenance (plan §5.3)
# ---------------------------------------------------------------------------

_TRUNCATION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "claude_code_read_limit",
        re.compile(r"File content \(\d[\d,]*\s*tokens?\) exceeds maximum allowed", re.I),
    ),
    ("lines_truncated", re.compile(r"\[\s*\.{3}\s*\d[\d,]*\s+lines? truncated\s*\.{3}\s*\]", re.I)),
    ("omitted_lines", re.compile(r"\[\.{3}\s*omitted \d[\d,]* of \d[\d,]* lines\s*\.{3}\]", re.I)),
    (
        "output_truncated",
        re.compile(r"\(?\b(?:output|content|response|result)s? (?:was )?truncated\b", re.I),
    ),
    (
        "truncated_marker",
        re.compile(r"\[\s*truncated\b[^\]]*\]|<truncated[^>]*>|\.{3}\s*truncated\s*\.{3}", re.I),
    ),
    (
        "chars_truncated",
        re.compile(r"\b\d[\d,]* (?:characters|chars|bytes) (?:truncated|omitted)\b", re.I),
    ),
    (
        "showing_first",
        re.compile(
            r"\b(?:showing|displaying) (?:the )?first \d[\d,]* (?:of \d[\d,]* )?(?:lines|results|items|matches)\b",
            re.I,
        ),
    ),
    (
        "more_results",
        re.compile(r"\b(?:and )?\d[\d,]* more (?:results|matches|lines|items|files)\b", re.I),
    ),
    (
        "pagination",
        re.compile(
            r"\b(?:next_page|next_cursor|has_more|nextPageToken)\b[\"']?\s*[:=]\s*(?:true|\"[^\"]+\"|'[^']+')",
            re.I,
        ),
    ),
)
_RANGE_INPUT_KEYS = (
    "offset",
    "limit",
    "line_range",
    "start_line",
    "end_line",
    "ranges",
    "head_limit",
)


@dataclass(frozen=True)
class Provenance:
    is_partial: bool = False
    reason: str = ""
    truncation_boundary: str = ""  # the verbatim notice text, preserved by transforms

    @property
    def complete(self) -> bool:
        return not self.is_partial


def detect_provenance(
    text: str, tool_name: str = "", tool_input: dict[str, Any] | None = None
) -> Provenance:
    """Did Headroom see the complete source for this tool result?

    A positive answer must be *proven*: any truncation notice, pagination
    cursor or range-limited invocation marks the input partial.
    """
    tool_input = tool_input or {}
    for key in _RANGE_INPUT_KEYS:
        value = tool_input.get(key)
        if value not in (None, "", 0, [], {}):
            return Provenance(True, f"range_input:{key}", "")
    if not text:
        return Provenance()
    # Notices sit at the head or tail; scanning the ends keeps this O(1)-ish
    # on very large outputs while still catching the in-body "[... N lines
    # truncated ...]" form most tools place near the cut.
    probe = text if len(text) <= 16_000 else text[:8_000] + "\n" + text[-8_000:]
    for reason, pattern in _TRUNCATION_PATTERNS:
        m = pattern.search(probe)
        if m:
            return Provenance(True, reason, m.group(0)[:200])
    return Provenance()


# ---------------------------------------------------------------------------
# TaskContext
# ---------------------------------------------------------------------------


def _approx_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4) if text else 0


_IMPORTANT_INPUT_KEYS = (
    "pattern",
    "query",
    "q",
    "regex",
    "search",
    "file_path",
    "path",
    "paths",
    "filename",
    "command",
    "cmd",
    "glob",
    "url",
    "symbol",
    "name",
    "id",
    "description",
)


def summarize_tool_input(tool_input: dict[str, Any], *, max_chars: int = 300) -> str:
    """Compact, deterministic summary of the important tool input fields."""
    if not tool_input:
        return ""
    parts: list[str] = []
    for key in _IMPORTANT_INPUT_KEYS:
        if key in tool_input:
            value = tool_input[key]
            if isinstance(value, (list, tuple)):
                value = " ".join(str(v) for v in value[:8])
            text = str(value).strip()
            if text:
                parts.append(f"{key}={text}")
    if not parts:
        for key in sorted(tool_input)[:4]:
            value = tool_input[key]
            if isinstance(value, (str, int, float)):
                parts.append(f"{key}={value}")
    summary = " ".join(parts)
    return summary[:max_chars]


@dataclass(frozen=True)
class TaskContext:
    """Provider-independent description of the current task (plan §5.1)."""

    workspace_key: str = ""
    provider: str = ""
    model: str = ""
    request_id: str = ""
    current_user_text: str = ""
    current_user_goal_text: str = ""
    tool_name: str = ""
    tool_call_id: str = ""
    tool_input_summary: str = ""
    file_paths: tuple[str, ...] = ()
    symbols: tuple[str, ...] = ()
    identifiers: tuple[str, ...] = ()
    urls: tuple[str, ...] = ()
    hashes: tuple[str, ...] = ()
    quoted: tuple[str, ...] = ()
    error_signals: tuple[str, ...] = ()
    exit_codes: tuple[str, ...] = ()
    numeric_literals_with_units: tuple[str, ...] = ()
    requested_operations: tuple[str, ...] = ()
    graph_neighbors: tuple[str, ...] = ()
    turn_kind: str = ""
    recent_tool_names: tuple[str, ...] = ()
    explicit_entities: tuple[str, ...] = ()
    cache_state: dict[str, Any] = field(default_factory=dict)

    @property
    def fingerprint(self) -> str:
        h = hashlib.sha256()
        h.update(self.relevance_query().encode("utf-8", "replace"))
        return h.hexdigest()[:16]

    def relevance_query(self, max_tokens: int = 192) -> str:
        """Canonical relevance query (plan §5.2), capped by tokens.

        Order: user text, tool name + important inputs, exact entities,
        unresolved errors, graph neighbors. Exact identifiers are never
        dropped before prose terms: the exact-term section is budgeted first
        and the prose is trimmed to whatever remains.
        """
        exact = list(self.explicit_entities)
        exact_text = " ".join(exact)
        tool_text = " ".join(p for p in (self.tool_name, self.tool_input_summary) if p)
        error_text = " ".join(self.error_signals[:3])
        graph_text = " ".join(self.graph_neighbors[:12])

        budget = max(16, max_tokens)
        sections: list[str] = []
        # Reserve exact terms first (cap at 60% of the budget so a pathological
        # entity list cannot starve the user's words entirely).
        exact_budget = min(_word_cost(exact_text), int(budget * 0.6))
        exact_kept = _truncate_tokens_words(exact_text, exact_budget)
        remaining = budget - _approx_tokens(exact_kept)
        prose = self.current_user_text.strip()
        prose_kept = _truncate_tokens_words(prose, max(0, int(remaining * 0.7)))
        if prose_kept:
            sections.append(prose_kept)
            remaining -= _approx_tokens(prose_kept)
        if tool_text and remaining > 0:
            kept = _truncate_tokens_words(tool_text, remaining)
            if kept:
                sections.append(kept)
                remaining -= _approx_tokens(kept)
        if exact_kept:
            sections.append(exact_kept)
        if error_text and remaining > 0:
            kept = _truncate_tokens_words(error_text, remaining)
            if kept:
                sections.append(kept)
                remaining -= _approx_tokens(kept)
        if graph_text and remaining > 0:
            kept = _truncate_tokens_words(graph_text, remaining)
            if kept:
                sections.append(kept)
        return "\n".join(sections)

    def block_query(self, tool_name: str = "", tool_input: dict[str, Any] | None = None) -> str:
        """Per-tool-result query: the task query plus that call's own inputs."""
        base = self.relevance_query()
        call = " ".join(p for p in (tool_name.strip(), summarize_tool_input(tool_input or {})) if p)
        return f"{base}\n{call}" if call and call not in base else base

    def relevance_bias_for(self, text: str) -> float:
        """Task-derived retention bias for a block (plan §7, item 5).

        >1.0 keeps more. Explicit user-named entities and active error
        locations are high; graph neighbors medium; unrelated content 1.0.
        """
        if not text:
            return 1.0
        bias = 1.0
        lowered = text.lower()
        named = [e for e in self.explicit_entities if len(e) >= 3]
        if any(e.lower() in lowered for e in named):
            bias = max(bias, 1.6)
        locations = [s for s in self.error_signals if len(s) >= 8]
        if any(loc.lower()[:60] in lowered for loc in locations):
            bias = max(bias, 1.5)
        if any(n.lower() in lowered for n in self.graph_neighbors if len(n) >= 3):
            bias = max(bias, 1.25)
        return bias


def _word_cost(text: str) -> int:
    return sum(_approx_tokens(w) + 1 for w in text.split())


def _truncate_tokens_words(text: str, max_tokens: int) -> str:
    if max_tokens <= 0 or not text:
        return ""
    if _approx_tokens(text) <= max_tokens:
        return text
    words = text.split()
    out: list[str] = []
    used = 0
    for word in words:
        cost = _approx_tokens(word) + 1
        if used + cost > max_tokens:
            break
        out.append(word)
        used += cost
    return " ".join(out)


_OPERATION_WORDS = (
    "fix",
    "debug",
    "explain",
    "refactor",
    "test",
    "implement",
    "add",
    "remove",
    "rename",
    "review",
    "find",
    "search",
    "list",
    "show",
    "run",
    "build",
    "deploy",
    "optimize",
    "document",
    "compare",
    "summarize",
    "migrate",
    "update",
    "delete",
    "create",
)


def _requested_operations(text: str) -> tuple[str, ...]:
    lowered = text.lower()
    return tuple(w for w in _OPERATION_WORDS if re.search(rf"\b{w}\w*\b", lowered))


def build_task_context(
    messages: list[dict[str, Any]],
    *,
    provider: str = "",
    model: str = "",
    workspace_key: str = "",
    request_id: str = "",
    graph_neighbors: Iterable[str] = (),
    max_user_chars: int = 4000,
) -> TaskContext:
    """Build the :class:`TaskContext` for a request.

    The *current* user text is the newest user message that carries prompt
    text (tool-result-only continuation turns are skipped, so a long agentic
    loop keeps conditioning on the task the user actually gave). The current
    tool is the newest tool invocation.
    """
    call_index = build_tool_call_index(messages)
    user_text = ""
    turn_kind = "unknown"
    if messages and isinstance(messages[-1], dict) and messages[-1].get("role") == "tool":
        turn_kind = "tool_continuation"  # OpenAI chat / Responses (Codex) shape
    for idx in range(len(messages) - 1, -1, -1):
        msg = messages[idx]
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        if idx == len(messages) - 1:
            turn_kind = "tool_continuation" if is_tool_result_only(msg) else "new_user_ask"
        if is_tool_result_only(msg):
            continue
        text = _strip_harness_noise(message_text(msg))
        if text.strip():
            user_text = text.strip()
            break
    # Entities come from the FULL user text (extract_entities bounds its own
    # scan); only the prose kept for the query is capped.
    full_user_text = user_text
    user_text = user_text[:max_user_chars]

    latest_call: ToolCall | None = None
    if call_index:
        latest_call = max(call_index.values(), key=lambda c: c.message_index)
    recent_tools = tuple(
        c.name
        for c in sorted(call_index.values(), key=lambda c: c.message_index, reverse=True)[:6]
        if c.name
    )

    user_entities = extract_entities(full_user_text)
    tool_summary = summarize_tool_input(latest_call.input) if latest_call else ""
    tool_entities = extract_entities(tool_summary) if tool_summary else Entities()

    # Unresolved errors from the immediately relevant (newest) tool results.
    error_signals: list[str] = []
    exit_codes: list[str] = []
    newest_results = list(iter_tool_results(messages, call_index, start=max(0, len(messages) - 2)))
    for ref in newest_results:
        ents = extract_entities(ref.text[-20_000:], max_items=8)
        error_signals.extend(ents.error_signals[:4])
        exit_codes.extend(ents.exit_codes[:2])

    explicit = _dedupe(
        [
            *user_entities.quoted,
            *user_entities.paths,
            *user_entities.identifiers,
            *user_entities.hashes,
            *user_entities.urls,
            *user_entities.symbols,
            *user_entities.file_locations,
            *tool_entities.paths,
            *tool_entities.identifiers,
        ]
    )[:48]

    goal = user_text
    if latest_call is not None:
        goal = f"{user_text}\n[tool:{latest_call.name}] {tool_summary}".strip()

    return TaskContext(
        workspace_key=workspace_key,
        provider=provider,
        model=model,
        request_id=request_id,
        current_user_text=user_text,
        current_user_goal_text=goal,
        tool_name=latest_call.name if latest_call else "",
        tool_call_id=latest_call.call_id if latest_call else "",
        tool_input_summary=tool_summary,
        file_paths=tuple(_dedupe([*user_entities.paths, *tool_entities.paths])),
        symbols=user_entities.symbols,
        identifiers=user_entities.identifiers,
        urls=user_entities.urls,
        hashes=user_entities.hashes,
        quoted=user_entities.quoted,
        error_signals=tuple(_dedupe(error_signals)[:8]),
        exit_codes=tuple(_dedupe(exit_codes)[:4]),
        numeric_literals_with_units=user_entities.numbers_with_units,
        requested_operations=_requested_operations(user_text),
        graph_neighbors=tuple(_dedupe(graph_neighbors)[:24]),
        turn_kind=turn_kind,
        recent_tool_names=recent_tools,
        explicit_entities=tuple(explicit),
    )


_SYSTEM_REMINDER_RE = re.compile(r"<system-reminder>.*?</system-reminder>", re.S)


def _strip_harness_noise(text: str) -> str:
    return _SYSTEM_REMINDER_RE.sub("", text)


EMPTY_TASK_CONTEXT = TaskContext()
