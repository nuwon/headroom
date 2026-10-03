"""Complexity-aware reasoning and output shaping (Optimization 10, plan §16).

``output_turn_policy.classify_turn`` is purely structural: any clean
tool-result continuation is "mechanical". A clean tool result can still carry
a failing invariant, contradictory evidence or a design discovery. This module
adds a semantic layer and separates two independent decisions:

* **reasoning effort** — how hard the model should think;
* **verbosity** — how much it should write.

A turn can legitimately be *high effort + low verbosity* (a subtle test
failure) or *low effort + low verbosity* (a file write succeeded).

Deterministic rules (never relaxed by the advisor):
effort is never lowered on a new user ask, an error/failure, conflicting
evidence, repeated failed attempts, a tool loop, or when the result introduces
new central files/symbols. JevK5 (``turn_complexity`` score 0–4) may raise
complexity readily; lowering needs a confident, agreeing answer.

Effort routing is applied only when explicitly enabled
(``HEADROOM_EFFORT_ROUTING=1``), only to effort fields the client already sent
(Anthropic ``output_config.effort``, OpenAI ``reasoning_effort`` /
``reasoning.effort``) and with per-conversation hysteresis: Headroom measured
that switching effort every turn costs more in prompt-cache rewrites than it
saves, so a conversation's routed level changes only on strong evidence.
"""

from __future__ import annotations

import hashlib
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from .output_turn_policy import TurnKind, classify_turn

_EXPLAIN_RE = re.compile(
    r"\b(?:why|explain|explanation|walk me through|how does|how do|what does|reason(?:ing)?|"
    r"teach|elaborate|in detail|detailed|compare|tradeoffs?|pros and cons|design|architecture)\b",
    re.I,
)
_TERSE_RE = re.compile(r"\b(?:just|only|briefly|short|tl;?dr|concise|quick(?:ly)?)\b", re.I)
_ERROR_RE = re.compile(
    r"\b(?:Traceback|Exception|Error:|ERROR|FAILED|FAIL\b|panic|fatal|assert(?:ion)? ?(?:error|failed))|exit (?:code|status)[: ]+[1-9]",
)
_PASS_RE = re.compile(r"\b(?:passed|success(?:ful)?|ok)\b", re.I)
_FAIL_RE = re.compile(r"\b(?:failed|failure|error)\b", re.I)
_IDENT_RE = re.compile(
    r"\b[A-Za-z_][A-Za-z0-9_]*(?:[./][A-Za-z_][A-Za-z0-9_]*)+\b|\b[A-Z][a-z]+(?:[A-Z][a-z0-9]*)+\b"
)
_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\s?(?:%|ms|s|MB|GB|KB|tokens|items)?\b")
_MUTATING_TOOLS = frozenset(
    {"edit", "write", "multiedit", "apply_patch", "str_replace_editor", "create_file"}
)
LEVELS = (
    "trivial continuation",
    "routine deterministic step",
    "moderate reasoning",
    "substantial reasoning",
    "high-risk/high-ambiguity reasoning",
)


@dataclass
class TurnComplexity:
    structural_turn_kind: str
    new_user_ask: bool = False
    error_present: bool = False
    new_error_signature: bool = False
    conflicting_evidence: bool = False
    novel_identifiers_count: int = 0
    changed_files: int = 0
    retrieval_count: int = 0
    failed_attempt_count: int = 0
    repeated_tool_loop_count: int = 0
    numeric_density: float = 0.0
    explanation_requested: bool = False
    terse_requested: bool = False
    score: int = 0
    advised_score: float | None = None
    reasons: list[str] = field(default_factory=list)

    @property
    def protected(self) -> bool:
        """True when effort must never be lowered (deterministic rules)."""
        return (
            self.new_user_ask
            or self.error_present
            or self.conflicting_evidence
            or self.failed_attempt_count >= 2
            or self.repeated_tool_loop_count >= 2
            or self.novel_identifiers_count >= 8
        )

    @property
    def semantic_turn_kind(self) -> TurnKind:
        """Mechanical only when structurally mechanical AND low complexity."""
        kind = TurnKind(self.structural_turn_kind)
        if kind is TurnKind.MECHANICAL_CONTINUATION and (self.score >= 2 or self.protected):
            return TurnKind.ERROR_CONTINUATION if self.error_present else TurnKind.UNKNOWN
        return kind

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if k != "reasons"} | {
            "reasons": list(self.reasons)
        }


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
                elif block.get("type") == "tool_result":
                    c = block.get("content")
                    parts.append(c if isinstance(c, str) else _text_of(c))
        return "\n".join(parts)
    return ""


def _tool_calls(msg: dict[str, Any]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    content = msg.get("content")
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                out.append((str(block.get("name") or ""), str(block.get("input"))))
    for call in msg.get("tool_calls") or []:
        if isinstance(call, dict):
            fn = call.get("function") or {}
            out.append((str(fn.get("name") or ""), str(fn.get("arguments"))))
    return out


def assess_turn(messages: list[dict[str, Any]], *, history_window: int = 24) -> TurnComplexity:
    """Deterministic complexity features + score (0–4) for the latest turn."""
    structural = classify_turn(messages)
    last = messages[-1] if messages and isinstance(messages[-1], dict) else {}
    latest_text = _text_of(last.get("content"))
    if structural is TurnKind.UNKNOWN and last.get("role") == "tool":
        # OpenAI chat / Responses (Codex) shape: a trailing tool message is a
        # tool continuation, structurally an error one if it reads as a failure.
        structural = (
            TurnKind.ERROR_CONTINUATION
            if _ERROR_RE.search(latest_text)
            else TurnKind.MECHANICAL_CONTINUATION
        )
    tc = TurnComplexity(structural_turn_kind=structural.value)
    if not messages:
        return tc
    tc.new_user_ask = structural is TurnKind.NEW_USER_ASK
    if tc.new_user_ask:
        tc.explanation_requested = bool(_EXPLAIN_RE.search(latest_text))
        tc.terse_requested = bool(_TERSE_RE.search(latest_text)) and not tc.explanation_requested
    else:
        # Explanation intent comes from the newest prompt text, not tool output.
        for msg in reversed(messages[:-1]):
            if isinstance(msg, dict) and msg.get("role") == "user":
                text = _text_of(msg.get("content")) if isinstance(msg.get("content"), str) else ""
                if text.strip():
                    tc.explanation_requested = bool(_EXPLAIN_RE.search(text))
                    break

    window = [m for m in messages[-history_window:-1] if isinstance(m, dict)]
    history_text = "\n".join(_text_of(m.get("content")) for m in window)
    errors_now = {m.group(0) for m in _ERROR_RE.finditer(latest_text)}
    tc.error_present = bool(errors_now) or structural is TurnKind.ERROR_CONTINUATION
    if tc.error_present:
        prior_errors = {m.group(0) for m in _ERROR_RE.finditer(history_text)}
        new_lines = [
            ln
            for ln in latest_text.splitlines()
            if _ERROR_RE.search(ln) and ln.strip() not in history_text
        ]
        tc.new_error_signature = bool(new_lines) or not (errors_now <= prior_errors)
    if (
        _PASS_RE.search(latest_text)
        and _FAIL_RE.search(latest_text)
        and re.search(r"exit (?:code|status)[: ]+0\b", latest_text)
    ):
        tc.conflicting_evidence = True  # "success" exit with failures reported
    if re.search(r"\b0 failed\b", latest_text) and re.search(r"\bFAILED\b", latest_text):
        tc.conflicting_evidence = True
    idents_now = set(_IDENT_RE.findall(latest_text[:200_000]))
    tc.novel_identifiers_count = sum(1 for i in idents_now if i not in history_text)
    nums = _NUMBER_RE.findall(latest_text[:50_000])
    tc.numeric_density = round(len(nums) / max(1, len(latest_text[:50_000]) / 100), 3)

    calls = [c for m in window if m.get("role") == "assistant" for c in _tool_calls(m)]
    tc.changed_files = sum(1 for name, _ in calls[-6:] if name.lower() in _MUTATING_TOOLS)
    tc.retrieval_count = sum(1 for name, _ in calls if name.endswith("headroom_retrieve"))
    signatures = [f"{n}:{hashlib.sha1(a.encode()).hexdigest()[:8]}" for n, a in calls]
    if signatures:
        last_sig = signatures[-1]
        tc.repeated_tool_loop_count = sum(1 for s in signatures[-8:] if s == last_sig) - 1
    consecutive = 0
    for m in reversed(window + [last]):
        if m.get("role") != "user" and m.get("role") != "tool":
            continue
        if _ERROR_RE.search(_text_of(m.get("content"))):
            consecutive += 1
        else:
            break
    tc.failed_attempt_count = consecutive

    score = 0
    if tc.new_user_ask:
        score += 2
        tc.reasons.append("new_user_ask")
    if tc.explanation_requested:
        score += 1
        tc.reasons.append("explanation_requested")
    if tc.error_present:
        score += 1 + int(tc.new_error_signature)
        tc.reasons.append("error")
    if tc.conflicting_evidence:
        score += 2
        tc.reasons.append("conflicting_evidence")
    if tc.failed_attempt_count >= 2 or tc.repeated_tool_loop_count >= 2:
        score += 1
        tc.reasons.append("repeated_failure_or_loop")
    if tc.novel_identifiers_count >= 8:
        score += 1
        tc.reasons.append("novel_identifiers")
    if tc.retrieval_count >= 2:
        score += 1
        tc.reasons.append("context_retrievals")
    tc.score = min(4, score)
    return tc


def advise_complexity(
    tc: TurnComplexity, messages: list[dict[str, Any]], advisor: Any
) -> TurnComplexity:
    """Fuse a JevK5 ``turn_complexity`` score: raise readily, lower cautiously."""
    if advisor is None or tc.score not in (1, 2):
        return tc
    from headroom.intelligence.models import DecisionFamily

    latest = _text_of(messages[-1].get("content")) if messages else ""
    state = f"Latest turn ({tc.structural_turn_kind}):\n{latest[:3000]}\nDeterministic signals: {', '.join(tc.reasons) or 'none'}"
    scores = advisor.score(
        DecisionFamily.TURN_COMPLEXITY,
        state,
        "How much reasoning does the assistant's next step require?",
        LEVELS,
    )
    if scores is None or scores.expected_score is None or scores.weight <= 0:
        return tc
    tc.advised_score = scores.expected_score
    fused = (1 - scores.weight) * tc.score + scores.weight * scores.expected_score
    if fused > tc.score:
        tc.score = min(4, round(fused + 0.25))
    elif not tc.protected and scores.confidence >= 0.8 and scores.expected_score <= 0.5:
        tc.score = max(0, tc.score - 1)
    return tc


# ------------------------------------------------------------------ decisions
EFFORT_ORDER = ("minimal", "low", "medium", "high", "xhigh", "max")


@dataclass(frozen=True)
class EffortDecision:
    target: str | None  # None = leave as is
    reason: str


def decide_effort(tc: TurnComplexity, requested: str | None) -> EffortDecision:
    """Effort for this turn relative to what the client asked for."""
    if not requested or requested not in EFFORT_ORDER:
        return EffortDecision(None, "no_client_effort")
    if tc.score >= 3 and EFFORT_ORDER.index(requested) < EFFORT_ORDER.index("high"):
        return EffortDecision("high", "complex_turn")
    if tc.protected or tc.semantic_turn_kind is not TurnKind.MECHANICAL_CONTINUATION:
        return EffortDecision(None, "protected")
    if tc.score == 0 and requested in ("medium", "high"):
        # An explicit xhigh/max is the user's call; only default-ish levels drop.
        return EffortDecision("low", "mechanical_low_complexity")
    return EffortDecision(None, "keep")


def decide_verbosity(tc: TurnComplexity, current_level: int) -> int:
    """Verbosity level (0–4, higher = terser), independent of effort."""
    if tc.explanation_requested:
        return min(current_level, 2)  # never "conclusions only" when asked why
    if tc.terse_requested:
        return max(current_level, 3)
    return current_level


class EffortRouter:
    """Applies effort decisions with per-conversation hysteresis."""

    _MAX = 2048

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state: OrderedDict[str, tuple[str, int]] = OrderedDict()  # key -> (level, streak)

    def route(self, key: str, decision: EffortDecision, requested: str) -> str | None:
        """Return the effort to send (or None to leave the client's value)."""
        target = decision.target or requested
        with self._lock:
            current, streak = self._state.get(key, (requested, 0))
            if target == current:
                self._state[key] = (current, 0)
                result = current
            elif decision.reason == "complex_turn":
                # Raising for a hard turn is immediate (quality first).
                self._state[key] = (target, 0)
                result = target
            else:
                # Lowering needs two consecutive agreeing turns.
                streak += 1
                if streak >= 2:
                    self._state[key] = (target, 0)
                    result = target
                else:
                    self._state[key] = (current, streak)
                    result = current
            self._state.move_to_end(key)
            while len(self._state) > self._MAX:
                self._state.popitem(last=False)
        return None if result == requested else result


def client_effort(body: dict[str, Any], provider: str) -> str | None:
    if provider == "anthropic":
        oc = body.get("output_config")
        return (
            oc.get("effort") if isinstance(oc, dict) and isinstance(oc.get("effort"), str) else None
        )
    if provider == "openai_chat":
        v = body.get("reasoning_effort")
        return v if isinstance(v, str) else None
    if provider == "openai_responses":
        r = body.get("reasoning")
        return r.get("effort") if isinstance(r, dict) and isinstance(r.get("effort"), str) else None
    return None


def apply_effort(body: dict[str, Any], provider: str, effort: str) -> bool:
    """Set an effort field the client already sent. Never injects one."""
    if client_effort(body, provider) is None:
        return False
    if provider == "anthropic":
        body["output_config"] = {**body["output_config"], "effort": effort}
    elif provider == "openai_chat":
        body["reasoning_effort"] = effort
    elif provider == "openai_responses":
        body["reasoning"] = {**body["reasoning"], "effort": effort}
    else:
        return False
    return True
