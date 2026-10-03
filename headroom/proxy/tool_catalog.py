"""Semantic progressive tool catalog (Optimization 9, plan §15).

For requests that do not get native deferral (OpenAI chat completions, Codex /
OpenCode on the Responses API, Anthropic through a third-party upstream or a
cloud backend), large tool lists ship every full schema on every turn. This
module keeps the *useful* schemas full and turns the rest into compact
catalog entries that remain directly callable:

* name kept; description cut to its first sentence;
* parameters reduced to the **required** properties with their types
  (enums and array item types kept) — optional parameters are omitted;
* typed/server tools, ``strict`` function tools and anything not shaped like a
  plain function are never touched.

Materialized (full) set = core coding tools ∪ tools already used in the
conversation ∪ the top-K tools most relevant to the task (BM25 over
name+description, exact-name mentions boosted; JevK5 may re-rank the top ≤16
ambiguous candidates).

Cache safety: the tools array sits at the front of the prompt, so the set is
*sticky* per conversation. It is chosen once, and grows (a newly used tool
becoming full) only when the prompt cache is cold, so a schema reshuffle never
costs more in cache rewrites than it saves. Native tool search
(``defer_loading`` / ``tool_search``) stays preferred wherever it applies.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import threading
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

_SENTENCE_RE = re.compile(r"^(.+?[.!?])(?:\s|$)", re.S)
_MAX_DESC = 140
_COMPACT_NOTE = " (optional params omitted)"
_SESSIONS_MAX = 512


@dataclass(frozen=True)
class CatalogResult:
    tools: list[Any]
    changed: bool
    materialized: frozenset[str]
    compacted: int
    bytes_before: int
    bytes_after: int
    grew: bool = False

    @property
    def bytes_saved(self) -> int:
        return max(0, self.bytes_before - self.bytes_after)


def _shape(tool: Any) -> str | None:
    """'anthropic' | 'chat' | 'responses' | None (not a plain function tool)."""
    if not isinstance(tool, dict):
        return None
    if tool.get("strict") is True or (
        isinstance(tool.get("function"), dict) and tool["function"].get("strict")
    ):
        return None
    if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
        return "chat"
    if tool.get("type") == "function" and "parameters" in tool and "name" in tool:
        return "responses"
    if "input_schema" in tool and "name" in tool and tool.get("type") in (None, "custom"):
        return "anthropic"
    return None


def tool_name(tool: Any) -> str:
    if not isinstance(tool, dict):
        return ""
    fn = tool.get("function")
    if isinstance(fn, dict):
        return str(fn.get("name") or "")
    return str(tool.get("name") or "")


def _desc(tool: dict[str, Any]) -> str:
    fn = tool.get("function")
    holder = fn if isinstance(fn, dict) else tool
    return str(holder.get("description") or "")


def _schema(tool: dict[str, Any]) -> dict[str, Any] | None:
    fn = tool.get("function")
    if isinstance(fn, dict):
        s = fn.get("parameters")
    else:
        s = tool.get("input_schema", tool.get("parameters"))
    return s if isinstance(s, dict) else None


def _first_sentence(text: str) -> str:
    text = " ".join(text.split())
    m = _SENTENCE_RE.match(text)
    out = m.group(1) if m else text
    if len(out) > _MAX_DESC:
        out = out[: _MAX_DESC - 1].rstrip() + "…"
    return out


def compact_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Required-only projection of a JSON schema (types/enums/items kept)."""
    if not isinstance(schema, dict):
        return {"type": "object", "properties": {}}
    required = [r for r in schema.get("required", []) if isinstance(r, str)]
    raw_props = schema.get("properties")
    props: dict[str, Any] = raw_props if isinstance(raw_props, dict) else {}
    out_props: dict[str, Any] = {}
    for name in required:
        spec = props.get(name)
        if not isinstance(spec, dict):
            out_props[name] = {}
            continue
        small: dict[str, Any] = {}
        for key in ("type", "enum", "const", "format", "anyOf", "oneOf"):
            if key in spec:
                small[key] = spec[key]
        if spec.get("type") == "object" and isinstance(spec.get("properties"), dict):
            small.update(compact_schema(spec))
        if spec.get("type") == "array" and isinstance(spec.get("items"), dict):
            items = spec["items"]
            small["items"] = (
                compact_schema(items)
                if items.get("type") == "object"
                else {k: items[k] for k in ("type", "enum") if k in items}
            )
        out_props[name] = small
    out: dict[str, Any] = {"type": "object", "properties": out_props}
    if required:
        out["required"] = required
    return out


def compact_tool(tool: dict[str, Any]) -> dict[str, Any]:
    shape = _shape(tool)
    if shape is None:
        return tool
    out = copy.deepcopy(tool)
    desc = _first_sentence(_desc(tool)) + _COMPACT_NOTE
    schema = compact_schema(_schema(tool) or {})
    if shape == "chat":
        out["function"]["description"] = desc
        out["function"]["parameters"] = schema
    elif shape == "responses":
        out["description"] = desc
        out["parameters"] = schema
    else:
        out["description"] = desc
        out["input_schema"] = schema
    return out


def used_tool_names(messages: Iterable[dict[str, Any]]) -> set[str]:
    used: set[str] = set()
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                if (
                    isinstance(block, dict)
                    and block.get("type") == "tool_use"
                    and block.get("name")
                ):
                    used.add(str(block["name"]))
        for call in msg.get("tool_calls") or []:
            if isinstance(call, dict):
                fn = call.get("function") or {}
                if fn.get("name"):
                    used.add(str(fn["name"]))
        if msg.get("type") == "function_call" and msg.get("name"):
            used.add(str(msg["name"]))
    return used


def _norm(name: str) -> str:
    return name.lower().lstrip("_")


class ProgressiveToolCatalog:
    def __init__(self, *, top_k: int = 5, min_tools: int = 12) -> None:
        self.top_k = max(1, top_k)
        self.min_tools = max(2, min_tools)
        self._lock = threading.Lock()
        self._sessions: OrderedDict[str, frozenset[str]] = OrderedDict()

    def rank(
        self,
        tools: list[Any],
        query: str,
        *,
        exclude: set[str],
        user_text: str = "",
        advisor: Any | None = None,
        goal: str = "",
    ) -> list[str]:
        """Names of the top-K relevant tools not already in ``exclude``."""
        from headroom.relevance import BM25Scorer

        candidates = [
            t for t in tools if _shape(t) and tool_name(t) and _norm(tool_name(t)) not in exclude
        ]
        if not candidates or not query.strip():
            return []
        docs = [
            f"{tool_name(t).replace('_', ' ')} {tool_name(t)} {_desc(t)[:400]}" for t in candidates
        ]
        try:
            scores = [s.score for s in BM25Scorer().score_batch(docs, query)]
        except Exception:  # noqa: BLE001
            scores = [0.0] * len(docs)
        lowered = user_text.lower()
        for i, t in enumerate(candidates):
            name = tool_name(t)
            if name and name.lower() in lowered:
                scores[i] += 1.0
        order = sorted(range(len(candidates)), key=lambda i: (-scores[i], tool_name(candidates[i])))
        order = [i for i in order if scores[i] > 0]
        if advisor is not None and len(order) > self.top_k:
            shortlist = order[:16]
            options = {
                tool_name(candidates[i]): _first_sentence(_desc(candidates[i]))
                or tool_name(candidates[i])
                for i in shortlist
            }
            from headroom.intelligence.models import DecisionFamily

            advice = advisor.choose(
                DecisionFamily.TOOL_MATERIALIZATION,
                f"Task goal: {goal[:600]}",
                "Which tool is most likely required for the next actions of this task?",
                options,
            )
            if advice is not None and advice.weight > 0:
                top = max(scores[i] for i in shortlist) or 1.0
                fused = {
                    i: (1 - advice.weight) * (scores[i] / top)
                    + advice.weight * advice.probabilities.get(tool_name(candidates[i]), 0.0)
                    for i in shortlist
                }
                order = (
                    sorted(shortlist, key=lambda i: (-fused[i], tool_name(candidates[i])))
                    + order[16:]
                )
        return [tool_name(candidates[i]) for i in order[: self.top_k]]

    def apply(
        self,
        tools: list[Any] | None,
        *,
        messages: list[dict[str, Any]],
        query: str,
        core: frozenset[str],
        session_key: str | None,
        cache_cold: bool,
        user_text: str = "",
        advisor: Any | None = None,
        goal: str = "",
    ) -> CatalogResult:
        tools = list(tools or [])
        before = len(json.dumps(tools, ensure_ascii=False, separators=(",", ":")))
        function_tools = [t for t in tools if _shape(t)]
        if len(function_tools) < self.min_tools:
            return CatalogResult(tools, False, frozenset(), 0, before, before)
        present = {_norm(tool_name(t)) for t in function_tools}
        used = {_norm(n) for n in used_tool_names(messages)}
        key = session_key or _conversation_key(messages, tools)
        grew = False
        with self._lock:
            sticky = self._sessions.get(key)
        if sticky is None:
            base = {_norm(c) for c in core} | used
            top = self.rank(
                tools, query, exclude=base, user_text=user_text, advisor=advisor, goal=goal
            )
            materialized = frozenset((base | {_norm(n) for n in top}) & present)
        else:
            materialized = sticky
            missing = (used & present) - sticky
            if missing and cache_cold:
                materialized = frozenset(sticky | missing)
                grew = True
        with self._lock:
            self._sessions[key] = materialized
            self._sessions.move_to_end(key)
            while len(self._sessions) > _SESSIONS_MAX:
                self._sessions.popitem(last=False)
        out: list[Any] = []
        compacted = 0
        for t in tools:
            if _shape(t) and _norm(tool_name(t)) not in materialized:
                out.append(compact_tool(t))
                compacted += 1
            else:
                out.append(t)
        after = len(json.dumps(out, ensure_ascii=False, separators=(",", ":")))
        if after >= before:
            return CatalogResult(tools, False, materialized, 0, before, before, grew)
        return CatalogResult(out, True, materialized, compacted, before, after, grew)


def _conversation_key(messages: list[dict[str, Any]], tools: list[Any]) -> str:
    """Stable per-conversation key when the caller has no session id."""
    first_user = next(
        (
            json.dumps(m.get("content"), sort_keys=True, ensure_ascii=False)[:2000]
            for m in messages
            if m.get("role") == "user"
        ),
        "",
    )
    names = ",".join(sorted(tool_name(t) for t in tools))
    return hashlib.sha256(f"{first_user}|{names}".encode("utf-8", "replace")).hexdigest()[:24]
