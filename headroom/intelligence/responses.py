"""Intelligence for the OpenAI Responses wire (Codex).

The Responses compression path (``_compress_openai_responses_payload``) does
not run ``TransformPipeline``: it extracts text slots as ``CompressionUnit``
objects, routes each one through ContentRouter and caches the result keyed on
the unit, ``context`` and ``bias`` included. That path has no frozen-prefix
tracker. Prompt-cache stability comes from determinism alone: the same item
text has to compress to the same bytes on every turn.

A per-turn task query or learned bias would break that determinism, so this
adapter pins them:

* the first time an item's text is routed, the (context, bias, task) used is
  pinned by text hash, and every later turn reuses the pin. The unit-cache key
  and the router output stay identical;
* delta encoding and admission run over a chat-shaped *view* of the items, and
  changed tool outputs are spliced back by ``call_id``. The prep transform's
  own memo pins its per-(call_id, content hash) decisions the same way.

Every entry point fails open and returns the input unchanged.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from .messages import responses_items_to_messages

logger = logging.getLogger(__name__)

RESPONSES_OUTPUT_TYPES = frozenset(
    {
        "function_call_output",
        "custom_tool_call_output",
        "local_shell_call_output",
        "apply_patch_call_output",
    }
)
_TEXT_PART_TYPES = frozenset({"input_text", "output_text"})


@dataclass(frozen=True)
class UnitPin:
    """Routing inputs pinned for one item text (first sighting wins)."""

    context: str
    bias: float
    task: Any = None  # TaskContext for the invariant guard / arbiter, or None


@dataclass
class ResponsesTurn:
    """Per-request view: chat-shaped messages + TaskContext + query."""

    messages: list[dict[str, Any]]
    task: Any
    query: str
    request_id: str = ""
    prep_tags: list[str] = field(default_factory=list)


def _text_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:32]


class ResponsesIntelligence:
    """Pins and prep for one proxy's Responses traffic (thread-safe)."""

    PIN_MAX = 20_000

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime
        self._pins: OrderedDict[str, UnitPin] = OrderedDict()
        self._lock = threading.Lock()

    # ------------------------------------------------------------ per request
    def begin(self, items: Any, *, model: str, request_id: str = "") -> ResponsesTurn | None:
        """Build the turn view, or None when nothing on this path is enabled."""
        try:
            cfg = self.runtime.config
            if not cfg.any_enabled:
                return None
            messages = responses_items_to_messages(items)
            if not messages:
                return None
            kwargs = self.runtime.prepare_request(
                messages, {"request_id": request_id}, provider="openai", model=model
            )
            query = str(kwargs.get("context") or "") if cfg.task_query else ""
            return ResponsesTurn(
                messages=messages,
                task=kwargs.get("task_context"),
                query=query,
                request_id=request_id,
            )
        except Exception:  # noqa: BLE001 - intelligence never breaks a request
            logger.debug("responses intelligence begin failed", exc_info=True)
            return None

    def pin_for(self, text: str, *, tool_name: str, turn: ResponsesTurn) -> UnitPin:
        """Return the pinned routing inputs for ``text`` (creating them once)."""
        key = _text_key(text)
        with self._lock:
            pin = self._pins.get(key)
            if pin is not None:
                self._pins.move_to_end(key)
                return pin
        cfg = self.runtime.config
        bias = 1.0
        learner = getattr(self.runtime, "learner", None)
        if learner is not None and tool_name:
            try:
                bias = float(learner.retention_bias(tool_name))
            except Exception:  # noqa: BLE001
                bias = 1.0
        review = cfg.invariant_guard or cfg.policy_budget or cfg.arbiter
        pin = UnitPin(context=turn.query, bias=bias, task=turn.task if review else None)
        with self._lock:
            existing = self._pins.get(key)
            if existing is not None:  # lost a race: the first pin wins
                return existing
            self._pins[key] = pin
            while len(self._pins) > self.PIN_MAX:
                self._pins.popitem(last=False)
        return pin

    def pin_count(self) -> int:
        with self._lock:
            return len(self._pins)

    # ------------------------------------------------------------------ prep
    def apply_prep(
        self,
        prep: Any,
        items: list[Any],
        turn: ResponsesTurn,
        tokenizer: Any,
        *,
        markers_ok: bool,
    ) -> list[Any] | None:
        """Run delta/admission over the items; return new items or None."""
        cfg = self.runtime.config
        if prep is None or not (cfg.delta or cfg.admission):
            return None
        try:
            result = prep.apply(
                turn.messages,
                tokenizer,
                task_context=turn.task,
                frozen_message_count=0,
                request_id=turn.request_id,
                cross_turn_dedup_recoverable=markers_ok,
            )
            if not result.transforms_applied or result.messages is turn.messages:
                return None
            changed = _changed_tool_outputs(turn.messages, result.messages)
            if not changed:
                return None
            new_items = splice_tool_outputs(items, changed)
            if new_items is None:
                return None
            turn.prep_tags = list(result.transforms_applied)
            return new_items
        except Exception:  # noqa: BLE001
            logger.debug("responses intelligence prep failed", exc_info=True)
            return None


def _changed_tool_outputs(
    before: list[dict[str, Any]], after: list[dict[str, Any]]
) -> dict[str, str]:
    """call_id -> new text for every ``role: tool`` message the prep rewrote."""
    if len(before) != len(after):
        return {}
    changed: dict[str, str] = {}
    for old, new in zip(before, after):
        if old is new or new.get("role") != "tool":
            continue
        content = new.get("content")
        if isinstance(content, list):
            content = "\n".join(str(p.get("text", "")) for p in content if isinstance(p, dict))
        if not isinstance(content, str) or content == old.get("content"):
            continue
        call_id = str(new.get("tool_call_id") or "")
        if call_id:
            changed[call_id] = content
    return changed


def splice_tool_outputs(items: list[Any], changed: dict[str, str]) -> list[Any] | None:
    """Write rewritten outputs back into Responses items, preserving shape.

    Only string outputs and single-text-part list outputs are rewritten; a
    ``call_id`` appearing on more than one output item is ambiguous and skipped.
    Untouched items are shared, so their bytes are identical by construction.
    """
    counts: dict[str, int] = {}
    for item in items:
        if isinstance(item, dict) and item.get("type") in RESPONSES_OUTPUT_TYPES:
            cid = item.get("call_id")
            if isinstance(cid, str):
                counts[cid] = counts.get(cid, 0) + 1
    out: list[Any] | None = None
    for idx, item in enumerate(items):
        if not isinstance(item, dict) or item.get("type") not in RESPONSES_OUTPUT_TYPES:
            continue
        cid = item.get("call_id")
        if not isinstance(cid, str) or cid not in changed or counts.get(cid) != 1:
            continue
        text = changed[cid]
        output = item.get("output")
        if isinstance(output, str):
            new_item = {**item, "output": text}
        elif (
            isinstance(output, list)
            and len(output) == 1
            and isinstance(output[0], dict)
            and output[0].get("type") in _TEXT_PART_TYPES
        ):
            new_item = {**item, "output": [{**output[0], "text": text}]}
        else:
            continue
        if out is None:
            out = list(items)
        out[idx] = new_item
    return out
