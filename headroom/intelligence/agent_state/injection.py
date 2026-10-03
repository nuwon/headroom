"""Cache-safe live-turn injection of the agent-state block (plan §5.8, §12.1).

The block goes into the live turn, never into cache-hot history. The
complication is that the client resends its own history without Headroom's
earlier insertions. If an insertion were dropped on the next turn, the bytes
the provider cached would change and the prompt cache would bust from that
point on.

The :class:`StickyInjector` therefore memoizes every insertion as
``(anchor index, anchor hash, block text)`` and re-applies all of them on
every request. History then reads byte-identically to what was sent before,
and a new block is added only when the model-relevant state changed, so
history grows only on change.

* **Anthropic**: the block is appended as a ``text`` block to the anchor (the
  last ``user`` message of the turn it was created on). Text after
  ``tool_result`` blocks is valid, and no cache breakpoint is added.
* **Responses (Codex)**: a ``user`` message item is inserted right after the
  anchor item. With ``previous_response_id`` the server holds history, so only
  the new block is appended and nothing is replayed.

Anchors are hashed in a canonical form (``cache_control`` removed) so the
client moving its breakpoint does not orphan an anchor. A memo entry whose
anchor no longer matches (compaction, edited history) is skipped, never
guessed. Memo entries are only committed after the handler confirms the
mutated body is what actually goes on the wire.
"""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass
from typing import Any

MAX_MEMO_ENTRIES = 256


def _strip_cache_control(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _strip_cache_control(v) for k, v in value.items() if k != "cache_control"}
    if isinstance(value, list):
        return [_strip_cache_control(v) for v in value]
    return value


def anchor_hash(message: Any) -> str:
    canon = json.dumps(
        _strip_cache_control(message), sort_keys=True, ensure_ascii=False, default=str
    )
    return hashlib.sha256(canon.encode("utf-8", "surrogatepass")).hexdigest()[:24]


@dataclass(frozen=True)
class MemoEntry:
    index: int
    anchor: str
    text: str
    state_hash: str

    def to_json(self) -> list[Any]:
        return [self.index, self.anchor, self.text, self.state_hash]

    @classmethod
    def from_json(cls, raw: Any) -> MemoEntry | None:
        try:
            index, anchor, text, state_hash = raw
            return cls(int(index), str(anchor), str(text), str(state_hash))
        except (TypeError, ValueError):
            return None


@dataclass
class PendingInjection:
    session_key: str
    entry: MemoEntry | None  # the new insertion (None when only replaying)
    replayed: int


class StickyInjector:
    """Per-session memo of insertions; thread-safe."""

    def __init__(self) -> None:
        self._memo: dict[str, list[MemoEntry]] = {}
        self._lock = threading.Lock()

    def load(self, session_key: str, raw: Any) -> None:
        entries = [e for e in (MemoEntry.from_json(r) for r in (raw or [])) if e is not None]
        with self._lock:
            self._memo.setdefault(session_key, entries)

    def entries(self, session_key: str) -> list[MemoEntry]:
        with self._lock:
            return list(self._memo.get(session_key, ()))

    def last_state_hash(self, session_key: str) -> str:
        entries = self.entries(session_key)
        return entries[-1].state_hash if entries else ""

    def commit(self, pending: PendingInjection) -> list[MemoEntry]:
        with self._lock:
            entries = self._memo.setdefault(pending.session_key, [])
            if pending.entry is not None and pending.entry not in entries:
                entries.append(pending.entry)
                if len(entries) > MAX_MEMO_ENTRIES:
                    del entries[: len(entries) - MAX_MEMO_ENTRIES]
            return list(entries)

    def forget(self, session_key: str) -> None:
        with self._lock:
            self._memo.pop(session_key, None)

    # ------------------------------------------------------------ Anthropic
    def apply_anthropic(
        self,
        session_key: str,
        client_messages: list[dict[str, Any]],
        outgoing: list[dict[str, Any]],
        new_block: str | None,
        state_hash: str,
        *,
        only_if_orphaned: bool = False,
    ) -> tuple[list[dict[str, Any]], PendingInjection] | None:
        """Return ``(messages, pending)`` or None when nothing changes.

        ``only_if_orphaned``: the state did not change, so ``new_block`` is
        inserted only when no earlier insertion survives in this history
        (compaction or a rewritten history dropped them).
        """
        if len(client_messages) != len(outgoing) or not outgoing:
            return None
        plan: dict[int, list[str]] = {}
        replayed = 0
        for entry in self.entries(session_key):
            i = entry.index
            if 0 <= i < len(client_messages) and anchor_hash(client_messages[i]) == entry.anchor:
                plan.setdefault(i, []).append(entry.text)
                replayed += 1
        new_entry: MemoEntry | None = None
        if only_if_orphaned and replayed:
            new_block = None
        if new_block:
            last = len(outgoing) - 1
            if outgoing[last].get("role") == "user":
                new_entry = MemoEntry(
                    last, anchor_hash(client_messages[last]), new_block, state_hash
                )
                if new_entry.text not in plan.get(last, []):
                    plan.setdefault(last, []).append(new_block)
                else:
                    new_entry = None
        if not plan:
            return None
        out = list(outgoing)
        for i, texts in plan.items():
            msg = out[i]
            content = msg.get("content")
            if isinstance(content, str):
                blocks: list[Any] = [{"type": "text", "text": content}] if content else []
            elif isinstance(content, list):
                blocks = list(content)
            else:
                continue
            blocks.extend({"type": "text", "text": t} for t in texts)
            out[i] = {**msg, "content": blocks}
        return out, PendingInjection(session_key, new_entry, replayed)

    # ------------------------------------------------------------ Responses
    def apply_responses(
        self,
        session_key: str,
        client_items: list[Any],
        outgoing: list[Any],
        new_block: str | None,
        state_hash: str,
        *,
        incremental: bool,
        only_if_orphaned: bool = False,
    ) -> tuple[list[Any], PendingInjection] | None:
        if len(client_items) != len(outgoing):
            return None
        plan: dict[int, list[str]] = {}
        replayed = 0
        if not incremental:
            for entry in self.entries(session_key):
                i = entry.index
                if 0 <= i < len(client_items) and anchor_hash(client_items[i]) == entry.anchor:
                    plan.setdefault(i, []).append(entry.text)
                    replayed += 1
        new_entry: MemoEntry | None = None
        if only_if_orphaned and (replayed or incremental):
            new_block = None
        if new_block and outgoing:
            last = len(outgoing) - 1
            new_entry = MemoEntry(last, anchor_hash(client_items[last]), new_block, state_hash)
            if new_block in plan.get(last, []):
                new_entry = None
            else:
                plan.setdefault(last, []).append(new_block)
        if not plan:
            return None
        out = list(outgoing)
        for i in sorted(plan, reverse=True):  # insert from the end: earlier indices stay valid
            for text in reversed(plan[i]):
                out.insert(i + 1, responses_state_item(text))
        return out, PendingInjection(session_key, new_entry, replayed)


def responses_state_item(text: str) -> dict[str, Any]:
    return {
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": text}],
    }
