"""Provider-shape-aware helpers for walking tool calls/results in messages.

The proxy pipeline sees two wire shapes:

* Anthropic: ``assistant`` messages carry ``tool_use`` blocks
  (``{"type": "tool_use", "id", "name", "input"}``) and the following ``user``
  message carries ``tool_result`` blocks (``{"type": "tool_result",
  "tool_use_id", "content"}``) whose content is a string or a list of
  ``{"type": "text"}`` blocks.
* OpenAI chat: ``assistant`` messages carry ``tool_calls`` and results are
  ``{"role": "tool", "tool_call_id", "content"}`` messages.

These helpers read both without changing either. Writers return new dicts and
preserve the original container shape (string vs list-of-text) so the wire
format of every untouched byte stays identical.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolCall:
    call_id: str
    name: str
    input: dict[str, Any] = field(default_factory=dict)
    message_index: int = -1


@dataclass(frozen=True)
class ToolResultRef:
    """Location + content of a single tool result inside a message list."""

    message_index: int
    block_index: int | None  # None for an OpenAI ``role: tool`` string message
    call_id: str
    tool_name: str
    tool_input: dict[str, Any]
    text: str
    list_form: bool  # content was a list of text blocks
    is_error: bool = False
    has_cache_control: bool = False


def _decode_args(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            decoded = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return {}
        return decoded if isinstance(decoded, dict) else {}
    return {}


def build_tool_call_index(messages: list[dict[str, Any]]) -> dict[str, ToolCall]:
    """``{call_id: ToolCall}`` for every tool invocation in ``messages``."""
    index: dict[str, ToolCall] = {}
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    bid = block.get("id")
                    if isinstance(bid, str):
                        index[bid] = ToolCall(
                            call_id=bid,
                            name=str(block.get("name") or ""),
                            input=_decode_args(block.get("input")),
                            message_index=i,
                        )
        tool_calls = msg.get("tool_calls")
        if isinstance(tool_calls, list):
            for call in tool_calls:
                if not isinstance(call, dict):
                    continue
                cid = call.get("id")
                if not isinstance(cid, str):
                    continue
                fn = call.get("function") or {}
                index[cid] = ToolCall(
                    call_id=cid,
                    name=str(fn.get("name") or ""),
                    input=_decode_args(fn.get("arguments")),
                    message_index=i,
                )
    return index


def _flatten_tool_content(content: Any) -> tuple[str | None, bool]:
    if isinstance(content, str):
        return content, False
    if (
        isinstance(content, list)
        and content
        and all(isinstance(b, dict) and b.get("type") == "text" for b in content)
    ):
        return "".join(str(b.get("text", "")) for b in content), True
    return None, False


def iter_tool_results(
    messages: list[dict[str, Any]],
    call_index: dict[str, ToolCall] | None = None,
    *,
    start: int = 0,
) -> Iterator[ToolResultRef]:
    """Yield every text tool result at or after message ``start``."""
    index = build_tool_call_index(messages) if call_index is None else call_index
    for i in range(max(0, start), len(messages)):
        msg = messages[i]
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        content = msg.get("content")
        if role == "tool":
            text, list_form = _flatten_tool_content(content)
            if text is None:
                continue
            cid = str(msg.get("tool_call_id") or "")
            call = index.get(cid)
            yield ToolResultRef(
                message_index=i,
                block_index=None,
                call_id=cid,
                tool_name=call.name if call else str(msg.get("name") or ""),
                tool_input=call.input if call else {},
                text=text,
                list_form=list_form,
            )
            continue
        if isinstance(content, list):
            for b_idx, block in enumerate(content):
                if not isinstance(block, dict) or block.get("type") != "tool_result":
                    continue
                text, list_form = _flatten_tool_content(block.get("content", ""))
                if text is None:
                    continue
                cid = str(block.get("tool_use_id") or "")
                call = index.get(cid)
                yield ToolResultRef(
                    message_index=i,
                    block_index=b_idx,
                    call_id=cid,
                    tool_name=call.name if call else "",
                    tool_input=call.input if call else {},
                    text=text,
                    list_form=list_form,
                    is_error=block.get("is_error") is True,
                    has_cache_control="cache_control" in block,
                )


def replace_tool_result_text(
    messages: list[dict[str, Any]], ref: ToolResultRef, new_text: str
) -> list[dict[str, Any]]:
    """Return a shallow-copied message list with ``ref``'s text replaced.

    Only the touched message (and block) are copied; every other message
    object is shared, so untouched bytes are identical by construction.
    """
    out = list(messages)
    msg = out[ref.message_index]
    if ref.block_index is None:
        new_content: Any = [{"type": "text", "text": new_text}] if ref.list_form else new_text
        out[ref.message_index] = {**msg, "content": new_content}
        return out
    blocks = list(msg.get("content") or [])
    block = blocks[ref.block_index]
    blocks[ref.block_index] = {
        **block,
        "content": [{"type": "text", "text": new_text}] if ref.list_form else new_text,
    }
    out[ref.message_index] = {**msg, "content": blocks}
    return out


def message_text(msg: dict[str, Any]) -> str:
    """Plain user/assistant text of a message (tool results excluded)."""
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "\n".join(parts)
    return ""


def is_tool_result_only(msg: dict[str, Any]) -> bool:
    """True when a user message carries tool results and no prompt text."""
    if msg.get("role") == "tool":
        return True
    content = msg.get("content")
    if not isinstance(content, list) or not content:
        return False
    saw_result = False
    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "tool_result":
            saw_result = True
        elif btype == "text" and str(block.get("text", "")).strip():
            text = str(block.get("text", "")).strip()
            # Claude Code appends <system-reminder> blocks to tool-result turns;
            # they are harness chatter, not a user ask.
            if not (text.startswith("<system-reminder>") and text.endswith("</system-reminder>")):
                return False
    return saw_result


def responses_items_to_messages(items: Any) -> list[dict[str, Any]]:
    """Adapt OpenAI Responses ``input`` items (Codex) to chat-shaped messages.

    Read-only view for analysis (TaskContext, complexity): ``message`` items
    become role/content messages, ``function_call``/``custom_tool_call``/
    ``local_shell_call`` become assistant ``tool_calls`` and their ``*_output``
    items become ``role: tool`` messages. Never used to rewrite the wire body.
    """
    if isinstance(items, str):
        return [{"role": "user", "content": items}]
    if not isinstance(items, list):
        return []
    out: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        itype = item.get("type", "message" if "role" in item else None)
        if itype == "message":
            content = item.get("content")
            if isinstance(content, list):
                text = "\n".join(
                    str(part.get("text", ""))
                    for part in content
                    if isinstance(part, dict)
                    and part.get("type") in ("input_text", "output_text", "text")
                )
            else:
                text = content if isinstance(content, str) else ""
            role = item.get("role", "user")
            out.append({"role": "system" if role == "developer" else role, "content": text})
        elif itype in ("function_call", "custom_tool_call", "local_shell_call"):
            call_id = str(item.get("call_id") or item.get("id") or "")
            name = str(item.get("name") or ("local_shell" if itype == "local_shell_call" else ""))
            if itype == "local_shell_call":
                action = item.get("action") or {}
                args: Any = json.dumps({"command": action.get("command", [])})
            else:
                args = item.get("arguments", item.get("input", "{}"))
                if not isinstance(args, str):
                    args = json.dumps(args)
            out.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {"name": name, "arguments": args},
                        }
                    ],
                }
            )
        elif itype in (
            "function_call_output",
            "custom_tool_call_output",
            "local_shell_call_output",
            "apply_patch_call_output",
        ):
            output = item.get("output", "")
            if isinstance(output, list):
                output = "\n".join(str(p.get("text", "")) for p in output if isinstance(p, dict))
            elif not isinstance(output, str):
                output = json.dumps(output)
            out.append(
                {"role": "tool", "tool_call_id": str(item.get("call_id") or ""), "content": output}
            )
    return out
