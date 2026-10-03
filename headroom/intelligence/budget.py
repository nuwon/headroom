"""Global information-per-token budget allocator (Optimization 8, plan §14).

Each compressor used to decide alone how much of its block to keep. Under
context pressure that lets one huge, low-value output crowd out the error
trace or the user-named evidence. The allocator looks at every live block at
once and expresses its decision through the existing per-message ``bias``
channel (``>1`` keep more, ``<1`` compress harder), so every compressor keeps
its own safety floors.

Algorithm:

1. ``available = context_limit - frozen_prefix - reserved_output - margin``;
2. mandatory minima first: blocks that carry the user's explicit entities,
   current errors/exit codes or the newest tool result are pinned high;
3. optional blocks are ranked by ``value / tokens`` (relevance, recency,
   novelty, error importance) and granted budget in that order;
4. a diversity cap stops one source taking more than ``diversity_cap`` of the
   live budget, so independent evidence is not starved;
5. recoverable blocks (CCR-backed) can be pushed harder than irrecoverable
   ones at equal value.

Below ``pressure_threshold`` (live tokens well inside the budget) the
allocator only boosts mandatory blocks; it never makes compression more
aggressive when there is room.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .messages import build_tool_call_index, iter_tool_results
from .resources import content_hash, resource_for
from .task_context import TaskContext

MIN_BIAS = 0.6
MAX_BIAS = 1.6
DEFAULT_OUTPUT_RESERVE = 8192
SAFETY_MARGIN = 0.05


@dataclass
class BlockValue:
    message_index: int
    tokens: int
    value: float
    mandatory: bool
    recoverable: bool
    source: str
    keep_fraction: float = 1.0
    bias: float = 1.0


@dataclass
class Allocation:
    biases: dict[int, float]
    available_tokens: int
    live_tokens: int
    pressure: float
    blocks: list[BlockValue]

    def summary(self) -> dict[str, Any]:
        return {
            "available_tokens": self.available_tokens,
            "live_tokens": self.live_tokens,
            "pressure": round(self.pressure, 3),
            "blocks": len(self.blocks),
            "boosted": sum(1 for b in self.blocks if b.bias > 1.0),
            "tightened": sum(1 for b in self.blocks if b.bias < 1.0),
        }


def _value(
    text: str,
    tokens: int,
    *,
    task: TaskContext,
    from_end: int,
    duplicate: bool,
    is_error: bool,
) -> tuple[float, bool]:
    """(value in [0, 1], mandatory)."""
    lowered = text.lower()
    entity_hits = sum(1 for e in task.explicit_entities if len(e) >= 3 and e.lower() in lowered)
    error = is_error or any(
        sig.lower()[:60] in lowered for sig in task.error_signals if len(sig) >= 8
    )
    mandatory = bool(entity_hits) or error or from_end <= 1
    relevance = min(1.0, 0.25 + 0.25 * entity_hits)
    graph = 0.15 if any(n.lower() in lowered for n in task.graph_neighbors if len(n) >= 3) else 0.0
    recency = 1.0 / (1.0 + 0.15 * max(0, from_end - 1))
    novelty = 0.4 if duplicate else 1.0
    value = (
        0.45 * relevance + 0.25 * recency + 0.15 * novelty + 0.15 * (1.0 if error else 0.0)
    ) + graph
    return min(1.0, value), mandatory


def allocate(
    messages: list[dict[str, Any]],
    *,
    count_tokens: Callable[[str], int],
    task: TaskContext,
    context_limit: int,
    frozen_message_count: int = 0,
    output_reserve: int | None = None,
    pressure_threshold: float = 0.5,
    diversity_cap: float = 0.6,
    advisor: Any | None = None,
) -> Allocation:
    """Compute per-message biases for the live zone."""
    reserve = DEFAULT_OUTPUT_RESERVE if output_reserve is None else max(0, int(output_reserve))
    call_index = build_tool_call_index(messages)
    blocks: list[BlockValue] = []
    seen_resources: dict[str, int] = {}
    seen_content: set[str] = set()
    frozen_tokens = 0
    for msg in messages[:frozen_message_count]:
        frozen_tokens += count_tokens(str(msg.get("content", "")))
    n = len(messages)
    for ref in iter_tool_results(messages, call_index):
        res = resource_for(ref.tool_name, ref.tool_input)
        h = content_hash(ref.text)
        duplicate = h in seen_content or (res is not None and res.identity in seen_resources)
        seen_content.add(h)
        if res is not None:
            seen_resources[res.identity] = ref.message_index
        if ref.message_index < frozen_message_count:
            continue
        tokens = count_tokens(ref.text)
        if tokens <= 0:
            continue
        value, mandatory = _value(
            ref.text,
            tokens,
            task=task,
            from_end=n - ref.message_index,
            duplicate=duplicate,
            is_error=ref.is_error,
        )
        blocks.append(
            BlockValue(
                message_index=ref.message_index,
                tokens=tokens,
                value=value,
                mandatory=mandatory,
                recoverable=True,  # CCR-backed compressors; read protection is separate
                source=(res.kind if res else (ref.tool_name or "tool")),
            )
        )
    # Several blocks may live in one message; aggregate per message index.
    per_msg: dict[int, BlockValue] = {}
    for b in blocks:
        cur = per_msg.get(b.message_index)
        if cur is None:
            per_msg[b.message_index] = b
        else:
            cur.tokens += b.tokens
            cur.value = max(cur.value, b.value)
            cur.mandatory = cur.mandatory or b.mandatory
    blocks = list(per_msg.values())
    live_tokens = sum(b.tokens for b in blocks)
    available = int(context_limit * (1 - SAFETY_MARGIN)) - frozen_tokens - reserve
    available = max(0, available)
    pressure = (live_tokens / available) if available else 1.0

    _maybe_advise(blocks, messages, task, advisor)

    biases: dict[int, float] = {}
    if not blocks:
        return Allocation(biases, available, live_tokens, pressure, blocks)

    if pressure < pressure_threshold:
        # Room to spare: only protect what matters; never tighten.
        for b in blocks:
            if b.mandatory:
                b.bias = 1.3
                biases[b.message_index] = b.bias
        return Allocation(biases, available, live_tokens, pressure, blocks)

    # Under pressure: steer live content toward ``pressure_threshold`` of the
    # available budget (headroom for the turns still to come). Mandatory
    # blocks are granted first, then optional ones by value density.
    target = available * pressure_threshold
    remaining = float(target)
    cap = diversity_cap * target
    mandatory = sorted(
        (b for b in blocks if b.mandatory), key=lambda b: (-b.value, b.message_index)
    )
    optional = sorted(
        (b for b in blocks if not b.mandatory),
        key=lambda b: (-(b.value / max(1, b.tokens)), b.message_index),
    )
    for group in (mandatory, optional):
        for b in group:
            grant = min(float(b.tokens), cap, max(0.0, remaining))
            b.keep_fraction = grant / b.tokens if b.tokens else 1.0
            remaining -= grant
    for b in blocks:
        if b.mandatory:
            b.bias = MAX_BIAS if b.keep_fraction >= 0.5 else 1.3
        else:
            # keep_fraction 1 -> 1.0 (default); 0 -> MIN_BIAS (compress hard)
            b.bias = round(MIN_BIAS + (1.0 - MIN_BIAS) * max(0.0, min(1.0, b.keep_fraction)), 4)
        if b.bias != 1.0:
            biases[b.message_index] = b.bias
    return Allocation(biases, available, live_tokens, pressure, blocks)


def _maybe_advise(
    blocks: list[BlockValue], messages: list[dict[str, Any]], task: TaskContext, advisor: Any
) -> None:
    """Let JevK5 score the importance of a few large, non-mandatory groups."""
    if advisor is None:
        return
    from .models import DecisionFamily

    candidates = sorted((b for b in blocks if not b.mandatory), key=lambda b: -b.tokens)[:3]
    for b in candidates:
        msg = messages[b.message_index]
        content = msg.get("content")
        preview = ""
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    c = block.get("content")
                    preview = c if isinstance(c, str) else ""
                    break
        elif isinstance(content, str):
            preview = content
        state = f"Task goal: {task.current_user_goal_text[:500]}\nContent group ({b.source}, {b.tokens} tokens), start:\n{preview[:1500]}"
        scores = advisor.score(
            DecisionFamily.GROUP_IMPORTANCE,
            state,
            "How important is this content group for completing the current task?",
            ("irrelevant", "marginal", "useful", "important", "essential"),
        )
        if scores is None or scores.weight <= 0 or scores.expected_score is None:
            continue
        advised = scores.expected_score / 4.0
        b.value = (1 - scores.weight) * b.value + scores.weight * advised
