"""Pre-context tool-result admission + delta stage (Optimizations 4, 5 and 14).

Runs as a pipeline stage *before* the ContentRouter, on the live zone only
(messages at or after ``frozen_message_count``; cached-prefix bytes are never
touched). For each tool result it builds a :class:`ToolResultEnvelope` and
decides, deterministically:

* **exact**        — small, error-carrying, protected, cached or already
                     compressed output passes through untouched;
* **delta**        — a re-read/re-run of a resource seen earlier in the
                     conversation is replaced by what changed;
* **externalize**  — a very large result is replaced by a task-relevant
                     exact-excerpt preview plus a CCR marker, after the exact
                     original was stored and a retrieval self-test passed;
* **defer**        — everything else goes to the type-specific compressors
                     (the ContentRouter, where the arbiter can still choose
                     externalization as one candidate).

Every rewrite goes through the shared invariant guard + policy admission
(:class:`~headroom.intelligence.arbiter.ArbiterSession`). If storing the
original fails, the original is forwarded. The stage is a pure function of the
message list plus idempotent content-addressed CCR writes, so re-running it on
the same conversation produces the same bytes (prompt-cache safe).
"""

from __future__ import annotations

import logging
import re
import threading
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from headroom.config import DEFAULT_VERBATIM_EXCLUDE_TOOLS, TransformResult, is_tool_excluded
from headroom.tokenizer import Tokenizer
from headroom.transforms.base import Transform

from .arbiter import TIER_DELTA, TIER_INDEXED, ArbiterSession, Candidate
from .config import IntelligenceConfig
from .delta import compute_delta, render_delta
from .messages import (
    ToolResultRef,
    build_tool_call_index,
    iter_tool_results,
    replace_tool_result_text,
)
from .models import DecisionFamily
from .policy import classify_change
from .resources import ResourceRef, content_hash, resource_for
from .task_context import EMPTY_TASK_CONTEXT, Provenance, TaskContext, detect_provenance

logger = logging.getLogger(__name__)

_ALREADY_COMPRESSED = (
    "Retrieve more: hash=",
    "Retrieve original: hash=",
    "<<ccr:",
    "headroom delta:",
    "headroom preview",
)
_EXIT_RE = re.compile(
    r"(?:exit(?:ed)?(?: with)?(?: code| status)?[:= ]+|exit_code[\"']?\s*[:=]\s*)(-?\d{1,3})", re.I
)


@dataclass(frozen=True)
class ToolResultEnvelope:
    tool_name: str
    tool_call_id: str
    workspace_key: str
    resource: ResourceRef | None
    provenance: Provenance
    content: str
    exit_code: int | None
    is_error: bool
    metadata: dict[str, Any] = field(default_factory=dict)


def envelope_for(ref: ToolResultRef, workspace_key: str = "") -> ToolResultEnvelope:
    m = _EXIT_RE.search(ref.text[-4000:]) if ref.text else None
    return ToolResultEnvelope(
        tool_name=ref.tool_name,
        tool_call_id=ref.call_id,
        workspace_key=workspace_key,
        resource=resource_for(ref.tool_name, ref.tool_input),
        provenance=detect_provenance(ref.text, ref.tool_name, ref.tool_input),
        content=ref.text,
        exit_code=int(m.group(1)) if m else None,
        is_error=ref.is_error,
    )


def _marker(hash_key: str, what: str) -> str:
    return (
        f"[{what} — exact original stored. Retrieve original: hash={hash_key} "
        f'(or headroom_retrieve(hash="{hash_key}", query="…") for specific parts)]'
    )


def store_and_verify(
    store: Any,
    original: str,
    *,
    compressed: str,
    original_tokens: int,
    compressed_tokens: int,
    tool_name: str | None,
    tool_call_id: str | None,
    query: str,
    strategy: str,
    partial: bool,
) -> str | None:
    """Store the exact original and prove it retrievable. ``None`` on failure."""
    if store is None:
        return None
    try:
        h = store.store(
            original,
            compressed,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            query_context=query[:500] if query else None,
            compression_strategy=f"{strategy}{':partial' if partial else ''}",
        )
        verify = getattr(store, "verify_exact", None)
        if callable(verify):
            if not verify(h, original):
                return None
        elif not store.exists(h):
            return None
        return str(h)
    except Exception as exc:  # noqa: BLE001 - storage failure => forward original
        logger.warning(
            "intelligence: CCR store failed (%s); forwarding original", type(exc).__name__
        )
        return None


def build_preview(
    original: str,
    task: TaskContext,
    *,
    preview_chars: int,
    provenance: Provenance,
) -> tuple[str, int, int]:
    """Query-aware preview of exact excerpts: ``(text, spans_shown, total)``."""
    from headroom.ccr.span_index import INDEX_CACHE, rank_spans

    index = INDEX_CACHE.get_or_build(content_hash(original), original)
    spans = index.spans
    if not spans:
        return original[:preview_chars], 1, 1
    chosen: set[int] = {0}
    ranked = rank_spans(index, original, task.relevance_query(), exact_terms=task.explicit_entities)
    for r in ranked:
        chosen.add(r.span.ordinal)
    for s in spans:
        if s.item_type == "failure_block":
            chosen.add(s.ordinal)
    chosen.add(spans[-1].ordinal)
    priority = (
        [0]
        + [r.span.ordinal for r in ranked]
        + [s.ordinal for s in spans if s.item_type == "failure_block"]
        + [spans[-1].ordinal]
    )
    kept: list[int] = []
    used = 0
    for ordinal in dict.fromkeys(priority):
        if ordinal not in chosen:
            continue
        size = spans[ordinal].end - spans[ordinal].start
        if used + size > preview_chars and kept:
            continue
        kept.append(ordinal)
        used += size
    kept.sort()
    parts: list[str] = []
    prev = -1
    for ordinal in kept:
        gap = ordinal - prev - 1
        if gap > 0:
            parts.append(f"… [{gap} span{'s' if gap != 1 else ''} omitted] …\n")
        span = spans[ordinal]
        text = original[span.start : span.end]
        parts.append(text if text.endswith("\n") else text + "\n")
        prev = ordinal
    tail_gap = len(spans) - 1 - prev
    if tail_gap > 0:
        parts.append(f"… [{tail_gap} span{'s' if tail_gap != 1 else ''} omitted] …\n")
    body = "".join(parts)
    if (
        provenance.is_partial
        and provenance.truncation_boundary
        and provenance.truncation_boundary not in body
    ):
        body += f"[partial input: {provenance.truncation_boundary}]\n"
    return body, len(kept), len(spans)


AdvisorFn = Callable[..., Any]


class IntelligencePrepTransform(Transform):
    """Delta + pre-context admission over the live zone (pipeline stage)."""

    name = "intelligence_prep"

    _MEMO_MAX = 4096

    def __init__(
        self,
        config: IntelligenceConfig,
        *,
        store_provider: Callable[[], Any] | None = None,
        advisor_provider: Callable[[], Any] | None = None,
        feedback: Any | None = None,
        markers_enabled: bool = True,
    ) -> None:
        self.config = config
        self._store_provider = store_provider
        self._advisor_provider = advisor_provider
        self._feedback = feedback
        self.markers_enabled = markers_enabled
        # (tool_call_id, content hash) -> (rendered, tag) | None. The first
        # decision for a tool result is pinned: later turns (with a different
        # task query) must reproduce the same bytes or the prompt cache busts.
        self._memo: OrderedDict[tuple[str, str], tuple[str, str] | None] = OrderedDict()
        self._memo_lock = threading.Lock()
        # Tool calls already observed (feedback re-access signals fire once).
        self._observed_calls: OrderedDict[str, None] = OrderedDict()

    def should_apply(
        self, messages: list[dict[str, Any]], tokenizer: Tokenizer, **kwargs: Any
    ) -> bool:
        return bool(messages) and (self.config.delta or self.config.admission)

    def _store(self, kwargs: dict[str, Any]) -> Any:
        injected = kwargs.get("compression_store")
        if injected is not None:
            return injected
        if self._store_provider is not None:
            return self._store_provider()
        try:
            from headroom.cache.compression_store import get_compression_store

            return get_compression_store()
        except Exception:  # noqa: BLE001
            return None

    # ------------------------------------------------------------------ apply
    def apply(
        self, messages: list[dict[str, Any]], tokenizer: Tokenizer, **kwargs: Any
    ) -> TransformResult:
        tokens_before = tokenizer.count_messages(messages)
        task: TaskContext = kwargs.get("task_context") or EMPTY_TASK_CONTEXT
        frozen = int(kwargs.get("frozen_message_count") or 0)
        policy = kwargs.get("compression_policy")
        prefix_replay = kwargs.get("prefix_replay_guaranteed") is True
        ccr_markers_allowed = (
            self.markers_enabled and kwargs.get("cross_turn_dedup_recoverable", True) is not False
        )
        call_index = build_tool_call_index(messages)
        seen: dict[str, tuple[int, ToolResultRef, ToolResultEnvelope]] = {}
        out = messages
        applied: list[str] = []
        warnings: list[str] = []
        last_index = len(messages) - 1

        for ref in list(iter_tool_results(messages, call_index)):
            env = envelope_for(ref, task.workspace_key)
            res = env.resource
            if self._feedback is not None and res is not None and ref.call_id:
                with self._memo_lock:
                    first_sighting = ref.call_id not in self._observed_calls
                    if first_sighting:
                        self._observed_calls[ref.call_id] = None
                        while len(self._observed_calls) > self._MEMO_MAX:
                            self._observed_calls.popitem(last=False)
                if first_sighting:
                    try:
                        self._feedback.observe_access(res)
                    except Exception:  # noqa: BLE001
                        pass
            prior = seen.get(res.identity) if res is not None else None
            if res is not None and not self._is_compressed(ref.text):
                # Only a complete, uncompressed occurrence can be a delta base.
                if env.provenance.complete:
                    seen[res.identity] = (ref.message_index, ref, env)
            if ref.message_index < frozen:
                continue  # cached prefix: base material only, never rewritten
            if ref.has_cache_control and not (prefix_replay and ref.message_index == last_index):
                continue
            if not ccr_markers_allowed:
                continue  # no retrieval path: never emit recovery-dependent forms
            memo_key = (
                ref.call_id or f"msg{ref.message_index}:{ref.block_index}",
                content_hash(ref.text),
            )
            with self._memo_lock:
                pinned = memo_key in self._memo
                decision = self._memo.get(memo_key) if pinned else None
                if pinned:
                    self._memo.move_to_end(memo_key)
            if not pinned:
                decision = self._decide(ref, env, prior, tokenizer, task, policy, kwargs)
                with self._memo_lock:
                    self._memo[memo_key] = decision
                    while len(self._memo) > self._MEMO_MAX:
                        self._memo.popitem(last=False)
            if decision is None:
                continue
            text, tag = decision
            if pinned:
                self._refresh_store(ref, text, tokenizer, kwargs)
            out = replace_tool_result_text(out, ref, text)
            applied.append(tag)
            if self._feedback is not None and res is not None and not pinned:
                try:
                    self._feedback.observe_rewrite(res, tag)
                    from .feedback import feature_key
                    from .invariants import ccr_hashes_in

                    for h in ccr_hashes_in(text):
                        self._feedback.note_compressed(
                            h, feature_key(ref.tool_name, tag, kind=res.kind)
                        )
                except Exception:  # noqa: BLE001
                    pass

        tokens_after = tokenizer.count_messages(out) if applied else tokens_before
        return TransformResult(
            messages=out,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            transforms_applied=applied,
            warnings=warnings,
        )

    def _refresh_store(
        self, ref: ToolResultRef, rendered: str, tokenizer: Tokenizer, kwargs: dict[str, Any]
    ) -> None:
        """Re-store a pinned rewrite's original so its marker outlives the CCR TTL.

        Content-addressed: the same original maps to the same hash, so this
        only refreshes the entry's age; the forwarded bytes do not change.
        """
        from .invariants import ccr_hashes_in

        store = self._store(kwargs)
        if store is None:
            return
        for h in ccr_hashes_in(rendered):
            try:
                status = store.get_entry_status(h)
                fresh = (
                    status.get("status") == "available"
                    and float(status.get("age_seconds", 0.0))
                    < float(status.get("ttl_seconds", 0.0)) / 2
                )
                if not fresh:
                    store.store(
                        ref.text,
                        rendered,
                        original_tokens=tokenizer.count_text(ref.text),
                        compressed_tokens=tokenizer.count_text(rendered),
                        tool_name=ref.tool_name or None,
                        tool_call_id=ref.call_id or None,
                        compression_strategy="intelligence_refresh",
                    )
            except Exception:  # noqa: BLE001
                logger.debug("intelligence: CCR refresh failed", exc_info=True)

    @staticmethod
    def _is_compressed(text: str) -> bool:
        return any(m in text for m in _ALREADY_COMPRESSED)

    def _protected(self, ref: ToolResultRef, res: ResourceRef | None) -> bool:
        if ref.tool_name and is_tool_excluded(ref.tool_name, DEFAULT_VERBATIM_EXCLUDE_TOOLS):
            return True
        if ref.tool_name and is_tool_excluded(ref.tool_name, ("headroom_retrieve",)):
            return True
        return False

    def _decide(
        self,
        ref: ToolResultRef,
        env: ToolResultEnvelope,
        prior: tuple[int, ToolResultRef, ToolResultEnvelope] | None,
        tokenizer: Tokenizer,
        task: TaskContext,
        policy: Any,
        kwargs: dict[str, Any],
    ) -> tuple[str, str] | None:
        text = ref.text
        if not text or self._is_compressed(text) or self._protected(ref, env.resource):
            return None
        res = env.resource
        tokens = tokenizer.count_text(text)
        if tokens < min(self.config.delta_min_tokens, self.config.admission_min_tokens):
            return None
        # Errors stay exact unless large (the router's log compressor keeps
        # error lines in big logs; small failures are the agent's evidence).
        if (
            env.is_error or (env.exit_code not in (None, 0))
        ) and tokens < self.config.admission_min_tokens:
            return None
        is_read = bool(res and res.is_read)
        candidates: list[Candidate] = []
        store = self._store(kwargs)
        query = task.relevance_query()

        # --- delta -------------------------------------------------------
        if (
            self.config.delta
            and prior is not None
            and res is not None
            and (not is_read or self.config.delta_reads)
            and env.provenance.complete
            and tokens >= self.config.delta_min_tokens
        ):
            base_idx, base_ref, _base_env = prior
            if base_ref.message_index != ref.message_index:
                body = compute_delta(base_ref.text, text, res.kind)
                if body is not None:
                    changed_tokens = (
                        tokenizer.count_text(body.changed_text) if body.changed_text else 0
                    )
                    stability = classify_change(policy, changed_tokens)
                    # A change touching most of the output is not clearer as a delta.
                    if body.strategy == "identical" or changed_tokens <= 0.5 * tokens:
                        rendered = render_delta(
                            body,
                            label=res.label,
                            base_message_index=base_idx,
                            base_hash=content_hash(base_ref.text),
                            current_hash=content_hash(text),
                            stability=stability,
                        )
                        h = store_and_verify(
                            store,
                            text,
                            compressed=rendered,
                            original_tokens=tokens,
                            compressed_tokens=tokenizer.count_text(rendered),
                            tool_name=ref.tool_name or None,
                            tool_call_id=ref.call_id or None,
                            query=query,
                            strategy=f"delta_{body.strategy}",
                            partial=False,
                        )
                        if h is not None:
                            full = rendered + "\n" + _marker(h, "complete current output")
                            claims = "\n".join(
                                ln for ln in body.body.split("\n") if not ln.startswith("- ")
                            )
                            candidates.append(
                                Candidate(
                                    f"delta_{body.strategy}",
                                    f"delta:{body.strategy}",
                                    full,
                                    tier=TIER_DELTA,
                                    claims_text=claims,
                                )
                            )

        # --- externalization --------------------------------------------
        if (
            self.config.admission
            and not is_read
            and tokens >= self.config.admission_min_tokens
            and self._prefer_preview(env, tokens, task)
        ):
            preview_chars = self.config.admission_preview_tokens * 4
            body, shown, total = build_preview(
                text, task, preview_chars=preview_chars, provenance=env.provenance
            )
            header = (
                f"headroom preview: {shown} of {total} exact excerpts of this {ref.tool_name or 'tool'} output "
                f"({tokens} tokens) selected for the current task"
                + (" [partial input]" if env.provenance.is_partial else "")
                + "."
            )
            preview = f"{header}\n{body}"
            h = store_and_verify(
                store,
                text,
                compressed=preview,
                original_tokens=tokens,
                compressed_tokens=tokenizer.count_text(preview),
                tool_name=ref.tool_name or None,
                tool_call_id=ref.call_id or None,
                query=query,
                strategy="indexed_preview",
                partial=env.provenance.is_partial,
            )
            if h is not None:
                candidates.append(
                    Candidate(
                        "indexed_preview",
                        "indexed_preview",
                        preview + _marker(h, f"{total - shown} excerpts omitted"),
                        tier=TIER_INDEXED,
                    )
                )

        if not candidates:
            return None
        session = ArbiterSession(
            text,
            count_tokens=tokenizer.count_text,
            task=task,
            policy=policy,
            weights=self.config.arbiter_weights,
            store_has=(lambda h: bool(store.exists(h))) if store is not None else None,
            provenance=env.provenance,
            enforce_invariants=True,
            enforce_policy=True,
        )
        session.prepare(candidates)
        decision = session.select(None)
        if decision.kept_original:
            logger.debug(
                "intelligence admission kept original (%s): %s",
                ref.tool_name,
                decision.rejections,
            )
            return None
        return decision.selected.content, f"intel:{decision.selected.strategy}"

    def _prefer_preview(self, env: ToolResultEnvelope, tokens: int, task: TaskContext) -> bool:
        """Deterministic admission mode, optionally refined by JevK5.

        Structured outputs (JSON, search, logs, diffs) default to the
        type-specific compressors unless they are very large; prose/code/other
        default to the indexed preview. Inside the ambiguous band the advisor
        may choose between the two (both are safe).
        """
        from headroom.ccr.span_index import detect_kind

        kind = detect_kind(env.content[:20_000])
        structured = kind in ("json_array", "json_object", "search", "log", "diff")
        default = not structured or tokens >= 4 * self.config.admission_min_tokens
        if self._advisor_provider is None or tokens >= 4 * self.config.admission_min_tokens:
            return default
        advisor = self._advisor_provider()
        if advisor is None:
            return default
        state = (
            f"Task goal: {task.current_user_goal_text[:500] or 'unknown'}\n"
            f"Tool: {env.tool_name}; output kind: {kind}; size: {tokens} tokens; "
            f"error output: {env.is_error or env.exit_code not in (None, 0)}."
        )
        scores = advisor.choose(
            DecisionFamily.ADMISSION_MODE,
            state,
            "Which representation should this tool output enter the context as?",
            {
                "structural_compact": "type-aware compaction that keeps the overall structure inline",
                "indexed_preview": "only task-relevant exact excerpts inline; the rest searchable on demand",
            },
        )
        if scores is None or scores.weight <= 0:
            return default
        p_preview = scores.probabilities.get("indexed_preview", 0.5)
        # Blend with the deterministic prior exactly like the arbiter does.
        blended = (1 - scores.weight) * (1.0 if default else 0.0) + scores.weight * p_preview
        return blended >= 0.5
