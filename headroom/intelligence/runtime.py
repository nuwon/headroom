"""Composition root for the intelligence layer.

One :class:`IntelligenceRuntime` is built per proxy from the resolved
:class:`IntelligenceConfig` and injected into the pipelines, the
ContentRouter and the provider handlers. It owns the long-lived
collaborators (advisor, retention learner, code graphs, speculative pool,
tool catalog, effort router) and the payload-free metrics, and exposes
:meth:`prepare_request` — the single per-request entry point that builds the
TaskContext and derives the pipeline kwargs (relevance query, per-message
biases).

Every method fails open: an exception here never reaches the request path.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from .config import IntelligenceConfig
from .task_context import TaskContext, build_task_context

logger = logging.getLogger(__name__)


@dataclass
class IntelligenceMetrics:
    """Counters only — never content, queries or state."""

    requests: int = 0
    task_query_applied: int = 0
    bias_messages: int = 0
    arbiter_decisions: int = 0
    arbiter_kept_original: int = 0
    arbiter_advised: int = 0
    tiers: Counter = field(default_factory=Counter)
    rejections: Counter = field(default_factory=Counter)
    prep_rewrites: Counter = field(default_factory=Counter)
    tokens_saved_by_source: Counter = field(default_factory=Counter)
    catalog_applied: int = 0
    catalog_bytes_saved: int = 0
    effort_changes: Counter = field(default_factory=Counter)
    semantic_turn_kinds: Counter = field(default_factory=Counter)
    prepare_ms_total: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def bump(self, name: str, key: str | None = None, amount: int | float = 1) -> None:
        with self._lock:
            target = getattr(self, name)
            if isinstance(target, Counter):
                counter: Any = target
                counter[key or "_"] += amount
            else:
                setattr(self, name, target + amount)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            out: dict[str, Any] = {}
            for k, v in self.__dict__.items():
                if k.startswith("_"):
                    continue
                out[k] = dict(v) if isinstance(v, Counter) else v
            return out


class IntelligenceRuntime:
    def __init__(
        self, config: IntelligenceConfig, *, workspace_roots: tuple[str, ...] | None = None
    ) -> None:
        self.config = config
        self.metrics = IntelligenceMetrics()
        self.workspace_roots = workspace_roots if workspace_roots is not None else (os.getcwd(),)
        self.advisor: Any | None = None
        if config.advisor_enabled:
            from .advisor import get_advisor

            self.advisor = get_advisor(config)
        self.learner: Any | None = None
        if config.feedback:
            from .feedback import RetentionLearner

            self.learner = RetentionLearner(
                alpha=config.feedback_alpha,
                min_observations=config.feedback_min_observations,
                advisor_provider=lambda: self.advisor,
            )
            from headroom.cache.compression_store import add_retrieval_listener

            add_retrieval_listener(self.learner.on_retrieval)
        self.graphs: Any | None = None
        if config.graph:
            from .graph import GraphRegistry

            self.graphs = GraphRegistry(max_files=config.graph_max_files)
        self.speculative: Any | None = None
        if config.speculative:
            from .speculative import SpeculativePreparer

            self.speculative = SpeculativePreparer(
                workers=config.speculative_workers, max_pending=config.speculative_max_pending
            )
        self.catalog: Any | None = None
        if config.tool_catalog:
            from headroom.proxy.tool_catalog import ProgressiveToolCatalog

            self.catalog = ProgressiveToolCatalog(
                top_k=config.tool_catalog_top_k, min_tools=config.tool_catalog_min_tools
            )
        self.effort_router: Any | None = None
        if config.effort_routing:
            from headroom.proxy.output_complexity import EffortRouter

            self.effort_router = EffortRouter()
        self._sweep_counter = 0

    # ----------------------------------------------------------- lifecycle
    def start(self) -> None:
        """Proxy startup: begin the JevK5 service in the background (never blocks)."""
        if self.advisor is not None:
            try:
                self.advisor.ensure_started(background=True)
            except Exception:  # noqa: BLE001
                logger.warning(
                    "JevK5 advisor start failed; continuing deterministically", exc_info=True
                )

    def shutdown(self) -> None:
        if self.learner is not None:
            try:
                self.learner.save()
                from headroom.cache.compression_store import remove_retrieval_listener

                remove_retrieval_listener(self.learner.on_retrieval)
            except Exception:  # noqa: BLE001
                pass
        if self.speculative is not None:
            self.speculative.shutdown()
        try:
            from .jevk5_service import stop_all_owned

            stop_all_owned()
        except Exception:  # noqa: BLE001
            pass

    def advisor_or_none(self) -> Any | None:
        advisor = self.advisor
        if advisor is None:
            return None
        try:
            return advisor if advisor.available() else None
        except Exception:  # noqa: BLE001
            return None

    # ----------------------------------------------------------- per request
    def prepare_request(
        self,
        messages: list[dict[str, Any]],
        kwargs: dict[str, Any],
        *,
        provider: str = "",
        model: str = "",
    ) -> dict[str, Any]:
        """Return updated pipeline kwargs (TaskContext, query, biases)."""
        started = time.perf_counter()
        out = dict(kwargs)
        try:
            self.metrics.bump("requests")
            frozen = int(out.get("frozen_message_count") or 0)
            workspace = str(out.get("workspace_key") or "")
            if self.speculative is not None:
                self.speculative.prepare(messages, start=frozen)
            neighbors: list[str] = []
            graph = None
            if self.graphs is not None:
                graph = self.graphs.for_workspace(workspace or "_default")
                graph.ingest_messages(messages, roots=self.workspace_roots)
            task = build_task_context(
                messages,
                provider=provider,
                model=model,
                workspace_key=workspace,
                request_id=str(out.get("request_id") or ""),
            )
            if graph is not None:
                files, symbols = graph.neighborhood(
                    files=task.file_paths, symbols=[*task.symbols, *task.quoted]
                )
                neighbors = [*symbols, *[f.rsplit("/", 1)[-1] for f in files]]
                if neighbors:
                    from dataclasses import replace

                    task = replace(task, graph_neighbors=tuple(dict.fromkeys(neighbors))[:24])
            # Phase 2: the TaskState view (in-scope paths, task-owned changes)
            # joins the exact terms; TaskState itself stays session-scoped.
            from .agent_state.runtime import task_state_terms

            state_terms = task_state_terms(messages)
            if state_terms:
                from dataclasses import replace as _replace

                task = _replace(
                    task,
                    explicit_entities=tuple(dict.fromkeys((*task.explicit_entities, *state_terms)))[
                        :48
                    ],
                )
            out["task_context"] = task
            if self.config.task_query:
                query = task.relevance_query()
                if query.strip():
                    out["context"] = query
                    self.metrics.bump("task_query_applied")
            biases: dict[int, float] = dict(out.get("biases") or {})
            if self.config.budget_allocator and out.get("model_limit"):
                from .budget import allocate

                alloc = allocate(
                    messages,
                    count_tokens=_approx_tokens,
                    task=task,
                    context_limit=int(out["model_limit"]),
                    frozen_message_count=frozen,
                    output_reserve=out.get("output_buffer"),
                    pressure_threshold=self.config.budget_pressure_threshold,
                    diversity_cap=self.config.budget_diversity_cap,
                    advisor=self.advisor_or_none(),
                )
                for idx, b in alloc.biases.items():
                    biases[idx] = biases.get(idx, 1.0) * b
            if self.learner is not None:
                self._learned_biases(messages, frozen, biases)
                self._sweep_counter += 1
                if self._sweep_counter % 50 == 0:
                    self.learner.sweep()
            if biases:
                out["biases"] = biases
                self.metrics.bump("bias_messages", amount=len(biases))
        except Exception:  # noqa: BLE001 - intelligence never breaks a request
            logger.warning(
                "intelligence prepare_request failed; using deterministic kwargs", exc_info=True
            )
            return dict(kwargs)
        finally:
            self.metrics.bump("prepare_ms_total", amount=(time.perf_counter() - started) * 1000.0)
        return out

    def _learned_biases(
        self, messages: list[dict[str, Any]], frozen: int, biases: dict[int, float]
    ) -> None:
        from .messages import build_tool_call_index, iter_tool_results

        learner = self.learner
        if learner is None:
            return
        for ref in iter_tool_results(messages, build_tool_call_index(messages), start=frozen):
            if not ref.tool_name:
                continue
            b = learner.retention_bias(ref.tool_name)
            if b != 1.0:
                biases[ref.message_index] = biases.get(ref.message_index, 1.0) * b

    def request_scope(self) -> Any:
        """Context manager bounding advisor calls for one request."""
        import contextlib

        if self.advisor is None:
            return contextlib.nullcontext()
        return self.advisor.request_scope()

    def status(self) -> dict[str, Any]:
        out: dict[str, Any] = {"config": self.config.to_dict(), "metrics": self.metrics.snapshot()}
        if self.advisor is not None:
            out["advisor"] = self.advisor.stats()
        if self.learner is not None:
            out["retention"] = self.learner.stats()
        if self.speculative is not None:
            out["speculative"] = self.speculative.stats()
        return out


def _approx_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4) if text else 0


_RUNTIME: IntelligenceRuntime | None = None
_RUNTIME_LOCK = threading.Lock()


def install_runtime(runtime: IntelligenceRuntime | None) -> None:
    global _RUNTIME
    with _RUNTIME_LOCK:
        _RUNTIME = runtime


def current_runtime() -> IntelligenceRuntime | None:
    return _RUNTIME


def task_context_from(kwargs: dict[str, Any]) -> TaskContext | None:
    task = kwargs.get("task_context")
    return task if isinstance(task, TaskContext) else None
