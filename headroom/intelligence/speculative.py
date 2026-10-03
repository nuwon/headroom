"""Speculative / background candidate preparation (Optimization 15, plan §20).

Work that does not depend on the final decision is started as soon as the
request's tool results are known, on a *bounded* pool, and cached by content
hash so the synchronous path picks it up if ready:

* content hashes and truncation provenance;
* invariant extraction;
* CCR span-index chunking (used by externalization previews and later
  indexed retrieval);
* code-graph ingestion of file contents.

Final invariant verdicts, candidate selection, CCR replacement, request
mutation, JevK5 decisions and output shaping always run on the synchronous
path with the final TaskContext, so results are identical whether or not a
speculative job finished first. A failed or cancelled job only loses an
optimization; nothing ever waits on this pool.
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

from .invariants import InvariantSet, extract_invariants
from .messages import build_tool_call_index, iter_tool_results
from .resources import content_hash
from .task_context import Provenance, detect_provenance

logger = logging.getLogger(__name__)

_CACHE_MAX = 2048
MIN_CHARS = 2000


class SpeculativePreparer:
    def __init__(self, *, workers: int = 2, max_pending: int = 64) -> None:
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, workers), thread_name_prefix="headroom-spec"
        )
        self._max_pending = max(1, max_pending)
        self._lock = threading.Lock()
        self._pending: dict[str, Future[Any]] = {}
        self._invariants: OrderedDict[str, InvariantSet] = OrderedDict()
        self._provenance: OrderedDict[str, Provenance] = OrderedDict()
        self.submitted = 0
        self.completed = 0
        self.failed = 0
        self.rejected = 0

    # ----------------------------------------------------------------- cache
    def _remember(self, cache: OrderedDict[str, Any], key: str, value: Any) -> None:
        with self._lock:
            cache[key] = value
            cache.move_to_end(key)
            while len(cache) > _CACHE_MAX:
                cache.popitem(last=False)

    def invariants_for(self, text: str) -> InvariantSet | None:
        """Precomputed task-independent invariants (no user entities) if ready."""
        with self._lock:
            return self._invariants.get(content_hash(text))

    def provenance_for(self, text: str) -> Provenance | None:
        with self._lock:
            return self._provenance.get(content_hash(text))

    # ---------------------------------------------------------------- submit
    def _job(
        self,
        key: str,
        text: str,
        tool_name: str,
        tool_input: dict[str, Any],
        extra: Callable[[str], None] | None,
    ) -> None:
        try:
            prov = detect_provenance(text, tool_name, tool_input)
            self._remember(self._provenance, key, prov)
            self._remember(self._invariants, key, extract_invariants(text, None, prov))
            from headroom.ccr.span_index import INDEX_CACHE

            INDEX_CACHE.prime(key, text)
            if extra is not None:
                extra(text)
            with self._lock:
                self.completed += 1
        except Exception:  # noqa: BLE001 - speculation only loses an optimization
            with self._lock:
                self.failed += 1
            logger.debug("speculative job failed", exc_info=True)
        finally:
            with self._lock:
                self._pending.pop(key, None)

    def prepare(
        self,
        messages: list[dict[str, Any]],
        *,
        start: int = 0,
        extra: Callable[[str], None] | None = None,
    ) -> int:
        """Queue preparation for live-zone tool results. Never blocks."""
        queued = 0
        for ref in iter_tool_results(messages, build_tool_call_index(messages), start=start):
            if len(ref.text) < MIN_CHARS:
                continue
            key = content_hash(ref.text)
            with self._lock:
                if key in self._pending or key in self._invariants:
                    continue
                if len(self._pending) >= self._max_pending:
                    self.rejected += 1
                    continue
                self.submitted += 1
                try:
                    fut = self._executor.submit(
                        self._job, key, ref.text, ref.tool_name, ref.tool_input, extra
                    )
                except RuntimeError:  # executor shut down
                    return queued
                self._pending[key] = fut
            queued += 1
        return queued

    def cancel_pending(self) -> int:
        with self._lock:
            futures = list(self._pending.values())
        return sum(1 for f in futures if f.cancel())

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "submitted": self.submitted,
                "completed": self.completed,
                "failed": self.failed,
                "rejected": self.rejected,
                "pending": len(self._pending),
                "cached_invariants": len(self._invariants),
            }

    def shutdown(self) -> None:
        self.cancel_pending()
        self._executor.shutdown(wait=False, cancel_futures=True)
