"""Retrieval-feedback retention learning (Optimization 12, plan §19).

Future behavior is supervision for what should have been kept earlier:

"retain more" (outcome 0)
    a CCR retrieval of something a strategy omitted; a re-read of a resource
    right after it was compressed; a re-run of the same search.
"can compress more" (outcome 1)
    compressed content that ages past half its TTL without ever being
    retrieved.

Priors are learned per *generalized* feature key — tool signature hash,
content kind, strategy, provider family — never raw content or queries:

    prior_new = clamp((1 - alpha) * prior_old + alpha * outcome, 0.05, 0.95)

A prior only influences decisions after ``min_observations``; before that it
reads as the neutral 0.5. Priors adjust arbiter utility, span-ranking priors
and compressor retention bias *within* the hard policy cap — they never
disable an invariant or raise ``max_lossy_ratio``.

When a JevK5-advised choice is later retrieved, the advisor is told
(:meth:`DecisionAdvisor.record_bad_outcome`) so its per-family trust (and
therefore its weight) decreases.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from .models import DecisionFamily
from .resources import ResourceRef
from .state import intelligence_dir, read_json, write_json_atomic

logger = logging.getLogger(__name__)

MIN_PRIOR = 0.05
MAX_PRIOR = 0.95
NEUTRAL = 0.5
_TRACK_MAX = 4096
_SAVE_EVERY = 25


def _tool_signature(tool_name: str | None) -> str:
    return hashlib.sha256((tool_name or "").lower().encode()).hexdigest()[:12]


def feature_key(
    tool_name: str | None, strategy: str | None, kind: str = "", provider: str = ""
) -> str:
    """Generalized learning key — hashes and categories only."""
    strat = (strategy or "unknown").split(":")[0]
    return f"{_tool_signature(tool_name)}|{kind or '-'}|{strat}|{provider or '-'}"


@dataclass
class Prior:
    value: float = NEUTRAL
    observations: int = 0

    def update(self, outcome: float, alpha: float) -> None:
        self.value = min(MAX_PRIOR, max(MIN_PRIOR, (1 - alpha) * self.value + alpha * outcome))
        self.observations += 1


class RetentionLearner:
    def __init__(
        self,
        *,
        alpha: float = 0.15,
        min_observations: int = 5,
        persist: bool = True,
        advisor_provider: Any | None = None,
    ) -> None:
        self.alpha = alpha
        self.min_observations = min_observations
        self.persist = persist
        self._advisor_provider = advisor_provider
        self._lock = threading.Lock()
        self._priors: dict[str, Prior] = {}
        # entry hash -> (feature key, created_at, advised)
        self._tracked: OrderedDict[str, tuple[str, float, bool]] = OrderedDict()
        # resource identity -> (feature key, compressed_at)
        self._recent_rewrites: OrderedDict[str, tuple[str, float]] = OrderedDict()
        self._dirty = 0
        if persist:
            self._load()

    # ------------------------------------------------------------ persistence
    def _path(self):  # type: ignore[no-untyped-def]
        return intelligence_dir() / "retention_priors.json"

    def _load(self) -> None:
        data = read_json(self._path()) or {}
        for key, raw in (data.get("priors") or {}).items():
            if isinstance(raw, dict):
                self._priors[key] = Prior(
                    float(raw.get("value", NEUTRAL)), int(raw.get("observations", 0))
                )

    def save(self) -> None:
        if not self.persist:
            return
        with self._lock:
            payload = {
                "version": 1,
                "priors": {
                    k: {"value": round(p.value, 6), "observations": p.observations}
                    for k, p in self._priors.items()
                },
            }
            self._dirty = 0
        try:
            write_json_atomic(self._path(), payload)
        except OSError as exc:
            logger.debug("retention priors not saved: %s", exc)

    def _bump(self, key: str, outcome: float) -> None:
        with self._lock:
            self._priors.setdefault(key, Prior()).update(outcome, self.alpha)
            self._dirty += 1
            due = self._dirty >= _SAVE_EVERY
        if due:
            self.save()

    # -------------------------------------------------------------- signals
    def note_compressed(self, entry_hash: str, key: str, *, advised: bool = False) -> None:
        with self._lock:
            self._tracked[entry_hash] = (key, time.time(), advised)
            while len(self._tracked) > _TRACK_MAX:
                self._tracked.popitem(last=False)

    def on_retrieval(self, info: dict[str, Any]) -> None:
        """CompressionStore retrieval listener (payload-free metadata)."""
        h = str(info.get("hash") or "")
        with self._lock:
            tracked = self._tracked.pop(h, None)
        key = (
            tracked[0]
            if tracked
            else feature_key(info.get("tool_name"), info.get("compression_strategy"))
        )
        # A targeted search retrieval is a weaker "kept too little" signal.
        outcome = 0.0 if info.get("retrieval_type", "full") == "full" else 0.3
        self._bump(key, outcome)
        if tracked and tracked[2] and self._advisor_provider is not None:
            advisor = self._advisor_provider()
            if advisor is not None:
                advisor.record_bad_outcome(DecisionFamily.TRANSFORM_CANDIDATE, retrieval=True)

    def observe_rewrite(self, resource: ResourceRef, tag: str) -> None:
        """Called by the admission stage when it rewrites a resource."""
        with self._lock:
            self._recent_rewrites[resource.identity] = (
                feature_key(None, tag, kind=resource.kind),
                time.time(),
            )
            while len(self._recent_rewrites) > _TRACK_MAX:
                self._recent_rewrites.popitem(last=False)

    def observe_access(self, resource: ResourceRef, *, window_s: float = 600.0) -> None:
        """A resource was read/searched/listed again: penalize a recent rewrite.

        Re-running a command (tests, builds) after an edit is normal workflow,
        not evidence that context was missing, so ``cmd``/``tool`` resources
        never count.
        """
        if resource.kind not in ("file", "search", "list"):
            return
        with self._lock:
            hit = self._recent_rewrites.pop(resource.identity, None)
        if hit is not None and time.time() - hit[1] <= window_s:
            self._bump(hit[0], 0.0)

    def sweep(self, *, ttl_s: float = 1800.0) -> int:
        """Reward entries that aged past half the TTL without a retrieval."""
        cutoff = time.time() - ttl_s / 2
        rewarded: list[str] = []
        with self._lock:
            for h, (key, created, _adv) in list(self._tracked.items()):
                if created <= cutoff:
                    rewarded.append(key)
                    del self._tracked[h]
        for key in rewarded:
            self._bump(key, 1.0)
        return len(rewarded)

    # -------------------------------------------------------------- queries
    def prior(self, key: str) -> float:
        with self._lock:
            p = self._priors.get(key)
        if p is None or p.observations < self.min_observations:
            return NEUTRAL
        return p.value

    def prior_for_strategy(
        self, tool_name: str | None, strategy: str, kind: str = "", provider: str = ""
    ) -> float:
        return self.prior(feature_key(tool_name, strategy, kind, provider))

    def retention_bias(
        self, tool_name: str | None, strategy: str = "default", kind: str = "", provider: str = ""
    ) -> float:
        """Compressor bias from the prior: low prior (often retrieved) keeps more.

        Bounded to [0.8, 1.45] so learning can only nudge a compressor, never
        override its floors or the policy cap.
        """
        p = self.prior_for_strategy(tool_name, strategy, kind, provider)
        return round(min(1.45, max(0.8, 1.0 + (NEUTRAL - p))), 4)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            active = {
                k: p for k, p in self._priors.items() if p.observations >= self.min_observations
            }
            return {
                "keys": len(self._priors),
                "active_keys": len(active),
                "tracked_entries": len(self._tracked),
                "mean_prior": round(sum(p.value for p in active.values()) / len(active), 4)
                if active
                else NEUTRAL,
            }
