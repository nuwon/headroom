"""DecisionAdvisor — the single JevK5 abstraction used everywhere (plan §4.10).

Hard rules enforced here, not at call sites:

* deadline per call (``HEADROOM_JEVK5_TIMEOUT_MS``) — the request path never
  waits longer, whatever the model is doing;
* at most ``HEADROOM_JEVK5_MAX_CALLS_PER_REQUEST`` calls per Headroom request
  (a :func:`request_scope` sets the budget; calls outside a scope get a
  one-call budget);
* identical decisions are de-duplicated by hashing the canonical
  ``(state, question)`` and cached for ``…_DECISION_CACHE_TTL_SECONDS``;
* decision state is bounded (``HEADROOM_JEVK5_MAX_STATE_TOKENS``, default
  ~6k tokens) — oversized state is cut, never sent unbounded;
* predominantly non-English state is not sent (JevK5 is English-only);
* below ``HEADROOM_JEVK5_MIN_CONFIDENCE`` the answer carries weight 0;
* per-family trust calibration only ever *lowers* the configured weight.

Nothing raw is logged: telemetry is family, status, latency, confidence and
counts. Any failure returns ``None`` and the caller's deterministic path runs.
"""

from __future__ import annotations

import contextlib
import contextvars
import hashlib
import json
import logging
import threading
import time
from collections import OrderedDict
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from typing import Any

from .config import MAX_ADVISORY_WEIGHT, IntelligenceConfig, JevK5Settings
from .jevk5_client import DecisionClient, GGUFDecisionClient, SystemOneClient
from .models import AdvisoryRequest, AdvisoryScores, DecisionFamily, QuestionType

logger = logging.getLogger(__name__)

_CACHE_MAX = 512
_MIN_TRUST_OBSERVATIONS = 10


class RequestBudget:
    def __init__(self, calls: int) -> None:
        self._left = max(0, int(calls))
        self._lock = threading.Lock()

    def take(self) -> bool:
        with self._lock:
            if self._left <= 0:
                return False
            self._left -= 1
            return True

    @property
    def remaining(self) -> int:
        return self._left


_BUDGET: contextvars.ContextVar[RequestBudget | None] = contextvars.ContextVar(
    "headroom_jevk5_budget", default=None
)


@dataclass
class FamilyStats:
    calls: int = 0
    ok: int = 0
    timeouts: int = 0
    errors: int = 0
    low_confidence: int = 0
    cache_hits: int = 0
    skipped: int = 0
    agreement: int = 0
    disagreement: int = 0
    later_retrieval_after_aggressive: int = 0
    corrections: int = 0
    latency_ms_total: float = 0.0
    confidence_total: float = 0.0

    def trust(self) -> float:
        """Multiplier in [0, 1] applied to the configured advisory weight."""
        observed = self.ok + self.timeouts + self.errors
        if observed < _MIN_TRUST_OBSERVATIONS:
            return 1.0
        failure = (self.timeouts + self.errors) / observed
        bad_outcomes = self.later_retrieval_after_aggressive + self.corrections
        outcome_rate = bad_outcomes / max(1, self.ok)
        return max(0.0, min(1.0, 1.0 - 1.5 * failure - 2.0 * outcome_rate))

    def to_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__.items())
        d["trust"] = round(self.trust(), 4)
        d["avg_latency_ms"] = round(self.latency_ms_total / self.ok, 2) if self.ok else 0.0
        d["avg_confidence"] = round(self.confidence_total / self.ok, 4) if self.ok else 0.0
        return d


def _is_mostly_english(text: str) -> bool:
    letters = [c for c in text[:4000] if c.isalpha()]
    if len(letters) < 20:
        return True
    ascii_letters = sum(1 for c in letters if c.isascii())
    return ascii_letters / len(letters) >= 0.7


def _bound_state(state: str, max_tokens: int) -> str:
    max_chars = max_tokens * 4
    if len(state) <= max_chars:
        return state
    head = max_chars * 2 // 3
    tail = max_chars - head - 40
    return f"{state[:head]}\n[… {len(state) - head - tail} chars omitted …]\n{state[-tail:]}"


class DecisionAdvisor:
    """Thread-safe, fail-open JevK5 advisor."""

    def __init__(
        self,
        config: IntelligenceConfig,
        *,
        client: DecisionClient | None = None,
        service: Any | None = None,
    ) -> None:
        self.config = config
        self.settings: JevK5Settings = config.jevk5
        self._client = client
        self._service = service
        self._lock = threading.Lock()
        self._cache: OrderedDict[str, tuple[float, dict[str, Any]]] = OrderedDict()
        self._stats: dict[str, FamilyStats] = {}
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="headroom-jevk5")
        self._inflight = threading.BoundedSemaphore(2)
        self._start_attempted = False
        self._unavailable_warned = False

    # ---------------------------------------------------------------- plumbing
    @contextlib.contextmanager
    def request_scope(self) -> Iterator[RequestBudget]:
        budget = RequestBudget(self.settings.max_calls_per_request)
        token = _BUDGET.set(budget)
        try:
            yield budget
        finally:
            _BUDGET.reset(token)

    def _stats_for(self, family: DecisionFamily) -> FamilyStats:
        st = self._stats.get(family.value)
        if st is None:
            st = self._stats[family.value] = FamilyStats()
        return st

    def ensure_started(self, *, background: bool = True) -> None:
        """Kick off the local service once (auto/on + autostart)."""
        if self._client is not None or self.settings.mode == "off":
            return
        with self._lock:
            if self._start_attempted:
                return
            self._start_attempted = True
        if self._service is None:
            from .jevk5_service import get_service

            self._service = get_service(self.settings)
        if not self.settings.autostart and not self.settings.external:
            return
        try:
            status = self._service.start(wait=not background, wait_timeout_s=300.0)
        except Exception as exc:  # noqa: BLE001 - startup must never break the proxy
            logger.warning("JevK5 advisor unavailable (%s); continuing deterministically", exc)
            return
        if not status.usable and status.state != "starting":
            if self.settings.mode == "on":
                logger.error("JevK5 (HEADROOM_JEVK5=on) unavailable: %s", status.reason)
            else:
                logger.warning(
                    "JevK5 advisor not started (%s); Headroom continues deterministically",
                    status.reason,
                )

    def _resolve_client(self) -> DecisionClient | None:
        if self._client is not None:
            return self._client
        timeout = self.settings.timeout_ms / 1000.0
        if self.settings.external:
            self._client = SystemOneClient(self.settings.url, timeout_s=timeout)
            return self._client
        if self._service is None:
            return None
        url = self._service.url()
        if not url:
            return None
        self._client = GGUFDecisionClient(
            url,
            temperature=self.settings.temperature,
            knockout_temperature=self.settings.knockout_temperature,
            top_k=self.settings.top_k,
            timeout_s=timeout,
        )
        return self._client

    def available(self) -> bool:
        if self.settings.mode == "off":
            return False
        return self._resolve_client() is not None

    # ------------------------------------------------------------------ advise
    def advise(self, request: AdvisoryRequest) -> AdvisoryScores | None:
        family = request.family
        stats = self._stats_for(family)
        if self.settings.mode == "off" or self.settings.max_calls_per_request <= 0:
            return None
        if not _is_mostly_english(request.state + " " + request.instructions):
            stats.skipped += 1
            return None
        state = _bound_state(request.state, self.settings.max_state_tokens)
        question = request.to_question()
        key = hashlib.sha256(
            json.dumps({"s": state, "q": question}, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()
        now = time.monotonic()
        with self._lock:
            hit = self._cache.get(key)
            if hit is not None and hit[0] > now:
                self._cache.move_to_end(key)
                stats.cache_hits += 1
                return self._scores(request, hit[1], 0.0, cached=True)
        budget = _BUDGET.get()
        if budget is None:
            budget = RequestBudget(1)
        if not budget.take():
            stats.skipped += 1
            return None
        client = self._resolve_client()
        if client is None:
            stats.skipped += 1
            return None
        if not self._inflight.acquire(blocking=False):
            stats.skipped += 1  # both workers busy: never queue behind a slow call
            return None
        stats.calls += 1
        started = time.perf_counter()
        future = self._executor.submit(client.decide, state, question)
        future.add_done_callback(lambda _f: self._inflight.release())
        try:
            answer = future.result(timeout=self.settings.timeout_ms / 1000.0)
        except FutureTimeout:
            stats.timeouts += 1
            logger.debug("jevk5 family=%s status=timeout", family.value)
            return None
        except Exception as exc:  # noqa: BLE001
            stats.errors += 1
            logger.debug("jevk5 family=%s status=error type=%s", family.value, type(exc).__name__)
            if self._service is not None and not self.settings.external:
                # Drop a dead client so the next request re-resolves the URL.
                self._client = None
            return None
        latency_ms = (time.perf_counter() - started) * 1000.0
        stats.ok += 1
        stats.latency_ms_total += latency_ms
        conf = float(answer.get("confidence", 0.0) or 0.0)
        stats.confidence_total += conf
        if self.settings.decision_cache_ttl_seconds > 0:
            with self._lock:
                self._cache[key] = (now + self.settings.decision_cache_ttl_seconds, answer)
                while len(self._cache) > _CACHE_MAX:
                    self._cache.popitem(last=False)
        scores = self._scores(request, answer, latency_ms)
        logger.debug(
            "jevk5 family=%s status=ok latency_ms=%.1f confidence=%.3f weight=%.3f",
            family.value,
            latency_ms,
            conf,
            scores.weight,
        )
        return scores

    def _scores(
        self,
        request: AdvisoryRequest,
        answer: dict[str, Any],
        latency_ms: float,
        *,
        cached: bool = False,
    ) -> AdvisoryScores:
        family = request.family
        stats = self._stats_for(family)
        conf = float(answer.get("confidence", 0.0) or 0.0)
        probs_raw = answer.get("probabilities") or {}
        probs = (
            {str(k): float(v) for k, v in probs_raw.items()} if isinstance(probs_raw, dict) else {}
        )
        noul = answer.get("noul")
        if request.qtype is QuestionType.NOUL and noul is not None:
            probs = {"true": float(noul), "false": 1.0 - float(noul)}
        weight = min(MAX_ADVISORY_WEIGHT, self.settings.advisory_weight) * stats.trust()
        if conf < self.settings.min_confidence:
            if not cached:
                stats.low_confidence += 1
            weight = 0.0
        return AdvisoryScores(
            family=family,
            probabilities=probs,
            confidence=conf,
            latency_ms=latency_ms,
            input_tokens=int(answer.get("input_tokens", 0) or 0),
            weight=weight,
            expected_score=float(answer["score"]) if answer.get("score") is not None else None,
            noul=float(noul) if noul is not None else None,
            cached=cached,
        )

    # -------------------------------------------------------- convenience API
    def choose(
        self, family: DecisionFamily, state: str, instruction: str, options: dict[str, str]
    ) -> AdvisoryScores | None:
        if len(options) < 2:
            return None
        return self.advise(
            AdvisoryRequest(family, QuestionType.CHOICE, state, instruction, options=options)
        )

    def noul(self, family: DecisionFamily, state: str, proposition: str) -> AdvisoryScores | None:
        return self.advise(AdvisoryRequest(family, QuestionType.NOUL, state, proposition))

    def score(
        self, family: DecisionFamily, state: str, instruction: str, levels: tuple[str, ...]
    ) -> AdvisoryScores | None:
        return self.advise(
            AdvisoryRequest(family, QuestionType.SCORE, state, instruction, levels=levels)
        )

    # ----------------------------------------------------- outcome feedback
    def record_agreement(self, family: DecisionFamily, agreed: bool) -> None:
        stats = self._stats_for(family)
        if agreed:
            stats.agreement += 1
        else:
            stats.disagreement += 1

    def record_bad_outcome(self, family: DecisionFamily, *, retrieval: bool = True) -> None:
        stats = self._stats_for(family)
        if retrieval:
            stats.later_retrieval_after_aggressive += 1
        else:
            stats.corrections += 1

    def stats(self) -> dict[str, Any]:
        service = None
        if self._service is not None:
            with contextlib.suppress(Exception):
                service = self._service.status.to_dict()
        return {
            "mode": self.settings.mode,
            "external": self.settings.external,
            "available": self._client is not None,
            "service": service,
            "families": {k: v.to_dict() for k, v in sorted(self._stats.items())},
        }

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)


@dataclass
class _AdvisorHolder:
    advisor: DecisionAdvisor | None = None
    key: tuple[Any, ...] = field(default_factory=tuple)


_HOLDER = _AdvisorHolder()
_HOLDER_LOCK = threading.Lock()


def get_advisor(config: IntelligenceConfig) -> DecisionAdvisor | None:
    """Process-wide advisor for ``config`` (``None`` when JevK5 is off)."""
    if not config.advisor_enabled:
        return None
    key = (config.jevk5,)
    with _HOLDER_LOCK:
        if _HOLDER.advisor is None or _HOLDER.key != key:
            if _HOLDER.advisor is not None:
                _HOLDER.advisor.shutdown()
            _HOLDER.advisor = DecisionAdvisor(config)
            _HOLDER.key = key
        return _HOLDER.advisor


def set_advisor(advisor: DecisionAdvisor | None) -> None:
    """Install an advisor (tests / embedding applications)."""
    with _HOLDER_LOCK:
        _HOLDER.advisor = advisor
        _HOLDER.key = (advisor.config.jevk5,) if advisor is not None else ()


def current_advisor() -> DecisionAdvisor | None:
    return _HOLDER.advisor
