"""Multi-candidate Transform Arbiter (Optimization 6, plan §13 + §31).

The arbiter stops committing to the first applicable compressor. Callers
generate several valid representations of one block (type-specific, lossless,
conservative, delta, indexed externalization …); the arbiter

1. measures every candidate (tokens, drop ratios, invariant report, policy
   admission, relevance retention, recoverability, cache cost, prior, latency);
2. applies the HARD prefilter (policy, invariants, positive savings, cache net
   cost, valid markers, partial-input provenance);
3. scores survivors with the deterministic utility (weights live in
   :class:`~headroom.intelligence.config.ArbiterWeights`, summing to 1.0);
4. optionally fuses a JevK5 advisory probability,
   ``final = (1 - w) * deterministic + w * p`` with ``w <= 0.40``;
5. applies the Pareto rule: the original is the quality baseline and wins
   whenever no compressed candidate is both safe and positive-value.

The session is split into ``prepare()`` / ``advisory_request()`` /
``select()`` so an orchestration layer can obtain advice asynchronously
without ever putting network I/O inside the pure selection logic. The
deterministic path is identical whether advice is absent, timed out or zero
weighted.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from .config import MAX_ADVISORY_WEIGHT, ArbiterWeights
from .invariants import (
    InvariantReport,
    InvariantSet,
    ccr_hashes_in,
    extract_invariants,
    validate_candidate,
)
from .models import AdvisoryRequest, AdvisoryScores, DecisionFamily, QuestionType
from .policy import PolicyLike, admit_candidate
from .task_context import _STOP_WORDS, EMPTY_TASK_CONTEXT, Provenance, TaskContext

# Backoff ladder tiers (plan §9). Lower = more aggressive.
TIER_AGGRESSIVE = 0
TIER_CONSERVATIVE = 1
TIER_STRUCTURAL = 2
TIER_DELTA = 3
TIER_INDEXED = 4
TIER_ORIGINAL = 5

TIER_NAMES = {
    TIER_AGGRESSIVE: "aggressive_lossy",
    TIER_CONSERVATIVE: "conservative_lossy",
    TIER_STRUCTURAL: "structural_lossless",
    TIER_DELTA: "delta",
    TIER_INDEXED: "indexed_externalization",
    TIER_ORIGINAL: "original",
}

#: Top-two deterministic scores closer than this are "ambiguous" (plan §13).
AMBIGUITY_MARGIN = 0.15
#: Non-recoverable candidates must keep at least this share of soft invariants.
MIN_SOFT_RECALL_UNRECOVERABLE = 0.25
#: Latency at which the latency component bottoms out.
LATENCY_SATURATION_MS = 250.0

_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.\-/]{2,}")


@dataclass
class Candidate:
    candidate_id: str
    strategy: str
    content: str
    tier: int = TIER_AGGRESSIVE
    lossless: bool = False
    latency_ms: float = 0.0
    # Filled by ArbiterSession.prepare()
    original_tokens: int = 0
    candidate_tokens: int = 0
    tokens_saved: int = 0
    wire_drop_ratio: float = 0.0
    irreversible_drop_ratio: float = 0.0
    report: InvariantReport | None = None
    policy_allowed: bool = True
    policy_reason: str = "ok"
    relevance_retention: float = 1.0
    recoverability_score: float = 0.0
    cache_cost_score: float = 1.0
    learned_prior_score: float = 0.5
    latency_score: float = 1.0
    deterministic_utility: float = 0.0
    advisory_p: float | None = None
    final_utility: float = 0.0
    rejected_reason: str | None = None
    ccr_hashes: tuple[str, ...] = ()

    @property
    def is_original(self) -> bool:
        return self.tier == TIER_ORIGINAL

    @property
    def safe(self) -> bool:
        return self.rejected_reason is None

    def descriptor(self) -> dict[str, object]:
        """Payload-free record for telemetry (no candidate content)."""
        return {
            "id": self.candidate_id,
            "strategy": self.strategy,
            "tier": TIER_NAMES.get(self.tier, str(self.tier)),
            "tokens": self.candidate_tokens,
            "saved": self.tokens_saved,
            "wire_drop": self.wire_drop_ratio,
            "irreversible_drop": self.irreversible_drop_ratio,
            "utility": round(self.final_utility, 4),
            "rejected": self.rejected_reason,
        }


@dataclass
class ArbiterDecision:
    selected: Candidate
    candidates: list[Candidate]
    rejections: dict[str, int] = field(default_factory=dict)
    advised: bool = False
    advisory: AdvisoryScores | None = None
    invariants: InvariantSet | None = None

    @property
    def kept_original(self) -> bool:
        return self.selected.is_original

    @property
    def tier_name(self) -> str:
        return TIER_NAMES.get(self.selected.tier, "unknown")


def _query_terms(task: TaskContext) -> tuple[list[str], list[str]]:
    exact = [e.lower() for e in task.explicit_entities if len(e) >= 2]
    prose: list[str] = []
    for word in _WORD_RE.findall(f"{task.current_user_text} {task.tool_input_summary}"):
        w = word.lower().strip(".-/")
        if len(w) >= 3 and w not in _STOP_WORDS and w not in prose and w not in exact:
            prose.append(w)
    return exact, prose[:40]


class ArbiterSession:
    """Prepare → (optional advice) → select, for one content block."""

    def __init__(
        self,
        original: str,
        *,
        count_tokens: Callable[[str], int],
        task: TaskContext | None = None,
        policy: PolicyLike | None = None,
        weights: ArbiterWeights | None = None,
        store_has: Callable[[str], bool] | None = None,
        prior_lookup: Callable[[str], float] | None = None,
        cache_penalty_tokens: int = 0,
        provenance: Provenance | None = None,
        content_type: str = "",
        enforce_invariants: bool = True,
        enforce_policy: bool = True,
        high_complexity: bool = False,
        invariants: InvariantSet | None = None,
    ) -> None:
        self.original = original
        self.count_tokens = count_tokens
        self.task = task or EMPTY_TASK_CONTEXT
        self.policy = policy
        self.weights = weights or ArbiterWeights()
        self.weights.validate()
        self.store_has = store_has
        self.prior_lookup = prior_lookup
        self.cache_penalty_tokens = max(0, int(cache_penalty_tokens))
        self.content_type = content_type
        self.enforce_invariants = enforce_invariants
        self.enforce_policy = enforce_policy
        self.high_complexity = high_complexity
        self.invariants = invariants or extract_invariants(original, self.task, provenance)
        self.original_tokens = max(1, count_tokens(original)) if original else 0
        self._exact, self._prose = _query_terms(self.task)
        lowered = original.lower()
        self._relevant_exact = [t for t in self._exact if t in lowered]
        self._relevant_prose = [t for t in self._prose if t in lowered]
        self.candidates: list[Candidate] = []

    # ------------------------------------------------------------------ prepare
    def _retention(self, content: str) -> float:
        if not self._relevant_exact and not self._relevant_prose:
            return 1.0
        lowered = content.lower()
        weight_total = 2.0 * len(self._relevant_exact) + len(self._relevant_prose)
        kept = 2.0 * sum(1 for t in self._relevant_exact if t in lowered) + sum(
            1 for t in self._relevant_prose if t in lowered
        )
        return kept / weight_total if weight_total else 1.0

    def _measure(self, cand: Candidate) -> None:
        cand.original_tokens = self.original_tokens
        if cand.is_original:
            cand.candidate_tokens = self.original_tokens
            cand.report = InvariantReport(True, 1.0, True, True, True, True)
            cand.recoverability_score = 1.0
            cand.relevance_retention = 1.0
            return
        cand.candidate_tokens = max(0, self.count_tokens(cand.content))
        cand.tokens_saved = max(0, cand.original_tokens - cand.candidate_tokens)
        cand.ccr_hashes = ccr_hashes_in(cand.content)
        report = validate_candidate(
            self.original,
            cand.content,
            self.invariants,
            original_tokens=cand.original_tokens,
            candidate_tokens=cand.candidate_tokens,
            lossless=cand.lossless,
            store_has=self.store_has,
        )
        cand.report = report
        marker_valid = "retrieval_marker_unresolvable" not in report.violations
        admission = admit_candidate(
            self.policy,
            cand.original_tokens,
            cand.candidate_tokens,
            recoverable=report.recoverable,
            lossless=cand.lossless,
            marker_valid=marker_valid,
        )
        cand.wire_drop_ratio = admission.ratios.wire_drop_ratio
        cand.irreversible_drop_ratio = admission.ratios.irreversible_drop_ratio
        cand.policy_allowed = admission.allowed
        cand.policy_reason = admission.reason
        cand.relevance_retention = self._retention(cand.content)
        if cand.lossless:
            cand.recoverability_score = 1.0
        elif report.recoverable:
            cand.recoverability_score = 0.9
        else:
            cand.recoverability_score = max(0.0, 1.0 - cand.irreversible_drop_ratio)
        if self.cache_penalty_tokens:
            cand.cache_cost_score = max(
                0.0, 1.0 - self.cache_penalty_tokens / max(1, cand.tokens_saved)
            )
        if self.prior_lookup is not None:
            try:
                cand.learned_prior_score = min(
                    1.0, max(0.0, float(self.prior_lookup(cand.strategy)))
                )
            except Exception:  # noqa: BLE001 - a broken prior is a neutral prior
                cand.learned_prior_score = 0.5
        cand.latency_score = max(0.0, 1.0 - cand.latency_ms / LATENCY_SATURATION_MS)

    def _prefilter(self, cand: Candidate) -> None:
        if cand.is_original:
            return
        report = cand.report
        assert report is not None
        if cand.tokens_saved <= 0:
            cand.rejected_reason = "non_positive_savings"
        elif self.enforce_policy and not cand.policy_allowed:
            cand.rejected_reason = f"policy:{cand.policy_reason}"
        elif self.enforce_invariants and not report.hard_invariants_preserved:
            cand.rejected_reason = f"invariant:{report.violations[0]}"
        elif (
            self.enforce_invariants
            and not report.recoverable
            and not cand.lossless
            and report.soft_invariant_recall < MIN_SOFT_RECALL_UNRECOVERABLE
        ):
            cand.rejected_reason = "invariant:soft_recall_floor"
        elif self.cache_penalty_tokens and self.cache_penalty_tokens >= cand.tokens_saved:
            # plan §25: cache cost >= savings keeps the byte-stable form.
            cand.rejected_reason = "cache_net_cost"

    def _utility(self, cand: Candidate) -> float:
        w = self.weights
        report = cand.report
        soft = report.soft_invariant_recall if report else 1.0
        savings = (cand.tokens_saved / cand.original_tokens) if cand.original_tokens else 0.0
        return (
            w.relevance_retention * cand.relevance_retention
            + w.soft_invariant_recall * soft
            + w.recoverability * cand.recoverability_score
            + w.token_savings * savings
            + w.cache_cost * cand.cache_cost_score
            + w.learned_prior * cand.learned_prior_score
            + w.latency * cand.latency_score
        )

    def prepare(self, candidates: Iterable[Candidate]) -> list[Candidate]:
        seen_content: set[str] = set()
        prepared: list[Candidate] = []
        have_original = False
        for cand in candidates:
            if cand.is_original:
                if have_original:
                    continue
                have_original = True
            elif cand.content in seen_content or cand.content == self.original:
                continue
            seen_content.add(cand.content)
            self._measure(cand)
            self._prefilter(cand)
            cand.deterministic_utility = self._utility(cand)
            cand.final_utility = cand.deterministic_utility
            prepared.append(cand)
        if not have_original:
            orig = Candidate("original", "original", self.original, tier=TIER_ORIGINAL)
            self._measure(orig)
            orig.deterministic_utility = self._utility(orig)
            orig.final_utility = orig.deterministic_utility
            prepared.append(orig)
        self.candidates = prepared
        return prepared

    # ------------------------------------------------------------------ advice
    def safe_compressed(self) -> list[Candidate]:
        return [c for c in self.candidates if c.safe and not c.is_original]

    def advisory_request(self, *, max_options: int = 6) -> AdvisoryRequest | None:
        """Return a JevK5 question when advice has expected value (plan §13)."""
        safe = sorted(self.safe_compressed(), key=lambda c: -c.deterministic_utility)
        if len(safe) < 2:
            return None
        top, second = safe[0], safe[1]
        close = abs(top.deterministic_utility - second.deterministic_utility) <= AMBIGUITY_MARGIN
        prose = self.content_type in ("plain_text", "text", "mixed", "html", "")
        tradeoff = any(
            a.tokens_saved > b.tokens_saved
            and (a.report.soft_invariant_recall if a.report else 1)
            < (b.report.soft_invariant_recall if b.report else 1)
            for a in safe[:3]
            for b in safe[:3]
        )
        if not (close or prose or self.high_complexity or tradeoff):
            return None
        options: dict[str, str] = {}
        total_exact = len(self._relevant_exact)
        for cand in safe[:max_options]:
            kept = sum(1 for t in self._relevant_exact if t in cand.content.lower())
            preview = " ".join(cand.content[:200].split())
            options[cand.candidate_id] = (
                f"{cand.strategy}: saves {cand.tokens_saved} of {cand.original_tokens} tokens "
                f"({100 * cand.wire_drop_ratio:.0f}%); keeps {kept}/{total_exact} task entities; "
                f"{'exactly recoverable' if (cand.report and cand.report.recoverable) or cand.lossless else 'not recoverable'}; "
                f"preview: {preview[:160]}"
            )
        goal = self.task.current_user_goal_text[:600] or "(no explicit user goal)"
        state = (
            f"Task goal: {goal}\n"
            f"Content type: {self.content_type or 'unknown'}; original size {self.original_tokens} tokens.\n"
            f"Task entities: {', '.join(self._relevant_exact[:12]) or 'none'}"
        )
        return AdvisoryRequest(
            family=DecisionFamily.TRANSFORM_CANDIDATE,
            qtype=QuestionType.CHOICE,
            state=state,
            instructions=(
                "Which candidate representation best preserves the information needed for the "
                "current task while minimizing irrelevant context?"
            ),
            options=options,
        )

    # ------------------------------------------------------------------ select
    def select(self, advisory: AdvisoryScores | None = None) -> ArbiterDecision:
        if not self.candidates:
            raise RuntimeError("prepare() must run before select()")
        weight = 0.0
        if advisory is not None and advisory.probabilities:
            weight = min(MAX_ADVISORY_WEIGHT, max(0.0, advisory.weight))
        for cand in self.candidates:
            cand.advisory_p = None
            cand.final_utility = cand.deterministic_utility
            if weight > 0.0 and advisory is not None and not cand.is_original and cand.safe:
                p = advisory.probabilities.get(cand.candidate_id)
                if p is not None:
                    cand.advisory_p = p
                    cand.final_utility = (1 - weight) * cand.deterministic_utility + weight * p
        safe = self.safe_compressed()
        rejections: dict[str, int] = {}
        for cand in self.candidates:
            if cand.rejected_reason:
                rejections[cand.rejected_reason] = rejections.get(cand.rejected_reason, 0) + 1
        if safe:
            safe.sort(
                key=lambda c: (-round(c.final_utility, 9), c.candidate_tokens, c.candidate_id)
            )
            selected = safe[0]
        else:
            selected = next(c for c in self.candidates if c.is_original)
        return ArbiterDecision(
            selected=selected,
            candidates=self.candidates,
            rejections=rejections,
            advised=weight > 0.0,
            advisory=advisory if weight > 0.0 else None,
            invariants=self.invariants,
        )


def arbitrate(
    original: str,
    candidates: Iterable[Candidate],
    *,
    count_tokens: Callable[[str], int],
    advisor: Callable[[AdvisoryRequest], AdvisoryScores | None] | None = None,
    **session_kwargs: object,
) -> ArbiterDecision:
    """Convenience one-shot: prepare, ask the advisor if useful, select."""
    session = ArbiterSession(original, count_tokens=count_tokens, **session_kwargs)  # type: ignore[arg-type]
    session.prepare(candidates)
    advice: AdvisoryScores | None = None
    if advisor is not None:
        request = session.advisory_request()
        if request is not None:
            try:
                advice = advisor(request)
            except Exception:  # noqa: BLE001 - advice is optional by contract
                advice = None
    return session.select(advice)
