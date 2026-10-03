"""Central compression-policy admission (Optimization 13, plan §6).

``CompressionPolicy.max_lossy_ratio`` and ``volatile_token_threshold`` were
plumbed but never consumed. This module is the ONE place that turns them into
runtime decisions; every lossy path (router, arbiter, admission, delta) calls
:func:`admit_candidate` — there are no transform-specific copies.

Two drop ratios are tracked separately:

``wire_drop_ratio``
    Fraction of tokens removed from the wire.
``irreversible_drop_ratio``
    Fraction of tokens that cannot be brought back. Lossless folds and exact
    CCR externalization (original stored, marker verified) are ~0 here even
    when the wire drop is very high.

Gate: ``irreversible_drop_ratio <= policy.max_lossy_ratio``. Exact-recovery
candidates may exceed the cap on the wire, but only when their retrieval
marker is valid.

The Rust twin lives in ``crates/headroom-core/src/compression_policy.rs``
(``drop_ratios`` / ``admit_lossy`` / ``classify_change``) and the two are kept
in lockstep by ``tests/test_intelligence_policy_parity.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class PolicyLike(Protocol):
    @property
    def max_lossy_ratio(self) -> float: ...

    @property
    def volatile_token_threshold(self) -> int: ...


@dataclass(frozen=True)
class DropRatios:
    wire_drop_ratio: float
    irreversible_drop_ratio: float


@dataclass(frozen=True)
class AdmissionDecision:
    allowed: bool
    reason: str
    ratios: DropRatios


def drop_ratios(
    original_tokens: int,
    candidate_tokens: int,
    *,
    recoverable: bool,
    lossless: bool,
) -> DropRatios:
    """Compute wire vs irreversible drop for a candidate."""
    orig = max(0, int(original_tokens))
    cand = max(0, int(candidate_tokens))
    if orig == 0:
        return DropRatios(0.0, 0.0)
    wire = max(0, orig - cand) / orig
    irreversible = 0.0 if (lossless or recoverable) else wire
    return DropRatios(round(wire, 6), round(irreversible, 6))


def admit_candidate(
    policy: PolicyLike | None,
    original_tokens: int,
    candidate_tokens: int,
    *,
    recoverable: bool,
    lossless: bool,
    marker_valid: bool = True,
) -> AdmissionDecision:
    """The single policy admission gate for every lossy candidate.

    Order of checks (identical in Rust):

    1. non-positive token savings      -> reject ``non_positive_savings``
    2. recovery claimed, marker broken -> reject ``invalid_marker``
    3. irreversible drop > cap         -> reject ``max_lossy_ratio``
    """
    effective_recoverable = recoverable and marker_valid
    ratios = drop_ratios(
        original_tokens,
        candidate_tokens,
        recoverable=effective_recoverable,
        lossless=lossless,
    )
    if candidate_tokens >= original_tokens:
        return AdmissionDecision(False, "non_positive_savings", ratios)
    if recoverable and not marker_valid and not lossless:
        return AdmissionDecision(False, "invalid_marker", ratios)
    cap = 0.45 if policy is None else float(policy.max_lossy_ratio)
    cap = min(1.0, max(0.0, cap))
    if ratios.irreversible_drop_ratio > cap + 1e-9:
        return AdmissionDecision(False, "max_lossy_ratio", ratios)
    return AdmissionDecision(True, "ok", ratios)


def classify_change(policy: PolicyLike | None, changed_tokens: int) -> str:
    """Classify a content change against ``volatile_token_threshold``.

    ``"stable"``: the change is at or below the policy threshold — treat the
    content as cache-stable noise (e.g. a timestamp or counter ticking). The
    delta encoder treats such resources as near-identical and represents them
    by reference + the small change set.

    ``"volatile"``: the change exceeds the threshold — the content genuinely
    moved and cache-stability heuristics must not assume otherwise.

    Same semantics in Python and Rust (Subscription flags earlier at 32
    tokens, PAYG tolerates 128).
    """
    threshold = 128 if policy is None else int(policy.volatile_token_threshold)
    return "stable" if max(0, int(changed_tokens)) <= threshold else "volatile"
