"""Finite JevK5 classifications for the agent-state layer.

JevK5 is advisory (plan §0.7). It is asked only finite questions with lettered
options, and its answer is used only when the Phase 1 advisor trusts it
(``weight > 0``) and its probability clears the caller's threshold. It never
authors text, never hard-blocks, and never overrides an authority rule. When
it is unavailable, times out or has no budget left, the caller gets ``None``
and the deterministic path continues (Scenario I).
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def classify(
    advisor: Any | None,
    family: Any,
    state: str,
    instruction: str,
    options: dict[str, str],
) -> tuple[str, float] | None:
    """``(option_id, probability)`` from JevK5, or None (fail-open)."""
    if advisor is None or len(options) < 2:
        return None
    try:
        scores = advisor.choose(family, state[:1600], instruction, options)
    except Exception:  # noqa: BLE001
        logger.debug("agent-state advisor call failed", exc_info=True)
        return None
    if scores is None or getattr(scores, "weight", 0.0) <= 0.0:
        return None
    top = scores.top()
    if top is None:
        return None
    prob = float(scores.probabilities.get(top, 0.0))
    return top, prob
