"""Decision models shared by the arbiter, the advisor and the gateway.

The boundary between deterministic selection and the (optional) JevK5
advisor is a pair of plain data objects (plan §31):

* :class:`AdvisoryRequest` — compact descriptors only (never raw payloads);
* :class:`AdvisoryScores` — probabilities keyed by option id, plus the
  confidence/latency metadata used for trust calibration.

Deterministic selection is identical whether ``AdvisoryScores`` is ``None``
(advisor absent, timed out, low confidence) or present with weight 0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class DecisionFamily(str, Enum):
    """Named JevK5 decision families (plan §22) — telemetry + trust keys."""

    TRANSFORM_CANDIDATE = "transform_candidate"
    RETRIEVAL_RERANK = "retrieval_rerank"
    TOOL_MATERIALIZATION = "tool_materialization"
    TURN_COMPLEXITY = "turn_complexity"
    GROUP_IMPORTANCE = "group_importance"
    SEMANTIC_REDUNDANCY = "semantic_redundancy"
    ADMISSION_MODE = "admission_mode"
    GRAPH_NEIGHBORHOOD = "graph_neighborhood"
    # Phase 2 agent-state classifications (finite choices only; never authoring).
    TASK_STATE = "task_state"
    EVIDENCE_CONFLICT = "evidence_conflict"
    TOOL_CONTRACT = "tool_contract"
    SCOPE_NECESSITY = "scope_necessity"
    TEST_IMPACT = "test_impact"


class QuestionType(str, Enum):
    NOUL = "noul"
    CHOICE = "choice"
    SCORE = "score"


@dataclass(frozen=True)
class AdvisoryRequest:
    """One typed question for the decision model."""

    family: DecisionFamily
    qtype: QuestionType
    state: str
    instructions: str
    # choice: option id -> short description; score: ordered level texts;
    # noul: optional {"true": ..., "false": ...} descriptions.
    options: dict[str, str] = field(default_factory=dict)
    levels: tuple[str, ...] = ()

    def to_question(self) -> dict[str, Any]:
        q: dict[str, Any] = {"type": self.qtype.value, "instructions": self.instructions}
        if self.qtype is QuestionType.CHOICE:
            q["criteria"] = dict(self.options)
        elif self.qtype is QuestionType.SCORE:
            q["criteria"] = list(self.levels)
        elif self.options:
            q["criteria"] = dict(self.options)
        return q


@dataclass(frozen=True)
class AdvisoryScores:
    """A calibrated answer from the advisor."""

    family: DecisionFamily
    probabilities: dict[str, float]
    confidence: float
    latency_ms: float = 0.0
    input_tokens: int = 0
    # Effective weight the caller should apply (already trust-calibrated and
    # clamped to the configured maximum). 0 means "advisory only, ignore".
    weight: float = 0.0
    expected_score: float | None = None
    noul: float | None = None
    cached: bool = False

    def top(self) -> str | None:
        if not self.probabilities:
            return None
        return max(sorted(self.probabilities), key=lambda k: self.probabilities[k])


@dataclass
class AdvisorOutcome:
    """Why an advisory call did or did not produce a usable answer."""

    family: DecisionFamily
    status: str  # ok | unavailable | timeout | budget | low_confidence | error | skipped
    latency_ms: float = 0.0
    confidence: float = 0.0
