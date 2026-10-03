"""Feature 16: Task State Compiler (plan §5).

This keeps compact, typed, durable task state (goal, binding constraints,
acceptance criteria, decisions, progress, blockers) so the model does not
rebuild it from conversation history on every turn. It is not a summary and
not ``TaskContext``. TaskContext is request-scoped and serves relevance;
TaskState is task-scoped and serves executive continuity.

Extraction makes no frontier-model calls (plan §5.5):

* **Tier A, deterministic.** Explicit user text supplies the goal,
  constraints, acceptance criteria, decisions and subgoals. Tool evidence
  supplies test and build results. Agent completion declarations are read
  explicitly.
* **Tier B, JevK5.** Finite questions only: how a message relates to the goal,
  and whether two constraints conflict. JevK5 never writes atom text.
* **Tier C, conservative.** When a case is ambiguous the state is left as it
  was.

Every change goes through :func:`apply_transition`, which enforces the
authority rules of §5.4 and is persisted as revision ``N+1`` in one
transaction. Published :class:`TaskState` objects are immutable.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Iterable
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any

from .advice import classify
from .events import AgentEvent, EventType
from .ids import stable_id
from .store import dumps, loads

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
MIN_TRANSITION_CONFIDENCE = 0.65
NEW_TASK_CONFIDENCE = 0.80


class AtomKind(str, Enum):
    GOAL = "GOAL"
    SUBGOAL = "SUBGOAL"
    CONSTRAINT = "CONSTRAINT"
    CRITERION = "CRITERION"
    DECISION = "DECISION"
    COMPLETED = "COMPLETED"
    BLOCKER = "BLOCKER"
    QUESTION = "QUESTION"
    ASSUMPTION = "ASSUMPTION"


class AtomState(str, Enum):
    PROPOSED = "PROPOSED"
    ACTIVE = "ACTIVE"
    SATISFIED = "SATISFIED"
    VIOLATED = "VIOLATED"
    SUPERSEDED = "SUPERSEDED"
    REJECTED = "REJECTED"


class Origin(str, Enum):
    USER = "USER"
    TOOL = "TOOL"
    AGENT = "AGENT"
    DERIVED = "DERIVED"


class TaskStatus(str, Enum):
    ACTIVE = "ACTIVE"
    BLOCKED = "BLOCKED"
    COMPLETED = "COMPLETED"
    ABANDONED = "ABANDONED"


_LABEL_PREFIX = {
    AtomKind.GOAL: "G",
    AtomKind.SUBGOAL: "S",
    AtomKind.CONSTRAINT: "C",
    AtomKind.CRITERION: "A",
    AtomKind.DECISION: "D",
    AtomKind.COMPLETED: "K",
    AtomKind.BLOCKER: "B",
    AtomKind.QUESTION: "Q",
    AtomKind.ASSUMPTION: "M",
}


@dataclass(frozen=True)
class StateAtom:
    atom_id: str
    kind: AtomKind
    text: str
    normalized_key: str
    state: AtomState
    origin: Origin
    confidence: float
    label: str = ""
    evidence_ids: tuple[str, ...] = ()
    source_event_ids: tuple[str, ...] = ()
    supersedes_atom_id: str | None = None
    hard: bool = False
    blocking: bool = False
    required: bool = True
    verify: str = ""  # test | build | deliverable | ""
    targets: tuple[str, ...] = ()  # paths / test ids a criterion names
    key: str = ""  # identity for tool-derived atoms (e.g. blocker per command)
    created_at: float = 0.0
    updated_at: float = 0.0

    def to_json(self) -> dict[str, Any]:
        return {
            "atom_id": self.atom_id,
            "kind": self.kind.value,
            "text": self.text,
            "normalized_key": self.normalized_key,
            "state": self.state.value,
            "origin": self.origin.value,
            "confidence": round(self.confidence, 3),
            "label": self.label,
            "evidence_ids": list(self.evidence_ids),
            "source_event_ids": list(self.source_event_ids),
            "supersedes_atom_id": self.supersedes_atom_id,
            "hard": self.hard,
            "blocking": self.blocking,
            "required": self.required,
            "verify": self.verify,
            "targets": list(self.targets),
            "key": self.key,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> StateAtom:
        return cls(
            atom_id=str(d["atom_id"]),
            kind=AtomKind(d["kind"]),
            text=str(d.get("text", "")),
            normalized_key=str(d.get("normalized_key", "")),
            state=AtomState(d.get("state", "ACTIVE")),
            origin=Origin(d.get("origin", "USER")),
            confidence=float(d.get("confidence", 1.0)),
            label=str(d.get("label", "")),
            evidence_ids=tuple(d.get("evidence_ids") or ()),
            source_event_ids=tuple(d.get("source_event_ids") or ()),
            supersedes_atom_id=d.get("supersedes_atom_id"),
            hard=bool(d.get("hard", False)),
            blocking=bool(d.get("blocking", False)),
            required=bool(d.get("required", True)),
            verify=str(d.get("verify", "")),
            targets=tuple(d.get("targets") or ()),
            key=str(d.get("key", "")),
            created_at=float(d.get("created_at", 0.0)),
            updated_at=float(d.get("updated_at", 0.0)),
        )


@dataclass(frozen=True)
class TaskState:
    workspace_id: str
    session_id: str
    task_id: str
    revision: int
    status: TaskStatus
    atoms: tuple[StateAtom, ...]
    created_at: float
    updated_at: float
    declared_complete: bool = False
    schema_version: int = SCHEMA_VERSION
    counters: tuple[tuple[str, int], ...] = ()

    # -------------------------------------------------------------- views
    def by_kind(self, kind: AtomKind, *, live: bool = True) -> list[StateAtom]:
        out = [a for a in self.atoms if a.kind is kind]
        if live:
            out = [a for a in out if a.state not in (AtomState.SUPERSEDED, AtomState.REJECTED)]
        return out

    @property
    def primary_goal(self) -> StateAtom | None:
        goals = self.by_kind(AtomKind.GOAL)
        return goals[-1] if goals else None

    @property
    def subgoals(self) -> list[StateAtom]:
        return self.by_kind(AtomKind.SUBGOAL)

    @property
    def constraints(self) -> list[StateAtom]:
        return self.by_kind(AtomKind.CONSTRAINT)

    @property
    def acceptance_criteria(self) -> list[StateAtom]:
        return self.by_kind(AtomKind.CRITERION)

    @property
    def decisions(self) -> list[StateAtom]:
        return self.by_kind(AtomKind.DECISION)

    @property
    def completed_items(self) -> list[StateAtom]:
        return self.by_kind(AtomKind.COMPLETED)

    @property
    def blockers(self) -> list[StateAtom]:
        return [a for a in self.by_kind(AtomKind.BLOCKER) if a.state is AtomState.ACTIVE]

    @property
    def unresolved_questions(self) -> list[StateAtom]:
        return [a for a in self.by_kind(AtomKind.QUESTION) if a.state is AtomState.ACTIVE]

    @property
    def assumptions(self) -> list[StateAtom]:
        return self.by_kind(AtomKind.ASSUMPTION)

    def atom(self, atom_id: str) -> StateAtom | None:
        for a in self.atoms:
            if a.atom_id == atom_id:
                return a
        return None

    def find_key(self, kind: AtomKind, key: str) -> StateAtom | None:
        for a in self.atoms:
            if a.kind is kind and (a.key == key or a.normalized_key == key):
                return a
        return None

    def counter(self, prefix: str) -> int:
        return dict(self.counters).get(prefix, 0)

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "workspace_id": self.workspace_id,
            "session_id": self.session_id,
            "task_id": self.task_id,
            "revision": self.revision,
            "status": self.status.value,
            "atoms": [a.to_json() for a in self.atoms],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "declared_complete": self.declared_complete,
            "counters": dict(self.counters),
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> TaskState:
        return cls(
            workspace_id=str(d.get("workspace_id", "")),
            session_id=str(d.get("session_id", "")),
            task_id=str(d["task_id"]),
            revision=int(d.get("revision", 0)),
            status=TaskStatus(d.get("status", "ACTIVE")),
            atoms=tuple(StateAtom.from_json(a) for a in d.get("atoms") or ()),
            created_at=float(d.get("created_at", 0.0)),
            updated_at=float(d.get("updated_at", 0.0)),
            declared_complete=bool(d.get("declared_complete", False)),
            schema_version=int(d.get("schema_version", SCHEMA_VERSION)),
            counters=tuple(sorted((d.get("counters") or {}).items())),
        )


# ------------------------------------------------------------ transitions
class TransitionType(str, Enum):
    ADD_ATOM = "ADD_ATOM"
    SET_ATOM_STATE = "SET_ATOM_STATE"
    SUPERSEDE = "SUPERSEDE"
    ATTACH_EVIDENCE = "ATTACH_EVIDENCE"
    SET_STATUS = "SET_STATUS"
    DECLARE_COMPLETE = "DECLARE_COMPLETE"


@dataclass(frozen=True)
class StateTransition:
    transition_id: str
    transition_type: TransitionType
    reason_code: str
    confidence: float = 1.0
    atom_id: str | None = None
    new_atom: StateAtom | None = None
    new_state: AtomState | None = None
    new_status: TaskStatus | None = None
    source_event_ids: tuple[str, ...] = ()
    supporting_evidence_ids: tuple[str, ...] = ()
    declared: bool | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "transition_id": self.transition_id,
            "transition_type": self.transition_type.value,
            "reason_code": self.reason_code,
            "confidence": self.confidence,
            "atom_id": self.atom_id,
            "new_atom": self.new_atom.to_json() if self.new_atom else None,
            "new_state": self.new_state.value if self.new_state else None,
            "new_status": self.new_status.value if self.new_status else None,
            "source_event_ids": list(self.source_event_ids),
            "supporting_evidence_ids": list(self.supporting_evidence_ids),
            "declared": self.declared,
        }


class InvalidTransition(ValueError):
    """A transition violates an authority rule (plan §5.4)."""


_DESTRUCTIVE_STATES = (AtomState.SATISFIED, AtomState.VIOLATED, AtomState.REJECTED)


def completion_blockers(state: TaskState) -> list[str]:
    """Why ``state`` cannot be COMPLETED (empty when it can), per rule 8."""
    reasons = []
    for a in state.acceptance_criteria:
        if a.required and a.state is not AtomState.SATISFIED:
            reasons.append(f"criterion {a.label} not satisfied")
    for a in state.constraints:
        if a.hard and a.state is AtomState.VIOLATED:
            reasons.append(f"hard constraint {a.label} violated")
    for a in state.blockers:
        if a.blocking:
            reasons.append(f"blocker {a.label} unresolved")
    return reasons


def apply_transition(
    state: TaskState, tr: StateTransition, *, now: float | None = None
) -> TaskState:
    """Return the new state for ``tr``; raise :class:`InvalidTransition` on a rule breach."""
    now = time.time() if now is None else now
    atoms = list(state.atoms)
    counters = dict(state.counters)
    t = tr.transition_type

    def index_of(atom_id: str | None) -> int:
        for i, a in enumerate(atoms):
            if a.atom_id == atom_id:
                return i
        raise InvalidTransition(f"unknown atom {atom_id}")

    if t is TransitionType.ADD_ATOM:
        atom = tr.new_atom
        if atom is None:
            raise InvalidTransition("ADD_ATOM without atom")
        if any(a.atom_id == atom.atom_id for a in atoms):
            raise InvalidTransition("duplicate atom id")
        prefix = _LABEL_PREFIX[atom.kind]
        counters[prefix] = counters.get(prefix, 0) + 1
        atoms.append(
            replace(
                atom,
                label=atom.label or f"{prefix}{counters[prefix]}",
                created_at=now,
                updated_at=now,
            )
        )
    elif t is TransitionType.SET_ATOM_STATE:
        i = index_of(tr.atom_id)
        atom = atoms[i]
        new_state = tr.new_state
        if new_state is None:
            raise InvalidTransition("SET_ATOM_STATE without state")
        if new_state in _DESTRUCTIVE_STATES and tr.confidence < MIN_TRANSITION_CONFIDENCE:
            raise InvalidTransition("confidence below 0.65 for a destructive/completion transition")
        if atom.state is AtomState.SUPERSEDED:
            raise InvalidTransition("superseded atoms are immutable")
        if (
            atom.kind is AtomKind.CRITERION
            and new_state is AtomState.SATISFIED
            and atom.verify in ("test", "build")
            and not tr.supporting_evidence_ids
        ):
            # Rules 3/4: a factual criterion needs tool evidence, never prose alone.
            raise InvalidTransition("factual criterion requires supporting evidence")
        if (
            atom.kind is AtomKind.CONSTRAINT
            and atom.origin is Origin.USER
            and new_state
            in (
                AtomState.REJECTED,
                AtomState.SUPERSEDED,
            )
        ):
            raise InvalidTransition("user constraints change only by a newer user instruction")
        atoms[i] = replace(
            atom,
            state=new_state,
            evidence_ids=tuple(dict.fromkeys((*atom.evidence_ids, *tr.supporting_evidence_ids))),
            source_event_ids=tuple(dict.fromkeys((*atom.source_event_ids, *tr.source_event_ids))),
            updated_at=now,
        )
    elif t is TransitionType.SUPERSEDE:
        i = index_of(tr.atom_id)
        old = atoms[i]
        new = tr.new_atom
        if new is None:
            raise InvalidTransition("SUPERSEDE without replacement")
        if old.origin is Origin.USER and new.origin is not Origin.USER:
            raise InvalidTransition("only a newer explicit user instruction supersedes a user atom")
        if old.state is AtomState.SUPERSEDED:
            raise InvalidTransition("already superseded")
        if tr.confidence < MIN_TRANSITION_CONFIDENCE:
            raise InvalidTransition("confidence below 0.65 for supersession")
        atoms[i] = replace(old, state=AtomState.SUPERSEDED, updated_at=now)
        prefix = _LABEL_PREFIX[new.kind]
        counters[prefix] = counters.get(prefix, 0) + 1
        atoms.append(
            replace(
                new,
                supersedes_atom_id=old.atom_id,
                label=new.label or f"{prefix}{counters[prefix]}",
                created_at=now,
                updated_at=now,
            )
        )
    elif t is TransitionType.ATTACH_EVIDENCE:
        i = index_of(tr.atom_id)
        atom = atoms[i]
        atoms[i] = replace(
            atom,
            evidence_ids=tuple(dict.fromkeys((*atom.evidence_ids, *tr.supporting_evidence_ids))),
            updated_at=now,
        )
    elif t is TransitionType.SET_STATUS:
        status = tr.new_status
        if status is None:
            raise InvalidTransition("SET_STATUS without status")
        if status in (TaskStatus.COMPLETED, TaskStatus.ABANDONED) and (
            tr.confidence < MIN_TRANSITION_CONFIDENCE
        ):
            raise InvalidTransition("confidence below 0.65 for completion")
        if status is TaskStatus.COMPLETED:
            reasons = completion_blockers(state)
            if reasons:
                raise InvalidTransition("; ".join(reasons))
        return replace(
            state,
            status=status,
            revision=state.revision + 1,
            updated_at=now,
        )
    elif t is TransitionType.DECLARE_COMPLETE:
        return replace(
            state,
            declared_complete=bool(tr.declared),
            revision=state.revision + 1,
            updated_at=now,
        )
    else:  # pragma: no cover - enum exhaustive
        raise InvalidTransition(f"unknown transition {t}")
    return replace(
        state,
        atoms=tuple(atoms),
        counters=tuple(sorted(counters.items())),
        revision=state.revision + 1,
        updated_at=now,
    )


# ------------------------------------------------------------- extraction
_CODE_FENCE_RE = re.compile(r"(?s)```.*?(?:```|$)")
_BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)]|\[[ xX]\])\s+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9`\"'(])|;\s+")

_CRITERION_RE = re.compile(
    r"(?i)\b(?:tests? (?:should|must|need to|needs to|have to|has to|will) (?:still )?pass"
    r"|all (?:the )?tests? (?:pass|passing|green)|(?:suite|build|ci|it) (?:should|must) (?:pass|succeed|be green|compile|build)"
    r"|(?:should|must) (?:compile|build cleanly|pass)|acceptance criteri|done when|definition of done"
    r"|verify (?:that|it)|ensure (?:that )?(?:the |all )?(?:tests?|build|it|this)"
    r"|make sure (?:that )?(?:the |all )?(?:tests?|build|it|everything) (?:pass|works?|compiles?|builds?|succeeds?)"
    r"|(?:should|must) (?:return|output|print|produce|emit|show|report|handle)\b"
    r"|expected (?:output|result|behaviou?r)|passes? (?:on|under) (?:windows|linux|macos))"
)
_HARD_RE = re.compile(
    r"(?i)(?:^\s*(?:do not|don't|dont|never|must not|mustn't|cannot|can't|should not|shouldn't|avoid|no )\b"
    r"|\b(?:must|must not|never|always|required|is required|are required|under no circumstances|mandatory|non-negotiable)\b)"
)
_CONSTRAINT_RE = re.compile(
    r"(?i)(?:^\s*(?:do not|don't|dont|never|must not|mustn't|cannot|can't|should not|shouldn't|avoid|only|keep|preserve|without|no )\b"
    r"|\b(?:must|must not|never|always|required|preserve|without (?:changing|modifying|breaking|touching|adding|removing)"
    r"|only (?:use|modify|change|touch|edit)|keep (?:the|it|them|all|existing|backward)|backward[- ]compatib"
    r"|first-class|cross-platform|on (?:both )?windows and linux|windows and linux)\b)"
)
_NEGATIVE_RE = re.compile(
    r"(?i)\b(?:not|never|don't|dont|no|without|avoid|mustn't|shouldn't|cannot|can't)\b"
)
_DECISION_RE = re.compile(
    r"(?i)(?:\b(?:instead of|rather than|go with|stick with|we(?:'ll| will) use|let'?s use|decided to|i prefer|prefer using)\b"
    r"|^\s*use \S+(?: \S+){0,4} (?:for|as|to) )"
)
_SUBGOAL_VERB_RE = re.compile(
    r"(?i)^\s*(?:please\s+)?(?:also\s+)?(?:add|implement|create|fix|update|write|refactor|remove|rename|make|support|"
    r"change|move|extract|document|wire|port|migrate|introduce|replace|handle|expose|build|set up|setup)\b"
)
_NEW_TASK_RE = re.compile(
    r"(?i)^\s*(?:new task|next task|different (?:task|question|topic|thing)|unrelated[,:]|switch(?:ing)? (?:gears|to)"
    r"|now (?:let'?s|lets) (?:work on|move on to|switch to|do something)|forget (?:about )?(?:that|the previous|it)"
    r"|start(?:ing)? (?:a )?(?:new|fresh)|moving on[,:]?|separate (?:task|question)|on a different note)"
)
_CORRECTION_RE = re.compile(
    r"(?i)^\s*(?:actually|wait|no[,.!]|nope|instead|correction|scratch that|i meant|sorry,? i meant|change of plan"
    r"|on second thought|that'?s (?:wrong|not right|not what i)|not quite|hold on|stop)"
)
_CORRECTION_PREFIX_RE = re.compile(
    r"(?i)^\s*(?:actually|wait|no|nope|instead|correction|scratch that|i meant|sorry,? i meant|change of plan"
    r"|on second thought|hold on)[\s,:.!-]*"
)
_ACK_RE = re.compile(
    r"(?i)^\s*(?:ok(?:ay)?|k|yes|yep|yeah|y|sure|continue|go on|go ahead|proceed|keep going|sounds good|great|thanks"
    r"|thank you|lgtm|do it|please continue|carry on|resume|next)[\s.!,]*(?:please)?[\s.!]*$"
)
_CONTINUATION_RE = re.compile(
    r"(?i)^\s*(?:also|and|now|next|then|additionally|one more|another|can you also|please also|plus|furthermore"
    r"|great,? (?:now|next|also|can)|thanks,? (?:now|next|also|can)|ok(?:ay)?,? (?:now|next|also))\b"
)
_COMPLETION_DECL_RE = re.compile(
    r"(?i)(?:\ball (?:the )?(?:changes|tasks?|items|requested changes|steps) (?:are |have been )?(?:now )?(?:complete|completed|done|implemented|finished)\b"
    r"|\b(?:the )?(?:task|implementation|feature|fix|change|work|refactor) is (?:now )?(?:complete|done|finished)\b"
    r"|\bi(?:'ve| have) (?:now )?(?:completed|finished|implemented) (?:the|all|everything|this)\b"
    r"|\beverything (?:is|has been) (?:now )?(?:done|implemented|complete|in place)\b"
    r"|^\s*done[.!]?\s*$)"
)
_TEST_WORD_RE = re.compile(
    r"(?i)\b(?:tests?|pytest|unit ?tests?|integration tests?|suite|cargo test|jest|vitest|ctest|specs?)\b"
)
_BUILD_WORD_RE = re.compile(
    r"(?i)\b(?:build|builds|compile|compiles|lint|linter|type-?check|mypy|clippy|tsc)\b"
)
_PATH_TOKEN_RE = re.compile(
    r"(?<![\w/])((?:[\w.-]+/)*[\w.-]+\.(?:py|rs|ts|tsx|js|jsx|go|cpp|cc|c|h|hpp|java|kt|rb|cs|toml|json|ya?ml|md|cmake|txt))(?:::[\w\[\]-]+)*"
)
_STOP = frozenset(
    "a an the to of in on for and or but is are be been being it its this that these those with without "
    "do does did not don't dont never must should shall will would can can't cannot may might please "
    "make sure ensure keep always only use using any all each every we i you they our your their "
    "from by as at into onto so if then than also just still".split()
)
_GOAL_MAX_CHARS = 320
_ATOM_MAX_CHARS = 220


def _strip_code(text: str) -> str:
    text = _CODE_FENCE_RE.sub(" ", text or "")
    return "\n".join(ln for ln in text.splitlines() if not ln.lstrip().startswith(">"))


def split_units(text: str) -> list[tuple[str, bool]]:
    """``(unit text, is_bullet)`` for each bullet line or sentence (code excluded)."""
    out: list[tuple[str, bool]] = []
    for line in _strip_code(text).splitlines():
        s = line.strip()
        if not s:
            continue
        m = _BULLET_RE.match(s)
        if m:
            body = s[m.end() :].strip()
            if body:
                out.append((body, True))
            continue
        for sent in _SENTENCE_SPLIT_RE.split(s):
            sent = sent.strip()
            if sent:
                out.append((sent, False))
    return out


def _words(text: str) -> list[str]:
    out = []
    for w in re.findall(r"[a-z0-9_./-]+", (text or "").lower()):
        if w in _STOP or len(w) < 2:
            continue
        for suf in ("ing", "ed", "es", "s"):
            if len(w) > 4 and w.endswith(suf):
                w = w[: -len(suf)]
                break
        out.append(w)
    return out


def normalize_key(text: str) -> str:
    """Order-insensitive content words plus polarity ("never X" and "X" differ)."""
    polarity = "neg:" if _NEGATIVE_RE.search(text or "") else ""
    return (polarity + " ".join(sorted(set(_words(text)))))[:200]


def jaccard(a: str, b: str) -> float:
    wa, wb = set(_words(a)), set(_words(b))
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def _clip(text: str, limit: int) -> str:
    t = " ".join((text or "").split())
    if len(t) <= limit:
        return t
    cut = t[:limit].rsplit(" ", 1)[0]
    return cut.rstrip(",;:") + " …"


def _goal_line(goal: str, atoms: Iterable[Any]) -> str:
    """The goal without the sentences that are already their own atom.

    The first message usually states the constraints and acceptance criteria
    too, and those are listed under their own headings. Repeating them on the
    goal line only costs tokens. Kept sentences stay verbatim and in order;
    superseded atoms count too, so a corrected constraint does not resurface
    here. If every sentence is an atom, the goal is shown unchanged.
    """
    keys = {
        a.normalized_key
        for a in atoms
        if a.kind in (AtomKind.CONSTRAINT, AtomKind.CRITERION, AtomKind.DECISION)
    }
    if not keys:
        return goal
    kept: list[str] = []
    for unit, _bullet in split_units(goal):
        bare = _LABEL_PREFIX_RE.sub("", unit.strip(), count=1)
        if normalize_key(_clip(bare, _ATOM_MAX_CHARS)) in keys:
            continue
        kept.append(unit.strip())
    return " ".join(kept) if kept else goal


def goal_text(message: str) -> str:
    """The user's own words: first paragraph, verbatim, clipped on a word boundary."""
    body = _strip_code(message).strip()
    para = re.split(r"\n\s*\n", body, maxsplit=1)[0] if body else ""
    return _clip(para or body, _GOAL_MAX_CHARS)


@dataclass(frozen=True)
class ExtractedUnit:
    kind: AtomKind
    text: str
    hard: bool = False
    verify: str = ""
    targets: tuple[str, ...] = ()


_LABEL_PREFIX_RE = re.compile(
    r"(?i)^\s*(?:constraints?|requirements?|rules?|notes?|important|acceptance criteria|criteria|goals?"
    r"|context|also|additionally)\s*[:\-\u2013]\s+"
)


def classify_unit(text: str, *, bullet: bool, in_goal_message: bool) -> ExtractedUnit | None:
    t = _LABEL_PREFIX_RE.sub("", text.strip(), count=1)
    if len(t) < 6:
        return None
    if _CRITERION_RE.search(t) and not (_NEGATIVE_RE.search(t) and not _TEST_WORD_RE.search(t)):
        verify = (
            "test"
            if _TEST_WORD_RE.search(t)
            else ("build" if _BUILD_WORD_RE.search(t) else "deliverable")
        )
        targets = tuple(dict.fromkeys(m.group(0) for m in _PATH_TOKEN_RE.finditer(t)))
        return ExtractedUnit(
            AtomKind.CRITERION, _clip(t, _ATOM_MAX_CHARS), verify=verify, targets=targets
        )
    if _CONSTRAINT_RE.search(t):
        return ExtractedUnit(
            AtomKind.CONSTRAINT, _clip(t, _ATOM_MAX_CHARS), hard=bool(_HARD_RE.search(t))
        )
    if _DECISION_RE.search(t):
        return ExtractedUnit(AtomKind.DECISION, _clip(t, _ATOM_MAX_CHARS))
    if bullet and in_goal_message:
        return ExtractedUnit(AtomKind.SUBGOAL, _clip(t, _ATOM_MAX_CHARS))
    if not in_goal_message and _SUBGOAL_VERB_RE.match(t):
        return ExtractedUnit(AtomKind.SUBGOAL, _clip(t, _ATOM_MAX_CHARS))
    return None


class Relation(str, Enum):
    KEEP = "keep"
    REFINE = "refine"
    REPLACE = "replace"
    ADD_SUBGOAL = "add_subgoal"


# ------------------------------------------------------------------ engine
class TaskStateCompiler:
    """Per-session compiler. Reads and writes go through the workspace store."""

    MAX_COMPLETED = 12

    def __init__(self, runtime: Any) -> None:
        self.rt = runtime
        self.store = runtime.store
        self.state: TaskState | None = None
        self.transitions = 0
        self.tokens_injected = 0
        self._load()

    @property
    def task_id(self) -> str | None:
        return self.state.task_id if self.state is not None else None

    # --------------------------------------------------------- persistence
    def _load(self) -> None:
        row = self.store.query_one(
            "SELECT state_json FROM tasks WHERE session_key = ? ORDER BY updated_at DESC LIMIT 1",
            (self.rt.session_key,),
        )
        if row is None:
            return
        try:
            self.state = TaskState.from_json(loads(row["state_json"], {}))
        except (KeyError, ValueError, TypeError):
            logger.debug("task state unreadable; starting fresh", exc_info=True)
            self.state = None

    def refresh(self) -> None:
        """Reload when another process (the MCP macro executor) advanced the task."""
        if self.state is None:
            self._load()
            return
        row = self.store.query_one(
            "SELECT revision, state_json FROM tasks WHERE task_id = ?", (self.state.task_id,)
        )
        newer = self.store.query_one(
            "SELECT task_id FROM tasks WHERE session_key = ? AND updated_at > ? AND task_id != ? LIMIT 1",
            (self.rt.session_key, self.state.updated_at, self.state.task_id),
        )
        if newer is not None or (row is not None and int(row["revision"]) > self.state.revision):
            self._load()

    def _commit(self, new: TaskState, transitions: list[StateTransition]) -> None:
        state_json = dumps(new.to_json())

        def run(c: Any) -> None:
            c.execute(
                "INSERT INTO tasks(task_id, session_key, revision, status, state_json, created_at, "
                "updated_at) VALUES (?,?,?,?,?,?,?) ON CONFLICT(task_id) DO UPDATE SET "
                "revision=excluded.revision, status=excluded.status, state_json=excluded.state_json, "
                "updated_at=excluded.updated_at",
                (
                    new.task_id,
                    self.rt.session_key,
                    new.revision,
                    new.status.value,
                    state_json,
                    new.created_at,
                    new.updated_at,
                ),
            )
            rev = new.revision - len(transitions)
            for tr in transitions:
                rev += 1
                c.execute(
                    "INSERT OR IGNORE INTO task_transitions(task_id, revision, transition_id, "
                    "transition_json, created_at) VALUES (?,?,?,?,?)",
                    (new.task_id, rev, tr.transition_id, dumps(tr.to_json()), time.time()),
                )

        self.store.write(run)
        self.state = new
        self.transitions += len(transitions)
        self.rt.metrics.bump("task_state_transitions", len(transitions))

    def _apply(self, transitions: list[StateTransition]) -> bool:
        """Validate then commit a batch atomically; invalid ones are dropped."""
        if self.state is None or not transitions:
            return False
        state = self.state
        applied: list[StateTransition] = []
        seen = self._seen_transition_ids()
        for tr in transitions:
            if tr.transition_id in seen:
                continue  # idempotent replay
            try:
                state = apply_transition(state, tr)
            except InvalidTransition as exc:
                logger.debug("task-state transition rejected (%s): %s", tr.reason_code, exc)
                self.rt.metrics.bump("task_state_rejected")
                continue
            applied.append(tr)
            seen.add(tr.transition_id)
        if not applied:
            return False
        self._commit(state, applied)
        return True

    def _seen_transition_ids(self) -> set[str]:
        if self.state is None:
            return set()
        rows = self.store.query(
            "SELECT transition_id FROM task_transitions WHERE task_id = ?", (self.state.task_id,)
        )
        return {r["transition_id"] for r in rows}

    # ------------------------------------------------------------- helpers
    def _atom(
        self,
        kind: AtomKind,
        text: str,
        *,
        origin: Origin,
        ev: AgentEvent,
        confidence: float = 0.95,
        hard: bool = False,
        verify: str = "",
        targets: tuple[str, ...] = (),
        key: str = "",
        blocking: bool = False,
        state: AtomState = AtomState.ACTIVE,
        evidence_ids: tuple[str, ...] = (),
    ) -> StateAtom:
        nkey = key or normalize_key(text)
        return StateAtom(
            atom_id=stable_id(ev.event_id, kind.value, nkey, n=16),
            kind=kind,
            text=text,
            normalized_key=nkey,
            state=state,
            origin=origin,
            confidence=confidence,
            hard=hard,
            verify=verify,
            targets=targets,
            key=key,
            blocking=blocking,
            evidence_ids=evidence_ids,
            source_event_ids=(ev.event_id,),
        )

    def _tr(
        self, ev: AgentEvent, ttype: TransitionType, reason: str, salt: str = "", **kw: Any
    ) -> StateTransition:
        return StateTransition(
            transition_id=stable_id(ev.event_id, ttype.value, reason, salt, n=20),
            transition_type=ttype,
            reason_code=reason,
            source_event_ids=(ev.event_id,),
            **kw,
        )

    def _new_task(self, ev: AgentEvent, text: str) -> None:
        now = time.time()
        task_id = "T" + stable_id(self.rt.session_key, ev.event_id, n=10)
        prior = self.state
        if prior is not None and prior.status in (TaskStatus.ACTIVE, TaskStatus.BLOCKED):
            # The old task is not deleted; it is closed as abandoned/replaced.
            close = self._tr(
                ev,
                TransitionType.SET_STATUS,
                "REPLACED_BY_NEW_TASK",
                new_status=TaskStatus.ABANDONED,
                confidence=1.0,
            )
            self._commit(apply_transition(prior, close, now=now), [close])
        base = TaskState(
            workspace_id=self.rt.workspace.workspace_id,
            session_id=self.rt.session_key,
            task_id=task_id,
            revision=0,
            status=TaskStatus.ACTIVE,
            atoms=(),
            created_at=now,
            updated_at=now,
        )
        self.state = base
        transitions = [
            self._tr(
                ev,
                TransitionType.ADD_ATOM,
                "USER_GOAL",
                new_atom=self._atom(AtomKind.GOAL, goal_text(text), origin=Origin.USER, ev=ev),
            )
        ]
        transitions.extend(self._extract_atoms(ev, text, in_goal_message=True))
        # Commit the fresh task even if extraction found only the goal.
        state = base
        applied = []
        for tr in transitions:
            try:
                state = apply_transition(state, tr)
                applied.append(tr)
            except InvalidTransition:
                continue
        self._commit(state, applied)
        self.rt.metrics.bump("task_state_new_tasks")
        self._emit_changed(ev, "NEW_TASK")
        if self.rt.scope is not None:
            self.rt.scope.on_new_task()

    def _extract_atoms(
        self, ev: AgentEvent, text: str, *, in_goal_message: bool
    ) -> list[StateTransition]:
        out: list[StateTransition] = []
        state = self.state
        existing_keys = {a.normalized_key for a in state.atoms} if state else set()
        for unit_text, bullet in split_units(text):
            unit = classify_unit(unit_text, bullet=bullet, in_goal_message=in_goal_message)
            if unit is None:
                continue
            nkey = normalize_key(unit.text)
            if not nkey or nkey in existing_keys:
                continue
            existing_keys.add(nkey)
            atom = self._atom(
                unit.kind,
                unit.text,
                origin=Origin.USER,
                ev=ev,
                hard=unit.hard,
                verify=unit.verify,
                targets=unit.targets,
            )
            out.append(
                self._tr(
                    ev, TransitionType.ADD_ATOM, f"USER_{unit.kind.value}", nkey, new_atom=atom
                )
            )
        return out

    def _emit_changed(self, ev: AgentEvent, reason: str) -> None:
        if self.state is None:
            return
        change = AgentEvent(
            event_id=stable_id(ev.event_id, "TASK_STATE_CHANGED", self.state.revision),
            workspace_id=self.rt.workspace.workspace_id,
            session_id=self.rt.session_key,
            task_id=self.state.task_id,
            event_type=EventType.TASK_STATE_CHANGED,
            metadata={
                "revision": self.state.revision,
                "reason": reason,
                "status": self.state.status.value,
            },
        )
        self.rt.persist_events([change])

    # ------------------------------------------------------------- events
    def classify_relation(self, text: str) -> tuple[Relation, float]:
        state = self.state
        if state is None:
            return Relation.REPLACE, 1.0
        if _ACK_RE.match(text):
            return Relation.KEEP, 0.99
        if _NEW_TASK_RE.match(text):
            return Relation.REPLACE, 0.95
        if _CORRECTION_RE.match(text):
            return Relation.REFINE, 0.95
        if state.status is TaskStatus.COMPLETED:
            if _CONTINUATION_RE.match(text):
                return Relation.ADD_SUBGOAL, 0.85
            return Relation.REPLACE, 0.85
        if _CONTINUATION_RE.match(text) or _SUBGOAL_VERB_RE.match(text):
            return Relation.ADD_SUBGOAL, 0.85
        # Tier B: JevK5 for a substantive message with no explicit marker.
        if len(text.split()) >= 8:
            from headroom.intelligence.models import DecisionFamily

            goal = state.primary_goal.text if state.primary_goal else ""
            answer = classify(
                self.rt.advisor(),
                DecisionFamily.TASK_STATE,
                f"Active goal: {goal}\nNew user message: {text[:800]}",
                "Does this user message modify the active goal?",
                {"A": "keep", "B": "refine", "C": "replace", "D": "add_subgoal"},
            )
            if answer is not None:
                choice, prob = answer
                rel = {
                    "A": Relation.KEEP,
                    "B": Relation.REFINE,
                    "C": Relation.REPLACE,
                    "D": Relation.ADD_SUBGOAL,
                }[choice]
                if rel is Relation.REPLACE and prob < NEW_TASK_CONFIDENCE:
                    return Relation.ADD_SUBGOAL, prob  # §5.10: retain and add a subgoal
                return rel, prob
        # Tier C: keep the task; extract whatever explicit atoms the message has.
        return Relation.KEEP, 0.6

    def on_user_message(self, ev: AgentEvent) -> None:
        text = str(ev.transient.get("text") or "")
        if not text:
            return
        if self.state is None:
            self._new_task(ev, text)
            return
        relation, confidence = self.classify_relation(text)
        if relation is Relation.REPLACE and confidence >= NEW_TASK_CONFIDENCE:
            self._new_task(ev, text)
            return
        transitions: list[StateTransition] = []
        if self.state.declared_complete:
            transitions.append(
                self._tr(ev, TransitionType.DECLARE_COMPLETE, "USER_FOLLOWUP", declared=False)
            )
        if self.state.status is TaskStatus.COMPLETED and relation is not Relation.KEEP:
            transitions.append(
                self._tr(
                    ev,
                    TransitionType.SET_STATUS,
                    "CONTINUATION",
                    new_status=TaskStatus.ACTIVE,
                    confidence=confidence,
                )
            )
        if relation is Relation.KEEP and _ACK_RE.match(text):
            self._apply(transitions)
            return
        body = _CORRECTION_PREFIX_RE.sub("", text, count=1) if relation is Relation.REFINE else text
        new_atoms = self._extract_atoms(ev, body, in_goal_message=False)
        if relation is Relation.ADD_SUBGOAL and not any(
            t.new_atom is not None and t.new_atom.kind is AtomKind.SUBGOAL for t in new_atoms
        ):
            sub = self._atom(
                AtomKind.SUBGOAL, goal_text(text), origin=Origin.USER, ev=ev, confidence=confidence
            )
            if sub.normalized_key not in {a.normalized_key for a in self.state.atoms}:
                new_atoms.append(
                    self._tr(ev, TransitionType.ADD_ATOM, "USER_SUBGOAL_MESSAGE", new_atom=sub)
                )
        transitions.extend(
            self._resolve_conflicts(ev, new_atoms, correction=relation is Relation.REFINE)
        )
        if self._apply(transitions):
            self._emit_changed(ev, f"USER_{relation.value.upper()}")

    def _resolve_conflicts(
        self, ev: AgentEvent, proposed: list[StateTransition], *, correction: bool
    ) -> list[StateTransition]:
        """Turn ADD_ATOMs that conflict with live user atoms into SUPERSEDEs (rules 1, 2, 7)."""
        state = self.state
        assert state is not None
        out: list[StateTransition] = []
        consumed: set[str] = set()
        for tr in proposed:
            atom = tr.new_atom
            if (
                tr.transition_type is not TransitionType.ADD_ATOM
                or atom is None
                or atom.kind
                not in (
                    AtomKind.CONSTRAINT,
                    AtomKind.DECISION,
                    AtomKind.CRITERION,
                )
            ):
                out.append(tr)
                continue
            best: StateAtom | None = None
            best_j = 0.0
            kinds = (
                (AtomKind.CONSTRAINT, AtomKind.DECISION)
                if atom.kind in (AtomKind.CONSTRAINT, AtomKind.DECISION)
                else (atom.kind,)
            )
            for old in [a for k in kinds for a in state.by_kind(k)]:
                if old.origin is not Origin.USER or old.atom_id in consumed:
                    continue
                j = jaccard(old.text, atom.text)
                if j > best_j:
                    best, best_j = old, j
            if best is None or best_j < 0.3:
                out.append(tr)
                continue
            polarity_flip = bool(_NEGATIVE_RE.search(best.text)) != bool(
                _NEGATIVE_RE.search(atom.text)
            )
            if best_j >= 0.5 and (correction or polarity_flip or best_j >= 0.6):
                conf = 0.95 if (correction or polarity_flip) else 0.85
                out.append(
                    self._tr(
                        ev,
                        TransitionType.SUPERSEDE,
                        "USER_SUPERSEDES",
                        best.atom_id,
                        atom_id=best.atom_id,
                        new_atom=atom,
                        confidence=conf,
                    )
                )
                consumed.add(best.atom_id)
                continue
            # Ambiguous overlap: Tier B, else keep both (Tier C).
            from headroom.intelligence.models import DecisionFamily

            answer = classify(
                self.rt.advisor(),
                DecisionFamily.TASK_STATE,
                f"Existing {best.label}: {best.text}\nNew: {atom.text}",
                f"Does the new constraint conflict with {best.label}?",
                {"A": "no", "B": "supersedes", "C": "conflicts_without_superseding"},
            )
            if answer is not None and answer[0] == "B" and answer[1] >= MIN_TRANSITION_CONFIDENCE:
                out.append(
                    self._tr(
                        ev,
                        TransitionType.SUPERSEDE,
                        "JEVK5_SUPERSEDES",
                        best.atom_id,
                        atom_id=best.atom_id,
                        new_atom=atom,
                        confidence=answer[1],
                    )
                )
                consumed.add(best.atom_id)
                continue
            out.append(tr)
            if answer is not None and answer[0] == "C":
                q = self._atom(
                    AtomKind.QUESTION,
                    f"New instruction may conflict with {best.label}; both kept.",
                    origin=Origin.DERIVED,
                    ev=ev,
                    confidence=answer[1],
                    key=f"conflict:{best.atom_id}",
                )
                out.append(
                    self._tr(
                        ev, TransitionType.ADD_ATOM, "POSSIBLE_CONFLICT", best.atom_id, new_atom=q
                    )
                )
        return out

    def on_assistant_message(self, ev: AgentEvent) -> None:
        state = self.state
        if state is None or state.status is not TaskStatus.ACTIVE:
            return
        text = str(ev.transient.get("text") or "")
        tail = text[-600:]
        if not _COMPLETION_DECL_RE.search(tail):
            return
        transitions: list[StateTransition] = [
            self._tr(ev, TransitionType.DECLARE_COMPLETE, "AGENT_DECLARES_COMPLETE", declared=True)
        ]
        # Rule 4: a pure deliverable can be satisfied by the statement producing it;
        # factual (test/build) criteria never are.
        for crit in state.acceptance_criteria:
            if crit.verify == "deliverable" and crit.state is AtomState.ACTIVE:
                transitions.append(
                    self._tr(
                        ev,
                        TransitionType.SET_ATOM_STATE,
                        "AGENT_DELIVERABLE",
                        crit.atom_id,
                        atom_id=crit.atom_id,
                        new_state=AtomState.SATISFIED,
                        confidence=0.7,
                    )
                )
        self._apply(transitions)
        self._maybe_complete(ev)

    def on_tool_result(self, ev: AgentEvent, records: list[Any]) -> None:
        state = self.state
        if state is None:
            return
        transitions: list[StateTransition] = []
        if state.declared_complete:
            transitions.append(
                self._tr(ev, TransitionType.DECLARE_COMPLETE, "AGENT_RESUMED_WORK", declared=False)
            )
        inv = ev.transient.get("invocation")
        wrote = (
            bool(inv is not None and (inv.paths_written or inv.paths_deleted))
            and ev.success is not False
        )
        if wrote and inv is not None:
            # A task-owned change invalidates earlier verification of factual criteria.
            for crit in state.acceptance_criteria:
                if crit.verify in ("test", "build") and crit.state is AtomState.SATISFIED:
                    transitions.append(
                        self._tr(
                            ev,
                            TransitionType.SET_ATOM_STATE,
                            "REVERIFY_AFTER_CHANGE",
                            crit.atom_id,
                            atom_id=crit.atom_id,
                            new_state=AtomState.ACTIVE,
                            confidence=0.99,
                        )
                    )
            rel = [self.rt.workspace.relpath(p) for p in (*inv.paths_written, *inv.paths_deleted)][
                :3
            ]
            if rel:
                key = "edit:" + ",".join(sorted(rel))
                if state.find_key(AtomKind.COMPLETED, key) is None:
                    verb = "deleted" if inv.paths_deleted and not inv.paths_written else "edited"
                    done = self._atom(
                        AtomKind.COMPLETED,
                        f"{verb} {', '.join(rel)}",
                        origin=Origin.TOOL,
                        ev=ev,
                        confidence=0.99,
                        key=key,
                        state=AtomState.SATISFIED,
                    )
                    transitions.append(
                        self._tr(ev, TransitionType.ADD_ATOM, "TOOL_EDIT", key, new_atom=done)
                    )
        for rec in records:
            transitions.extend(self._from_evidence(ev, rec))
        if self._apply(transitions):
            self._emit_changed(ev, "TOOL_EVIDENCE")
        self._maybe_complete(ev)

    def _from_evidence(self, ev: AgentEvent, rec: Any) -> list[StateTransition]:
        """Evidence-backed criterion and blocker transitions (rule 3)."""
        state = self.state
        assert state is not None
        out: list[StateTransition] = []
        ctype = getattr(rec, "claim_type", "")
        if ctype not in ("test_aggregate", "build_status"):
            return out
        value = rec.value if isinstance(rec.value, dict) else {}
        passed = bool(value.get("ok"))
        command = str(value.get("command", ""))[:120]
        verify = "test" if ctype == "test_aggregate" else "build"
        blocker_key = f"{verify}:{rec.subject}"
        blocker = state.find_key(AtomKind.BLOCKER, blocker_key)
        if passed:
            passed_ids = {str(x) for x in value.get("passed_ids") or []}
            for b in state.blockers:
                if not b.key.startswith(f"{verify}:"):
                    continue
                same_command = b.key.split("#", 1)[0] == blocker_key
                # A narrower re-run resolves the blocker only when every test
                # that failed is now observed passing.
                covered = bool(b.targets) and all(
                    t in passed_ids or t.split("::")[-1] in passed_ids for t in b.targets
                )
                if same_command or covered or value.get("full_suite"):
                    out.append(
                        self._tr(
                            ev,
                            TransitionType.SET_ATOM_STATE,
                            "BLOCKER_RESOLVED",
                            b.atom_id,
                            atom_id=b.atom_id,
                            new_state=AtomState.SATISFIED,
                            confidence=rec.confidence,
                            supporting_evidence_ids=(rec.evidence_id,),
                        )
                    )
            for crit in state.acceptance_criteria:
                if crit.verify != verify or crit.state is AtomState.SATISFIED:
                    continue
                if crit.targets and not _targets_covered(crit.targets, value, command):
                    continue
                out.append(
                    self._tr(
                        ev,
                        TransitionType.SET_ATOM_STATE,
                        "EVIDENCE_SATISFIES",
                        crit.atom_id,
                        atom_id=crit.atom_id,
                        new_state=AtomState.SATISFIED,
                        confidence=rec.confidence,
                        supporting_evidence_ids=(rec.evidence_id,),
                    )
                )
        else:
            failed = int(value.get("failed", 0) or 0) + int(value.get("errors", 0) or 0)
            summary = f"{command or verify}: " + (f"{failed} failing" if failed else "failed")
            failed_ids = tuple(str(x) for x in (value.get("failed_ids") or [])[:20])
            preexisting = {str(x) for x in value.get("preexisting") or []}
            # A failure confirmed to predate the task is reported, never a task blocker (§9.10).
            is_blocking = not (failed_ids and set(failed_ids) <= preexisting)
            if not is_blocking:
                summary += " (pre-existing)"
            if blocker is None:
                atom = self._atom(
                    AtomKind.BLOCKER,
                    _clip(summary, 160),
                    origin=Origin.TOOL,
                    ev=ev,
                    confidence=rec.confidence,
                    key=blocker_key,
                    blocking=is_blocking,
                    evidence_ids=(rec.evidence_id,),
                    targets=failed_ids,
                )
                out.append(
                    self._tr(
                        ev, TransitionType.ADD_ATOM, "TOOL_FAILURE", blocker_key, new_atom=atom
                    )
                )
            elif blocker.state is not AtomState.ACTIVE:
                # Re-opened: a resolved blocker failed again.
                atom = self._atom(
                    AtomKind.BLOCKER,
                    _clip(summary, 160),
                    origin=Origin.TOOL,
                    ev=ev,
                    confidence=rec.confidence,
                    key=blocker_key + f"#{ev.event_id[:6]}",
                    blocking=is_blocking,
                    evidence_ids=(rec.evidence_id,),
                    targets=failed_ids,
                )
                out.append(
                    self._tr(
                        ev,
                        TransitionType.ADD_ATOM,
                        "TOOL_FAILURE_AGAIN",
                        blocker_key,
                        new_atom=atom,
                    )
                )
            for crit in state.acceptance_criteria:
                if crit.verify == verify and crit.state is AtomState.SATISFIED:
                    out.append(
                        self._tr(
                            ev,
                            TransitionType.SET_ATOM_STATE,
                            "EVIDENCE_REGRESSION",
                            crit.atom_id,
                            atom_id=crit.atom_id,
                            new_state=AtomState.ACTIVE,
                            confidence=rec.confidence,
                            supporting_evidence_ids=(rec.evidence_id,),
                        )
                    )
        return out

    def mark_constraint_violated(
        self, ev: AgentEvent, atom_id: str, evidence_ids: tuple[str, ...] = ()
    ) -> None:
        if self.state is None:
            return
        self._apply(
            [
                self._tr(
                    ev,
                    TransitionType.SET_ATOM_STATE,
                    "CONSTRAINT_VIOLATED",
                    atom_id,
                    atom_id=atom_id,
                    new_state=AtomState.VIOLATED,
                    confidence=0.99,
                    supporting_evidence_ids=evidence_ids,
                )
            ]
        )

    def _maybe_complete(self, ev: AgentEvent) -> None:
        state = self.state
        if state is None or state.status is TaskStatus.COMPLETED or not state.declared_complete:
            return
        extra = self.rt.scope.blocking_violations() if self.rt.scope is not None else []
        if extra or completion_blockers(state):
            if state.status is TaskStatus.ACTIVE and any(b.blocking for b in state.blockers):
                self._apply(
                    [
                        self._tr(
                            ev,
                            TransitionType.SET_STATUS,
                            "BLOCKED",
                            new_status=TaskStatus.BLOCKED,
                            confidence=0.99,
                        )
                    ]
                )
            return
        if self._apply(
            [
                self._tr(
                    ev,
                    TransitionType.SET_STATUS,
                    "ALL_CRITERIA_SATISFIED",
                    new_status=TaskStatus.COMPLETED,
                    confidence=0.9,
                )
            ]
        ):
            self.rt.metrics.bump("task_state_completed")
            self._emit_changed(ev, "COMPLETED")
            if self.rt.scope is not None:
                self.rt.scope.freeze()

    def note_injected(self, tokens: int) -> None:
        self.tokens_injected += tokens
        self.rt.metrics.bump("task_state_tokens_injected", tokens)

    # ------------------------------------------------------------ rendering
    def wants_injection(self) -> bool:
        """Plan §5.9: inject for multi-turn work, binding atoms, blockers, long sessions."""
        state = self.state
        if state is None or state.status is TaskStatus.ABANDONED:
            return False
        return bool(
            self.rt.user_turns > 1
            or state.constraints
            or [c for c in state.acceptance_criteria if c.state is not AtomState.SATISFIED]
            or state.blockers
            or state.decisions
            or self.rt.history_tokens > 8000
        )

    def sections(self) -> list[tuple[int, str, list[str]]]:
        """``(priority, heading, lines)``, lower priority sorts first and survives longest."""
        state = self.state
        if state is None:
            return []
        out: list[tuple[int, str, list[str]]] = []
        blockers = [f"- {b.label} {b.text}" for b in state.blockers]
        if blockers:
            out.append((10, "blocked", blockers))
        questions = [f"- {q.label} {q.text}" for q in state.unresolved_questions]
        if questions:
            out.append((15, "unresolved", questions))
        cons = []
        for c in state.constraints:
            if c.state is AtomState.VIOLATED:
                cons.append(f"- {c.label} VIOLATED: {c.text}")
            else:
                cons.append(f"- {c.label} {c.text}")
        if cons:
            out.append((20, "constraints", cons))
        goal = state.primary_goal
        if goal is not None:
            out.append((30, "goal", [_goal_line(goal.text, state.atoms)]))
        pending = [
            f"- {a.label} {a.text}"
            for a in state.acceptance_criteria
            if a.state is not AtomState.SATISFIED
        ]
        subs = [
            f"- {s.label} {s.text}" for s in state.subgoals if s.state is not AtomState.SATISFIED
        ]
        if pending:
            out.append((40, "acceptance", pending))
        if subs:
            out.append((45, "pending", subs[:8]))
        decisions = [f"- {d.label} {d.text}" for d in state.decisions]
        if decisions:
            out.append((50, "decisions", decisions[:8]))
        done = [a.label for a in state.acceptance_criteria if a.state is AtomState.SATISFIED]
        done += [c.text for c in state.completed_items[-self.MAX_COMPLETED :]]
        if done:
            out.append((90, "done", ["; ".join(done)]))
        return out

    def header_attrs(self) -> dict[str, str]:
        state = self.state
        if state is None:
            return {}
        attrs = {"task": state.task_id, "revision": str(state.revision)}
        if state.status is not TaskStatus.ACTIVE:
            attrs["status"] = state.status.value.lower()
        return attrs


def _targets_covered(targets: tuple[str, ...], value: dict[str, Any], command: str) -> bool:
    if not targets:
        return True
    haystack = " ".join([command, *[str(x) for x in value.get("passed_ids") or []]])
    if value.get("full_suite"):
        return True
    return all(t.split("::")[0] in haystack for t in targets)
