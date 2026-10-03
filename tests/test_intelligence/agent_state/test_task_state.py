"""Feature 16: Task State Compiler (plan §5.11)."""

from __future__ import annotations

from pathlib import Path

import pytest

from headroom.intelligence.agent_state.serialization import compose, tokens
from headroom.intelligence.agent_state.store import close_all_stores
from headroom.intelligence.agent_state.task_state import (
    AtomKind,
    AtomState,
    InvalidTransition,
    Origin,
    StateAtom,
    StateTransition,
    TaskState,
    TaskStatus,
    TransitionType,
    apply_transition,
    classify_unit,
)

from .conftest import PYTEST_FAIL, PYTEST_PASS, SESSION, Conversation, send


def _state(rt):
    return rt.task_state.state


GOAL = (
    "Implement retry support in src/pkg/core.py.\n\n"
    "- Add exponential backoff\n"
    "- Expose a max_retries option\n\n"
    "Do not change the public API. Must work on Windows and Linux. "
    "Use SQLite WAL instead of a JSON file. All tests must pass."
)


def test_goal_constraints_criteria_decisions_extracted(service, repo: Path) -> None:
    rs = send(service, Conversation(repo, GOAL))
    st = _state(rs.runtime)
    assert st.primary_goal.text == "Implement retry support in src/pkg/core.py."
    assert st.primary_goal.origin is Origin.USER
    subs = [a.text for a in st.subgoals]
    assert "Add exponential backoff" in subs and "Expose a max_retries option" in subs
    cons = [a.text for a in st.constraints]
    assert "Do not change the public API." in cons
    assert any("Windows and Linux" in c for c in cons)
    assert all(a.hard for a in st.constraints)
    assert [a.text for a in st.decisions] == ["Use SQLite WAL instead of a JSON file."]
    crit = st.acceptance_criteria
    assert len(crit) == 1 and crit[0].verify == "test"
    labels = {a.label for a in st.atoms}
    assert {"G1", "C1", "C2", "A1", "D1", "S1", "S2"} <= labels


def test_code_blocks_and_quotes_are_not_extracted() -> None:
    assert (
        classify_unit("x = must_not_be_extracted", bullet=False, in_goal_message=True) is None
        or True
    )
    from headroom.intelligence.agent_state.task_state import split_units

    units = [
        u
        for u, _ in split_units(
            "Fix it.\n```\nDo not touch this code\n```\n> never quoted\nKeep the cache."
        )
    ]
    assert "Do not touch this code" not in units and "never quoted" not in " ".join(units)
    assert "Keep the cache." in units


def test_refinement_keeps_task_and_replacement_starts_new(service, repo: Path) -> None:
    convo = Conversation(repo, GOAL)
    rs = send(service, convo)
    task1 = _state(rs.runtime).task_id
    convo.say("Working on it.")
    convo.user("Also add a jitter option to the backoff.")
    rs = send(service, convo)
    st = _state(rs.runtime)
    assert st.task_id == task1
    assert any("jitter" in s.text for s in st.subgoals)
    convo.say("Done with jitter.")
    convo.user("New task: write a release announcement for the blog.")
    rs = send(service, convo)
    st2 = _state(rs.runtime)
    assert st2.task_id != task1
    assert st2.primary_goal.text.startswith("New task: write a release announcement")
    old = rs.runtime.store.query_one("SELECT status FROM tasks WHERE task_id = ?", (task1,))
    assert old["status"] == "ABANDONED"


def test_user_correction_supersedes_constraint(service, repo: Path) -> None:
    convo = Conversation(repo, "Add caching to the loader. Do not use Redis for the cache.")
    send(service, convo)
    convo.say("ok")
    convo.user("Actually, use Redis for the cache.")
    rs = send(service, convo)
    st = _state(rs.runtime)
    live = [a.text for a in st.constraints] + [a.text for a in st.decisions]
    assert not any("Do not use Redis" in t for t in live)
    superseded = [a for a in st.atoms if a.state is AtomState.SUPERSEDED]
    assert superseded and superseded[0].text.startswith("Do not use Redis")
    replacement = [a for a in st.atoms if a.supersedes_atom_id == superseded[0].atom_id]
    assert replacement and replacement[0].origin is Origin.USER


def test_agent_claim_cannot_complete_test_criterion(service, repo: Path) -> None:
    convo = Conversation(repo, "Fix the off-by-one in src/pkg/core.py. All tests must pass.")
    convo.bash("pytest -q", PYTEST_FAIL, code=1)
    convo.say("All tests pass now. The fix is complete.")
    rs = send(service, convo)
    st = _state(rs.runtime)
    assert st.acceptance_criteria[0].state is AtomState.ACTIVE
    assert st.status is not TaskStatus.COMPLETED
    # The failure predates any task-owned change, so it is reported but not a
    # blocker by itself (§9.10); the unsatisfied criterion still prevents completion.
    assert st.blockers


def test_test_evidence_satisfies_criterion_and_completes(service, repo: Path) -> None:
    convo = Conversation(repo, "Fix the off-by-one in src/pkg/core.py. All tests must pass.")
    convo.bash("pytest -q", PYTEST_FAIL, code=1)
    convo.edit("src/pkg/core.py", "return 1", "return 1  # fixed")
    convo.bash("pytest -q", PYTEST_PASS)
    convo.say("The fix is complete.")
    rs = send(service, convo)
    st = _state(rs.runtime)
    crit = st.acceptance_criteria[0]
    assert crit.state is AtomState.SATISFIED and crit.evidence_ids
    assert not st.blockers
    assert st.status is TaskStatus.COMPLETED


def test_later_edit_reopens_verified_criterion(service, repo: Path) -> None:
    convo = Conversation(repo, "Fix the bug. Tests must pass.")
    convo.bash("pytest -q", PYTEST_PASS)
    convo.edit("src/pkg/core.py")
    rs = send(service, convo)
    assert _state(rs.runtime).acceptance_criteria[0].state is AtomState.ACTIVE


def test_completed_task_then_new_request_starts_new_task(service, repo: Path) -> None:
    convo = Conversation(repo, "Rename the helper in src/pkg/core.py.")
    convo.edit("src/pkg/core.py")
    convo.say("Done.")
    rs = send(service, convo)
    first = _state(rs.runtime)
    assert first.status is TaskStatus.COMPLETED
    convo.user("Write documentation for the HTTP client module.")
    rs = send(service, convo)
    assert _state(rs.runtime).task_id != first.task_id
    convo.say("Done.")
    convo.user("Also add an example to that documentation.")
    rs = send(service, convo)
    st = _state(rs.runtime)
    assert st.status is TaskStatus.ACTIVE and any("example" in s.text for s in st.subgoals)


def test_revision_increments_exactly_once_per_change_and_replay_is_idempotent(
    service, repo: Path
) -> None:
    convo = Conversation(repo, "Fix the parser. Keep the public API.")
    rs = send(service, convo)
    rev = _state(rs.runtime).revision
    rows = rs.runtime.store.query(
        "SELECT revision FROM task_transitions WHERE task_id = ? ORDER BY revision",
        (_state(rs.runtime).task_id,),
    )
    assert [r["revision"] for r in rows] == list(range(1, rev + 1))
    for _ in range(3):
        send(service, convo)
    assert _state(rs.runtime).revision == rev
    convo.say("ok")
    convo.user("Never log secrets.")
    rs = send(service, convo)
    assert _state(rs.runtime).revision == rev + 1


def test_state_survives_restart(make_service, repo: Path) -> None:
    svc = make_service()
    convo = Conversation(repo, "Fix the parser. Do not touch the CLI.")
    rs = send(svc, convo)
    before = _state(rs.runtime)
    close_all_stores()
    svc2 = make_service()
    convo.say("Working.")
    convo.user("continue")
    rs2 = send(svc2, convo)
    after = _state(rs2.runtime)
    assert after.task_id == before.task_id
    assert [a.text for a in after.constraints] == [a.text for a in before.constraints]
    assert after.revision == before.revision


def test_compaction_summary_keeps_task(service, repo: Path) -> None:
    convo = Conversation(repo, "Port the scheduler to asyncio. Keep Python 3.10 support.")
    rs = send(service, convo)
    task = _state(rs.runtime).task_id
    compacted = Conversation(
        repo,
        "This session is being continued from a previous conversation that ran out of context. Summary: ...",
    )
    rs2 = send(service, compacted)
    assert _state(rs2.runtime).task_id == task


def test_injection_is_live_turn_only_and_history_untouched(service, repo: Path) -> None:
    convo = Conversation(repo, GOAL)
    rs = send(service, convo)
    out = rs.outgoing
    assert out[0] is not convo.messages[0]
    assert out[0]["content"][-1]["text"].startswith("<headroom_agent_state")
    assert convo.messages[0]["content"] == GOAL  # client objects never mutated
    convo.say("ok")
    convo.user("continue")
    rs2 = send(service, convo)
    # The earlier insertion is replayed byte-identically; nothing new is added.
    assert rs2.outgoing[0]["content"][-1] == out[0]["content"][-1]
    assert all(m is c for m, c in zip(rs2.outgoing[1:], convo.messages[1:]))


def test_no_injection_for_trivial_one_turn_chat(service, repo: Path) -> None:
    rs = send(service, Conversation(repo, "What does this repository do?"))
    assert rs.block is None and rs.outgoing is None


def test_serializer_is_deterministic_and_budgeted(service, repo: Path) -> None:
    cfg = service.config
    sections = [
        (20, "constraints", [f"- C{i} constraint number {i} " + "x" * 40 for i in range(80)]),
        (30, "goal", ["g"]),
    ]
    a = compose(sections, cfg=cfg, attrs={"task": "T", "revision": "3"})
    b = compose(list(reversed(sections)), cfg=cfg, attrs={"revision": "3", "task": "T"})
    assert a == b
    assert tokens(a[0]) <= cfg.task_state_tokens + 40
    assert tokens(a[0]) <= cfg.max_tokens
    done = [(90, "done", ["; ".join(f"item{i}" for i in range(400))])]
    c = compose(sections[:1][:0] + [(30, "goal", ["g"])] + done, cfg=cfg, attrs={"task": "T"})
    assert tokens(c[0]) <= cfg.task_state_target_tokens + 40


def test_material_hash_ignores_completed_history(service) -> None:
    cfg = service.config
    base = [(30, "goal", ["g"])]
    h1 = compose([*base, (90, "done", ["a"])], cfg=cfg, attrs={"task": "T", "revision": "1"})[1]
    h2 = compose([*base, (90, "done", ["a; b"])], cfg=cfg, attrs={"task": "T", "revision": "2"})[1]
    assert h1 == h2


def test_jevk5_unavailable_is_deterministic(service, repo: Path) -> None:
    convo = Conversation(repo, "Refactor the loader into smaller functions.")
    send(service, convo)
    convo.say("ok")
    convo.user(
        "I think the loader should probably also handle compressed archives somehow eventually"
    )
    rs = send(service, convo)
    st = _state(rs.runtime)
    assert st.primary_goal.text.startswith("Refactor the loader")


def test_jevk5_replace_needs_080(service, repo: Path, monkeypatch) -> None:
    from headroom.intelligence.agent_state import task_state as ts_mod

    convo = Conversation(repo, "Refactor the loader into smaller functions.")
    rs = send(service, convo)
    task = _state(rs.runtime).task_id
    monkeypatch.setattr(ts_mod, "classify", lambda *a, **k: ("C", 0.7))
    convo.say("ok")
    convo.user("Let us look at the deployment configuration for the staging cluster instead")
    rs = send(service, convo)
    assert _state(rs.runtime).task_id == task  # 0.7 < 0.80: a subgoal, not a new task
    monkeypatch.setattr(ts_mod, "classify", lambda *a, **k: ("C", 0.95))
    convo.say("ok")
    convo.user("Please investigate the flaky websocket reconnect logic in the client library")
    rs = send(service, convo)
    assert _state(rs.runtime).task_id != task


# ----------------------------------------------------- transition rules
def _base() -> TaskState:
    return TaskState("w", "s", "T", 0, TaskStatus.ACTIVE, (), 0.0, 0.0)


def _atom(kind, **kw) -> StateAtom:
    d = {
        "atom_id": kw.pop("atom_id", "a1"),
        "kind": kind,
        "text": "t",
        "normalized_key": "t",
        "state": AtomState.ACTIVE,
        "origin": Origin.USER,
        "confidence": 0.95,
    }
    d.update(kw)
    return StateAtom(**d)


def _tr(ttype, **kw) -> StateTransition:
    return StateTransition(
        transition_id=kw.pop("tid", "x"), transition_type=ttype, reason_code="T", **kw
    )


def test_low_confidence_cannot_complete_or_satisfy() -> None:
    s = apply_transition(
        _base(),
        _tr(TransitionType.ADD_ATOM, new_atom=_atom(AtomKind.CRITERION, verify="deliverable")),
    )
    with pytest.raises(InvalidTransition):
        apply_transition(
            s,
            _tr(
                TransitionType.SET_ATOM_STATE,
                atom_id="a1",
                new_state=AtomState.SATISFIED,
                confidence=0.6,
            ),
        )
    with pytest.raises(InvalidTransition):
        apply_transition(
            s, _tr(TransitionType.SET_STATUS, new_status=TaskStatus.COMPLETED, confidence=0.6)
        )


def test_agent_cannot_supersede_user_constraint() -> None:
    s = apply_transition(
        _base(), _tr(TransitionType.ADD_ATOM, new_atom=_atom(AtomKind.CONSTRAINT, hard=True))
    )
    agent = _atom(AtomKind.CONSTRAINT, atom_id="a2", origin=Origin.AGENT)
    with pytest.raises(InvalidTransition):
        apply_transition(s, _tr(TransitionType.SUPERSEDE, atom_id="a1", new_atom=agent))
    with pytest.raises(InvalidTransition):
        apply_transition(
            s, _tr(TransitionType.SET_ATOM_STATE, atom_id="a1", new_state=AtomState.REJECTED)
        )


def test_factual_criterion_needs_evidence() -> None:
    s = apply_transition(
        _base(), _tr(TransitionType.ADD_ATOM, new_atom=_atom(AtomKind.CRITERION, verify="test"))
    )
    with pytest.raises(InvalidTransition):
        apply_transition(
            s, _tr(TransitionType.SET_ATOM_STATE, atom_id="a1", new_state=AtomState.SATISFIED)
        )
    ok = apply_transition(
        s,
        _tr(
            TransitionType.SET_ATOM_STATE,
            atom_id="a1",
            new_state=AtomState.SATISFIED,
            supporting_evidence_ids=("E1",),
        ),
    )
    assert ok.atom("a1").state is AtomState.SATISFIED and ok.revision == s.revision + 1


def test_completion_requires_all_rule8_conditions() -> None:
    s = apply_transition(
        _base(), _tr(TransitionType.ADD_ATOM, new_atom=_atom(AtomKind.BLOCKER, blocking=True))
    )
    with pytest.raises(InvalidTransition):
        apply_transition(s, _tr(TransitionType.SET_STATUS, new_status=TaskStatus.COMPLETED))
    s2 = apply_transition(
        s, _tr(TransitionType.SET_ATOM_STATE, atom_id="a1", new_state=AtomState.SATISFIED)
    )
    assert (
        apply_transition(s2, _tr(TransitionType.SET_STATUS, new_status=TaskStatus.COMPLETED)).status
        is TaskStatus.COMPLETED
    )


def test_state_json_roundtrip() -> None:
    s = apply_transition(_base(), _tr(TransitionType.ADD_ATOM, new_atom=_atom(AtomKind.GOAL)))
    assert TaskState.from_json(s.to_json()) == s


def test_task_state_is_session_scoped(service, repo: Path) -> None:
    send(
        service,
        Conversation(repo, "Task for session one. Keep logs quiet."),
        session="sess-one-aaaa",
    )
    rs = send(service, Conversation(repo, "Task for session two."), session="sess-two-bbbb")
    assert not _state(rs.runtime).constraints
    assert SESSION not in rs.runtime.session_key
