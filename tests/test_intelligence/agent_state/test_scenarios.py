"""Plan §19 end-to-end scenarios A-J (same code paths the proxy drives)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from headroom.intelligence.agent_state.scope import ScopeClass
from headroom.intelligence.agent_state.serialization import tokens
from headroom.intelligence.agent_state.store import close_all_stores
from headroom.intelligence.agent_state.task_state import AtomState, TaskStatus
from headroom.intelligence.agent_state.test_impact import tiers_to_run
from headroom.intelligence.agent_state.workflows import WorkflowExecutor, eligible_macros

from .conftest import PYTEST_FAIL, PYTEST_PASS, Conversation, hook, send

LONG_GOAL = (
    "Add request retries to src/pkg/core.py.\n\n"
    "Constraints: Do not change the public API. Must work on Windows and Linux. "
    "Never log credentials. Keep Python 3.10 compatibility.\n\n"
    "Acceptance criteria:\n"
    "- All tests must pass.\n"
    "- The build must compile cleanly.\n"
    "- It should return the last error after the final retry.\n"
    "- It should handle a zero retry count.\n"
    "- Make sure the README documents it is complete.\n"
)


def test_scenario_a_long_task_state_continuity(service, repo: Path) -> None:
    convo = Conversation(repo, LONG_GOAL)
    rs = send(service, convo)
    st = rs.runtime.task_state.state
    assert len(st.constraints) == 4 and len(st.acceptance_criteria) == 5
    for i in range(22):  # >20 tool turns
        if i % 3 == 0:
            convo.edit("src/pkg/core.py", f"o{i}", f"n{i}")
        elif i % 3 == 1:
            convo.read("src/pkg/core.py")
        else:
            convo.bash("git status --short", " M src/pkg/core.py")
    send(service, convo)
    # Early conversation becomes cold: the client compacts history.
    compacted = Conversation(
        repo,
        "This session is being continued from a previous conversation that ran out of context. Summary: retries in progress.",
    )
    compacted.user(
        "Actually, Python 3.10 support is no longer needed; keep Python 3.12 compatibility."
    )
    rs = send(service, compacted)
    st = rs.runtime.task_state.state
    live = [c.text for c in st.constraints]
    assert not any("3.10" in t for t in live) and any("3.12" in t for t in live)
    assert len(live) == 4
    assert any(a.state is AtomState.SUPERSEDED and "3.10" in a.text for a in st.atoms)
    block = rs.block or ""
    assert tokens(block) <= service.config.max_tokens
    for crit in st.acceptance_criteria:
        if crit.state is AtomState.SATISFIED and crit.verify in ("test", "build"):
            assert crit.evidence_ids


def test_scenario_b_evidence_beats_unsupported_claim(service, repo: Path) -> None:
    convo = Conversation(repo, "Fix src/pkg/core.py. All tests must pass.")
    convo.edit("src/pkg/core.py")
    convo.say("I ran the suite and all tests pass.")
    convo.bash("python -m pytest -q", PYTEST_FAIL.replace("test_core", "test_iface"), code=1)
    convo.say("Done, the task is complete.")
    rs = send(service, convo)
    led = rs.runtime.evidence
    assert led.get_claim("tests_passing|latest").source_kind == "TEST"
    assert led.contradictions()
    st = rs.runtime.task_state.state
    assert st.status is not TaskStatus.COMPLETED
    assert st.acceptance_criteria[0].state is AtomState.ACTIVE


def test_scenario_c_invalid_tool_call(service, repo: Path, tmp_path: Path) -> None:
    outside = tmp_path / "outside" / "x.txt"
    # Without a pre-exec hook: the call already ran; it is recorded and surfaced,
    # never reported as blocked.
    convo = Conversation(repo, "Fix src/pkg/core.py.")
    convo.call("Write", {"file_path": str(outside), "content": "x"}, "File created successfully")
    rs = send(service, convo)
    rows = rs.runtime.store.query("SELECT enforced FROM validations")
    assert rows and all(r["enforced"] != "blocked" for r in rows)
    assert "OUTSIDE_PROJECT" in (rs.block or "")
    # With the host hook: blocked before the write happens.
    answer = hook(
        service, repo, "Write", {"file_path": str(outside), "content": "x"}, tool_use_id="toolu_c2"
    )
    assert answer["decision"] == "deny" and answer["enforced"] == "blocked"
    missing = hook(
        service,
        repo,
        "Edit",
        {"file_path": str(repo / "src/pkg/missing.py"), "old_string": "a", "new_string": "b"},
        tool_use_id="toolu_c3",
    )
    assert missing["decision"] == "deny" and "does not exist" in missing["reason"]


def test_scenario_d_legitimate_scope_expansion(service, repo: Path) -> None:
    convo = Conversation(repo, "Add retry support to src/pkg/core.py.")
    rs = send(service, convo)
    sc = rs.runtime.scope
    assert (
        sc.classify_path(str(repo / "src/other/cost.py"), op="write").scope is ScopeClass.UNRELATED
    )
    convo.edit("src/pkg/core.py")
    convo.bash(
        "python -m pytest -q",
        'Traceback (most recent call last):\n  File "src/other/cost.py", line 1, in <module>\nTypeError: interface changed\n1 failed in 0.1s',
        code=1,
    )
    rs = send(service, convo)
    cls = rs.runtime.scope.classify_path(str(repo / "src/other/cost.py"), op="write")
    assert cls.scope is ScopeClass.DEPENDENCY_SCOPE and cls.reason == "TEST_DEPENDENCY"
    assert (
        hook(
            service,
            repo,
            "Edit",
            {"file_path": str(repo / "src/other/cost.py"), "old_string": "R", "new_string": "S"},
            tool_use_id="toolu_d",
        )["decision"]
        == "allow"
    )


def test_scenario_e_unrelated_drift(service, repo: Path) -> None:
    (repo / "src" / "zeta").mkdir()
    (repo / "src" / "zeta" / "cleanup.py").write_text("x = 1\n")
    convo = Conversation(repo, "Add retry support to src/pkg/core.py.")
    convo.read("src/zeta/cleanup.py")  # reading never authorizes writing
    rs = send(service, convo)
    answer = hook(
        service,
        repo,
        "Edit",
        {"file_path": str(repo / "src/zeta/cleanup.py"), "old_string": "x", "new_string": "y"},
    )
    assert answer["decision"] == "allow" and "no supported relation" in answer["context"]
    assert (
        rs.runtime.scope.classify_path(str(repo / "src/zeta/cleanup.py"), op="write").scope
        is ScopeClass.UNRELATED
    )


def test_scenario_f_test_impact_escalation(service, repo: Path) -> None:
    cfg = service.config
    convo = Conversation(repo, "Tweak the helper in src/pkg/iface.py.")
    convo.edit("src/pkg/iface.py")
    rs = send(service, convo)
    planner = rs.runtime.test_impact
    leaf = planner.plan(force=True)
    assert (
        tiers_to_run(
            leaf.risk_score,
            t2=cfg.test_risk_tier2,
            t3=cfg.test_risk_tier3,
            mandatory=bool(leaf.mandatory_tier3),
            criteria_demand_tier2=False,
        )
        == 1
    )
    # A public, platform-specific interface change plus an evidenced dependency
    # raises risk into tier 2.
    (repo / "src" / "pkg" / "api.py").write_text(
        "import msvcrt  # Windows console API\ndef public(): ...\n"
    )
    convo.edit("src/pkg/api.py")
    rs.runtime.scope.add_expansion("src/other/cost.py", "INTERFACE_DEPENDENCY", _ev())
    convo.edit("src/other/cost.py")
    send(service, convo)
    iface = planner.plan(force=True)
    assert iface.components["interface_risk"] == 1.0
    assert (
        tiers_to_run(
            iface.risk_score,
            t2=cfg.test_risk_tier2,
            t3=cfg.test_risk_tier3,
            mandatory=bool(iface.mandatory_tier3),
            criteria_demand_tier2=False,
        )
        == 2
    )
    # A shared serialization change forces tier 3 regardless of score.
    (repo / "src" / "pkg" / "serialization.py").write_text("")
    convo.edit("src/pkg/serialization.py")
    send(service, convo)
    wire = planner.plan(force=True)
    assert "WIRE_PROTOCOL_OR_SERIALIZATION" in wire.mandatory_tier3 and wire.max_tier == 3
    # A tier-1 failure stops broader execution.
    (repo / "tests" / "test_iface.py").write_text(
        "from pkg.iface import g\n\n\ndef test_g():\n    assert g() == 3\n"
    )
    rs.runtime.scope._status_cache = None
    out = planner.run_plan(planner.plan(force=True))
    assert out["status"] == "failed" and [t["tier"] for t in out["tiers"]] == [1]


def _ev():
    from headroom.intelligence.agent_state.events import AgentEvent, EventType

    return AgentEvent(
        event_id="scenario-f", workspace_id="w", session_id="s", event_type=EventType.USER_MESSAGE
    )


@pytest.fixture
def venv_path(monkeypatch):
    monkeypatch.setenv(
        "PATH", os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", "")
    )


def _verify_loop(convo: Conversation, times: int) -> None:
    for i in range(times):
        convo.edit("src/pkg/core.py", f"a{i}", f"b{i}")
        convo.bash("git diff --stat", " 1 file changed")
        convo.bash("python -m pytest -q tests/test_core.py", PYTEST_PASS)
        convo.bash("git status --short", " M src/pkg/core.py")
    convo.edit("src/pkg/core.py", "z", "y")
    convo.say("a")
    convo.say("b")


def test_scenario_g_learned_safe_macro(service, repo: Path, venv_path) -> None:
    convo = Conversation(repo, "Iterate on src/pkg/core.py.")
    _verify_loop(convo, 3)
    rs = send(service, convo)
    learned = [
        m
        for m in eligible_macros(rs.runtime.store, rs.runtime.workspace.workspace_id)
        if m.origin == "learned"
    ]
    assert len(learned) == 1
    from headroom.intelligence.agent_state import mcp

    spec = mcp.tool_spec(eligible_macros(rs.runtime.store, rs.runtime.workspace.workspace_id))
    assert learned[0].name in spec["inputSchema"]["properties"]["macro"]["enum"]
    before = rs.runtime.store.query_one("SELECT COUNT(*) AS n FROM events")["n"]
    result = WorkflowExecutor(rs.runtime).run(learned[0].name)
    assert result["status"] == "success"
    after = rs.runtime.store.query_one("SELECT COUNT(*) AS n FROM events")["n"]
    assert after - before >= 7  # proposed + finished per step (+ derived) + WORKFLOW_RUN
    assert result["evidence_refs"]


def test_scenario_h_macro_invalidation(service, repo: Path, venv_path) -> None:
    convo = Conversation(repo, "Iterate on src/pkg/core.py.")
    _verify_loop(convo, 3)
    rs = send(service, convo)
    macro = [
        m
        for m in eligible_macros(rs.runtime.store, rs.runtime.workspace.workspace_id)
        if m.origin == "learned"
    ][0]
    (repo / "pyproject.toml").write_text(
        "[tool.pytest.ini_options]\npythonpath = ['src']\nmarkers = ['slow']\n"
    )
    rs.runtime.workspace.facts.pop("test_project", None)
    from headroom.intelligence.agent_state.workflows import MacroError

    with pytest.raises(MacroError, match="invalidated"):
        WorkflowExecutor(rs.runtime).run(macro.name)
    convo.bash("git status --short", "")  # ordinary tools remain usable
    assert send(service, convo) is not None


class _DeadAdvisor:
    def choose(self, *a, **k):
        raise ConnectionError("llama-server is down")


class _DeadIntel:
    graphs = None

    def advisor_or_none(self):
        return _DeadAdvisor()


def test_scenario_i_jevk5_outage(make_service, repo: Path) -> None:
    svc = make_service(intelligence=_DeadIntel())
    convo = Conversation(repo, "Fix src/pkg/core.py. Do not modify src/other/. Tests must pass.")
    convo.say("ok")
    convo.user(
        "I believe the retry policy should probably be reconsidered at some later point in time"
    )
    convo.bash("pytest -q", PYTEST_FAIL, code=1)
    rs = send(svc, convo)
    st = rs.runtime.task_state.state
    assert st.primary_goal.text.startswith("Fix src/pkg/core.py")
    assert st.blockers
    assert (
        hook(
            svc,
            repo,
            "Edit",
            {"file_path": str(repo / "src/other/cost.py"), "old_string": "R", "new_string": "S"},
        )["decision"]
        == "deny"
    )
    (repo / "src" / "zeta").mkdir()
    (repo / "src" / "zeta" / "z.py").write_text("")
    assert (
        rs.runtime.scope.classify_path(str(repo / "src/zeta/z.py"), op="write").scope
        is ScopeClass.UNRELATED
    )
    assert svc.metrics.counters["errors"] == 0


def test_scenario_j_restart(make_service, repo: Path, tmp_path: Path, venv_path) -> None:
    svc = make_service()
    convo = Conversation(repo, "Fix src/pkg/core.py. Never log secrets.")
    _verify_loop(convo, 3)
    rs = send(svc, convo)
    task = rs.runtime.task_state.state.task_id
    svc.shutdown()
    close_all_stores()
    # Same workspace, same agent session: the unfinished task resumes; workspace
    # evidence and macros persist.
    svc2 = make_service()
    convo.user("continue")
    rs2 = send(svc2, convo)
    assert rs2.runtime.task_state.state.task_id == task
    assert rs2.runtime.evidence.get_claim("tests_passing|latest") is not None
    assert any(
        m.origin == "learned"
        for m in eligible_macros(rs2.runtime.store, rs2.runtime.workspace.workspace_id)
    )
    ended = rs2.runtime.store.query(
        "SELECT event_type FROM events WHERE event_type = 'SESSION_END'"
    )
    assert ended
    # A different workspace receives none of it.
    other = tmp_path / "other-ws"
    (other / ".git").mkdir(parents=True)
    rs3 = send(svc2, Conversation(other, "Something else."))
    assert rs3.runtime.evidence.get_claim("tests_passing|latest") is None
    assert not any(
        m.origin == "learned"
        for m in eligible_macros(rs3.runtime.store, rs3.runtime.workspace.workspace_id)
    )
    assert rs3.runtime.task_state.state.task_id != task
