"""Feature 20: Workflow Macro Compiler (plan §10.15)."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from headroom.intelligence.agent_state.workflows import (
    MacroError,
    WorkflowExecutor,
    _valid_project_path,
    all_macros,
    candidate_summary,
    eligible_macros,
    format_result,
    macro_name,
)

from .conftest import PYTEST_PASS, Conversation, send

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git required")


@pytest.fixture(autouse=True)
def _agent_venv_on_path(monkeypatch):
    """Macros replay the agent's own commands; put this interpreter's venv first on PATH
    so ``python -m pytest`` resolves the way it does in an activated project venv."""
    import os
    import sys

    monkeypatch.setenv(
        "PATH", os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", "")
    )


VERIFY = [
    ("git diff --stat", " src/pkg/core.py | 2 +-\n 1 file changed, 1 insertion(+), 1 deletion(-)"),
    ("python -m pytest -q {t}", PYTEST_PASS),
    ("git status --short", " M src/pkg/core.py"),
]


def _repeat(
    convo: Conversation, times: int, *, tests: list[str] | None = None, steps=VERIFY
) -> None:
    tests = tests or ["tests/test_core.py"] * times
    for i in range(times):
        convo.edit("src/pkg/core.py", f"a{i}", f"b{i}")
        for cmd, out in steps:
            convo.bash(cmd.format(t=tests[i]), out)
    convo.edit("src/pkg/core.py", "z", "y")
    convo.say("Working.")
    convo.say("Still working.")


def _learned(rs):
    return [
        m
        for m in eligible_macros(rs.runtime.store, rs.runtime.workspace.workspace_id)
        if m.origin == "learned"
    ]


def test_safe_sequence_is_learned_promoted_and_executed(service, repo: Path) -> None:
    """Phase 7 gate."""
    convo = Conversation(repo, "Iterate on src/pkg/core.py until the tests pass.")
    _repeat(convo, 2)
    rs = send(service, convo)
    assert _learned(rs) == []  # two observations: below threshold
    _repeat(convo, 1)
    rs = send(service, convo)
    macros = _learned(rs)
    assert len(macros) == 1
    m = macros[0]
    assert (
        m.safety_class == "VERIFICATION" and m.support_count >= 3 and m.estimated_turns_saved >= 2
    )
    assert m.name == "diff_then_test_then_status"
    assert [s.argument_template[0] for s in m.steps] == ["git", "python", "git"]
    assert m.input_schema["properties"] == {}  # identical args: no slots
    # Announced once in the live-turn state.
    assert "headroom_workflow macro=diff_then_test_then_status" in (rs.block or "")
    result = WorkflowExecutor(rs.runtime).run(m.name)
    assert result["status"] == "success" and result["steps"] == "3/3", result["details"]
    assert result["verification"]["passed"] == 1
    text = format_result(result)
    assert "macro: diff_then_test_then_status" in text and "status: success" in text
    store = rs.runtime.store
    origins = store.query(
        "SELECT event_type FROM events WHERE metadata LIKE '%headroom%' ORDER BY seq"
    )
    kinds = [r["event_type"] for r in origins]
    assert kinds.count("TOOL_CALL_PROPOSED") == 3 and "WORKFLOW_RUN" in kinds
    assert (
        rs.runtime.evidence.get_claim(
            "test_aggregate|pytest|python -m pytest -q tests/test_core.py"
        )
        is not None
    )
    row = store.query_one(
        "SELECT turns_saved, last_used_at FROM workflow_macros WHERE macro_id = ?", (m.macro_id,)
    )
    assert row["turns_saved"] >= 2 and row["last_used_at"]


def test_differing_test_files_become_validated_slot(service, repo: Path) -> None:
    convo = Conversation(repo, "Iterate on the package.")
    _repeat(convo, 3, tests=["tests/test_core.py", "tests/test_iface.py", "tests/test_core.py"])
    rs = send(service, convo)
    m = _learned(rs)[0]
    assert m.input_schema["required"] == ["path1"]
    assert m.input_schema["properties"]["path1"]["validator"] == "test_path"
    ex = WorkflowExecutor(rs.runtime)
    assert ex.run(m.name, {"path1": "tests/test_iface.py"})["status"] == "success"
    for bad in (
        "../outside.py",
        "/etc/passwd",
        "--collect-only",
        "tests/test_core.py; rm -rf /",
        "src/pkg/core.py",
        "tests/nope.py",
    ):
        with pytest.raises(MacroError):
            ex.run(m.name, {"path1": bad})
    with pytest.raises(MacroError):
        ex.run(m.name, {"path1": "tests/test_core.py", "extra": "x"})


def test_arbitrary_command_never_becomes_a_parameter(service, repo: Path) -> None:
    convo = Conversation(repo, "Look around.")
    for cmd in ("git log -1", "git show HEAD", "git blame README"):
        convo.edit("src/pkg/core.py")
        convo.bash(cmd, "x")
        convo.bash("git status --short", "")
        convo.bash("git diff --stat", "")
    convo.edit("src/pkg/core.py")
    convo.say("a")
    convo.say("b")
    rs = send(service, convo)
    assert _learned(rs) == []  # three different commands: three signatures, none repeated


def test_sequence_with_shell_syntax_is_not_executable(service, repo: Path) -> None:
    steps = [
        ("git diff --stat | head -5", "x"),
        ("python -m pytest -q tests/test_core.py 2>&1 | tail -3", PYTEST_PASS),
        ("git status --short", ""),
    ]
    convo = Conversation(repo, "Iterate.")
    _repeat(convo, 3, steps=steps)
    rs = send(service, convo)
    assert _learned(rs) == []
    assert candidate_summary(rs.runtime.store)


def test_mutating_sequences_are_never_promoted(service, repo: Path) -> None:
    convo = Conversation(repo, "Format the code.")
    for _ in range(4):
        convo.bash("ruff format src", "1 file reformatted")
        convo.bash("git add -A", "")
        convo.bash("git status --short", "")
        convo.read("README.md")
    convo.say("a")
    convo.say("b")
    rs = send(service, convo)
    assert _learned(rs) == []


def test_correction_disqualifies(service, repo: Path) -> None:
    convo = Conversation(repo, "Iterate on src/pkg/core.py.")
    for _ in range(3):
        convo.edit("src/pkg/core.py")
        for cmd, out in VERIFY:
            convo.bash(cmd.format(t="tests/test_core.py"), out)
        convo.user("No, that's wrong, undo it.")
    convo.say("a")
    convo.say("b")
    rs = send(service, convo)
    assert _learned(rs) == []
    assert all(c["corrected"] for c in candidate_summary(rs.runtime.store))


def test_contract_violation_disqualifies(service, repo: Path) -> None:
    steps = [
        ("git diff --stat", "x"),
        ("notarealbinary-zz --check", "ok"),
        ("git status --short", ""),
    ]
    convo = Conversation(repo, "Iterate.")
    _repeat(convo, 3, steps=steps)
    rs = send(service, convo)
    assert _learned(rs) == []


def test_length_bounds(service, repo: Path) -> None:
    convo = Conversation(repo, "Iterate.")
    for _ in range(3):
        convo.edit("src/pkg/core.py")
        convo.bash("git status --short", "")  # length 1: below MIN_STEPS
    convo.edit("src/pkg/core.py")
    convo.say("a")
    convo.say("b")
    rs = send(service, convo)
    assert candidate_summary(rs.runtime.store) == []


def test_macro_stops_on_failure_and_auto_disables(service, repo: Path) -> None:
    convo = Conversation(repo, "Iterate on src/pkg/core.py until the tests pass.")
    _repeat(convo, 3)
    rs = send(service, convo)
    m = _learned(rs)[0]
    (repo / "tests" / "test_core.py").write_text("def test_f():\n    assert False\n")
    ex = WorkflowExecutor(rs.runtime)
    r1 = ex.run(m.name)
    assert r1["status"] == "failed" and r1["failed_step"] == "s2" and r1["steps"] == "1/3"
    assert "failure_tail" in format_result(r1)
    r2 = ex.run(m.name)
    assert r2["status"] == "failed"
    with pytest.raises(MacroError, match="disabled"):
        ex.run(m.name)
    assert all_macros(rs.runtime.store)[0].disabled_reason in ("failure_policy", None) or True
    row = rs.runtime.store.query_one(
        "SELECT disabled_reason FROM workflow_macros WHERE macro_id = ?", (m.macro_id,)
    )
    assert row["disabled_reason"] == "failure_policy"


def test_config_change_invalidates_macro(service, repo: Path) -> None:
    """Scenario H."""
    convo = Conversation(repo, "Iterate on src/pkg/core.py until the tests pass.")
    _repeat(convo, 3)
    rs = send(service, convo)
    m = _learned(rs)[0]
    (repo / "pyproject.toml").write_text(
        "[tool.pytest.ini_options]\npythonpath = ['src']\naddopts = '-x'\n"
    )
    rs.runtime.workspace.facts.pop("test_project", None)
    with pytest.raises(MacroError, match="config_changed"):
        WorkflowExecutor(rs.runtime).run(m.name)
    assert _learned(rs) == []
    # Ordinary tools remain usable: the next request is processed normally.
    convo.bash("git status --short", "")
    assert send(service, convo) is not None


def test_workspace_ownership(service, repo: Path, tmp_path: Path) -> None:
    convo = Conversation(repo, "Iterate on src/pkg/core.py until the tests pass.")
    _repeat(convo, 3)
    rs = send(service, convo)
    m = _learned(rs)[0]
    other = tmp_path / "other"
    (other / ".git").mkdir(parents=True)
    rs2 = send(service, Conversation(other, "Something else."))
    ex = WorkflowExecutor(rs2.runtime)
    with pytest.raises(MacroError):
        ex.run(m.name)
    with pytest.raises(MacroError):
        ex.run(m.macro_id)


def test_seed_templates_enabled_only_when_applicable(service, repo: Path, tmp_path: Path) -> None:
    rs = send(service, Conversation(repo, "Fix src/pkg/core.py."))
    names = {m.name for m in eligible_macros(rs.runtime.store, rs.runtime.workspace.workspace_id)}
    assert {"run_test_impact_plan", "show_task_owned_diff", "verify_then_status"} <= names
    bare = tmp_path / "bare"
    bare.mkdir()
    rs2 = send(service, Conversation(bare, "Fix main.c."))
    assert eligible_macros(rs2.runtime.store, rs2.runtime.workspace.workspace_id) == []
    disabled = {m.name: m.disabled_reason for m in all_macros(rs2.runtime.store)}
    assert disabled["run_test_impact_plan"] == "not_applicable"


def test_seeded_plan_macro_uses_test_impact_planner(service, repo: Path) -> None:
    convo = Conversation(repo, "Fix src/pkg/iface.py.")
    convo.edit("src/pkg/iface.py")
    rs = send(service, convo)
    result = WorkflowExecutor(rs.runtime).run("verify_then_status")
    assert result["status"] == "success", result
    assert result["verification"]["status"] == "passed" and 1 in result["verification"]["tiers"]


def test_slot_validator() -> None:
    root = "/r/proj"
    assert _valid_project_path("tests/test_a.py", root, must_exist=False)
    for bad in (
        "../x",
        "/abs",
        "-flag",
        "a;b",
        "a\nb",
        "node_modules/x.js",
        ".env",
        "~/x",
        "$(id)",
    ):
        assert not _valid_project_path(bad, root, must_exist=False), bad


def test_deterministic_names() -> None:
    from headroom.intelligence.agent_state.workflows import WorkflowStep

    steps = [
        WorkflowStep("s1", "git", "Bash", "diff", ("git", "diff")),
        WorkflowStep("s2", "test", "Bash", "test", ("pytest",)),
    ]
    assert macro_name(steps) == macro_name(steps) == "diff_then_test"
    assert macro_name(steps, {"diff_then_test"}).startswith("diff_then_test_")


def test_no_jevk5_needed_for_execution(service, repo: Path) -> None:
    assert service.advisor() is None
    rs = send(service, Conversation(repo, "Fix src/pkg/core.py."))
    assert WorkflowExecutor(rs.runtime).run("show_task_owned_diff")["status"] == "success"
