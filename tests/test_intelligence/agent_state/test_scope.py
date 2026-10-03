"""Feature 23: Change-Scope / Drift Firewall (plan §8.14)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from headroom.intelligence.agent_state.scope import ContractMode, ScopeClass, extract_paths

from .conftest import Conversation, git, hook, send


def _cls(rs, rel: str, *, op: str = "write"):
    sc = rs.runtime.scope
    return sc.classify_path(str(Path(sc.root) / rel), op=op)


def test_extract_paths() -> None:
    assert extract_paths("Fix src/pkg/core.py and docs/ but not README.md") == [
        "src/pkg/core.py",
        "docs/",
        "README.md",
    ]


def test_explicit_scope_and_contract(service, repo: Path) -> None:
    rs = send(
        service,
        Conversation(repo, "Fix the retry bug in src/pkg/core.py. Do not modify src/other/."),
    )
    c = rs.runtime.scope.contract
    assert c.explicit_in_scope_paths == ("src/pkg/core.py",)
    assert c.explicit_out_of_scope_paths == ("src/other/",)
    assert c.mode is ContractMode.ACTIVE and c.task_class == "small"
    assert _cls(rs, "src/pkg/core.py").scope is ScopeClass.IN_SCOPE
    assert _cls(rs, "src/pkg/iface.py").scope is ScopeClass.IN_SCOPE  # expected subsystem
    excl = _cls(rs, "src/other/cost.py")
    assert (
        excl.scope is ScopeClass.FORBIDDEN and excl.reason == "USER_EXCLUDED" and excl.deterministic
    )


def test_test_file_of_target_is_dependency(service, repo: Path) -> None:
    rs = send(service, Conversation(repo, "Fix the retry bug in src/pkg/core.py."))
    cls = _cls(rs, "tests/test_core.py")
    assert cls.scope is ScopeClass.DEPENDENCY_SCOPE and cls.expansion == "TEST_DEPENDENCY"


def test_dependency_via_graph(make_service, repo: Path) -> None:
    class Graph:
        def neighborhood(self, files, symbols):
            return (["src/other/cost.py"] if "src/pkg/core.py" in files else []), []

    class Intel:
        graphs = type("G", (), {"for_workspace": staticmethod(lambda key: Graph())})()

        def advisor_or_none(self):
            return None

    svc = make_service(intelligence=Intel())
    rs = send(svc, Conversation(repo, "Fix the retry bug in src/pkg/core.py."))
    cls = _cls(rs, "src/other/cost.py")
    assert cls.scope is ScopeClass.DEPENDENCY_SCOPE and cls.reason == "GRAPH_DIRECT_DEPENDENCY"


def test_dependency_via_failing_build_and_tempting_unrelated_edit(service, repo: Path) -> None:
    """Phase 5 gate: unrelated drift is flagged, an evidenced dependency is allowed."""
    (repo / "src" / "billing").mkdir()
    (repo / "src" / "billing" / "rates.py").write_text("X = 1\n")
    (repo / "src" / "net").mkdir()
    (repo / "src" / "net" / "wire.py").write_text("Y = 2\n")
    convo = Conversation(repo, "Add retry support to src/pkg/core.py.")
    convo.edit("src/pkg/core.py")
    convo.bash(
        "python -m pytest -q",
        'Traceback (most recent call last):\n  File "src/net/wire.py", line 1, in <module>\nTypeError: retry() missing argument\n1 failed in 0.1s',
        code=1,
    )
    rs = send(service, convo)
    dep = _cls(rs, "src/net/wire.py")
    assert dep.scope is ScopeClass.DEPENDENCY_SCOPE and dep.reason == "TEST_DEPENDENCY"
    row = rs.runtime.store.query_one(
        "SELECT evidence_ids FROM scope_expansions WHERE path = 'src/net/wire.py'"
    )
    assert row is not None and "E" in row["evidence_ids"]
    drift = _cls(rs, "src/billing/rates.py")
    assert drift.scope is ScopeClass.UNRELATED
    convo.edit("src/billing/rates.py")
    rs = send(service, convo)
    assert "src/billing/rates.py" in (rs.block or "") and "no supported relation" in rs.block
    assert "src/net/wire.py" not in (rs.block or "")


def test_read_does_not_authorize_write(service, repo: Path) -> None:
    (repo / "src" / "billing").mkdir()
    (repo / "src" / "billing" / "rates.py").write_text("X = 1\n")
    convo = Conversation(repo, "Add retry support to src/pkg/core.py.")
    convo.read("src/billing/rates.py")
    rs = send(service, convo)
    assert _cls(rs, "src/billing/rates.py").scope is ScopeClass.UNRELATED


def test_learning_mode_allows_first_edits_then_activates(service, repo: Path) -> None:
    convo = Conversation(repo, "The pricing calculation is wrong, please investigate and fix it.")
    rs = send(service, convo)
    assert rs.runtime.scope.contract.mode is ContractMode.LEARNING
    assert _cls(rs, "src/other/cost.py").reason == "LEARNING_CANDIDATE"
    convo.edit("src/other/cost.py")
    convo.edit("src/other/cost.py", "c", "d")
    (repo / "src/other/rate.py").write_text("")
    convo.edit("src/other/rate.py")
    rs = send(service, convo)
    assert rs.runtime.scope.contract.mode is ContractMode.ACTIVE
    assert "src/other" in rs.runtime.scope.contract.expected_subsystems


def test_dirty_baseline_is_not_blamed(service, repo: Path) -> None:
    (repo / "src" / "other" / "cost.py").write_text("RATE = 4  # user's own edit\n")
    (repo / "notes.txt").write_text("scratch\n")
    convo = Conversation(repo, "Fix the retry bug in src/pkg/core.py.")
    rs = send(service, convo)
    base = rs.runtime.scope.baseline()
    assert "src/other/cost.py" in base["dirty"] and "notes.txt" in base["untracked"]
    changes = dict(rs.runtime.scope.task_owned_changes(refresh_git=True))
    assert "src/other/cost.py" not in changes and "notes.txt" not in changes
    # The dirty file changes again during the task: now it is task-owned.
    (repo / "src" / "other" / "cost.py").write_text("RATE = 5\n")
    rs.runtime.scope._status_cache = None
    assert "src/other/cost.py" in dict(rs.runtime.scope.task_owned_changes(refresh_git=True))


def test_generated_output_classified_separately(service, repo: Path) -> None:
    (repo / "build").mkdir()
    rs = send(service, Conversation(repo, "Fix src/pkg/core.py."))
    from headroom.intelligence.agent_state.families import normalize_tool_call

    build = normalize_tool_call("Bash", {"command": "make"}, default_cwd=str(repo))
    cls = rs.runtime.scope.classify_path(str(repo / "build/out.o"), op="write", inv=build)
    assert cls.scope is ScopeClass.GENERATED_EFFECT
    edit = rs.runtime.scope.classify_path(str(repo / "build/out.o"), op="write")
    assert edit.scope is ScopeClass.FORBIDDEN and edit.reason == "GENERATED_ARTIFACT"


def test_protected_paths(service, repo: Path) -> None:
    rs = send(service, Conversation(repo, "Fix src/pkg/core.py."))
    assert _cls(rs, ".git/config").reason == "GIT_INTERNALS"
    assert _cls(rs, "node_modules/x/index.js").reason == "INSTALLED_DEPENDENCY"
    assert _cls(rs, ".env").reason == "SECRET_FILE"
    lock = _cls(rs, "poetry.lock")
    assert lock.scope is ScopeClass.AMBIGUOUS and lock.reason == "LOCKFILE_WITHOUT_DEPENDENCY_TASK"
    rs2 = send(
        service,
        Conversation(repo, "Upgrade the requests dependency to 2.32."),
        session="dep-task-session",
    )
    assert _cls(rs2, "poetry.lock").scope is ScopeClass.DEPENDENCY_SCOPE


def test_user_named_secret_file_is_allowed(service, repo: Path) -> None:
    rs = send(service, Conversation(repo, "Add the new DSN to .env.local please."))
    assert _cls(rs, ".env.local").scope is ScopeClass.IN_SCOPE


def test_root_escape_blocked_by_hook(service, repo: Path, tmp_path: Path) -> None:
    send(service, Conversation(repo, "Fix src/pkg/core.py."))
    ans = hook(service, repo, "Write", {"file_path": str(tmp_path / "outside.txt"), "content": "x"})
    assert ans["decision"] == "deny" and "OUTSIDE_PROJECT" in ans["reason"]
    ans = hook(service, repo, "Bash", {"command": "rm -rf ../"}, tool_use_id="hk-rm")
    assert ans["decision"] == "deny"
    ans = hook(service, repo, "Bash", {"command": "rm -rf ."}, tool_use_id="hk-rm2")
    assert ans["decision"] == "deny" and "DELETES_PROJECT_ROOT" in ans["reason"]


@pytest.mark.skipif(
    sys.platform.startswith("win"), reason="symlink creation needs privileges on Windows"
)
def test_symlink_escape_is_resolved(service, repo: Path, tmp_path: Path) -> None:
    target = tmp_path / "secret_dir"
    target.mkdir()
    os.symlink(target, repo / "link")
    rs = send(service, Conversation(repo, "Fix src/pkg/core.py."))
    cls = _cls(rs, "link/file.txt")
    assert cls.scope is ScopeClass.FORBIDDEN and cls.reason == "OUTSIDE_PROJECT"


def test_windows_case_insensitive_matching(service, repo: Path, monkeypatch) -> None:
    rs = send(service, Conversation(repo, "Fix src/pkg/core.py. Do not modify src/other/."))
    sc = rs.runtime.scope
    monkeypatch.setattr(os, "name", "nt")
    assert sc._under("SRC/OTHER/cost.py", sc.contract.explicit_out_of_scope_paths)


def test_unrelated_warns_not_blocks(service, repo: Path) -> None:
    (repo / "src" / "billing").mkdir()
    (repo / "src" / "billing" / "rates.py").write_text("X = 1\n")
    send(service, Conversation(repo, "Fix the retry bug in src/pkg/core.py."))
    ans = hook(
        service,
        repo,
        "Edit",
        {"file_path": str(repo / "src/billing/rates.py"), "old_string": "X", "new_string": "Y"},
    )
    assert ans["decision"] == "allow" and "no supported relation" in ans["context"]


def test_scope_warn_mode_never_blocks(make_service, repo: Path) -> None:
    svc = make_service(env={"HEADROOM_SCOPE_MODE": "warn"})
    send(svc, Conversation(repo, "Fix src/pkg/core.py. Do not modify src/other/."))
    ans = hook(
        svc,
        repo,
        "Edit",
        {"file_path": str(repo / "src/other/cost.py"), "old_string": "R", "new_string": "S"},
    )
    assert ans["decision"] == "allow"


def test_frozen_state_flags_new_edits(service, repo: Path) -> None:
    convo = Conversation(repo, "Rename the helper in src/pkg/core.py.")
    convo.edit("src/pkg/core.py")
    convo.say("Done.")
    rs = send(service, convo)
    assert rs.runtime.scope.contract.mode is ContractMode.FROZEN
    convo.user("continue")
    convo.edit("src/pkg/iface.py")
    rs = send(service, convo)
    warnings = [dict(r) for r in rs.runtime.store.query("SELECT reason FROM scope_warnings")]
    assert any(w["reason"] == "FROZEN_NEW_EDIT" for w in warnings)


def test_warnings_are_deduplicated(service, repo: Path) -> None:
    (repo / "src" / "billing").mkdir()
    (repo / "src" / "billing" / "rates.py").write_text("X = 1\n")
    convo = Conversation(repo, "Fix the retry bug in src/pkg/core.py.")
    for _ in range(3):
        convo.edit("src/billing/rates.py")
    rs = send(service, convo)
    rows = rs.runtime.store.query(
        "SELECT * FROM scope_warnings WHERE path = 'src/billing/rates.py'"
    )
    assert len(rows) == 1
    assert rs.block.count("no supported relation") == 1
    convo.say("ok")
    convo.user("continue")
    rs = send(service, convo)
    assert "no supported relation" not in (rs.block or "")


def test_forbidden_write_is_violation_and_blocks_completion(service, repo: Path) -> None:
    convo = Conversation(repo, "Fix src/pkg/core.py. Do not modify src/other/.")
    convo.edit("src/other/cost.py")
    convo.say("Done.")
    rs = send(service, convo)
    assert rs.runtime.scope.blocking_violations()
    st = rs.runtime.task_state.state
    assert st.status.value != "COMPLETED"
    assert any(c.state.value == "VIOLATED" for c in st.constraints)


def test_change_budget_warning(service, repo: Path) -> None:
    convo = Conversation(repo, "Fix the typo in src/pkg/core.py.")
    for i in range(9):
        p = repo / "src" / "pkg" / f"m{i}.py"
        p.write_text("")
        convo.edit(f"src/pkg/m{i}.py")
    rs = send(service, convo)
    assert any(
        "Change budget exceeded" in r["message"]
        for r in rs.runtime.store.query("SELECT message FROM scope_warnings")
    )


def test_no_jevk5_required(service, repo: Path) -> None:
    assert service.advisor() is None
    rs = send(service, Conversation(repo, "Fix src/pkg/core.py."))
    assert _cls(rs, "src/pkg/core.py").scope is ScopeClass.IN_SCOPE


def test_jevk5_can_suppress_but_never_authorize(service, repo: Path, monkeypatch) -> None:
    from headroom.intelligence.agent_state import advice

    (repo / "src" / "billing").mkdir()
    (repo / "src" / "billing" / "rates.py").write_text("X = 1\n")
    monkeypatch.setattr(advice, "classify", lambda *a, **k: ("A", 0.95))
    send(service, Conversation(repo, "Fix src/pkg/core.py. Do not modify src/other/."))
    ans = hook(
        service,
        repo,
        "Edit",
        {"file_path": str(repo / "src/billing/rates.py"), "old_string": "X", "new_string": "Y"},
    )
    assert ans["decision"] == "allow" and "context" not in ans
    ans = hook(
        service,
        repo,
        "Edit",
        {"file_path": str(repo / "src/other/cost.py"), "old_string": "R", "new_string": "S"},
        tool_use_id="hk9",
    )
    assert ans["decision"] == "deny"


def test_expansion_reason_codes_are_closed(service, repo: Path) -> None:
    rs = send(service, Conversation(repo, "Fix src/pkg/core.py."))
    from headroom.intelligence.agent_state.events import AgentEvent, EventType

    ev = AgentEvent(
        event_id="e", workspace_id="w", session_id="s", event_type=EventType.USER_MESSAGE
    )
    with pytest.raises(ValueError):
        rs.runtime.scope.add_expansion("x.py", "because needed", ev)
    rs.runtime.scope.add_expansion("src/other/cost.py", "USER_EXPANSION", ev)
    assert _cls(rs, "src/other/cost.py").scope is ScopeClass.DEPENDENCY_SCOPE


def test_git_unavailable_repo_still_classifies(service, tmp_path: Path) -> None:
    root = tmp_path / "plain"
    (root / "src").mkdir(parents=True)
    rs = send(service, Conversation(root, "Fix src/main.py."))
    assert rs.runtime.scope.baseline()["head"] == ""
    assert _cls(rs, "src/main.py").scope is ScopeClass.IN_SCOPE


def test_committed_change_detection_uses_baseline(service, repo: Path) -> None:
    rs = send(service, Conversation(repo, "Fix src/pkg/core.py."))
    (repo / "src" / "pkg" / "core.py").write_text("changed\n")
    rs.runtime.scope._status_cache = None
    assert "src/pkg/core.py" in dict(rs.runtime.scope.task_owned_changes(refresh_git=True))
    git(repo, "status")
