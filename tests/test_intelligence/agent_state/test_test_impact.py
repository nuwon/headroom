"""Feature 24: Test Impact and Verification Planner (plan §9.13)."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

from headroom.intelligence.agent_state.test_impact import (
    CargoAdapter,
    CTestAdapter,
    GenericAdapter,
    JsAdapter,
    PytestAdapter,
    _walk,
    combine,
    decayed,
    display_argv,
    mandatory_tier3,
    package_manager,
    parse_collect_only,
    parse_ctest_json,
    parse_ctest_n,
    risk_components,
    risk_score,
    tiers_to_run,
)

from .conftest import PYTEST_FAIL, PYTEST_PASS, Conversation, send


# --------------------------------------------------------------- adapters
def test_pytest_adapter_detects_and_maps_static_imports(repo: Path) -> None:
    files = _walk(str(repo))
    a = PytestAdapter(str(repo), files)
    assert a.detect() >= 0.9
    tests = a.discover()
    assert {t.test_id for t in tests} == {"pytest:tests/test_core.py", "pytest:tests/test_iface.py"}
    edges = a.static_edges(["src/pkg/core.py"], tests)
    assert edges == {"pytest:tests/test_core.py": [("src/pkg/core.py", "STATIC", 0.80)]}
    cmd = a.select([type("S", (), {"target": "tests/test_core.py"})()])[0]
    assert cmd.argv[-3:] == ("pytest", "-q", "tests/test_core.py") and cmd.argv[1] == "-m"


def test_pytest_collect_only_parsing() -> None:
    text = "tests/test_a.py::test_one\ntests/test_a.py::TestX::test_two[param-1]\n\n2 tests collected in 0.01s\n"
    assert parse_collect_only(text) == [
        "tests/test_a.py::test_one",
        "tests/test_a.py::TestX::test_two[param-1]",
    ]


def test_cargo_workspace_package_selection(tmp_path: Path) -> None:
    (tmp_path / "Cargo.toml").write_text('[workspace]\nmembers = ["crates/*"]\n')
    for name in ("alpha", "beta"):
        d = tmp_path / "crates" / name
        (d / "src").mkdir(parents=True)
        (d / "tests").mkdir()
        (d / "Cargo.toml").write_text(f'[package]\nname = "{name}"\nversion = "0.1.0"\n')
        (d / "src" / "lib.rs").write_text("")
        (d / "tests" / "parse.rs").write_text("")
    a = CargoAdapter(str(tmp_path), _walk(str(tmp_path)))
    assert a.detect() > 0.9
    assert a.packages() == {"alpha": "crates/alpha", "beta": "crates/beta"}
    tests = a.discover()
    edges = a.static_edges(["crates/alpha/src/lib.rs"], tests)
    assert "cargo:alpha:lib" in edges and not any(k.startswith("cargo:beta") for k in edges)
    from headroom.intelligence.agent_state.test_impact import TestSelection

    cmds = a.select([TestSelection("cargo:alpha:test:parse", "cargo", "", 0.8)])
    assert cmds[0].argv[1:] == ("test", "-p", "alpha", "--test", "parse")
    assert a.full_suite().argv[1:] == ("test", "--workspace")


def test_ctest_json_and_n_fallback(tmp_path: Path) -> None:
    assert parse_ctest_json(
        '{"kind":"ctestInfo","tests":[{"name":"unit_parse"},{"name":"io"}]}'
    ) == ["unit_parse", "io"]
    assert parse_ctest_n(
        "Test project /b\n  Test #1: unit_parse\n  Test #2: io\n\nTotal Tests: 2"
    ) == ["unit_parse", "io"]
    (tmp_path / "CMakeLists.txt").write_text(
        "project(x)\nenable_testing()\nadd_test(NAME a COMMAND a)\n"
    )
    a = CTestAdapter(str(tmp_path), _walk(str(tmp_path)))
    assert a.detect() == 0.5  # no configured build tree: no configure is ever run
    assert a.discover() == []
    (tmp_path / "build").mkdir()
    (tmp_path / "build" / "CTestTestfile.cmake").write_text("")
    a = CTestAdapter(str(tmp_path), _walk(str(tmp_path)))
    assert a.detect() == 0.9


@pytest.mark.parametrize(
    ("lock", "pm", "prefix"),
    [
        ("pnpm-lock.yaml", "pnpm", ["exec", "vitest"]),
        ("yarn.lock", "yarn", ["vitest"]),
        ("bun.lockb", "bun", ["vitest"]),
        ("package-lock.json", "npm", ["--no-install", "vitest"]),
    ],
)
def test_js_package_manager_preserved(
    tmp_path: Path, lock: str, pm: str, prefix: list[str]
) -> None:
    (tmp_path / "package.json").write_text(
        json.dumps({"devDependencies": {"vitest": "1"}, "scripts": {"test": "vitest"}})
    )
    (tmp_path / lock).write_text("")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.ts").write_text("export const a = 1\n")
    (tmp_path / "src" / "a.test.ts").write_text("import { a } from './a'\n")
    files = _walk(str(tmp_path))
    assert package_manager(str(tmp_path), files) == pm
    adapter = JsAdapter(str(tmp_path), files)
    assert adapter.detect() == 0.9 and adapter.framework == "vitest"
    tests = adapter.discover()
    assert adapter.static_edges(["src/a.ts"], tests) == {
        "vitest:src/a.test.ts": [("src/a.ts", "STATIC", 0.80)]
    }
    from headroom.intelligence.agent_state.test_impact import TestSelection

    argv = adapter.select([TestSelection("vitest:src/a.test.ts", "vitest", "src/a.test.ts", 1.0)])[
        0
    ].argv
    assert Path(argv[0]).name.split(".")[0] in ({"npm": "npx", "bun": "bunx"}.get(pm, pm),)
    assert list(argv[1 : 1 + len(prefix)]) == prefix


def test_generic_adapter_never_invents_selectors(tmp_path: Path) -> None:
    assert GenericAdapter(str(tmp_path), []).detect() == 0.0
    (tmp_path / "Makefile").write_text("test:\n\t./run_tests\n")
    a = GenericAdapter(str(tmp_path), _walk(str(tmp_path)))
    assert a.detect() == 0.5 and a.select([]) == [] and a.full_suite().argv[1:] == ("test",)


def test_windows_commands_are_argv_not_shell(monkeypatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    assert (
        display_argv(("C:\\Py\\python.exe", "-m", "pytest", "tests\\a b.py"))
        == 'C:\\Py\\python.exe -m pytest "tests\\a b.py"'
    )


# ------------------------------------------------------------- math/risk
def test_edge_combination_math() -> None:
    assert combine([0.8]) == pytest.approx(0.8)
    assert combine([0.8, 0.45]) == pytest.approx(1 - 0.2 * 0.55)
    assert combine([1.0, 0.9, 0.9]) == 1.0
    assert combine([]) == 0.0


def test_historical_decay() -> None:
    now = time.time()
    assert decayed(0.8, now, now=now) == pytest.approx(0.8)
    assert decayed(0.8, now - 45 * 86400, now=now) == pytest.approx(0.4)
    assert decayed(0.8, now - 91 * 86400, now=now) == 0.0


def test_risk_score_fixture() -> None:
    comps = risk_components(
        changed=[("src/pkg/core.py", "IN_SCOPE"), ("src/api/routes.py", "DEPENDENCY_SCOPE")],
        expected_subsystems=("src/pkg",),
        unresolved_scope_warnings=0,
        historical_failure=0.3,
        evidence_confidences=[0.8, 0.6],
        change_files=2,
        change_lines=1200,
        budget=(8, 400),
    )
    assert comps == {
        "scope_risk": 0.5,
        "interface_risk": 1.0,
        "config_build_risk": 0.0,
        "historical_failure_risk": 0.3,
        "platform_risk": 0.0,
        "evidence_uncertainty": pytest.approx(0.3),
        "change_size_risk": 1.0,
    }
    assert risk_score(comps) == pytest.approx(
        0.25 * 0.5 + 0.20 * 1.0 + 0.15 * 0.3 + 0.10 * 0.3 + 0.05 * 1.0
    )


def test_risk_platform_and_config() -> None:
    comps = risk_components(
        changed=[("pyproject.toml", "IN_SCOPE"), ("src/win_console.py", "IN_SCOPE")],
        expected_subsystems=(),
        unresolved_scope_warnings=1,
        historical_failure=0.0,
        evidence_confidences=[],
        change_files=2,
        change_lines=10,
        budget=(20, 1500),
    )
    assert comps["config_build_risk"] == 1.0 and comps["platform_risk"] == 1.0
    assert comps["scope_risk"] == 1.0 and comps["evidence_uncertainty"] == 1.0


@pytest.mark.parametrize(
    ("risk", "expected"),
    [(0.0, 1), (0.3499, 1), (0.35, 2), (0.6999, 2), (0.70, 3), (0.95, 3)],
)
def test_tier_thresholds(risk: float, expected: int) -> None:
    assert (
        tiers_to_run(risk, t2=0.35, t3=0.70, mandatory=False, criteria_demand_tier2=False)
        == expected
    )


def test_feature_tasks_require_tier2() -> None:
    assert tiers_to_run(0.1, t2=0.35, t3=0.70, mandatory=False, criteria_demand_tier2=True) == 2


@pytest.mark.parametrize(
    ("path", "code"),
    [
        ("proto/api.proto", "WIRE_PROTOCOL_OR_SERIALIZATION"),
        ("app/serializers.py", "WIRE_PROTOCOL_OR_SERIALIZATION"),
        ("db/migrations/0003_add.py", "DATABASE_MIGRATION_OR_SCHEMA"),
        ("app/auth/tokens.py", "AUTH_CORE"),
        ("poetry.lock", "DEPENDENCY_RESOLUTION"),
        ("CMakeLists.txt", "ROOT_BUILD_CONFIG"),
        ("headroom/cli/main.py", "SHARED_CLI_PARSING"),
        ("headroom/proxy/server.py", "CORE_PROXY_MODEL"),
    ],
)
def test_mandatory_tier3_categories(path: str, code: str) -> None:
    assert code in mandatory_tier3([path])
    assert tiers_to_run(0.0, t2=0.35, t3=0.70, mandatory=True, criteria_demand_tier2=False) == 3


def test_leaf_change_is_not_mandatory_and_user_request_is() -> None:
    assert mandatory_tier3(["src/pkg/leaf.py"]) == []
    assert mandatory_tier3(["src/pkg/leaf.py"], user_text="please run the full test suite") == [
        "USER_REQUESTED_FULL_SUITE"
    ]


# ------------------------------------------------------------- planning
def test_plan_uses_task_owned_changes(service, repo: Path) -> None:
    (repo / "src" / "other" / "cost.py").write_text("RATE = 9  # pre-existing user edit\n")
    convo = Conversation(repo, "Fix the leaf helper in src/pkg/iface.py.")
    convo.edit("src/pkg/iface.py")
    rs = send(service, convo)
    plan = rs.runtime.test_impact.plan(force=True)
    assert plan.changed == ["src/pkg/iface.py"]
    assert [s.test_id for s in plan.tier1] == ["pytest:tests/test_iface.py"]
    assert plan.max_tier >= 1 and plan.commands[1][0].selections == ("tests/test_iface.py",)


def test_no_impacted_tests_reported(service, repo: Path) -> None:
    (repo / "README.md").write_text("docs\n")
    convo = Conversation(repo, "Fix the README wording.")
    convo.edit("README.md")
    rs = send(service, convo)
    plan = rs.runtime.test_impact.plan(force=True)
    assert plan.tier1 == [] and "NO_IMPACTED_TESTS_FOUND" in plan.rationale_codes


def test_plan_without_confident_adapter_invents_nothing(service, tmp_path: Path) -> None:
    root = tmp_path / "bare"
    (root / ".git").mkdir(parents=True)
    (root / "main.c").write_text("int main(){}\n")
    convo = Conversation(root, "Fix main.c.")
    convo.edit("main.c")
    rs = send(service, convo)
    plan = rs.runtime.test_impact.plan(force=True)
    assert plan.commands[1] == [] and plan.commands[2] == [] and plan.commands[3] == []


def test_plan_works_without_jevk5(service, repo: Path) -> None:
    assert service.advisor() is None
    convo = Conversation(repo, "Fix src/pkg/core.py.")
    convo.edit("src/pkg/core.py")
    assert send(service, convo).runtime.test_impact.plan(force=True) is not None


def test_historical_learning_failure_edge(service, repo: Path) -> None:
    convo = Conversation(repo, "Fix src/pkg/core.py. Tests must pass.")
    convo.edit("src/pkg/core.py")
    convo.bash("python -m pytest -q", PYTEST_FAIL, code=1)
    rs = send(service, convo)
    (repo / "src" / "pkg" / "core.py").write_text("def f():\n    return 1  # fixed\n")
    convo.edit("src/pkg/core.py", "x", "y")
    convo.bash(
        "python -m pytest -q", "..\n" + "tests/test_core.py::test_f PASSED\n2 passed in 0.1s"
    )
    rs = send(service, convo)
    edges = {
        (r["source_resource_id"], r["test_id"], r["edge_type"]): r["weight"]
        for r in rs.runtime.store.query("SELECT * FROM impact_edges")
    }
    assert edges[("file:src/pkg/core.py", "pytest:tests/test_core.py", "FAILURE")] == 0.90
    assert ("file:src/pkg/core.py", "pytest:tests/test_core.py", "HISTORICAL") in edges


def test_preexisting_failure_is_not_a_blocker(service, repo: Path) -> None:
    convo = Conversation(repo, "Add a docstring to src/pkg/iface.py. Tests must pass.")
    convo.bash("python -m pytest -q", PYTEST_FAIL, code=1)  # before any task-owned change
    convo.edit("src/pkg/iface.py")
    convo.bash("python -m pytest -q", PYTEST_FAIL, code=1)
    rs = send(service, convo)
    agg = rs.runtime.evidence.get_claim("test_aggregate|pytest|python -m pytest -q")
    assert agg.value["preexisting"] == ["tests/test_core.py::test_f"]
    blockers = rs.runtime.task_state.state.blockers
    assert (
        blockers and not any(b.blocking for b in blockers) and "pre-existing" in blockers[-1].text
    )


def _write_pass_fail_repo(repo: Path, *, fail: bool) -> None:
    body = "def test_g():\n    assert False\n" if fail else "def test_g():\n    assert True\n"
    (repo / "tests" / "test_iface.py").write_text("from pkg.iface import g\n\n\n" + body)


def test_staged_execution_stops_at_first_failure(service, repo: Path) -> None:
    convo = Conversation(repo, "Fix src/pkg/iface.py.")
    convo.edit("src/pkg/iface.py")
    rs = send(service, convo)
    planner = rs.runtime.test_impact
    _write_pass_fail_repo(repo, fail=True)
    plan = planner.plan(force=True)
    out = planner.run_plan(plan, max_tier=3)
    assert out["status"] == "failed" and [t["tier"] for t in out["tiers"]] == [1]
    assert out["evidence_refs"]
    _write_pass_fail_repo(repo, fail=False)
    out = planner.run_plan(planner.plan(force=True), max_tier=3)
    assert out["status"] == "passed" and [t["tier"] for t in out["tiers"]] == [1, 3]
    # Results feed the ledger and task state like any other test run.
    assert rs.runtime.evidence.get_claim("tests_passing|latest").value["ok"] is True


def test_flaky_test_gets_exactly_one_rerun(service, repo: Path, monkeypatch) -> None:
    from headroom.intelligence.agent_state import proc as proc_mod
    from headroom.intelligence.agent_state.proc import ProcResult

    convo = Conversation(repo, "Fix src/pkg/iface.py.")
    convo.edit("src/pkg/iface.py")
    rs = send(service, convo)
    planner = rs.runtime.test_impact
    planner.store.write(
        lambda c: c.execute(
            "INSERT INTO test_cases(test_id, framework, canonical_name, file_path, flake_score, last_status, history) VALUES ('pytest:tests/test_iface.py','pytest','x','tests/test_iface.py',0.7,'pass','[]')"
        )
    )
    calls = []
    outputs = iter(
        [
            "F\nFAILED tests/test_iface.py::test_g - x\n1 failed in 0.1s",
            ".\n1 passed in 0.1s",
            "never",
        ]
    )

    real = proc_mod.run_argv

    def fake(argv, *, cwd, timeout=0, env=None):
        if argv[0] == "git" or (len(argv) > 1 and argv[1] == "-c"):
            return real(argv, cwd=cwd, timeout=timeout)
        calls.append(argv)
        text = next(outputs)
        return ProcResult(tuple(argv), 1 if "failed" in text else 0, text, "", 1.0)

    monkeypatch.setattr(proc_mod, "run_argv", fake)
    plan = planner.plan(force=True)
    plan.required_non_test_checks = []
    out = planner.run_plan(plan, max_tier=1)
    assert len(calls) == 2 and out["status"] == "flaky"
    row = planner.store.query_one(
        "SELECT last_status FROM test_cases WHERE test_id = 'pytest:tests/test_iface.py'"
    )
    assert row["last_status"] == "flaky"


def test_verify_section_injected_only_when_actionable(service, repo: Path) -> None:
    convo = Conversation(repo, "Fix src/pkg/iface.py. Tests must pass.")
    convo.edit("src/pkg/iface.py")
    rs = send(service, convo)
    assert "verify:" not in (rs.block or "")
    convo.bash("python -m pytest -q tests/test_core.py", PYTEST_PASS)
    rs = send(service, convo)
    assert "verify:" in rs.block and "tests/test_iface.py" in rs.block and len(rs.block) // 4 < 1200


def test_phase6_gate_escalation(service, repo: Path) -> None:
    """Leaf change -> tier 1; interface change -> tier 2; serialization change -> tier 3."""
    convo = Conversation(repo, "Tweak the helper in src/pkg/iface.py.")
    convo.edit("src/pkg/iface.py")
    rs = send(service, convo)
    leaf = rs.runtime.test_impact.plan(force=True)
    assert not leaf.mandatory_tier3
    (repo / "src" / "pkg" / "serializer.py").write_text("")
    convo.edit("src/pkg/serializer.py")
    rs = send(service, convo)
    wire = rs.runtime.test_impact.plan(force=True)
    assert "WIRE_PROTOCOL_OR_SERIALIZATION" in wire.mandatory_tier3 and wire.max_tier == 3
    assert wire.risk_score >= leaf.risk_score
