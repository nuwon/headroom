"""Feature 17: Evidence Ledger with Provenance (plan §6.11)."""

from __future__ import annotations

import threading
import time
from pathlib import Path

from headroom.intelligence.agent_state.results import (
    parse_diagnostics,
    parse_exit_code,
    parse_test_output,
)
from headroom.intelligence.agent_state.store import close_all_stores

from .conftest import PYTEST_FAIL, PYTEST_PASS, Conversation, send


def _ledger(rs):
    return rs.runtime.evidence


# ------------------------------------------------------------- parsers
def test_exit_code_formats() -> None:
    assert parse_exit_code("Exit code 2\nboom") == 2
    assert parse_exit_code("Exit code: 1\nWall time: 0.2 seconds\nOutput:\n") == 1
    assert parse_exit_code("Process exited with code 0") == 0
    assert parse_exit_code('{"output":"x","metadata":{"exit_code":3}}') == 3
    assert parse_exit_code("nothing here") is None


def test_test_output_parsers() -> None:
    py = parse_test_output(PYTEST_FAIL, "pytest")
    assert (py.passed, py.failed, py.failed_ids) == (1, 1, ["tests/test_core.py::test_f"])
    cargo = parse_test_output(
        "test a::b ... ok\ntest a::c ... FAILED\ntest result: FAILED. 1 passed; 1 failed; 0 ignored; 0 measured",
        "cargo",
    )
    assert cargo.failed_ids == ["a::c"] and cargo.passed == 1
    jest = parse_test_output("FAIL src/a.test.ts\nTests:       1 failed, 3 passed, 4 total", "jest")
    assert jest.failed == 1 and jest.passed == 3 and jest.failed_ids == ["src/a.test.ts"]
    vitest = parse_test_output(
        " FAIL  src/x.test.ts > adds\n      Tests  1 failed | 5 passed (6)", "vitest"
    )
    assert vitest.failed == 1 and vitest.passed == 5
    ctest = parse_test_output(
        "  3 - unit_parse (Failed)\n75% tests passed, 1 tests failed out of 4", "ctest"
    )
    assert ctest.failed_ids == ["unit_parse"] and ctest.passed == 3
    crlf = parse_test_output("..\r\n2 passed in 0.01s\r\n", "pytest")
    assert crlf.passed == 2
    assert parse_test_output("random output", "pytest") is None


def test_diagnostics_parsers() -> None:
    gcc = parse_diagnostics("src/a.c:12:5: error: unknown type 'x'\nsrc/a.c:3:1: warning: unused")
    assert [(d.path, d.line) for d in gcc] == [("src/a.c", 12)]
    msvc = parse_diagnostics(r"C:\proj\a.cpp(7,3): error C2065: 'y': undeclared identifier")
    assert msvc[0].code == "C2065" and msvc[0].line == 7
    rust = parse_diagnostics("error[E0425]: cannot find value `z`\n --> src/lib.rs:4:9")
    assert rust[0].path == "src/lib.rs" and rust[0].code == "E0425"
    tb = parse_diagnostics(
        'Traceback (most recent call last):\n  File "pkg/x.py", line 9, in f\nValueError: bad'
    )
    assert tb[0].path == "pkg/x.py" and tb[0].code == "ValueError"


# ---------------------------------------------------------- extractors
def test_command_filesystem_test_extractors(service, repo: Path) -> None:
    convo = Conversation(repo, "Fix the bug. Tests must pass.")
    convo.read("src/pkg/core.py")
    convo.call(
        "Read",
        {"file_path": str(repo / "src/pkg/missing.py")},
        "<tool_use_error>File does not exist.</tool_use_error>",
        error=True,
    )
    convo.edit("src/pkg/core.py")
    convo.bash("python -m pytest -q", PYTEST_FAIL, code=1)
    rs = send(service, convo)
    led = _ledger(rs)
    assert led.get_claim("file_exists|src/pkg/core.py").value is True
    assert led.get_claim("file_exists|src/pkg/missing.py").value is False
    assert led.get_claim("file_hash|src/pkg/core.py") is not None
    assert led.get_claim("file_modified|src/pkg/core.py").source_kind == "FILESYSTEM"
    cmd = led.get_claim("command_exit|python -m pytest -q")
    assert cmd.value["exit_code"] == 1 and cmd.confidence == 0.99
    agg = led.get_claim("test_aggregate|pytest|python -m pytest -q")
    assert agg.value["failed"] == 1 and agg.value["ok"] is False and agg.source_kind == "TEST"
    status = led.get_claim("test_status|pytest|tests/test_core.py::test_f")
    assert status.value["state"] == "fail"


def test_git_and_build_extractors(service, repo: Path) -> None:
    convo = Conversation(repo, "Build the project.")
    convo.bash(
        "git status",
        "On branch main\nChanges not staged for commit:\n\tmodified:   src/pkg/core.py\n",
    )
    convo.bash("git rev-parse HEAD", "0123456789abcdef0123456789abcdef01234567")
    convo.bash(
        "git diff --stat", " src/pkg/core.py | 2 +-\n 1 file changed, 1 insertion(+), 1 deletion(-)"
    )
    convo.bash(
        "make", "src/a.c:12:5: error: unknown type name 'foo'\nmake: *** [all] Error 1", code=2
    )
    rs = send(service, convo)
    led = _ledger(rs)
    assert led.get_claim("git_branch|repo").value == "main"
    assert led.get_claim("git_dirty|repo").value["files"] == ["src/pkg/core.py"]
    assert led.get_claim("git_head|repo").value.startswith("0123456789")
    assert led.get_claim("git_diffstat|repo").value == {"files": 1, "insertions": 1, "deletions": 1}
    build = led.get_claim("build_status|make|make")
    assert build.value["ok"] is False and build.confidence == 0.98
    assert led.get_claim("compiler_error|src/a.c:12").value["line"] == 12


def test_unknown_output_records_only_generic_fact(service, repo: Path) -> None:
    convo = Conversation(repo, "Run the checks.")
    convo.bash("pytest -q", "something unparseable happened", code=3)
    rs = send(service, convo)
    led = _ledger(rs)
    assert led.get_claim("command_exit|pytest -q") is not None
    assert led.get_claim("test_aggregate|pytest|pytest -q") is None


def test_supersession_keeps_history(service, repo: Path) -> None:
    convo = Conversation(repo, "Fix it. Tests must pass.")
    convo.bash("pytest -q", PYTEST_FAIL, code=1)
    convo.bash("pytest -q", PYTEST_PASS)
    rs = send(service, convo)
    led = _ledger(rs)
    head = led.get_claim("test_aggregate|pytest|pytest -q")
    assert head.value["ok"] is True and head.supersedes_evidence_id
    old = led.get(head.supersedes_evidence_id)
    assert old.status == "SUPERSEDED" and old.value["ok"] is False


def test_tool_evidence_contradicts_unsupported_agent_claim(service, repo: Path) -> None:
    convo = Conversation(repo, "Fix it. Tests must pass.")
    convo.say("All tests pass now.")
    convo.bash("pytest -q", PYTEST_FAIL, code=1)
    rs = send(service, convo)
    led = _ledger(rs)
    head = led.get_claim("tests_passing|latest")
    assert head.source_kind == "TEST" and head.value["ok"] is False
    pairs = led.contradictions()
    assert pairs and {p.source_kind for p in pairs[0]} == {"TEST", "AGENT"}
    agent = next(r for r in pairs[0] if r.source_kind == "AGENT")
    assert agent.status == "CONTRADICTED" and agent.confidence == 0.55


def test_agent_claim_never_replaces_tool_head(service, repo: Path) -> None:
    convo = Conversation(repo, "Fix it.")
    convo.bash("pytest -q", PYTEST_FAIL, code=1)
    convo.say("The tests pass.")
    rs = send(service, convo)
    led = _ledger(rs)
    head = led.get_claim("tests_passing|latest")
    assert head.source_kind == "TEST" and head.value["ok"] is False
    assert any(
        r.source_kind == "AGENT" and r.status == "CONTRADICTED"
        for r in led.get_recent(20, ("tests_passing",))
    )


def test_user_fact_contradicted_by_probe(service, repo: Path) -> None:
    convo = Conversation(
        repo, "The dev server is listening on port 8090, please test the endpoint."
    )
    convo.bash(
        "curl -s http://127.0.0.1:8090/health",
        "curl: (7) Failed to connect to 127.0.0.1 port 8090: Connection refused",
        code=7,
    )
    rs = send(service, convo)
    led = _ledger(rs)
    head = led.get_claim("port_state|127.0.0.1:8090")
    assert head.source_kind == "TOOL" and head.value["state"] == "closed"
    user = [r for r in led.get_recent(10, ("port_state",)) if r.source_kind == "USER"][0]
    assert user.status == "CONTRADICTED" and user.confidence == 0.75
    assert led.get_conflicts(user.evidence_id)


def test_volatile_facts_go_stale_durable_do_not(service, repo: Path) -> None:
    convo = Conversation(repo, "Check the server")
    convo.bash("curl http://localhost:9000", "ok")
    convo.bash("pytest -q", PYTEST_PASS)
    rs = send(service, convo)
    led = _ledger(rs)
    port = led.get_claim("port_state|127.0.0.1:9000")
    assert port.valid_until is not None and port.valid_until - port.created_at == 30.0
    led.store.write(
        lambda c: c.execute(
            "UPDATE evidence SET valid_until = ? WHERE evidence_id = ?",
            (time.time() - 1, port.evidence_id),
        )
    )
    assert led.get_claim("port_state|127.0.0.1:9000").status == "STALE"
    agg = led.get_claim("test_aggregate|pytest|pytest -q")
    assert agg.valid_until is None and agg.status == "ACTIVE"


def test_file_hash_stale_after_mutation(service, repo: Path) -> None:
    convo = Conversation(repo, "Edit core")
    convo.read("src/pkg/core.py")
    rs = send(service, convo)
    led = _ledger(rs)
    first = led.get_claim("file_hash|src/pkg/core.py")
    (repo / "src/pkg/core.py").write_text("def f():\n    return 42\n")
    convo.edit("src/pkg/core.py")
    send(service, convo)
    assert led.get(first.evidence_id).status in ("STALE", "SUPERSEDED")
    assert led.get_claim("file_hash|src/pkg/core.py").value != first.value


def test_redaction_and_no_giant_payloads(service, repo: Path) -> None:
    convo = Conversation(repo, "Deploy")
    big = "x" * 200_000
    convo.bash(
        "export API_TOKEN=supersecretvalue123 && ./deploy.sh",
        f"API_TOKEN=supersecretvalue123\n{big}",
        code=1,
    )
    rs = send(service, convo)
    store = rs.runtime.store
    rows = store.query("SELECT * FROM evidence")
    dump = " ".join(str(dict(r)) for r in rows)
    events = " ".join(str(dict(r)) for r in store.query("SELECT * FROM events"))
    assert "supersecretvalue123" not in dump and "supersecretvalue123" not in events
    assert max(len(str(dict(r))) for r in rows) < 8000
    assert big[:5000] not in dump and big[:5000] not in events


def test_restart_preserves_ledger_and_workspaces_isolated(
    make_service, repo: Path, tmp_path: Path
) -> None:
    svc = make_service()
    convo = Conversation(repo, "Fix it")
    convo.bash("pytest -q", PYTEST_PASS)
    send(svc, convo)
    close_all_stores()
    svc2 = make_service()
    rs = send(svc2, convo)
    assert _ledger(rs).get_claim("test_aggregate|pytest|pytest -q").value["ok"] is True
    other = tmp_path / "elsewhere"
    (other / ".git").mkdir(parents=True)
    rs_other = send(svc2, Conversation(other, "Fix it"))
    assert _ledger(rs_other).get_claim("test_aggregate|pytest|pytest -q") is None


def test_concurrent_ledger_writes(service, repo: Path) -> None:
    rs = send(service, Conversation(repo, "Fix it"))
    led = _ledger(rs)

    def work(i: int) -> None:
        for j in range(20):
            led.record(
                led.make(
                    None,
                    claim_type="probe",
                    subject=f"s{i}",
                    predicate="p",
                    value=j,
                    display="d",
                    source_kind="TOOL",
                    confidence=0.99,
                    source_event_id=f"ev{i}-{j}",
                )
            )

    threads = [threading.Thread(target=work, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert (
        led.store.query_one("SELECT COUNT(*) AS n FROM evidence WHERE claim_type='probe'")["n"]
        == 80
    )
    for i in range(4):
        assert led.get_claim(f"probe|s{i}").value == 19


def test_query_api(service, repo: Path) -> None:
    convo = Conversation(repo, "Fix it")
    convo.bash("pytest -q", PYTEST_FAIL, code=1)
    rs = send(service, convo)
    led = _ledger(rs)
    keys = ["tests_passing|latest", "test_aggregate|pytest|pytest -q", "missing|key"]
    got = led.query_exact(keys)
    assert set(got) == set(keys[:2])
    assert led.get_active(subject="latest")
    assert led.get_recent(3, ("test_aggregate",))[0].claim_type == "test_aggregate"
    assert led.get_active(task_id=rs.runtime.task_id)


def test_criterion_references_evidence(service, repo: Path) -> None:
    convo = Conversation(repo, "Fix it. Tests must pass.")
    convo.bash("pytest -q", PYTEST_PASS)
    rs = send(service, convo)
    crit = rs.runtime.task_state.state.acceptance_criteria[0]
    assert crit.evidence_ids
    rec = _ledger(rs).get(crit.evidence_ids[0])
    assert rec is not None and rec.claim_type == "test_aggregate"


def test_raw_output_can_leave_context_but_facts_remain(service, repo: Path) -> None:
    """Phase 3 gate: after compaction drops the raw logs, the facts are still known."""
    convo = Conversation(repo, "Fix the failing build and tests.")
    convo.bash("make", "src/a.c:3:1: error: expected ';'\nmake: *** Error 1", code=2)
    convo.bash("pytest -q", PYTEST_FAIL, code=1)
    send(service, convo)
    compacted = Conversation(
        repo,
        "This session is being continued from a previous conversation that ran out of context.",
    )
    rs = send(service, compacted)
    led = _ledger(rs)
    assert led.get_claim("build_passing|latest").value["ok"] is False
    assert led.get_claim("test_status|pytest|tests/test_core.py::test_f").value["state"] == "fail"
    assert "failing" in (rs.block or "")
