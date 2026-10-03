"""Feature 19: Tool Contract and Argument Validator (plan §7.12)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from headroom.intelligence.agent_state.contracts import (
    RuleResult,
    apply_safe_repair,
    valid_ref_name,
)
from headroom.intelligence.agent_state.events import AgentEvent, EventType
from headroom.intelligence.agent_state.families import normalize_tool_call
from headroom.intelligence.agent_state.paths import escapes_root, within

from .conftest import Conversation, hook, send


def _validate(rs, name: str, inp: dict, *, source: str = "history", which=None):
    rt = rs.runtime
    inv = normalize_tool_call(name, inp, default_cwd=rt.workspace.root)
    ev = AgentEvent(
        event_id=f"ev-{name}-{time.time_ns()}",
        workspace_id=rt.workspace.workspace_id,
        session_id=rt.session_key,
        event_type=EventType.TOOL_CALL_PROPOSED,
        transient={"invocation": inv, "input": inp, "call_id": "c"},
    )
    return rt.contracts.validate(ev, source=source, which=which)


@pytest.fixture
def rs(service, repo: Path):
    return send(service, Conversation(repo, "Fix the parser in src/pkg/core.py."))


def _rules(outcome) -> set[str]:
    return {f.rule for f in outcome.findings}


# ----------------------------------------------------------------- paths
def test_missing_input_path_blocks_with_suggestion(rs, repo: Path) -> None:
    out = _validate(rs, "Read", {"file_path": str(repo / "src/core.py")})
    assert out.result is RuleResult.BLOCK
    assert "src/pkg/core.py" in out.message  # did-you-mean from the project index


def test_existing_path_passes_fast(rs, repo: Path) -> None:
    times = []
    for _ in range(30):
        out = _validate(rs, "Read", {"file_path": str(repo / "src/pkg/core.py")})
        assert out.result is RuleResult.PASS
        times.append(out.latency_ms)
    times.sort()
    assert times[len(times) // 2] < 10.0  # plan §12.2: < 10 ms median


def test_bad_cwd_and_cd_target(rs, repo: Path) -> None:
    out = _validate(rs, "exec_command", {"cmd": ["ls"], "workdir": str(repo / "nope")})
    assert out.result is RuleResult.BLOCK and "cwd:exists" in _rules(out)
    out = _validate(rs, "Bash", {"command": "cd does/not/exist && ls"})
    assert out.result is RuleResult.BLOCK and "cwd:cd_target" in _rules(out)
    file_cwd = _validate(
        rs, "exec_command", {"cmd": ["ls"], "workdir": str(repo / "pyproject.toml")}
    )
    assert "cwd:isdir" in _rules(file_cwd)


def test_unknown_executable_warns_and_builtins_are_exempt(rs) -> None:
    out = _validate(rs, "Bash", {"command": "definitely-not-a-real-tool-xyz --go"})
    assert out.result is RuleResult.WARN and "exe:resolve" in _rules(out)
    assert all(not f.deterministic for f in out.findings)
    assert (
        _validate(rs, "Bash", {"command": "cd src && echo hi && export A=1"}).result
        is RuleResult.PASS
    )
    # The agent's own environment (hook-reported) wins over the proxy's PATH.
    agent_env = _validate(
        rs, "Bash", {"command": "mytool run"}, which={"mytool": "/opt/bin/mytool"}
    )
    assert "exe:resolve" not in _rules(agent_env)


def test_relative_executable_path_must_exist(rs) -> None:
    out = _validate(rs, "Bash", {"command": "./scripts/missing.sh"})
    assert out.result is RuleResult.BLOCK and "exe:path" in _rules(out)


@pytest.mark.skipif(sys.platform.startswith("win"), reason="POSIX host behavior")
def test_windows_only_commands_on_posix(rs) -> None:
    assert "platform:cmd" in _rules(_validate(rs, "Bash", {"command": "cmd /c dir"}))
    assert "platform:exe" in _rules(_validate(rs, "Bash", {"command": "tool.exe --x"}))
    assert "platform:drive" in _rules(_validate(rs, "Bash", {"command": "C:\\tools\\x.exe"}))


def test_windows_path_semantics() -> None:
    assert within("C:/Repo/src/a.py", "c:/repo") or os.name != "nt"
    assert escapes_root("../outside.txt", "/r/proj", cwd="/r/proj")
    assert not escapes_root("src/a.py", "/r/proj", cwd="/r/proj")
    inv = normalize_tool_call(
        "exec_command",
        {"cmd": ["powershell", "-Command", "Remove-Item C:\\temp\\x"], "workdir": "C:\\repo"},
    )
    assert inv.paths_deleted == ("C:\\temp\\x",)


# ------------------------------------------------------------------- git
@pytest.mark.parametrize(
    ("ref", "ok"),
    [
        ("main", True),
        ("feature/x-1", True),
        ("a..b", False),
        ("bad ref", False),
        ("x.lock", False),
        ("-bad", False),
        ("a/.b", False),
        ("x~1", False),
        ("refs/heads/ok", True),
    ],
)
def test_ref_format(ref: str, ok: bool) -> None:
    assert valid_ref_name(ref) is ok


def test_git_validation(rs, repo: Path, tmp_path: Path) -> None:
    out = _validate(rs, "Bash", {"command": "git checkout -b 'bad..name'"})
    assert "git:ref" in _rules(out) and out.result is RuleResult.BLOCK
    assert _validate(rs, "Bash", {"command": "git checkout HEAD~1"}).result is RuleResult.PASS
    outside = tmp_path / "plain"
    outside.mkdir()
    out = _validate(rs, "exec_command", {"cmd": ["git", "status"], "workdir": str(outside)})
    assert "git:repo" in _rules(out)


# -------------------------------------------------------- selectors/args
def test_test_selector_and_npm_script(rs, repo: Path) -> None:
    out = _validate(rs, "Bash", {"command": "python -m pytest tests/test_nope.py::test_x"})
    assert "test:selector" in _rules(out)
    assert (
        _validate(rs, "Bash", {"command": "python -m pytest tests/test_core.py::test_f"}).result
        is RuleResult.PASS
    )
    (repo / "package.json").write_text(json.dumps({"scripts": {"build": "tsc"}}))
    assert "build:npm_script" in _rules(_validate(rs, "Bash", {"command": "npm run lint"}))
    assert _validate(rs, "Bash", {"command": "npm run build"}).result in (
        RuleResult.PASS,
        RuleResult.WARN,
    )


def test_mutually_exclusive_args_from_schema(rs) -> None:
    rs.runtime.tool_schemas["Fetch"] = {
        "x-mutually-exclusive": [["url", "file"]],
        "oneOf": [{"required": ["url"]}, {"required": ["file"]}],
    }
    out = _validate(rs, "Fetch", {"url": "u", "file": "f"})
    assert out.result is RuleResult.BLOCK and "schema:exclusive" in _rules(out)
    assert _validate(rs, "Fetch", {"url": "u"}).result is RuleResult.PASS


def test_noop_edit_blocked(rs, repo: Path) -> None:
    out = _validate(
        rs,
        "Edit",
        {"file_path": str(repo / "src/pkg/core.py"), "old_string": "x", "new_string": "x"},
    )
    assert "args:edit_noop" in _rules(out)


def test_regex_and_glob_syntax_warns(rs) -> None:
    out = _validate(rs, "Grep", {"pattern": "foo(bar"})
    assert out.result is RuleResult.WARN and "pattern:regex" in _rules(out)


def test_unknown_tool_fails_open(rs) -> None:
    out = _validate(rs, "BrandNewTool", {"anything": 1})
    assert out.result is RuleResult.PASS and out.enforced == "allowed"


# ---------------------------------------------------------------- repair
def test_relative_path_is_repairable_and_repair_is_safe(rs, repo: Path) -> None:
    out = _validate(rs, "Read", {"file_path": "src/pkg/core.py"})
    assert out.result is RuleResult.REPAIRABLE
    repair = next(f.repair for f in out.findings if f.repair)
    assert repair == {"file_path": str(repo / "src/pkg/core.py")}
    assert (
        apply_safe_repair({"file_path": "src/pkg/core.py"}, repair, cwd=str(repo), root=str(repo))
        == repair
    )


@pytest.mark.parametrize(
    "repair",
    [
        {"file_path": "/some/other/file.py"},  # changes target
        {"command": "pytest"},  # changes command intent
        {"branch": "main"},  # changes ref
        {"selector": "tests/x.py"},  # changes test selector
        {"port": "8080"},
    ],
)
def test_forbidden_repairs_are_refused(repair, repo: Path) -> None:
    raw = {
        "file_path": "src/pkg/core.py",
        "command": "pytest -k a",
        "branch": "dev",
        "selector": "a",
        "port": "1",
    }
    assert apply_safe_repair(raw, repair, cwd=str(repo), root=str(repo)) is None


def test_hook_never_rewrites_arguments(service, repo: Path) -> None:
    send(service, Conversation(repo, "Fix the parser."))
    ans = hook(service, repo, "Read", {"file_path": "src/pkg/core.py"})
    assert ans["decision"] == "allow" and "updated_input" not in ans
    rt = next(iter(service._runtimes.values()))
    assert rt.capabilities.can_rewrite_safe_args is False


# ----------------------------------------------------------- learned rules
def test_learned_rule_needs_three_failures_and_is_narrow(service, repo: Path) -> None:
    sub = repo / "sub"
    sub.mkdir()
    convo = Conversation(repo, "Build the native module.")
    for _i in range(2):
        convo.call(
            "exec_command",
            {"cmd": ["cmake", "--build", "."], "workdir": str(sub)},
            "Exit code: 1\nCMake Error: not a CMake build directory (missing CMakeLists.txt)",
        )
    rs = send(service, convo)
    assert not rs.runtime.contracts.learned_rules()
    convo.call(
        "exec_command",
        {"cmd": ["cmake", "--build", "."], "workdir": str(sub)},
        "Exit code: 1\nCMake Error: not a CMake build directory (missing CMakeLists.txt)",
    )
    rs = send(service, convo)
    rules = rs.runtime.contracts.learned_rules()
    assert len(rules) == 1 and rules[0]["executable"] == "cmake"
    assert "MISSING_PROJECT_ROOT" in rules[0]["reason"]
    hit = _validate(rs, "exec_command", {"cmd": ["cmake", "--build", "."], "workdir": str(sub)})
    assert hit.result is RuleResult.WARN and any(
        f.rule.startswith("learned:") for f in hit.findings
    )
    # Narrow: the same command where the predicate does not hold is not flagged.
    (repo / "CMakeLists.txt").write_text("project(x)\n")
    elsewhere = _validate(
        rs, "exec_command", {"cmd": ["cmake", "--build", "."], "workdir": str(repo)}
    )
    assert not any(f.rule.startswith("learned:") for f in elsewhere.findings)


def test_learned_rule_invalidated_by_counterexample(service, repo: Path) -> None:
    convo = Conversation(repo, "Run the thing.")
    for _ in range(3):
        convo.bash("weirdtool --mode x", "weirdtool: error: config missing", code=1)
    rs = send(service, convo)
    assert rs.runtime.contracts.learned_rules()
    convo.bash("weirdtool --mode x", "done")
    rs = send(service, convo)
    assert all(
        r["disabled_reason"] == "counterexample" for r in rs.runtime.contracts.learned_rules()
    )


def test_learned_rules_do_not_leak_across_workspaces(service, repo: Path, tmp_path: Path) -> None:
    convo = Conversation(repo, "Run the thing.")
    for _ in range(3):
        convo.bash("weirdtool --mode x", "weirdtool: error: config missing", code=1)
    assert send(service, convo).runtime.contracts.learned_rules()
    other = tmp_path / "w2"
    (other / ".git").mkdir(parents=True)
    rs2 = send(service, Conversation(other, "Run the thing."))
    assert rs2.runtime.contracts.learned_rules() == []


# -------------------------------------------------------- jevk5 / failopen
def test_jevk5_timeout_fails_open(rs, monkeypatch) -> None:
    from headroom.intelligence.agent_state import advice

    class Slow:
        def choose(self, *a, **k):
            raise TimeoutError("deadline")

    rs.runtime.tool_schemas["custom_tool"] = {"description": "Formats a date"}
    monkeypatch.setattr(rs.runtime, "advisor", lambda: Slow())
    out = _validate(rs, "custom_tool", {"x": 1})
    assert out.result is RuleResult.PASS
    assert advice.classify(Slow(), None, "s", "q", {"A": "a", "B": "b"}) is None


def test_jevk5_alone_can_only_warn(rs, monkeypatch) -> None:
    class Says:
        def choose(self, *a, **k):
            from headroom.intelligence.models import AdvisoryScores, DecisionFamily

            return AdvisoryScores(
                DecisionFamily.TOOL_CONTRACT, {"A": 0.01, "B": 0.98, "C": 0.01}, 0.98, weight=0.4
            )

    rs.runtime.tool_schemas["custom_tool"] = {"description": "Formats a date"}
    monkeypatch.setattr(rs.runtime, "advisor", lambda: Says())
    out = _validate(rs, "custom_tool", {"rm": "-rf /"})
    assert out.result is RuleResult.WARN and all(not f.deterministic for f in out.findings)


# ------------------------------------------------------- capability matrix
def test_history_only_is_never_reported_blocked(rs, repo: Path) -> None:
    out = _validate(rs, "Read", {"file_path": str(repo / "missing.py")}, source="history")
    assert out.result is RuleResult.BLOCK and out.enforced == "observed"
    assert rs.runtime.capabilities.can_block_before_execution is False
    out = _validate(rs, "Read", {"file_path": str(repo / "missing.py")}, source="hook")
    assert out.enforced == "warned"  # no hook has been seen for this session yet


def test_claude_hook_blocks_after_capability_seen(service, repo: Path) -> None:
    send(service, Conversation(repo, "Fix the parser."))
    ans = hook(service, repo, "Read", {"file_path": str(repo / "missing.py")})
    assert ans["decision"] == "deny" and ans["enforced"] == "blocked"
    assert "does not exist" in ans["reason"]
    ok = hook(
        service, repo, "Read", {"file_path": str(repo / "src/pkg/core.py")}, tool_use_id="hk2"
    )
    assert ok["decision"] == "allow"


def test_protect_mode_off_means_warn_only(make_service, repo: Path) -> None:
    svc = make_service(env={"HEADROOM_TOOL_CONTRACT_MODE": "warn"})
    send(svc, Conversation(repo, "Fix the parser."))
    ans = hook(svc, repo, "Read", {"file_path": str(repo / "missing.py")})
    assert (
        ans["decision"] == "allow"
        and ans["enforced"] == "warned"
        and "does not exist" in ans["context"]
    )
    svc2 = make_service(env={"HEADROOM_TOOL_CONTRACT_MODE": "observe"})
    send(svc2, Conversation(repo, "Fix the parser."))
    assert (
        hook(svc2, repo, "Read", {"file_path": str(repo / "missing.py")})["enforced"] == "observed"
    )


def test_codex_block_is_requested_until_verified(service, repo: Path) -> None:
    from headroom.intelligence.agent_state.runtime import RequestState  # noqa: F401

    ans = hook(
        service,
        repo,
        "shell",
        {"command": ["cat", "missing.txt"], "workdir": str(repo / "nope")},
        agent="codex",
        tool_use_id="call_1",
        session="codex-thread-1",
    )
    assert ans["decision"] == "deny" and ans["enforced"] == "block_requested"
    rt = next(iter(service._runtimes.values()))
    assert rt.capabilities.can_block_before_execution is False
    # History shows the call did not run (host returned the hook's reason) -> verified.
    items = [
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "read it"}]},
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "shell",
            "arguments": json.dumps(
                {"command": ["cat", "missing.txt"], "workdir": str(repo / "nope")}
            ),
        },
        {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": "Headroom blocked this shell call: working directory does not exist",
        },
    ]
    service.begin_responses(
        {
            "input": items,
            "tools": [{"type": "function", "name": "shell"}],
            "prompt_cache_key": "codex-thread-1",
        },
        client="codex",
        cwd=str(repo),
    )
    assert rt.capabilities.can_block_before_execution is True


def test_block_not_honored_downgrades_capability(service, repo: Path) -> None:
    send(service, Conversation(repo, "Fix the parser."))
    ans = hook(
        service, repo, "Read", {"file_path": str(repo / "missing.py")}, tool_use_id="toolu_0001"
    )
    assert ans["enforced"] == "blocked"
    convo = Conversation(repo, "Fix the parser.")
    convo.call(
        "Read", {"file_path": str(repo / "missing.py")}, "file contents that should not exist"
    )
    send(service, convo)
    rt = next(iter(service._runtimes.values()))
    assert rt.capabilities.can_block_before_execution is False and rt.capabilities.downgraded


# ------------------------------------------------- real hook client process
class _Handler(BaseHTTPRequestHandler):
    service = None

    def do_POST(self) -> None:  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        answer = type(self).service.on_pretool_hook(body)
        data = json.dumps(answer).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a) -> None:
        pass


def _client() -> str:
    import headroom.intelligence.agent_state as pkg

    return str(Path(pkg.__file__).parent / "hook_client.py")


def test_hook_client_end_to_end(service, repo: Path) -> None:
    send(service, Conversation(repo, "Fix the parser. Do not modify tests/test_core.py."))
    _Handler.service = service
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        port = str(server.server_address[1])
        payload = {
            "session_id": "11111111-2222-3333-4444-555555555555",
            "cwd": str(repo),
            "hook_event_name": "PreToolUse",
            "tool_name": "Edit",
            "tool_input": {
                "file_path": str(repo / "tests/test_core.py"),
                "old_string": "a",
                "new_string": "b",
            },
            "tool_use_id": "toolu_x",
        }
        proc = subprocess.run(
            [sys.executable, _client(), "--agent", "claude", "--port", port],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert proc.returncode == 2 and "USER_EXCLUDED" in proc.stderr
        payload["tool_input"] = {
            "file_path": str(repo / "src/pkg/core.py"),
            "old_string": "a",
            "new_string": "b",
        }
        payload["tool_use_id"] = "toolu_y"
        proc = subprocess.run(
            [sys.executable, _client(), "--agent", "claude", "--port", port],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert proc.returncode == 0
    finally:
        server.shutdown()


def test_hook_client_fails_open_without_proxy(repo: Path) -> None:
    payload = {
        "session_id": "s",
        "cwd": str(repo),
        "tool_name": "Bash",
        "tool_input": {"command": "rm -rf /"},
    }
    proc = subprocess.run(
        [sys.executable, _client(), "--agent", "claude", "--port", "1"],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0 and proc.stdout == "" and proc.stderr == ""
    proc = subprocess.run(
        [sys.executable, _client(), "--agent", "codex", "--port", "1"],
        input="not json",
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0


def test_hook_client_imports_no_headroom() -> None:
    src = Path(_client()).read_text()
    assert "import headroom" not in src and "from headroom" not in src
