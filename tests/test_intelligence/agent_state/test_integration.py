"""Phase 8: the agent-state layer in real proxy traffic, wrap, MCP and the CLI."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from headroom.intelligence.agent_state.config import (
    AgentStateConfig,
    disabled_agent_state_config,
)
from headroom.proxy.models import ProxyConfig
from headroom.proxy.server import create_app

from .conftest import SESSION, Conversation


def _app(agent_state=None, *, optimize: bool = True) -> TestClient:
    config = ProxyConfig(
        optimize=optimize,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        log_requests=False,
        ccr_inject_tool=False,
        ccr_handle_responses=False,
        ccr_context_tracking=False,
        image_optimize=False,
        agent_state=agent_state,
    )
    return TestClient(create_app(config), base_url="http://127.0.0.1", client=("127.0.0.1", 12345))


def _anthropic_reply(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet",
            "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 2},
        },
    )


def _claude_body(repo: Path, messages: list[dict]) -> dict:
    return {
        "model": "claude-sonnet-4-5",
        "max_tokens": 64,
        "system": f"You are Claude Code.\nWorking directory: {repo}\n",
        "metadata": {"user_id": f"user_x_account_y_session_{SESSION}"},
        "tools": [
            {
                "name": "Bash",
                "description": "run",
                "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}},
            }
        ],
        "messages": messages,
    }


CLAUDE = {
    "x-api-key": "test-key",
    "anthropic-version": "2023-06-01",
    "user-agent": "claude-cli/2.1.0 (external, cli)",
}


def _run_anthropic(client: TestClient, repo: Path, turns: list[list[dict]]) -> list[dict]:
    sent: list[dict] = []

    def provider(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return _anthropic_reply(request)

    proxy = client.app.state.proxy
    proxy.http_client = httpx.AsyncClient(transport=httpx.MockTransport(provider))
    for messages in turns:
        r = client.post(
            "/v1/messages", headers=CLAUDE, json=_claude_body(repo, copy.deepcopy(messages))
        )
        assert r.status_code == 200, r.text
    return sent


def test_claude_code_turns_get_state_and_history_is_byte_stable(repo: Path) -> None:
    turn1 = [
        {
            "role": "user",
            "content": "Fix the retry bug in src/pkg/core.py. Do not modify src/other/. All tests must pass.",
        }
    ]
    turn2 = [
        *turn1,
        {"role": "assistant", "content": [{"type": "text", "text": "On it."}]},
        {"role": "user", "content": "continue"},
    ]
    turn3 = [
        *turn2,
        {"role": "assistant", "content": [{"type": "text", "text": "Still going."}]},
        {"role": "user", "content": "keep going"},
    ]
    with _app() as client:
        sent = _run_anthropic(client, repo, [turn1, turn2, turn3])
    first = sent[0]["messages"][0]["content"]
    assert isinstance(first, list) and first[-1]["text"].startswith("<headroom_agent_state")
    assert "Do not modify src/other/." in first[-1]["text"]
    # Replay: the earlier insertion is re-applied byte-identically; nothing new.
    # Cache-control markers are the proxy's own moving breakpoint; content is compared.
    from headroom.intelligence.agent_state.injection import _strip_cache_control as strip

    assert strip(sent[1]["messages"][0]) == strip(sent[0]["messages"][0])
    assert strip(sent[2]["messages"][:3]) == strip(sent[1]["messages"][:3])
    assert sum("<headroom_agent_state" in json.dumps(m) for m in sent[2]["messages"]) == 1
    # The client's own content is otherwise untouched.
    assert strip(sent[2]["messages"][1:]) == turn3[1:]


def test_optimize_off_ingests_but_never_mutates(repo: Path) -> None:
    turn1 = [{"role": "user", "content": "Fix src/pkg/core.py. Do not modify src/other/."}]
    with _app(optimize=False) as client:
        sent = _run_anthropic(client, repo, [turn1])
        status = client.get("/v1/agent-state/status").json()
    assert sent[0]["messages"] == turn1
    assert status["sessions"] and status["sessions"][-1]["task"]


def test_feature_off_is_byte_identical(repo: Path) -> None:
    turn1 = [{"role": "user", "content": "Fix src/pkg/core.py. Do not modify src/other/."}]
    with _app(disabled_agent_state_config()) as client:
        assert client.app.state.proxy.agent_state is None
        sent = _run_anthropic(client, repo, [turn1])
    assert sent[0]["messages"] == turn1


def test_non_agent_client_is_untouched(repo: Path) -> None:
    turn1 = [{"role": "user", "content": "Fix src/pkg/core.py. Do not modify src/other/."}]
    sent: list[dict] = []
    with _app() as client:
        client.app.state.proxy.http_client = httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda req: (sent.append(json.loads(req.content)), _anthropic_reply(req))[1]
            )
        )
        body = _claude_body(repo, turn1)
        body.pop("metadata")
        r = client.post(
            "/v1/messages",
            headers={
                "x-api-key": "k",
                "anthropic-version": "2023-06-01",
                "user-agent": "my-app/1.0",
            },
            json=body,
        )
        assert r.status_code == 200
    assert sent[0]["messages"] == turn1


def test_hook_and_status_endpoints(repo: Path) -> None:
    turn1 = [{"role": "user", "content": "Fix src/pkg/core.py. Do not modify src/other/."}]
    with _app() as client:
        _run_anthropic(client, repo, [turn1])
        hook = {
            "agent": "claude",
            "hook": {
                "session_id": SESSION,
                "cwd": str(repo),
                "hook_event_name": "PreToolUse",
                "tool_name": "Edit",
                "tool_input": {
                    "file_path": str(repo / "src/other/cost.py"),
                    "old_string": "R",
                    "new_string": "S",
                },
                "tool_use_id": "toolu_h1",
            },
        }
        answer = client.post("/v1/agent-state/hook", json=hook).json()
        assert answer["decision"] == "deny" and "USER_EXCLUDED" in answer["reason"]
        assert (
            client.post("/v1/agent-state/hook", content=b"not json").json()["decision"] == "allow"
        )
        status = client.get("/v1/agent-state/status").json()
        assert status["enabled"] and status["metrics"]["counters"]["hook_blocked"] == 1
        caps = status["sessions"][-1]["capabilities"]
        assert caps["can_block_before_execution"] is True and caps["can_rewrite_safe_args"] is False
        intel = client.get("/v1/intelligence/status").json()
        assert "enabled" in intel


def test_bad_override_fails_at_startup(monkeypatch) -> None:
    monkeypatch.setenv("HEADROOM_SCOPE_MODE", "explode")
    from headroom.intelligence.agent_state.config import AgentStateConfigError

    with pytest.raises(AgentStateConfigError):
        create_app(ProxyConfig(log_requests=False, cost_tracking_enabled=False))


def _responses_reply(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "resp_1",
            "object": "response",
            "status": "completed",
            "model": "gpt-5-codex",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "ok"}],
                }
            ],
            "usage": {"input_tokens": 5, "output_tokens": 1, "total_tokens": 6},
        },
    )


def test_codex_responses_get_state_after_live_turn(repo: Path) -> None:
    items = [
        {
            "type": "message",
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": f"<environment_context>\n  <cwd>{repo}</cwd>\n</environment_context>",
                }
            ],
        },
        {
            "type": "message",
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "Fix src/pkg/core.py. Never edit tests/test_core.py.",
                }
            ],
        },
    ]
    sent: list[dict] = []
    with _app() as client:
        client.app.state.proxy.http_client = httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda req: (sent.append(json.loads(req.content)), _responses_reply(req))[1]
            )
        )
        body = {
            "model": "gpt-5-codex",
            "input": items,
            "tools": [{"type": "function", "name": "shell", "parameters": {"type": "object"}}],
            "prompt_cache_key": "codex-thread-9",
            "stream": False,
        }
        r = client.post(
            "/v1/responses",
            headers={
                "authorization": "Bearer k",
                "originator": "codex_cli_rs",
                "user-agent": "codex_cli_rs/0.50",
            },
            json=body,
        )
        assert r.status_code == 200, r.text
        later = [
            *items,
            {
                "type": "function_call",
                "call_id": "c1",
                "name": "shell",
                "arguments": json.dumps({"command": ["git", "status"], "workdir": str(repo)}),
            },
            {
                "type": "function_call_output",
                "call_id": "c1",
                "output": "Exit code: 0\nOutput:\nclean",
            },
        ]
        body2 = {**body, "input": later}
        r = client.post(
            "/v1/responses",
            headers={
                "authorization": "Bearer k",
                "originator": "codex_cli_rs",
                "user-agent": "codex_cli_rs/0.50",
            },
            json=body2,
        )
        assert r.status_code == 200
    first = sent[0]["input"]
    assert len(first) == 3 and "<headroom_agent_state" in first[2]["content"][0]["text"]
    # Turn 2 replays the insertion at the same position; history bytes are stable.
    assert sent[1]["input"][:3] == first


def test_codex_ws_frames_are_observed_but_never_changed_with_optimize_off(repo: Path) -> None:
    """optimize=False WS passthrough: the frame's bytes go upstream untouched, but
    Phase 2 still ingests it (hooks, evidence and workflow learning keep working)."""
    import asyncio
    from types import SimpleNamespace

    from headroom.intelligence.agent_state.runtime import AgentStateService
    from headroom.proxy.handlers.openai import OpenAIHandlerMixin

    payload = {
        "model": "gpt-5-codex",
        "prompt_cache_key": "codex-ws-thread",
        "tools": [{"type": "function", "name": "shell", "parameters": {"type": "object"}}],
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": f"<environment_context>\n  <cwd>{repo}</cwd>\n</environment_context>",
                    }
                ],
            },
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Fix src/pkg/core.py."}],
            },
            {
                "type": "function_call",
                "call_id": "w1",
                "name": "shell",
                "arguments": json.dumps({"command": ["git", "status"], "workdir": str(repo)}),
            },
            {"type": "function_call_output", "call_id": "w1", "output": "Exit code: 0\nOutput:\n"},
        ],
    }
    before = json.dumps(payload, sort_keys=True)
    service = AgentStateService(AgentStateConfig.from_env({}))
    observe = OpenAIHandlerMixin._observe_agent_state_ws_payload
    asyncio.run(observe(SimpleNamespace(agent_state=service), payload, "codex"))
    assert json.dumps(payload, sort_keys=True) == before
    assert service.metrics.counters["evidence_records_created"] >= 1
    # Fail-open no-ops: no service, a frame without an input list.
    asyncio.run(observe(SimpleNamespace(agent_state=None), payload, "codex"))
    asyncio.run(observe(SimpleNamespace(agent_state=service), {"type": "session.update"}, "codex"))
    service.shutdown()


# ----------------------------------------------------------------- wrap
def test_wrap_hook_install_and_remove(tmp_path: Path) -> None:
    from headroom.intelligence.agent_state import install

    settings = tmp_path / ".claude" / "settings.local.json"
    settings.parent.mkdir()
    settings.write_text(
        json.dumps(
            {
                "hooks": {
                    "PreToolUse": [
                        {"matcher": "Bash", "hooks": [{"type": "command", "command": "user-hook"}]}
                    ]
                },
                "env": {"A": "1"},
            }
        )
    )
    assert install.ensure_claude_hook(settings, 8787)
    assert not install.ensure_claude_hook(settings, 8787)  # idempotent
    assert install.ensure_claude_hook(settings, 9999)  # port change rewrites in place
    data = json.loads(settings.read_text())
    entries = data["hooks"]["PreToolUse"]
    assert len(entries) == 2 and entries[0]["hooks"][0]["command"] == "user-hook"
    ours = entries[1]
    assert (
        ours["matcher"] == install.CLAUDE_MATCHER and "--port 9999" in ours["hooks"][0]["command"]
    )
    assert (
        "hook_client.py" in ours["hooks"][0]["command"]
        and install.MARKER in ours["hooks"][0]["command"]
    )
    assert install.remove_claude_hook(settings)
    assert json.loads(settings.read_text()) == {
        "hooks": {
            "PreToolUse": [
                {"matcher": "Bash", "hooks": [{"type": "command", "command": "user-hook"}]}
            ]
        },
        "env": {"A": "1"},
    }
    hooks = tmp_path / "codex" / "hooks.json"
    assert install.ensure_codex_hook(hooks, 8787)
    assert (
        "--agent codex"
        in json.loads(hooks.read_text())["hooks"]["PreToolUse"][0]["hooks"][0]["command"]
    )
    assert install.remove_codex_hook(hooks) and json.loads(hooks.read_text()) == {}


def test_hooks_wanted_follows_flags() -> None:
    from headroom.intelligence.agent_state.install import hooks_wanted

    assert hooks_wanted({})
    assert hooks_wanted({"HEADROOM_SCOPE_FIREWALL": "off"})
    assert not hooks_wanted({"HEADROOM_SCOPE_FIREWALL": "off", "HEADROOM_TOOL_CONTRACTS": "off"})


def test_codex_launch_enables_hooks(monkeypatch, tmp_path: Path) -> None:
    from headroom.cli import wrap

    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex-home"))
    (tmp_path / "codex-home").mkdir()
    args, env, _ = wrap._codex_session_launch_settings(
        port=8787, codex_args=(), environ={"CODEX_HOME": str(tmp_path / "codex-home")}
    )
    assert "features.hooks=true" in args
    assert (tmp_path / "codex-home" / "hooks.json").exists()


# ------------------------------------------------------------------ MCP
def test_mcp_workflow_tool_listing_and_run(repo: Path, monkeypatch, service) -> None:
    from headroom.intelligence.agent_state import mcp

    from .conftest import send

    send(service, Conversation(repo, "Fix src/pkg/core.py."))
    monkeypatch.chdir(repo)
    macros = mcp.eligible()
    spec = mcp.tool_spec(macros)
    assert spec is not None and spec["name"] == "headroom_workflow"
    assert "show_task_owned_diff" in spec["inputSchema"]["properties"]["macro"]["enum"]
    assert len(spec["description"]) <= 1400
    out = mcp.run({"macro": "show_task_owned_diff"})
    assert out.startswith("macro: show_task_owned_diff") and "status: success" in out
    assert mcp.run({"macro": "nope"}).startswith("error:")
    assert mcp.tool_spec([]) is None


def test_mcp_server_lists_tool_only_when_eligible(repo: Path, tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("mcp")
    import asyncio

    from headroom.ccr.mcp_server import HeadroomMCPServer

    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.chdir(empty)  # not a project: no applicable macros
    server = HeadroomMCPServer(check_proxy=False)
    assert asyncio.run(server._workflow_tool_spec()) is None
    monkeypatch.chdir(repo)
    server2 = HeadroomMCPServer(check_proxy=False)
    spec = asyncio.run(server2._workflow_tool_spec())
    assert spec is not None and spec["name"] == "headroom_workflow"


# ------------------------------------------------------------------ CLI
def test_cli_diagnostics(repo: Path, service) -> None:
    from click.testing import CliRunner

    from headroom.cli.main import main

    from .conftest import send

    convo = Conversation(repo, "Fix src/pkg/core.py. Do not modify src/other/. Tests must pass.")
    convo.edit("src/pkg/core.py")
    convo.bash("pytest -q", "1 failed in 0.1s", code=1)
    send(service, convo)
    runner = CliRunner()
    for cmd in ("state", "evidence", "contracts", "scope", "verify", "workflows"):
        res = runner.invoke(main, ["intelligence", cmd, "--project", str(repo), "--json"])
        assert res.exit_code == 0, (cmd, res.output)
        data = json.loads(res.output)
        assert data["project"] == str(repo.resolve())
    state = json.loads(
        runner.invoke(main, ["intelligence", "state", "--project", str(repo), "--json"]).output
    )
    assert state["task"]["status"] in ("ACTIVE", "BLOCKED")
    text = runner.invoke(main, ["intelligence", "scope", "--project", str(repo)]).output
    assert "excluded: src/other/" in text
    ev = runner.invoke(
        main,
        ["intelligence", "evidence", "--project", str(repo), "--claim", "tests_passing|latest"],
    ).output
    assert "tests_passing|latest" in ev


def test_config_survives_worker_handoff() -> None:
    from headroom.proxy.server import _proxy_config_payload

    cfg = AgentStateConfig.from_env(
        {"HEADROOM_SCOPE_MODE": "warn", "HEADROOM_WORKFLOW_MACROS": "off"}
    )
    payload = _proxy_config_payload(ProxyConfig(agent_state=cfg))
    restored = AgentStateConfig.from_dict(payload["agent_state"])
    assert restored.scope_mode.value == "warn" and restored.workflows.value == "off"
