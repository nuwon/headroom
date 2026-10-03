"""Request-path wiring of the intelligence layer.

Covers the normal enablement routes (env / CLI / savings profile / wrap), the
Codex Responses adapter (pinned routing inputs, delta splice), the pipeline
integration and the proxy's status endpoint.
"""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest

from headroom.intelligence.config import IntelligenceConfig
from headroom.intelligence.responses import (
    ResponsesIntelligence,
    splice_tool_outputs,
)
from headroom.intelligence.runtime import IntelligenceRuntime


@pytest.fixture(autouse=True)
def _isolated_intelligence_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("HEADROOM_INTELLIGENCE_DIR", str(tmp_path / "intel"))
    monkeypatch.setenv("HEADROOM_JEVK5", "off")
    yield
    # A full-posture proxy flips process-wide switches; restore the defaults.
    from headroom.ccr.tool_injection import set_ccr_search_enabled
    from headroom.intelligence.runtime import install_runtime
    from headroom.proxy.interceptors import enable_rich_interception

    set_ccr_search_enabled(False)
    enable_rich_interception(False)
    install_runtime(None)


def _runtime(level: str = "full", **env: str) -> IntelligenceRuntime:
    cfg = IntelligenceConfig.from_env(
        {"HEADROOM_INTELLIGENCE": level, "HEADROOM_JEVK5": "off", **env}
    )
    return IntelligenceRuntime(cfg, workspace_roots=())


class _Counter:
    def count_text(self, text: str) -> int:
        return max(1, len(text) // 4) if text else 0

    def count_messages(self, messages) -> int:
        return sum(self.count_text(json.dumps(m, default=str)) for m in messages)


def _inventory(n: int = 80, *, bump: int = -1) -> str:
    rows = [
        {
            "id": i,
            "service": f"svc-{i:03d}",
            "status": "error" if i in (7, 41) else "ok",
            "latency_ms": 120 + i + (500 if i == bump else 0),
            "region": "us-east-1" if i % 2 else "eu-west-1",
        }
        for i in range(n)
    ]
    return json.dumps(rows, indent=1)


def _user(text: str) -> dict:
    return {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}


def _call(call_id: str, cmd: str) -> dict:
    return {
        "type": "function_call",
        "name": "exec_command",
        "call_id": call_id,
        "arguments": json.dumps({"cmd": cmd}),
    }


def _output(call_id: str, text: str) -> dict:
    return {"type": "function_call_output", "call_id": call_id, "output": text}


# --------------------------------------------------------------- splice helper
def test_splice_preserves_shapes_and_skips_ambiguous_call_ids():
    items = [
        _output("a", "old a"),
        {
            "type": "function_call_output",
            "call_id": "b",
            "output": [{"type": "input_text", "text": "old b"}],
        },
        _output("dup", "x"),
        _output("dup", "y"),
        {"type": "function_call_output", "call_id": "img", "output": [{"type": "input_image"}]},
    ]
    out = splice_tool_outputs(items, {"a": "new a", "b": "new b", "dup": "z", "img": "nope"})
    assert out is not None
    assert out[0]["output"] == "new a"
    assert out[1]["output"] == [{"type": "input_text", "text": "new b"}]
    assert out[2] is items[2] and out[3] is items[3]  # ambiguous: untouched
    assert out[4] is items[4]  # non-text output: untouched
    assert items[0]["output"] == "old a"  # input never mutated
    assert splice_tool_outputs(items, {"missing": "t"}) is None


# ------------------------------------------------------------------ unit pins
def test_pins_are_first_sighting_and_survive_query_changes():
    runtime = _runtime("safe")
    intel = ResponsesIntelligence(runtime)
    turn1 = intel.begin(
        [
            _user("find the failing services in the inventory"),
            _call("c1", "inv"),
            _output("c1", "x"),
        ],
        model="gpt-5",
    )
    turn2 = intel.begin([_user("now review the billing latency")], model="gpt-5")
    assert turn1 is not None and turn2 is not None
    assert "failing" in turn1.query and "billing" in turn2.query
    first = intel.pin_for("payload text", tool_name="exec_command", turn=turn1)
    again = intel.pin_for("payload text", tool_name="exec_command", turn=turn2)
    assert again == first and again.context == turn1.query
    assert first.task is turn1.task  # review features on in "safe"
    fresh = intel.pin_for("other text", tool_name="exec_command", turn=turn2)
    assert fresh.context == turn2.query


def test_begin_returns_none_when_everything_is_off():
    cfg = IntelligenceConfig.from_env({"HEADROOM_INTELLIGENCE": "off", "HEADROOM_JEVK5": "off"})
    intel = ResponsesIntelligence(IntelligenceRuntime(cfg, workspace_roots=()))
    assert intel.begin([_user("hello")], model="gpt-5") is None


# ------------------------------------------------- Codex Responses end-to-end
def _responses_handler(runtime: IntelligenceRuntime):
    from headroom.intelligence.admission import IntelligencePrepTransform
    from headroom.proxy.handlers.openai import OpenAIHandlerMixin
    from headroom.transforms.content_router import ContentRouter, ContentRouterConfig

    router = ContentRouter(ContentRouterConfig(intelligence=runtime))
    prep = IntelligencePrepTransform(runtime.config, feedback=runtime.learner)
    handler = OpenAIHandlerMixin()
    handler.openai_pipeline = SimpleNamespace(transforms=[prep, router])
    handler.openai_provider = SimpleNamespace(get_token_counter=lambda _m: _Counter())
    handler.intelligence = runtime
    return handler


def _compress(handler, items):
    payload = {"model": "gpt-5", "input": copy.deepcopy(items)}
    out, modified, saved, transforms, *_ = handler._compress_openai_responses_payload(
        payload, model="gpt-5", request_id="t"
    )
    return out, modified, saved, transforms


def test_codex_history_bytes_are_stable_across_turns_with_new_queries():
    runtime = _runtime("safe")
    handler = _responses_handler(runtime)
    turn1 = [
        _user("which services in the inventory are failing?"),
        _call("c1", "curl -s http://inventory/api/services"),
        _output("c1", _inventory()),
    ]
    out1, *_ = _compress(handler, turn1)
    turn2 = [
        *turn1,
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "svc-007 and svc-041."}],
        },
        _user("ok, now explain the eu-west-1 latency distribution for svc-060"),
    ]
    out2, *_ = _compress(handler, turn2)
    # The historical tool output must forward identical bytes on every turn
    # even though the task query changed (prompt-cache stability).
    assert out2["input"][2] == out1["input"][2]
    assert handler._responses_intel.pin_count() >= 1
    assert runtime.metrics.snapshot()["requests"] == 2


def test_codex_repeated_command_output_becomes_delta():
    runtime = _runtime("full")
    handler = _responses_handler(runtime)
    items = [
        _user("watch the service inventory and tell me what changes"),
        _call("c1", "kubectl get svc -o json"),
        _output("c1", _inventory(120)),
        _call("c2", "kubectl get svc -o json"),
        _output("c2", _inventory(120, bump=33)),
    ]
    out, modified, saved, transforms = _compress(handler, items)
    second = out["input"][4]["output"]
    assert modified
    assert second.startswith("headroom delta") or "delta" in second[:200]
    assert "svc-033" in second  # the changed row survives verbatim
    assert any("intelligence" in t or "delta" in t for t in transforms)
    assert saved > 0
    # Pinned: replaying the same request reproduces the same bytes.
    out_again, *_ = _compress(handler, items)
    assert out_again["input"][4] == out["input"][4]


def test_codex_path_untouched_when_runtime_absent():
    from headroom.proxy.handlers.openai import OpenAIHandlerMixin
    from headroom.transforms.content_router import ContentRouter, ContentRouterConfig

    handler = OpenAIHandlerMixin()
    handler.openai_pipeline = SimpleNamespace(transforms=[ContentRouter(ContentRouterConfig())])
    handler.openai_provider = SimpleNamespace(get_token_counter=lambda _m: _Counter())
    assert handler._openai_responses_intelligence({"input": []}, model="m", request_id="r") is None


# --------------------------------------------------------------- pipeline path
def test_pipeline_prepares_task_context_and_query():
    from headroom.config import HeadroomConfig
    from headroom.transforms.pipeline import TransformPipeline

    runtime = _runtime("safe")
    seen: dict = {}

    class _Probe:
        name = "probe"

        def should_apply(self, messages, tokenizer, **kwargs):
            return True

        def apply(self, messages, tokenizer, **kwargs):
            from headroom.config import TransformResult

            seen.update(kwargs)
            return TransformResult(
                messages=messages, tokens_before=0, tokens_after=0, transforms_applied=[]
            )

    pipeline = TransformPipeline(HeadroomConfig(), transforms=[_Probe()], intelligence=runtime)
    messages = [{"role": "user", "content": "fix test_parse_header in src/headroom/http.py"}]
    pipeline.apply(messages, model="gpt-4o", model_limit=128_000)
    task = seen.get("task_context")
    assert task is not None
    assert "test_parse_header" in seen.get("context", "")
    assert any(p.endswith("http.py") for p in task.file_paths)


# ------------------------------------------------------------- enablement routes
def test_proxy_status_endpoint_and_health_posture(monkeypatch):
    from fastapi.testclient import TestClient

    from headroom.proxy.models import ProxyConfig
    from headroom.proxy.server import create_app

    config = ProxyConfig(
        optimize=True,
        image_optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        log_requests=False,
        intelligence=IntelligenceConfig.from_env(
            {"HEADROOM_INTELLIGENCE": "full", "HEADROOM_JEVK5": "off"}
        ),
    )
    app = create_app(config)
    proxy = app.state.proxy
    names = [type(t).__name__ for t in proxy.anthropic_pipeline.transforms]
    assert "IntelligencePrepTransform" in names and "ContentRouter" in names
    client = TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 12345))
    status = client.get("/v1/intelligence/status").json()
    assert status["config"]["level"] == "full"
    health = client.get("/health").json()
    assert health["config"]["intelligence"] == "full"


def test_wrap_intelligence_option_exports_env(monkeypatch):
    import click
    from click.testing import CliRunner

    from headroom.cli.wrap import _intelligence_options

    monkeypatch.delenv("HEADROOM_INTELLIGENCE", raising=False)
    monkeypatch.delenv("HEADROOM_JEVK5", raising=False)
    captured: dict = {}

    @click.command()
    @_intelligence_options
    def cmd() -> None:
        import os

        captured["posture"] = os.environ.get("HEADROOM_INTELLIGENCE")
        captured["jevk5"] = os.environ.get("HEADROOM_JEVK5")

    result = CliRunner().invoke(cmd, ["--intelligence", "FULL", "--jevk5", "auto"])
    assert result.exit_code == 0, result.output
    assert captured == {"posture": "full", "jevk5": "auto"}


def test_wrap_claude_and_codex_accept_intelligence_flags():
    from click.testing import CliRunner

    from headroom.cli.wrap import wrap

    for sub in ("claude", "codex"):
        result = CliRunner().invoke(wrap, [sub, "--help"])
        assert result.exit_code == 0
        assert "--intelligence" in result.output and "--jevk5" in result.output


def test_wrap_restarts_proxy_on_posture_mismatch(monkeypatch):
    from headroom.cli import wrap as wrap_mod

    monkeypatch.setenv("HEADROOM_SAVINGS_PROFILE", "coding")
    monkeypatch.setenv("HEADROOM_INTELLIGENCE", "on")  # alias of "safe"
    agent = "claude"
    base = {"savings_profile": "coding"}
    same = wrap_mod._agent_savings_config_mismatches({**base, "intelligence": "safe"}, agent)
    other = wrap_mod._agent_savings_config_mismatches({**base, "intelligence": "off"}, agent)
    assert "intelligence" not in same
    assert "intelligence" in other


def test_coding_profile_seeds_safe_posture_without_overriding_user():
    from headroom.agent_savings import apply_agent_savings_env_defaults

    env: dict[str, str] = {}
    apply_agent_savings_env_defaults(env, "coding")
    assert env["HEADROOM_INTELLIGENCE"] == "safe"
    env = {"HEADROOM_INTELLIGENCE": "off"}
    apply_agent_savings_env_defaults(env, "coding")
    assert env["HEADROOM_INTELLIGENCE"] == "off"


def _app_config(intelligence):
    from headroom.proxy.models import ProxyConfig

    return ProxyConfig(
        optimize=True,
        image_optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        log_requests=False,
        intelligence=intelligence,
    )


def test_proxy_without_intelligence_resets_process_switches():
    from headroom.ccr import tool_injection
    from headroom.proxy.interceptors import astgrep
    from headroom.proxy.server import create_app

    full = IntelligenceConfig.from_env({"HEADROOM_INTELLIGENCE": "full", "HEADROOM_JEVK5": "off"})
    create_app(_app_config(full))
    assert tool_injection._SEARCH_ENABLED and astgrep._RICH
    create_app(_app_config(None))
    assert not tool_injection._SEARCH_ENABLED and not astgrep._RICH


# ------------------------------------------------------- turn-routing adapters
@pytest.fixture
def installed():
    from headroom.intelligence.runtime import install_runtime

    def _install(level: str = "safe", **env: str) -> IntelligenceRuntime:
        rt = _runtime(level, **env)
        install_runtime(rt)
        return rt

    yield _install
    install_runtime(None)


def test_verbosity_never_terse_when_user_asks_why(installed):
    from headroom.intelligence.turn_routing import adjust_verbosity

    installed("safe")
    body = {
        "messages": [{"role": "user", "content": "why does the retry loop never exit? explain"}]
    }
    level, labels = adjust_verbosity(body, "anthropic", 4)
    assert level == 2 and labels == ["intel:verbosity:4->2"]
    level, labels = adjust_verbosity(body, "anthropic", 1)
    assert level == 1 and labels == []


def test_verbosity_untouched_without_runtime():
    from headroom.intelligence.turn_routing import adjust_verbosity

    body = {"messages": [{"role": "user", "content": "explain why"}]}
    assert adjust_verbosity(body, "anthropic", 4) == (4, [])


def test_effort_routing_is_opt_in_and_never_injects(installed):
    from headroom.intelligence.turn_routing import route_effort

    def mechanical_turn(effort: str | None) -> dict:
        body = {
            "messages": [
                {"role": "user", "content": "run the formatter"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "bash", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "c1", "content": "formatted 3 files"},
            ]
        }
        if effort:
            body["reasoning_effort"] = effort
        return body

    installed("full")  # effort routing is in no posture
    body = mechanical_turn("medium")
    assert route_effort(body, "openai_chat", "conv") == []
    assert body["reasoning_effort"] == "medium"

    installed("safe", HEADROOM_EFFORT_ROUTING="1")
    no_field = mechanical_turn(None)
    assert route_effort(no_field, "openai_chat", "conv") == []
    assert "reasoning_effort" not in no_field  # never injected
    # Lowering needs two consecutive agreeing mechanical turns (hysteresis).
    first = mechanical_turn("medium")
    assert route_effort(first, "openai_chat", "conv") == []
    second = mechanical_turn("medium")
    labels = route_effort(second, "openai_chat", "conv")
    assert second["reasoning_effort"] == "low" and labels


def test_effort_router_raises_back_to_client_level_immediately():
    from headroom.proxy.output_complexity import EffortDecision, EffortRouter

    router = EffortRouter()
    low = EffortDecision("low", "mechanical_low_complexity")
    keep = EffortDecision(None, "protected")
    assert router.route("k", low, "medium") is None  # first vote
    assert router.route("k", low, "medium") == "low"  # second vote lowers
    # A protected turn returns to the client's level at once, not after 2 votes.
    assert router.route("k", keep, "medium") is None


def test_tool_catalog_adapter_skips_native_deferral_and_compacts(installed):
    from headroom.intelligence.turn_routing import apply_tool_catalog

    rt = installed("full")
    tools = [
        {
            "type": "function",
            "function": {
                "name": f"tool_{i}",
                "description": f"Tool number {i} does a thing. More detail follows here.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "string"},
                        "b": {"type": "integer", "description": "x" * 80},
                    },
                    "required": ["a"],
                },
            },
        }
        for i in range(20)
    ]
    body = {
        "messages": [{"role": "user", "content": "use tool_3 please"}],
        "tools": copy.deepcopy(tools),
    }
    labels = apply_tool_catalog(body, "openai_chat", cache_cold=True)
    assert labels and rt.metrics.snapshot()["catalog_applied"] == 1
    by_name = {t["function"]["name"]: t for t in body["tools"]}
    assert by_name["tool_3"] == tools[3]  # mentioned tool stays full
    assert "b" not in by_name["tool_17"]["function"]["parameters"]["properties"]
    deferred = {
        "messages": body["messages"],
        "tools": [*copy.deepcopy(tools), {"type": "tool_search_tool_bm25"}],
    }
    assert apply_tool_catalog(deferred, "openai_chat", cache_cold=True) == []


def test_codex_workspace_from_environment_context():
    from headroom.intelligence.responses import codex_workspace

    items = [
        {
            "type": "message",
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "<environment_context>\n  <cwd>C:\\Users\\dev\\RepoA</cwd>\n  <shell>powershell</shell>\n</environment_context>",
                }
            ],
        },
        _user("fix the bug"),
    ]
    assert codex_workspace(items) == "c:/users/dev/repoa"
    assert codex_workspace([_user("no context")]) == ""
    assert codex_workspace("not a list") == ""
