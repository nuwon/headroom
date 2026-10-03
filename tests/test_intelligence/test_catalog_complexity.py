"""Progressive tool catalog (9) and complexity-aware output shaping (10)."""

from __future__ import annotations

import json

from headroom.intelligence.messages import responses_items_to_messages
from headroom.proxy.output_complexity import (
    EffortDecision,
    EffortRouter,
    apply_effort,
    assess_turn,
    client_effort,
    decide_effort,
    decide_verbosity,
)
from headroom.proxy.output_turn_policy import TurnKind
from headroom.proxy.tool_catalog import ProgressiveToolCatalog, compact_schema, compact_tool

CORE = frozenset({"bash", "read", "edit"})


def chat_tool(name: str, desc: str, required: list[str], optional: list[str]) -> dict:
    props = {
        p: {"type": "string", "description": f"The {p} parameter. " * 5}
        for p in required + optional
    }
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": desc + " Extra detail. " * 10,
            "parameters": {"type": "object", "properties": props, "required": required},
        },
    }


def catalog_tools(n: int = 30) -> list[dict]:
    tools = [chat_tool("bash", "Run a shell command.", ["command"], ["timeout"])]
    tools.append(
        chat_tool(
            "jira_create_issue",
            "Create a Jira issue in a project.",
            ["project", "summary"],
            ["labels", "assignee"],
        )
    )
    tools.append(
        chat_tool(
            "github_create_pr", "Open a GitHub pull request.", ["repo", "title"], ["body", "draft"]
        )
    )
    for i in range(n):
        tools.append(
            chat_tool(
                f"tool_{i}", f"Utility number {i} for misc things.", ["arg"], ["opt1", "opt2"]
            )
        )
    return tools


class TestCompaction:
    def test_required_only_projection(self):
        schema = {
            "type": "object",
            "properties": {
                "a": {"type": "string", "description": "x"},
                "b": {"type": "array", "items": {"type": "string"}, "description": "y"},
                "c": {
                    "type": "object",
                    "properties": {"d": {"type": "integer"}, "e": {"type": "string"}},
                    "required": ["d"],
                },
                "opt": {"type": "string"},
            },
            "required": ["a", "b", "c"],
        }
        out = compact_schema(schema)
        assert set(out["properties"]) == {"a", "b", "c"} and out["required"] == ["a", "b", "c"]
        assert out["properties"]["b"]["items"] == {"type": "string"}
        assert set(out["properties"]["c"]["properties"]) == {"d"}

    def test_strict_and_typed_tools_untouched(self):
        strict = {"type": "function", "name": "x", "strict": True, "parameters": {"type": "object"}}
        typed = {"type": "web_search_20250305", "name": "web_search"}
        assert compact_tool(strict) is strict and compact_tool(typed) is typed


class TestCatalog:
    def test_materializes_relevant_and_compacts_rest(self):
        cat = ProgressiveToolCatalog(top_k=2, min_tools=12)
        tools = catalog_tools()
        msgs = [{"role": "user", "content": "open a pull request on github for this fix"}]
        res = cat.apply(
            tools,
            messages=msgs,
            query=msgs[0]["content"],
            core=CORE,
            session_key="s1",
            cache_cold=True,
            user_text=msgs[0]["content"],
        )
        assert res.changed and res.bytes_saved > 0
        by = {t["function"]["name"]: t for t in res.tools}
        assert "draft" in by["github_create_pr"]["function"]["parameters"]["properties"]
        assert "timeout" in by["bash"]["function"]["parameters"]["properties"]  # core stays full
        assert set(by["tool_3"]["function"]["parameters"]["properties"]) == {
            "arg"
        }  # compact, callable
        assert [t["function"]["name"] for t in res.tools] == [
            t["function"]["name"] for t in tools
        ]  # order kept

    def test_sticky_set_is_cache_stable(self):
        cat = ProgressiveToolCatalog(top_k=2)
        tools = catalog_tools()
        m1 = [{"role": "user", "content": "create a jira issue"}]
        r1 = cat.apply(
            tools,
            messages=m1,
            query="create a jira issue",
            core=CORE,
            session_key="s",
            cache_cold=True,
        )
        m2 = m1 + [
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "now open a github pull request"},
        ]
        r2 = cat.apply(
            tools,
            messages=m2,
            query="open a github pull request",
            core=CORE,
            session_key="s",
            cache_cold=False,
        )
        assert json.dumps(r1.tools) == json.dumps(r2.tools)

    def test_used_tool_grows_only_when_cache_cold(self):
        cat = ProgressiveToolCatalog(top_k=1)
        tools = catalog_tools()
        m1 = [{"role": "user", "content": "create a jira issue"}]
        cat.apply(tools, messages=m1, query="jira", core=CORE, session_key="g", cache_cold=True)
        used = m1 + [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "1",
                        "type": "function",
                        "function": {"name": "tool_7", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "1", "content": "ok"},
        ]
        warm = cat.apply(
            tools, messages=used, query="x", core=CORE, session_key="g", cache_cold=False
        )
        assert "tool_7" not in warm.materialized and not warm.grew
        cold = cat.apply(
            tools, messages=used, query="x", core=CORE, session_key="g", cache_cold=True
        )
        assert "tool_7" in cold.materialized and cold.grew

    def test_small_tool_lists_untouched(self):
        res = ProgressiveToolCatalog().apply(
            catalog_tools(3), messages=[], query="x", core=CORE, session_key=None, cache_cold=True
        )
        assert not res.changed

    def test_responses_shape(self):
        tools = [
            {
                "type": "function",
                "name": f"mcp_{i}",
                "description": "Does thing. More.",
                "parameters": {
                    "type": "object",
                    "properties": {"a": {"type": "string"}, "b": {"type": "string"}},
                    "required": ["a"],
                },
                "strict": False,
            }
            for i in range(20)
        ]
        res = ProgressiveToolCatalog(top_k=1).apply(
            tools,
            messages=[{"role": "user", "content": "x"}],
            query="x",
            core=CORE,
            session_key="r",
            cache_cold=True,
        )
        assert res.changed and all(set(t["parameters"]["properties"]) == {"a"} for t in res.tools)


class TestComplexity:
    def test_clean_tool_result_with_conflict_is_not_mechanical(self):
        msgs = [
            {"role": "user", "content": "run the tests"},
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "t", "name": "Bash", "input": {"command": "pytest"}}
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t",
                        "content": "0 failed, 12 passed\ntest_x FAILED (flaky retry)\nexit code 0",
                    }
                ],
            },
        ]
        tc = assess_turn(msgs)
        assert tc.structural_turn_kind == TurnKind.MECHANICAL_CONTINUATION.value
        assert (
            tc.conflicting_evidence
            and tc.semantic_turn_kind is not TurnKind.MECHANICAL_CONTINUATION
        )
        assert decide_effort(tc, "medium").target in (None, "high")

    def test_simple_write_success_is_mechanical_low(self):
        msgs = [
            {"role": "user", "content": "write the file"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "w",
                        "name": "Write",
                        "input": {"file_path": "a.txt"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "w",
                        "content": "File written successfully.",
                    }
                ],
            },
        ]
        tc = assess_turn(msgs)
        assert tc.semantic_turn_kind is TurnKind.MECHANICAL_CONTINUATION and tc.score == 0
        assert decide_effort(tc, "medium") == EffortDecision("low", "mechanical_low_complexity")
        assert decide_effort(tc, "xhigh").target is None  # explicit high stays

    def test_new_architecture_question_high_effort_detailed(self):
        tc = assess_turn(
            [{"role": "user", "content": "Explain the architecture tradeoffs of our cache design"}]
        )
        assert tc.new_user_ask and tc.explanation_requested and tc.score >= 3
        assert decide_effort(tc, "low").target == "high"
        assert decide_verbosity(tc, 3) == 2

    def test_complex_failure_high_effort_concise(self):
        msgs = [
            {"role": "user", "content": "fix it"},
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "t", "name": "Bash", "input": {"command": "pytest"}}
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t",
                        "is_error": True,
                        "content": "Traceback (most recent call last):\nValueError: invariant broken in ledger.reconcile\nexit code 1",
                    }
                ],
            },
        ]
        tc = assess_turn(msgs)
        assert tc.error_present and tc.protected
        assert (
            decide_effort(tc, "medium").target is None
            or decide_effort(tc, "medium").target == "high"
        )
        assert decide_verbosity(tc, 3) == 3

    def test_codex_responses_items(self):
        items = [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "update docs"}],
            },
            {
                "type": "function_call",
                "call_id": "c",
                "name": "shell",
                "arguments": json.dumps({"command": ["bash", "-lc", "echo ok > f"]}),
            },
            {"type": "function_call_output", "call_id": "c", "output": "ok"},
        ]
        tc = assess_turn(responses_items_to_messages(items))
        assert tc.structural_turn_kind == TurnKind.MECHANICAL_CONTINUATION.value

    def test_effort_router_hysteresis_and_fields(self):
        r = EffortRouter()
        low = EffortDecision("low", "mechanical_low_complexity")
        assert r.route("k", low, "medium") is None  # first lowering vote held back
        assert r.route("k", low, "medium") == "low"  # second agreeing vote applies
        assert (
            r.route("k", EffortDecision("high", "complex_turn"), "medium") == "high"
        )  # raise immediately
        body = {"reasoning": {"effort": "medium", "summary": "auto"}}
        assert client_effort(body, "openai_responses") == "medium"
        assert apply_effort(body, "openai_responses", "high") and body["reasoning"] == {
            "effort": "high",
            "summary": "auto",
        }
        assert not apply_effort({}, "anthropic", "high")  # never injects a field
