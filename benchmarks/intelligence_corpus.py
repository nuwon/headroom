"""Deterministic eval corpus for the context-intelligence layer.

Each :class:`Scenario` is one coding-agent turn: the user's task, the tool
call the agent made and the (large) tool output it got back, plus the
**facts** an answer depends on, i.e. exact strings that must stay visible in
context or be recoverable through a retrieval marker. The same scenario renders
as a Claude Code request (Anthropic ``tool_use`` / ``tool_result``) and as a
Codex request (OpenAI Responses ``function_call`` / ``function_call_output``).

No randomness and no network: outputs are generated from fixed templates so
runs are comparable across machines and commits.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolTurn:
    name: str  # Claude Code tool name
    input: dict[str, Any]
    output: str
    codex_cmd: str  # the equivalent Codex exec_command


@dataclass(frozen=True)
class Scenario:
    name: str
    task: str
    turns: tuple[ToolTurn, ...]
    facts: tuple[str, ...]
    follow_up: str = ""  # optional later user question (multi-turn)
    tags: tuple[str, ...] = field(default_factory=tuple)

    # ------------------------------------------------------------ renderers
    def anthropic_messages(self) -> list[dict[str, Any]]:
        msgs: list[dict[str, Any]] = [{"role": "user", "content": self.task}]
        for i, turn in enumerate(self.turns):
            tid = f"toolu_{self.name}_{i}"
            msgs.append(
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": f"Running {turn.name}."},
                        {"type": "tool_use", "id": tid, "name": turn.name, "input": turn.input},
                    ],
                }
            )
            msgs.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": tid, "content": turn.output}
                    ],
                }
            )
        if self.follow_up:
            msgs.append({"role": "assistant", "content": "Done; what next?"})
            msgs.append({"role": "user", "content": self.follow_up})
        return msgs

    def responses_input(self) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": self.task}],
            }
        ]
        for i, turn in enumerate(self.turns):
            cid = f"call_{self.name}_{i}"
            items.append(
                {
                    "type": "function_call",
                    "name": "exec_command",
                    "call_id": cid,
                    "arguments": json.dumps({"cmd": turn.codex_cmd}),
                }
            )
            items.append({"type": "function_call_output", "call_id": cid, "output": turn.output})
        if self.follow_up:
            items.append(
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Done; what next?"}],
                }
            )
            items.append(
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": self.follow_up}],
                }
            )
        return items


# ---------------------------------------------------------------- generators
def _services_json(n: int = 160) -> str:
    rows = []
    for i in range(n):
        failing = i in (23, 117)
        rows.append(
            {
                "id": i,
                "service": f"svc-{i:03d}",
                "status": "error" if failing else "ok",
                "region": ("us-east-1", "eu-west-1", "ap-south-1")[i % 3],
                "latency_ms": 90 + (i * 7) % 60 + (900 if failing else 0),
                "last_error": (f"upstream timeout after 30s (pool={i % 4})" if failing else None),
                "owner": f"team-{i % 9}",
            }
        )
    return json.dumps(rows, indent=2)


def _pytest_output() -> str:
    lines = ["============================= test session starts =============================="]
    lines.append("platform linux -- Python 3.11.9, pytest-8.3.2")
    lines.append("collected 312 items")
    lines.append("")
    for i in range(311):
        lines.append(
            f"tests/test_module_{i // 40}.py::test_case_{i:03d} PASSED  [{(i * 100) // 312:3d}%]"
        )
    lines.append("tests/test_http.py::test_parse_header_folding FAILED  [100%]")
    lines.append("")
    lines.append("=================================== FAILURES ===================================")
    lines.append("_________________________ test_parse_header_folding _________________________")
    lines.append("")
    lines.append("    def test_parse_header_folding():")
    lines.append('        raw = b"X-Long: a\\r\\n b\\r\\n"')
    lines.append("        headers = parse_headers(raw)")
    lines.append('>       assert headers["X-Long"] == "a b"')
    lines.append("E       AssertionError: assert 'a' == 'a b'")
    lines.append("")
    lines.append("src/http/headers.py:88: AssertionError")
    lines.append("=========================== short test summary info ============================")
    lines.append(
        "FAILED tests/test_http.py::test_parse_header_folding - AssertionError: assert 'a' == 'a b'"
    )
    lines.append("======================== 1 failed, 311 passed in 41.27s ========================")
    return "\n".join(lines)


def _build_log() -> str:
    lines = []
    for i in range(260):
        lines.append(
            f"   Compiling crate_{i % 37} v0.{i % 9}.{i % 5} (/work/crates/crate_{i % 37})"
        )
        if i % 23 == 0:
            lines.append(f"warning: unused variable: `tmp_{i}`")
            lines.append(f"  --> crates/crate_{i % 37}/src/lib.rs:{10 + i}:9")
    lines.append("error[E0308]: mismatched types")
    lines.append("   --> crates/proxy/src/router.rs:214:17")
    lines.append("    |")
    lines.append("214 |         let n: u32 = config.max_retries;")
    lines.append("    |                ---   ^^^^^^^^^^^^^^^^^^ expected `u32`, found `usize`")
    lines.append("error: could not compile `proxy` (lib) due to 1 previous error")
    lines.append("Process exited with code 101")
    return "\n".join(lines)


def _grep_output() -> str:
    lines = []
    for f in range(40):
        for j in range(6):
            lines.append(
                f"src/mod_{f}/handler_{j}.py:{20 + j * 13}:    value = compute_{f}_{j}(request)"
            )
    lines.insert(97, "src/http/headers.py:71:def parse_headers(raw: bytes) -> dict[str, str]:")
    lines.insert(
        98, "src/http/headers.py:88:        folded = line.lstrip()  # obs-fold continuation"
    )
    lines.insert(150, "src/http/client.py:203:    headers = parse_headers(response_head)")
    return "\n".join(lines)


def _pods(state_of_7: str) -> str:
    header = "NAME                         READY   STATUS             RESTARTS   AGE"
    rows = [header]
    for i in range(120):
        status = state_of_7 if i == 7 else "Running"
        restarts = 14 if (i == 7 and state_of_7 != "Running") else i % 3
        rows.append(
            f"checkout-api-{i:03d}-7f9c6d   1/1     {status:<18} {restarts:<10} {3 + i % 20}d"
        )
    return "\n".join(rows)


def _long_doc() -> str:
    parts = ["# Platform API Guide", ""]
    for s in range(60):
        parts.append(f"## Section {s}: Topic {s}")
        parts.append(
            f"Topic {s} covers configuration surface {s}. It explains defaults, "
            f"failure handling and the operational runbook for subsystem {s}. "
            "Teams should read it before changing production settings."
        )
        parts.append("")
    parts.insert(
        200,
        "## Rate limits\nThe public API allows 600 requests per minute per key and a burst of 120. "
        "Exceeding it returns HTTP 429 with a Retry-After header of 17 seconds.\n",
    )
    return "\n".join(parts)


def _diff() -> str:
    out = []
    for f in range(14):
        out.append(f"diff --git a/src/pkg_{f}/util.py b/src/pkg_{f}/util.py")
        out.append("index 1a2b3c4..5d6e7f8 100644")
        out.append(f"--- a/src/pkg_{f}/util.py")
        out.append(f"+++ b/src/pkg_{f}/util.py")
        out.append(f"@@ -10,12 +10,14 @@ def helper_{f}(x):")
        for k in range(10):
            out.append(f"     line_{k} = transform_{f}(x, {k})")
        out.append(f"-    return line_9 + {f}")
        out.append(f"+    return normalize(line_9) + {f}")
    out.insert(30, "@@ -40,6 +42,9 @@ def retry_backoff(attempt):")
    out.insert(31, "-    delay = 2 ** attempt")
    out.insert(32, "+    delay = min(2 ** attempt, MAX_BACKOFF_SECONDS)")
    out.insert(33, "+    delay += jitter(attempt)")
    return "\n".join(out)


def _source_file() -> str:
    lines = ['"""Session store."""', "", "import time", ""]
    for i in range(70):
        lines.append(f"def helper_{i}(value):")
        lines.append(f'    """Helper {i}."""')
        lines.append(f"    return value * {i} + {i % 7}")
        lines.append("")
    lines.append("def evict_expired(store, now=None):")
    lines.append('    """Drop sessions whose ttl elapsed."""')
    lines.append("    now = now or time.time()")
    lines.append("    for key in list(store):")
    lines.append("        if store[key].expires_at < now:")
    lines.append("            del store[key]")
    return "\n".join(lines)


def build_corpus() -> list[Scenario]:
    return [
        Scenario(
            "api_json_failures",
            "Which services in the inventory are failing, and why?",
            (
                ToolTurn(
                    "Bash",
                    {"command": "curl -s http://inventory.internal/api/services"},
                    _services_json(),
                    "curl -s http://inventory.internal/api/services",
                ),
            ),
            ("svc-023", "svc-117", "upstream timeout after 30s"),
            tags=("json",),
        ),
        Scenario(
            "pytest_failure",
            "Run the test suite and fix the failing test.",
            (
                ToolTurn(
                    "Bash",
                    {"command": "pytest -q"},
                    _pytest_output(),
                    "pytest -q",
                ),
            ),
            (
                "test_parse_header_folding",
                "AssertionError: assert 'a' == 'a b'",
                "src/http/headers.py:88",
                "1 failed, 311 passed",
            ),
            tags=("test",),
        ),
        Scenario(
            "build_error",
            "The build is broken, find out why.",
            (
                ToolTurn(
                    "Bash",
                    {"command": "cargo build --workspace"},
                    _build_log(),
                    "cargo build --workspace",
                ),
            ),
            ("error[E0308]: mismatched types", "crates/proxy/src/router.rs:214:17", "101"),
            tags=("log",),
        ),
        Scenario(
            "grep_definition",
            "Where is `parse_headers` defined and who calls it?",
            (
                ToolTurn(
                    "Grep",
                    {"pattern": "parse_headers|compute_", "path": "src", "output_mode": "content"},
                    _grep_output(),
                    "rg -n 'parse_headers|compute_' src",
                ),
            ),
            # Content, not "path:line": a lossless fold regroups grep output by
            # directory ("src/http/" then "headers.py:71:…"), which keeps the
            # location but not the literal "src/http/headers.py:71" string.
            ("def parse_headers(raw: bytes)", "headers = parse_headers(response_head)"),
            tags=("search",),
        ),
        Scenario(
            "pods_changed",
            "Watch the checkout pods and tell me what changed between the two checks.",
            (
                ToolTurn(
                    "Bash",
                    {"command": "kubectl get pods -l app=checkout"},
                    _pods("Running"),
                    "kubectl get pods -l app=checkout",
                ),
                ToolTurn(
                    "Bash",
                    {"command": "kubectl get pods -l app=checkout"},
                    _pods("CrashLoopBackOff"),
                    "kubectl get pods -l app=checkout",
                ),
            ),
            ("checkout-api-007-7f9c6d", "CrashLoopBackOff"),
            tags=("delta",),
        ),
        Scenario(
            "doc_rate_limits",
            "What are the API rate limits and what happens when we exceed them?",
            (
                ToolTurn(
                    "WebFetch",
                    {"url": "https://docs.internal/platform-api", "prompt": "rate limits"},
                    _long_doc(),
                    "curl -s https://docs.internal/platform-api",
                ),
            ),
            ("600 requests per minute", "burst of 120", "HTTP 429", "17 seconds"),
            tags=("text",),
        ),
        Scenario(
            "diff_review",
            "Review the change to `retry_backoff` in this diff.",
            (
                ToolTurn(
                    "Bash",
                    {"command": "git diff HEAD~1"},
                    _diff(),
                    "git diff HEAD~1",
                ),
            ),
            ("def retry_backoff(attempt)", "MAX_BACKOFF_SECONDS", "jitter(attempt)"),
            tags=("diff",),
        ),
        Scenario(
            "code_read",
            "Explain how `evict_expired` decides what to delete in src/session_store.py.",
            (
                ToolTurn(
                    "Read",
                    {"file_path": "src/session_store.py"},
                    _source_file(),
                    "cat src/session_store.py",
                ),
            ),
            ("def evict_expired(store, now=None):", "store[key].expires_at < now"),
            tags=("code", "read"),
        ),
    ]
