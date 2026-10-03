"""Phase 2 agent-state benchmark: overhead, token cost, cache safety, savings.

Runs deterministic synthetic coding sessions through the real
:class:`AgentStateService` (the code the proxy calls for each Claude Code
request) and reports the plan §20 measurements:

* per-request deterministic overhead (median and p95) and hook validation
  latency;
* injected agent-state tokens as a share of the tokens sent;
* cache safety (every turn's history must be a byte-identical extension of the
  previous turn's outgoing messages);
* drift edits detected, and evidenced dependency edits allowed;
* tests selected versus the full suite for low-risk edits;
* workflow macro turns collapsed.

Usage::

    python benchmarks/agent_state_benchmark.py [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))


def _tokens(obj) -> int:  # noqa: ANN001
    return max(1, len(json.dumps(obj)) // 4)


def _state_tokens(messages: list) -> int:  # noqa: ANN001
    """Tokens of agent-state blocks present in an outgoing message list."""
    total = 0
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            for block in content:
                text = block.get("text", "") if isinstance(block, dict) else ""
                if text.startswith("<headroom_agent_state"):
                    total += _tokens(block)
    return total


def _pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * q))]


def _make_repo(base: Path, n_tests: int = 40) -> Path:
    root = base / "proj"
    (root / "src" / "pkg").mkdir(parents=True)
    (root / "src" / "billing").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "src" / "pkg" / "__init__.py").write_text("")
    for i in range(n_tests):
        (root / "src" / "pkg" / f"mod{i}.py").write_text(f"def f{i}():\n    return {i}\n")
        (root / "tests" / f"test_mod{i}.py").write_text(
            f"from pkg.mod{i} import f{i}\n\n\ndef test_{i}():\n    assert f{i}() == {i}\n"
        )
    (root / "src" / "billing" / "rates.py").write_text("RATE = 3\n")
    (root / "pyproject.toml").write_text("[tool.pytest.ini_options]\npythonpath = ['src']\n")
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "b",
        "GIT_AUTHOR_EMAIL": "b@e",
        "GIT_COMMITTER_NAME": "b",
        "GIT_COMMITTER_EMAIL": "b@e",
    }
    for args in (["init", "-q"], ["add", "-A"], ["commit", "-q", "-m", "init"]):
        subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, env=env)
    return root


def run() -> dict:
    tmp = Path(tempfile.mkdtemp(prefix="hr-agent-state-bench-"))
    os.environ["HEADROOM_AGENT_STATE_DIR"] = str(tmp / "state")
    # The session's commands resolve like an activated project venv.
    os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", "")
    from test_intelligence.agent_state.conftest import CLAUDE_HEADERS, Conversation, claude_body

    from headroom.intelligence.agent_state.config import AgentStateConfig
    from headroom.intelligence.agent_state.injection import _strip_cache_control
    from headroom.intelligence.agent_state.runtime import AgentStateService
    from headroom.intelligence.agent_state.workflows import WorkflowExecutor, eligible_macros

    repo = _make_repo(tmp)
    svc = AgentStateService(AgentStateConfig.from_env({}))
    goal = (
        "Add retry support to src/pkg/mod3.py. Do not change the public API. Must work on Windows and Linux. "
        "Never log credentials. Do not modify src/billing/. All tests must pass. "
        "It should handle a zero retry count."
    )
    convo = Conversation(repo, goal)
    request_ms: list[float] = []
    injected = 0
    sent_tokens = 0
    baseline_tokens = 0  # the identical replay with every Phase 2 feature off
    state_resent = 0  # agent-state tokens across every request (blocks stay in history)
    state_last = 0
    injections = 0
    prev_out = None
    cache_violations = 0
    turns = 0

    def send() -> object:
        nonlocal injected, sent_tokens, baseline_tokens, injections, prev_out, cache_violations
        nonlocal turns, state_resent, state_last
        baseline_tokens += _tokens(convo.messages)
        started = time.perf_counter()
        rs = svc.begin_anthropic(claude_body(), CLAUDE_HEADERS, convo.messages, cwd=str(repo))
        out = convo.messages
        if rs is not None:
            new = svc.apply_anthropic(rs, convo.messages, convo.messages)
            if new is not None:
                out = new
                svc.commit(rs)
                if rs.pending and rs.pending.entry is not None:
                    injections += 1
                    injected += max(1, len(rs.pending.entry.text) // 4)
        request_ms.append((time.perf_counter() - started) * 1000.0)
        if prev_out is not None:
            a = _strip_cache_control(prev_out)
            b = _strip_cache_control(out[: len(prev_out)])
            if a != b:
                cache_violations += 1
        prev_out = [dict(m) for m in out]
        sent_tokens += _tokens(out)
        state_last = _state_tokens(out)
        state_resent += state_last
        turns += 1
        return rs

    send()
    passing = "..\n2 passed in 0.05s"
    for i in range(40):
        k = i % 5
        if k == 0:
            convo.read(f"src/pkg/mod{i % 10}.py")
        elif k == 1:
            convo.edit("src/pkg/mod3.py", f"o{i}", f"n{i}")
        elif k == 2:
            convo.bash("python -m pytest -q tests/test_mod3.py", passing)
        elif k == 3:
            convo.bash("git status --short", " M src/pkg/mod3.py")
        else:
            convo.bash("git diff --stat", " 1 file changed")
        send()
    # Drift and a real dependency.
    convo.edit("src/billing/rates.py", "RATE", "RATE2")
    rs = send()
    drift_detected = rs is not None and "src/billing/rates.py" in (rs.block or "")
    convo.bash(
        "python -m pytest -q",
        'Traceback (most recent call last):\n  File "src/pkg/mod4.py", line 1, in <module>\nTypeError: retry\n1 failed in 0.1s',
        code=1,
    )
    rs = send()
    from headroom.intelligence.agent_state.scope import ScopeClass

    dep_allowed = rs.runtime.scope.classify_path(
        str(repo / "src/pkg/mod4.py"), op="write"
    ).scope in (ScopeClass.IN_SCOPE, ScopeClass.DEPENDENCY_SCOPE)
    # Hook validation latency over mixed calls.
    hook_ms: list[float] = []
    # (tool, input, deterministically invalid?): the last two are a write under the
    # excluded src/billing/ and an edit of a file that does not exist.
    calls = [
        (
            "Edit",
            {"file_path": str(repo / "src/pkg/mod3.py"), "old_string": "a", "new_string": "b"},
        ),
        ("Bash", {"command": "python -m pytest -q tests/test_mod3.py"}),
        ("Bash", {"command": "git status"}),
        ("Write", {"file_path": str(repo / "src/billing/new.py"), "content": "x"}),
        (
            "Edit",
            {"file_path": str(repo / "src/pkg/missing.py"), "old_string": "a", "new_string": "b"},
        ),
    ]
    invalid_calls = invalid_executed = valid_denied = 0
    for j in range(200):
        name, inp = calls[j % len(calls)]
        invalid = j % len(calls) >= 3
        t = time.perf_counter()
        decision = svc.on_pretool_hook(
            {
                "agent": "claude",
                "hook": {
                    "session_id": "11111111-2222-3333-4444-555555555555",
                    "cwd": str(repo),
                    "tool_name": name,
                    "tool_input": inp,
                    "tool_use_id": f"bench{j}",
                },
            }
        )
        hook_ms.append((time.perf_counter() - t) * 1000.0)
        denied = decision.get("decision") == "deny"
        invalid_calls += invalid
        invalid_executed += invalid and not denied
        valid_denied += (not invalid) and denied
    # Test impact on a low-risk leaf edit.
    planner = rs.runtime.test_impact
    plan = planner.plan(force=True)
    total_tests = plan.tests_considered
    selected = len(plan.tier1)
    # Real execution: the tier-1 plan versus the full suite an agent would otherwise run.
    t = time.perf_counter()
    tier1_run = planner.run_plan(plan, max_tier=1)
    tier1_s = time.perf_counter() - t
    t = time.perf_counter()
    full = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"],
        cwd=repo,
        capture_output=True,
        text=True,
    )
    full_s = time.perf_counter() - t
    from headroom.intelligence.agent_state.results import parse_test_output

    full_parsed = parse_test_output(full.stdout, "pytest")
    # Workflow macro: learned from the repeated verify loop above.
    macros = [
        m
        for m in eligible_macros(rs.runtime.store, rs.runtime.workspace.workspace_id)
        if m.origin == "learned"
    ]
    macro_result = None
    if macros:
        props = (macros[0].input_schema or {}).get("properties") or {}
        inputs = {k: (spec.get("examples") or [""])[0] for k, spec in props.items()}
        macro_result = WorkflowExecutor(rs.runtime).run(macros[0].name, inputs)
    snapshot = svc.metrics.snapshot()
    return {
        "requests": turns,
        "request_overhead_ms": {
            "median": round(statistics.median(request_ms), 2),
            "p95": round(_pct(request_ms, 0.95), 2),
            "first": round(request_ms[0], 2),
        },
        "hook_validation_ms": {
            "median": round(statistics.median(hook_ms), 3),
            "p95": round(_pct(hook_ms, 0.95), 3),
        },
        "baseline_tokens_sent": baseline_tokens,
        "injections": injections,
        "injected_tokens_first_send": injected,
        "tokens_sent": sent_tokens,
        "state_tokens_resent_total": state_resent,
        "state_tokens_in_last_request": state_last,
        "token_overhead_vs_baseline_pct": round(
            100.0 * (sent_tokens - baseline_tokens) / max(1, baseline_tokens), 1
        ),
        "cache_prefix_violations": cache_violations,
        "drift_detected": drift_detected,
        "dependency_allowed": dep_allowed,
        "tests_total": total_tests,
        "tests_selected_tier1": selected,
        "tests_avoided_pct": round(100.0 * (1 - selected / max(1, total_tests)), 1),
        "risk": plan.risk_score,
        "verification": {
            "tier1_status": tier1_run.get("status"),
            "tier1_tests_run": tier1_run.get("tests_run"),
            "tier1_seconds": round(tier1_s, 2),
            "full_suite_tests_run": full_parsed.total if full_parsed else None,
            "full_suite_seconds": round(full_s, 2),
        },
        "hook_calls": {
            "total": len(hook_ms),
            "invalid": invalid_calls,
            "invalid_executed_baseline": invalid_calls,
            "invalid_executed_with_phase2": invalid_executed,
            "valid_calls_denied": valid_denied,
        },
        "macros_promoted": [m.name for m in macros],
        "macro_status": (macro_result or {}).get("status"),
        "macro_steps_collapsed": (macro_result or {}).get("steps"),
        "counters": {
            k: v for k, v in snapshot["counters"].items() if not k.startswith("tool_validation_")
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    out = run()
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
