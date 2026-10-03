"""Test-runner interceptor: collapse passing-test noise, keep every failure.

Rich interception (Optimization 14). Matches a shell tool (Bash, shell,
PowerShell, …) whose command runs a test runner — pytest, unittest, cargo
test, go test, jest/vitest/npm test, dotnet test, mvn/gradle test — and whose
output is large. The rewrite:

* keeps every line that is not a plain "this test passed" line: failure
  names, assertions, stack traces, warnings, summary counts, durations and
  exit codes, in their original order and byte-exact;
* collapses runs of passing-test lines into ``[… N passing test lines
  collapsed …]``;
* stores the exact original in CCR first (verified) and appends the
  retrieval marker, so nothing is unrecoverable.

Registered only when rich interception is enabled
(``HEADROOM_RICH_INTERCEPTORS=1`` / ``HEADROOM_INTELLIGENCE=full``).
"""

from __future__ import annotations

import logging
import re
from typing import Any

from . import base

logger = logging.getLogger(__name__)

_SHELL_TOOLS = frozenset(
    {
        "bash",
        "shell",
        "local_shell",
        "shell_command",
        "powershell",
        "pwsh",
        "exec_command",
        "run_terminal_cmd",
    }
)
_RUNNER_RE = re.compile(
    r"(?i)(?:^|[\s;&|/\\])(?:pytest|py\.test|python[\d.]*(?:\.exe)?\s+-m\s+(?:pytest|unittest)|"
    r"cargo\s+(?:test|nextest)|go\s+test|npx\s+(?:jest|vitest)|jest|vitest|"
    r"npm\s+(?:run\s+)?test|pnpm\s+(?:run\s+)?test|yarn\s+test|bun\s+test|"
    r"dotnet\s+test|mvn\b[^\n]*\btest|gradlew?(?:\.bat)?\b[^\n]*\btest|tox|nox|"
    r"phpunit|rspec|mix\s+test|ctest)\b"
)
_PASS_LINE_RE = re.compile(
    r"^(?:"
    r".*::\S+\s+PASSED(?:\s+\[\s*\d+%\])?\s*$"  # pytest -v
    r"|test \S+ \.\.\. ok\s*$"  # cargo / unittest -v
    r"|\S+ \(\S+\) \.\.\. ok\s*$"  # unittest
    r"|\s*--- PASS: \S+.*$"  # go test -v
    r"|=== (?:RUN|PAUSE|CONT)\s+\S+\s*$"  # go test -v chatter
    r"|ok\s+\S+\s+[\d.]+s.*$"  # go package ok
    r"|\s*[✓✔√]\s.*$"  # jest/vitest/mocha
    r"|\s*PASS\s+\S+.*$"  # jest file pass
    r"|\s*Passed\s+\S+.*$"  # dotnet test -v
    r"|\S+\.py\s+\.+\s*(?:\[\s*\d+%\])?\s*$"  # pytest progress line of only dots
    r")"
)
_MIN_CHARS = 2000
_MIN_COLLAPSE = 3


def collapse_passing(text: str) -> tuple[str, int]:
    """Return ``(collapsed_text, passing_lines_collapsed)``."""
    lines = text.splitlines()
    out: list[str] = []
    run = 0
    collapsed = 0

    def flush() -> None:
        nonlocal run, collapsed
        if run >= _MIN_COLLAPSE:
            out.append(f"[… {run} passing test lines collapsed …]")
            collapsed += run
        else:
            out.extend(pending)
        pending.clear()
        run = 0

    pending: list[str] = []
    for line in lines:
        if _PASS_LINE_RE.match(line):
            pending.append(line)
            run += 1
            continue
        flush()
        out.append(line)
    flush()
    return "\n".join(out), collapsed


def _command(tool_input: dict[str, Any]) -> str:
    cmd = tool_input.get("command", tool_input.get("cmd", ""))
    if isinstance(cmd, list):
        cmd = " ".join(str(c) for c in cmd)
    return cmd if isinstance(cmd, str) else ""


class TestRunnerInterceptor:
    name = "test-runner"

    def matches(self, tool_name: str | None, tool_input: dict[str, Any], tool_output: str) -> bool:
        if not tool_name or tool_name.lower().split("__")[-1] not in _SHELL_TOOLS:
            return False
        if len(tool_output) < _MIN_CHARS:
            return False
        return bool(_RUNNER_RE.search(_command(tool_input)))

    def transform(
        self, tool_name: str | None, tool_input: dict[str, Any], tool_output: str
    ) -> str | None:
        collapsed, count = collapse_passing(tool_output)
        if count == 0 or len(collapsed) >= len(tool_output):
            return None
        try:
            from headroom.cache.compression_store import get_compression_store

            store = get_compression_store()
            h = store.store(
                tool_output,
                collapsed,
                tool_name=tool_name,
                compression_strategy="test_runner_collapse",
                query_context=_command(tool_input)[:300],
            )
            if not store.verify_exact(h, tool_output):
                return None
        except Exception as e:  # noqa: BLE001 - cannot prove recovery: keep original
            logger.debug("test-runner interceptor: CCR store failed: %s", e)
            return None
        return (
            f"{collapsed}\n[headroom: {count} passing-test lines collapsed; every failure, "
            f"summary and exit code above is verbatim. Retrieve original: hash={h}]"
        )

    def progressive_disclosure_key(
        self, tool_name: str | None, tool_input: dict[str, Any]
    ) -> str | None:
        return None  # every run is new evidence; never pass a later run through raw


_REGISTERED = False


def enable() -> None:
    """Register the interceptor (idempotent)."""
    global _REGISTERED
    if not _REGISTERED:
        base.register(TestRunnerInterceptor())
        _REGISTERED = True


def disable() -> None:
    """Unregister the interceptor (idempotent)."""
    global _REGISTERED
    base.INTERCEPTORS[:] = [
        i for i in base.INTERCEPTORS if not isinstance(i, TestRunnerInterceptor)
    ]
    _REGISTERED = False
