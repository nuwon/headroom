"""Deterministic parsers for tool outputs (exit codes, test/build results).

They are shared by the event normalizer, the evidence extractors, the test
impact planner and the workflow executor, so a ``pytest`` summary is
understood the same way everywhere. Every parser is a pure function of text.
When a parser does not recognize an output it returns nothing rather than a
guess (plan §16, "Evidence parser cannot understand output").

Formats covered:

* exit codes from Claude Code (``Exit code N``), Codex (``Exit code: N``,
  ``Process exited with code N``, JSON ``exit_code``);
* pytest / unittest, cargo test, jest / vitest, ctest, go test summaries plus
  per-test failure ids;
* compiler diagnostics from gcc/clang, MSVC, rustc/cargo, tsc, mypy, ruff and
  Python tracebacks.

CRLF output is handled throughout.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any

_EXIT_RES = (
    re.compile(r"(?im)^\s*Exit code:?\s*(-?\d+)\b"),
    re.compile(r"(?i)Process exited with code\s+(-?\d+)"),
    re.compile(r'"exit_code"\s*:\s*(-?\d+)'),
    re.compile(r"(?i)\bexited with (?:status|code)\s+(-?\d+)"),
)
_TOOL_ERROR_RE = re.compile(r"(?is)<tool_use_error>(.*?)</tool_use_error>")


def parse_exit_code(text: str) -> int | None:
    head = (text or "")[:4000]
    tail = (text or "")[-2000:]
    for rx in _EXIT_RES:
        m = rx.search(head) or rx.search(tail)
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                return None
    return None


def tool_use_error(text: str) -> str:
    m = _TOOL_ERROR_RE.search(text or "")
    return m.group(1).strip()[:400] if m else ""


def outcome_success(text: str, *, is_error: bool) -> tuple[bool | None, int | None]:
    """``(success, exit_code)`` for a tool result; ``success`` None when unknowable."""
    code = parse_exit_code(text)
    if code is not None:
        return code == 0, code
    if is_error or tool_use_error(text):
        return False, None
    stripped = (text or "").lstrip()
    if stripped.startswith("{") and '"exit_code"' in stripped[:200]:
        try:
            data = json.loads(stripped)
            code = int((data.get("metadata") or data).get("exit_code"))
            return code == 0, code
        except (ValueError, TypeError, AttributeError):
            pass
    return True, None


# ---------------------------------------------------------------- tests
@dataclass
class TestResult:
    framework: str
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    errors: int = 0
    failed_ids: list[str] = field(default_factory=list)
    passed_ids: list[str] = field(default_factory=list)
    duration_s: float | None = None
    failure_signatures: dict[str, str] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return self.passed + self.failed + self.errors + self.skipped

    @property
    def ok(self) -> bool:
        return self.failed == 0 and self.errors == 0 and (self.passed > 0 or self.skipped > 0)


_PYTEST_SUMMARY_RE = re.compile(
    r"(?m)^=+ (?P<body>(?:\d+ \w+(?:, )?)+)(?: in (?P<dur>[\d.]+)s)?.*=+\s*$"
)
_PYTEST_SHORT_RE = re.compile(
    r"(?m)^(?P<n>\d+) (?P<kind>passed|failed|error|errors|skipped|xfailed|xpassed|deselected)"
)
_PYTEST_COUNT_RE = re.compile(
    r"(\d+) (passed|failed|errors?|skipped|xfailed|xpassed|deselected|warnings?)"
)
_PYTEST_FAIL_RE = re.compile(r"(?m)^(?:FAILED|ERROR) (\S+?)(?:\s+-\s+(.*))?$")
_PYTEST_PASS_RE = re.compile(r"(?m)^(\S+::\S+) PASSED")
_UNITTEST_RAN_RE = re.compile(r"(?m)^Ran (\d+) tests? in ([\d.]+)s")
_UNITTEST_RESULT_RE = re.compile(r"(?m)^(OK|FAILED)(?: \((.*)\))?\s*$")
_UNITTEST_FAIL_RE = re.compile(r"(?m)^(?:FAIL|ERROR): (\S+) \(([^)]+)\)")
_CARGO_RESULT_RE = re.compile(
    r"(?m)^test result: (?:ok|FAILED)\. (\d+) passed; (\d+) failed; (\d+) ignored;"
)
_CARGO_FAIL_RE = re.compile(r"(?m)^test (\S+) \.\.\. FAILED")
_CARGO_PASS_RE = re.compile(r"(?m)^test (\S+) \.\.\. ok")
_JEST_TESTS_RE = re.compile(
    r"(?m)^Tests:\s+(?:(\d+) failed, )?(?:(\d+) skipped, )?(?:(\d+) todo, )?(?:(\d+) passed, )?(\d+) total"
)
_JEST_FAIL_FILE_RE = re.compile(r"(?m)^\s*FAIL\s+(\S+)")
_VITEST_TESTS_RE = re.compile(
    r"(?m)^\s*Tests\s+(?:(\d+) failed\s*\|\s*)?(?:(\d+) passed)?(?:\s*\|\s*(\d+) skipped)?\s*\((\d+)\)"
)
_VITEST_FAIL_RE = re.compile(r"(?m)^\s*(?:FAIL|×|✗)\s+(\S+\.(?:[jt]sx?|mjs|cjs))(?:\s+>\s+(.+))?$")
_CTEST_SUMMARY_RE = re.compile(r"(?m)(\d+)% tests passed, (\d+) tests failed out of (\d+)")
_CTEST_FAIL_RE = re.compile(r"(?m)^\s*\d+ - (\S+) \((Failed|Timeout|Not Run|SEGFAULT|Exception)\)")
_GO_FAIL_RE = re.compile(r"(?m)^--- FAIL: (\S+)")
_GO_PASS_RE = re.compile(r"(?m)^--- PASS: (\S+)")
_GO_PKG_RE = re.compile(r"(?m)^(ok|FAIL)\s+(\S+)")


def _last(iterator: Any) -> Any:
    last = None
    for last in iterator:  # noqa: B007
        pass
    return last


def _sig(text: str) -> str:
    norm = re.sub(r"0x[0-9a-fA-F]+|\d+", "N", text or "")
    norm = re.sub(r"(?:[A-Za-z]:)?[\\/][^\s:'\"]+", "<path>", norm)
    return hashlib.sha256(" ".join(norm.split())[:400].encode()).hexdigest()[:16]


def parse_test_output(text: str, framework: str = "") -> TestResult | None:
    """Parse a test-runner output. ``None`` when no known summary is present."""
    if not text:
        return None
    t = text.replace("\r\n", "\n")
    fw = (framework or "").lower()
    order = [fw] if fw in ("pytest", "unittest", "cargo", "jest", "vitest", "ctest", "go") else []
    order += ["pytest", "cargo", "vitest", "jest", "ctest", "go", "unittest"]
    for name in dict.fromkeys(order):
        parsed = _PARSERS[name](t)
        if parsed is not None:
            return parsed
    return None


def _parse_pytest(t: str) -> TestResult | None:
    summary = _last(_PYTEST_SUMMARY_RE.finditer(t))
    body = summary.group("body") if summary else ""
    dur = float(summary.group("dur")) if summary and summary.group("dur") else None
    if not body:
        # ``-q`` prints a bare "3 passed, 1 failed in 0.12s" line.
        lines = [ln for ln in t.splitlines() if _PYTEST_SHORT_RE.match(ln.strip())]
        if not lines:
            return None
        body = lines[-1]
        m = re.search(r"in ([\d.]+)s", body)
        dur = float(m.group(1)) if m else None
    counts = {k: int(n) for n, k in _PYTEST_COUNT_RE.findall(body)}
    if not counts:
        return None
    r = TestResult("pytest", duration_s=dur)
    r.passed = counts.get("passed", 0) + counts.get("xpassed", 0)
    r.failed = counts.get("failed", 0)
    r.errors = counts.get("error", 0) + counts.get("errors", 0)
    r.skipped = counts.get("skipped", 0) + counts.get("xfailed", 0)
    for m in _PYTEST_FAIL_RE.finditer(t):
        tid = m.group(1)
        r.failed_ids.append(tid)
        r.failure_signatures[tid] = _sig(m.group(2) or "")
    r.passed_ids = _PYTEST_PASS_RE.findall(t)[:500]
    if r.total == 0:
        return None
    return r


def _parse_unittest(t: str) -> TestResult | None:
    ran = _UNITTEST_RAN_RE.search(t)
    res = _last(_UNITTEST_RESULT_RE.finditer(t))
    if not ran or not res:
        return None
    r = TestResult("unittest", duration_s=float(ran.group(2)))
    total = int(ran.group(1))
    detail = res.group(2) or ""
    counts = dict(re.findall(r"(\w+)=(\d+)", detail))
    r.failed = int(counts.get("failures", 0))
    r.errors = int(counts.get("errors", 0))
    r.skipped = int(counts.get("skipped", 0))
    r.passed = max(0, total - r.failed - r.errors - r.skipped)
    for name, cls in _UNITTEST_FAIL_RE.findall(t):
        r.failed_ids.append(f"{cls}.{name}")
    return r


def _parse_cargo(t: str) -> TestResult | None:
    results = _CARGO_RESULT_RE.findall(t)
    if not results:
        return None
    r = TestResult("cargo")
    for p, f, i in results:
        r.passed += int(p)
        r.failed += int(f)
        r.skipped += int(i)
    r.failed_ids = _CARGO_FAIL_RE.findall(t)
    r.passed_ids = _CARGO_PASS_RE.findall(t)[:500]
    return r


def _parse_jest(t: str) -> TestResult | None:
    m = _last(_JEST_TESTS_RE.finditer(t))
    if m is None:
        return None
    r = TestResult("jest")
    r.failed = int(m.group(1) or 0)
    r.skipped = int(m.group(2) or 0) + int(m.group(3) or 0)
    r.passed = int(m.group(4) or 0)
    r.failed_ids = list(dict.fromkeys(_JEST_FAIL_FILE_RE.findall(t)))
    return r


def _parse_vitest(t: str) -> TestResult | None:
    m = _last(_VITEST_TESTS_RE.finditer(t))
    if m is None:
        return None
    r = TestResult("vitest")
    r.failed = int(m.group(1) or 0)
    r.passed = int(m.group(2) or 0)
    r.skipped = int(m.group(3) or 0)
    ids = []
    for f, name in _VITEST_FAIL_RE.findall(t):
        ids.append(f"{f} > {name}" if name else f)
    r.failed_ids = list(dict.fromkeys(ids))
    return r


def _parse_ctest(t: str) -> TestResult | None:
    m = _CTEST_SUMMARY_RE.search(t)
    if not m:
        return None
    total = int(m.group(3))
    failed = int(m.group(2))
    r = TestResult("ctest", passed=total - failed, failed=failed)
    r.failed_ids = [name for name, _ in _CTEST_FAIL_RE.findall(t)]
    return r


def _parse_go(t: str) -> TestResult | None:
    pkgs = _GO_PKG_RE.findall(t)
    fails = _GO_FAIL_RE.findall(t)
    passes = _GO_PASS_RE.findall(t)
    if not pkgs and not fails and not passes:
        return None
    r = TestResult("go")
    r.failed_ids = fails
    r.passed_ids = passes[:500]
    r.failed = len(fails) or sum(1 for status, _ in pkgs if status == "FAIL")
    r.passed = len(passes) or sum(1 for status, _ in pkgs if status == "ok")
    return r


_PARSERS = {
    "pytest": _parse_pytest,
    "unittest": _parse_unittest,
    "cargo": _parse_cargo,
    "jest": _parse_jest,
    "vitest": _parse_vitest,
    "ctest": _parse_ctest,
    "go": _parse_go,
}


# ---------------------------------------------------------------- builds
@dataclass(frozen=True)
class Diagnostic:
    path: str
    line: int
    code: str
    message: str
    severity: str = "error"


_DIAG_RES = (
    # gcc/clang/ruff/mypy/tsc-with-colon: path:line:col: error: msg
    re.compile(
        r"(?m)^(?P<path>(?:[A-Za-z]:)?[^\s:(][^:\n(]*?\.[A-Za-z0-9]+):(?P<line>\d+)(?::\d+)?:?\s+"
        r"(?P<sev>error|fatal error|warning)?:?\s*(?P<code>[A-Z]+\d+|error\[[A-Z]\d+\])?:?\s*(?P<msg>.+)$"
    ),
    # MSVC: path(line[,col]): error C1234: msg
    re.compile(
        r"(?m)^(?P<path>(?:[A-Za-z]:)?[^\s(][^(\n]*?\.[A-Za-z0-9]+)\((?P<line>\d+)(?:,\d+)?\):\s*"
        r"(?P<sev>error|fatal error|warning)\s+(?P<code>[A-Z]+\d+):\s*(?P<msg>.+)$"
    ),
)
_RUSTC_RE = re.compile(
    r"(?m)^(?P<sev>error|warning)(?:\[(?P<code>E\d+)\])?: (?P<msg>.+)\n\s*--> (?P<path>[^:\n]+):(?P<line>\d+)"
)
_PY_TRACE_RE = re.compile(r'(?m)^\s*File "(?P<path>[^"]+)", line (?P<line>\d+)')
_PY_EXC_RE = re.compile(r"(?m)^(?P<exc>[A-Za-z_][\w.]*(?:Error|Exception|Exit)):\s*(?P<msg>.*)$")


def parse_diagnostics(text: str, *, limit: int = 40) -> list[Diagnostic]:
    """Compiler/linter errors with path and line (warnings dropped)."""
    if not text:
        return []
    t = text.replace("\r\n", "\n")
    out: list[Diagnostic] = []
    seen: set[tuple[str, int, str]] = set()

    def add(path: str, line: int, code: str, msg: str, sev: str) -> None:
        if sev and "warning" in sev:
            return
        key = (path, line, code)
        if key in seen or len(out) >= limit:
            return
        seen.add(key)
        out.append(Diagnostic(path.strip(), line, code or "", msg.strip()[:200], "error"))

    for m in _RUSTC_RE.finditer(t):
        add(
            m.group("path"),
            int(m.group("line")),
            m.group("code") or "",
            m.group("msg"),
            m.group("sev"),
        )
    for rx in _DIAG_RES:
        for m in rx.finditer(t):
            sev = (m.group("sev") or "").lower()
            code = m.group("code") or ""
            msg = m.group("msg") or ""
            if not sev and not code and not re.search(r"(?i)\berror\b", msg):
                continue
            add(m.group("path"), int(m.group("line")), code, msg, sev)
    if "Traceback (most recent call last)" in t:
        frames = list(_PY_TRACE_RE.finditer(t))
        exc = _last(_PY_EXC_RE.finditer(t))
        if frames:
            last = frames[-1]
            add(
                last.group("path"),
                int(last.group("line")),
                exc.group("exc") if exc else "Traceback",
                exc.group("msg") if exc else "",
                "error",
            )
    return out


def failure_signature(text: str) -> str:
    """Normalized hash of the first error-looking line (numbers/paths elided)."""
    for line in (text or "").replace("\r\n", "\n").splitlines()[:200]:
        low = line.lower()
        if any(
            k in low
            for k in (
                "error",
                "fatal",
                "not found",
                "no such file",
                "cannot",
                "failed",
                "denied",
                "not recognized",
                "unknown",
                "invalid",
            )
        ):
            return _sig(line)
    first = next((ln for ln in (text or "").splitlines() if ln.strip()), "")
    return _sig(first) if first else ""


def first_error_line(text: str, limit: int = 160) -> str:
    for line in (text or "").replace("\r\n", "\n").splitlines()[:200]:
        low = line.lower()
        if "error" in low or "fatal" in low or "not found" in low or "no such file" in low:
            return line.strip()[:limit]
    return ""
