"""Rich tool-result interception (Optimization 14)."""

from __future__ import annotations

import pytest

from headroom.cache.compression_store import CompressionStore
from headroom.intelligence.invariants import ccr_hashes_in
from headroom.proxy.interceptors import astgrep
from headroom.proxy.interceptors.test_runner import TestRunnerInterceptor, collapse_passing


def _ast_grep_available() -> bool:
    try:
        from headroom import binaries

        binaries.resolve("ast-grep")
        return True
    except Exception:  # noqa: BLE001
        return False


needs_ast_grep = pytest.mark.skipif(not _ast_grep_available(), reason="ast-grep binary unavailable")


def numbered(src: str, arrow: bool = True) -> str:
    sep = "→" if arrow else "\t"
    return "".join(f"{i:>6}{sep}{line}\n" for i, line in enumerate(src.splitlines(), start=1))


PY_SRC = "\n".join(
    [
        "import os",
        "",
        *[
            f"def handler_{i}(request):\n    value = request.get({i})\n    return value * {i}\n"
            for i in range(40)
        ],
        "class Service:",
        '    """Coordinates handlers."""',
        "    def run(self):",
        "        return 1",
    ]
)


@pytest.fixture
def store(monkeypatch):
    s = CompressionStore(max_entries=100)
    monkeypatch.setattr("headroom.cache.compression_store.get_compression_store", lambda *a, **k: s)
    yield s
    astgrep.set_rich_mode(False)


@needs_ast_grep
class TestReadOutline:
    def test_numbered_read_keeps_real_line_numbers(self, store):
        out = astgrep.AstGrepReadOutline().transform(
            "Read", {"file_path": "svc.py"}, numbered(PY_SRC)
        )
        assert out is not None
        assert "→def handler_0(request):" in out
        assert "     3→def handler_0" in out or "3→def handler_0" in out
        assert "class Service" in out

    def test_tab_numbered_variant(self, store):
        out = astgrep.AstGrepReadOutline().transform(
            "Read", {"file_path": "svc.py"}, numbered(PY_SRC, arrow=False)
        )
        assert out is not None and "\tdef handler_1(request):" in out

    def test_partial_read_labelled_and_reminder_kept(self, store):
        text = (
            numbered(PY_SRC)
            + "\n<system-reminder>File truncated: showing first 2000 lines.</system-reminder>\n"
        )
        text = text.replace(
            "<system-reminder>File truncated",
            "[... 4000 lines truncated ...]\n<system-reminder>File truncated",
        )
        out = astgrep.AstGrepReadOutline().transform("Read", {"file_path": "svc.py"}, text)
        assert out is not None
        assert "partial input" in out
        assert out.rstrip().endswith("</system-reminder>")

    def test_rich_mode_stores_exact_original(self, store):
        astgrep.set_rich_mode(True)
        original = numbered(PY_SRC)
        out = astgrep.AstGrepReadOutline().transform("Read", {"file_path": "svc.py"}, original)
        assert out is not None
        h = ccr_hashes_in(out)[0]
        assert store.retrieve(h).original_content == original

    def test_rich_mode_store_failure_keeps_original(self, monkeypatch, store):
        astgrep.set_rich_mode(True)

        def boom(*a, **k):
            raise OSError("no disk")

        monkeypatch.setattr(store, "store", boom)
        assert (
            astgrep.AstGrepReadOutline().transform(
                "Read", {"file_path": "svc.py"}, numbered(PY_SRC)
            )
            is None
        )


PYTEST_OUT = "\n".join(
    [
        "============================= test session starts ==============================",
        *[
            f"tests/test_api.py::test_endpoint_{i} PASSED                       [{i % 100:>3}%]"
            for i in range(300)
        ],
        "tests/test_api.py::test_refund_flow FAILED                             [ 99%]",
        "=================================== FAILURES ===================================",
        "______________________________ test_refund_flow ______________________________",
        "    def test_refund_flow():",
        ">       assert refund(42) == 'ok'",
        "E       AssertionError: assert 'declined' == 'ok'",
        "tests/test_api.py:88: AssertionError",
        "=========================== short test summary info ============================",
        "FAILED tests/test_api.py::test_refund_flow - AssertionError: assert 'declined' == 'ok'",
        "======================== 1 failed, 300 passed in 4.21s =========================",
    ]
)


class TestTestRunner:
    def test_collapse_keeps_failures_and_summary(self):
        collapsed, count = collapse_passing(PYTEST_OUT)
        assert count == 300
        for needle in (
            "test_refund_flow FAILED",
            "AssertionError: assert 'declined' == 'ok'",
            "tests/test_api.py:88",
            "1 failed, 300 passed in 4.21s",
        ):
            assert needle in collapsed
        assert "PASSED" not in collapsed

    def test_matches_runners_across_shells(self):
        tr = TestRunnerInterceptor()
        for name, cmd in [
            ("Bash", "cd /repo && pytest -q"),
            ("shell", ["bash", "-lc", "cargo test"]),
            ("PowerShell", "python -m pytest tests"),
            ("shell", ["powershell.exe", "-Command", "dotnet test"]),
            ("Bash", "npm run test"),
        ]:
            assert tr.matches(name, {"command": cmd}, PYTEST_OUT), (name, cmd)
        assert not tr.matches("Bash", {"command": "ls -la"}, PYTEST_OUT)
        assert not tr.matches("Read", {"command": "pytest"}, PYTEST_OUT)

    def test_transform_stores_original(self, store):
        out = TestRunnerInterceptor().transform("Bash", {"command": "pytest"}, PYTEST_OUT)
        assert out is not None and len(out) < len(PYTEST_OUT) / 4
        h = ccr_hashes_in(out)[0]
        assert store.retrieve(h).original_content == PYTEST_OUT

    def test_go_and_jest(self):
        go = "\n".join(
            ["=== RUN   TestA", "--- PASS: TestA (0.00s)"] * 50
            + ["--- FAIL: TestB (0.01s)", "    b_test.go:12: boom", "FAIL", "exit status 1"]
        )
        c, n = collapse_passing(go)
        assert n == 100 and "--- FAIL: TestB" in c and "exit status 1" in c
        jest = "\n".join(
            [
                " PASS  src/a.test.ts",
                *[f"  ✓ renders row {i} (3 ms)" for i in range(40)],
                "  ✕ handles error (5 ms)",
                "Tests: 1 failed, 40 passed, 41 total",
            ]
        )
        c, n = collapse_passing(jest)
        assert "✕ handles error" in c and "Tests: 1 failed, 40 passed, 41 total" in c and n == 41
