"""Resource identity, cross-turn delta encoding and pre-context admission."""

from __future__ import annotations

import json

import pytest

from headroom.cache.compression_store import CompressionStore
from headroom.intelligence.admission import IntelligencePrepTransform
from headroom.intelligence.config import IntelligenceConfig
from headroom.intelligence.delta import compute_delta
from headroom.intelligence.invariants import ccr_hashes_in
from headroom.intelligence.resources import canonical_path, content_hash, resource_for
from headroom.intelligence.task_context import build_task_context
from headroom.proxy.auth_mode import AuthMode
from headroom.tokenizer import Tokenizer
from headroom.tokenizers import get_tokenizer
from headroom.transforms.compression_policy import policy_for_mode

TOK = Tokenizer(get_tokenizer("gpt-4o"), "gpt-4o")
PAYG = policy_for_mode(AuthMode.PAYG)


def cfg(**env: str) -> IntelligenceConfig:
    return IntelligenceConfig.from_env({"HEADROOM_INTELLIGENCE": "full", **env})


def tool_turn(
    call_id: str, name: str, inp: dict, result: str, *, is_error: bool = False
) -> list[dict]:
    block = {"type": "tool_result", "tool_use_id": call_id, "content": result}
    if is_error:
        block["is_error"] = True
    return [
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": call_id, "name": name, "input": inp}],
        },
        {"role": "user", "content": [block]},
    ]


def pytest_output(failing: list[str], passing: int = 400) -> str:
    lines = [f"tests/test_mod.py::test_case_{i} PASSED" for i in range(passing)]
    for f in failing:
        lines.append(f"tests/test_mod.py::{f} FAILED")
    lines.append(f"===== {len(failing)} failed, {passing} passed in 12.31s =====")
    return "\n".join(lines)


def run(transform, messages, **kw):
    ctx = build_task_context(messages)
    return transform.apply(messages, TOK, task_context=ctx, compression_policy=PAYG, **kw)


def result_text(messages, index):
    return messages[index]["content"][0]["content"]


# ------------------------------------------------------------------ identity
class TestResourceIdentity:
    def test_windows_and_posix_path_spellings(self):
        assert canonical_path("C:\\Repo\\Src\\A.py") == canonical_path("c:/repo/src/a.py")
        assert canonical_path("src/./pkg/../a.py") == "src/a.py"
        a = resource_for("Read", {"file_path": "C:\\Repo\\src\\a.py"})
        b = resource_for("Read", {"file_path": "c:/repo/SRC/a.py"})
        assert a and b and a.identity == b.identity and a.is_read

    def test_shell_read_shares_file_identity(self):
        a = resource_for("Read", {"file_path": "src/a.py"})
        b = resource_for("Bash", {"command": "cd /x && cat src/a.py"})
        c = resource_for("shell", {"command": ["powershell", "-Command", "Get-Content src\\a.py"]})
        assert b and b.kind == "file" and b.is_read and c and c.kind == "file"
        assert canonical_path("src/a.py") in b.identity and canonical_path("src/a.py") in c.identity
        assert a is not None

    def test_search_and_command_identity(self):
        g1 = resource_for("Grep", {"pattern": "foo", "path": "src"})
        g2 = resource_for("Grep", {"pattern": "foo", "path": "src/"})
        g3 = resource_for("Grep", {"pattern": "bar", "path": "src"})
        assert g1.identity == g2.identity != g3.identity
        c1 = resource_for("Bash", {"command": "cd /repo &&  pytest -q"})
        c2 = resource_for("Bash", {"command": "pytest   -q"})
        assert c1.identity == c2.identity and not c1.is_read

    def test_symlink_canonical(self, tmp_path):
        real = tmp_path / "real.py"
        real.write_text("x")
        link = tmp_path / "link.py"
        try:
            link.symlink_to(real)
        except (OSError, NotImplementedError):
            pytest.skip("symlinks unavailable")
        assert canonical_path(str(link)) == canonical_path(str(real))

    def test_crlf_hash_equal(self):
        assert content_hash("a\r\nb\r\n") == content_hash("a\nb\n")


# --------------------------------------------------------------------- delta
class TestDeltaBodies:
    def test_json_keyed(self):
        base = json.dumps([{"id": i, "v": i} for i in range(50)])
        cur = json.loads(base)
        cur[3]["v"] = 999
        cur.append({"id": 50, "v": 50})
        del cur[10]
        body = compute_delta(base, json.dumps(cur), "tool")
        assert body.strategy == "json_keyed"
        assert (body.added, body.removed, body.changed) == (1, 1, 1)
        assert '"v":999' in body.body and "- id=10" in body.body

    def test_line_set_for_search(self):
        base = "\n".join(f"src/a.py:{i}:x" for i in range(100))
        cur = base + "\nsrc/b.py:1:new"
        body = compute_delta(base, cur, "search")
        assert body.strategy == "line_set" and body.added == 1 and body.removed == 0

    def test_identical(self):
        assert compute_delta("a\r\nb", "a\nb", "cmd").strategy == "identical"


# ----------------------------------------------------------------- admission
class TestDeltaStage:
    def _convo(self, first: str, second: str, user: str = "fix the failing tests"):
        return [
            {"role": "user", "content": user},
            *tool_turn("t1", "Bash", {"command": "pytest -q"}, first),
            *tool_turn("t2", "Bash", {"command": "pytest -q"}, second),
        ]

    def test_repeated_test_run_becomes_delta_and_is_recoverable(self):
        store = CompressionStore(max_entries=50)
        tr = IntelligencePrepTransform(cfg(HEADROOM_ADMISSION="0"), store_provider=lambda: store)
        first = pytest_output(["test_case_a", "test_case_b"])
        second = pytest_output(["test_case_b"])
        msgs = self._convo(first, second)
        res = run(tr, msgs)
        out = result_text(res.messages, 4)
        assert out.startswith("headroom delta:")
        assert "test_case_a FAILED" in out  # the fixed test shows as removed
        assert "1 failed, 400 passed" in out  # new summary kept exactly
        assert TOK.count_text(out) < TOK.count_text(second) * 0.1
        hashes = ccr_hashes_in(out)
        assert hashes and store.retrieve(hashes[0]).original_content == second
        assert result_text(res.messages, 2) == first  # base untouched
        # Next turn, different task text: the same tool result keeps identical bytes.
        msgs2 = [
            *msgs,
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "now explain test_case_b in depth please"},
        ]
        res2 = run(tr, msgs2)
        assert result_text(res2.messages, 4) == out

    def test_partial_base_never_used(self):
        tr = IntelligencePrepTransform(
            cfg(HEADROOM_ADMISSION="0"), store_provider=lambda: CompressionStore()
        )
        first = pytest_output(["test_case_a"]) + "\n[... 900 lines truncated ...]"
        msgs = self._convo(first, pytest_output(["test_case_a"]))
        res = run(tr, msgs)
        assert not result_text(res.messages, 4).startswith("headroom delta:")

    def test_file_reads_protected_unless_opted_in(self):
        code = "\n".join(f"def f{i}(x):\n    return x + {i}" for i in range(300))
        code2 = code.replace("return x + 7\n", "return x + 70\n")
        msgs = [
            {"role": "user", "content": "update f7"},
            *tool_turn("r1", "Read", {"file_path": "src/m.py"}, code),
            *tool_turn("r2", "Read", {"file_path": "src/m.py"}, code2),
        ]
        store = CompressionStore()
        off = run(
            IntelligencePrepTransform(cfg(HEADROOM_ADMISSION="0"), store_provider=lambda: store),
            msgs,
        )
        assert result_text(off.messages, 4) == code2
        on = run(
            IntelligencePrepTransform(
                cfg(HEADROOM_ADMISSION="0", HEADROOM_DELTA_READS="1"), store_provider=lambda: store
            ),
            msgs,
        )
        out = result_text(on.messages, 4)
        assert out.startswith("headroom delta:") and "return x + 70" in out

    def test_frozen_and_cache_control_untouched(self):
        tr = IntelligencePrepTransform(
            cfg(HEADROOM_ADMISSION="0"), store_provider=lambda: CompressionStore()
        )
        first, second = pytest_output(["a"]), pytest_output(["b"])
        msgs = self._convo(first, second)
        frozen = run(tr, msgs, frozen_message_count=len(msgs))
        assert frozen.messages == msgs
        msgs_cc = self._convo(first, second)
        msgs_cc[4]["content"][0]["cache_control"] = {"type": "ephemeral"}
        assert (
            run(
                IntelligencePrepTransform(
                    cfg(HEADROOM_ADMISSION="0"), store_provider=lambda: CompressionStore()
                ),
                msgs_cc,
            ).messages
            == msgs_cc
        )

    def test_no_markers_when_retrieval_impossible(self):
        msgs = self._convo(pytest_output(["a"]), pytest_output(["b"]))
        tr = IntelligencePrepTransform(
            cfg(), store_provider=lambda: CompressionStore(), markers_enabled=False
        )
        assert run(tr, msgs).messages == msgs
        tr2 = IntelligencePrepTransform(cfg(), store_provider=lambda: CompressionStore())
        assert run(tr2, msgs, cross_turn_dedup_recoverable=False).messages == msgs

    def test_store_failure_forwards_original(self):
        class Broken(CompressionStore):
            def store(self, *a, **k):
                raise OSError("disk full")

        msgs = self._convo(pytest_output(["a"]), pytest_output(["b"]))
        tr = IntelligencePrepTransform(cfg(), store_provider=lambda: Broken())
        assert run(tr, msgs).messages == msgs


class TestExternalization:
    def test_large_prose_output_gets_task_relevant_preview(self):
        paras = [
            f"Section {i}: routine notes about module {i} and its configuration."
            for i in range(600)
        ]
        paras[417] = (
            "Section 417: the RATE_LIMIT_EXCEEDED error is raised by billing_gateway.charge()."
        )
        doc = "\n\n".join(paras)
        msgs = [
            {"role": "user", "content": "Where is RATE_LIMIT_EXCEEDED raised?"},
            *tool_turn("w1", "WebDocs", {"query": "rate limits"}, doc),
        ]
        store = CompressionStore()
        tr = IntelligencePrepTransform(cfg(HEADROOM_DELTA="0"), store_provider=lambda: store)
        res = run(tr, msgs)
        out = result_text(res.messages, 2)
        assert out.startswith("headroom preview:")
        assert "billing_gateway.charge()" in out
        assert res.tokens_after < res.tokens_before * 0.3
        h = ccr_hashes_in(out)[0]
        assert store.retrieve(h).original_content == doc

    def test_small_and_error_outputs_untouched(self):
        msgs = [
            {"role": "user", "content": "run it"},
            *tool_turn(
                "e1",
                "Bash",
                {"command": "make"},
                "error: boom\n" * 30 + "exit code 2",
                is_error=True,
            ),
        ]
        tr = IntelligencePrepTransform(cfg(), store_provider=lambda: CompressionStore())
        assert run(tr, msgs).messages == msgs

    def test_partial_output_keeps_truncation_notice(self):
        body = "\n\n".join(f"note {i} about things" for i in range(2000)) + "\n\n(output truncated)"
        msgs = [
            {"role": "user", "content": "summarize"},
            *tool_turn("p1", "Fetch", {"url": "x"}, body),
        ]
        res = run(
            IntelligencePrepTransform(
                cfg(HEADROOM_DELTA="0"), store_provider=lambda: CompressionStore()
            ),
            msgs,
        )
        out = result_text(res.messages, 2)
        assert "partial input" in out and "truncated" in out

    def test_read_never_externalized(self):
        code = "\n".join(f"def g{i}():\n    pass" for i in range(3000))
        msgs = [
            {"role": "user", "content": "look"},
            *tool_turn("r", "Read", {"file_path": "big.py"}, code),
        ]
        res = run(IntelligencePrepTransform(cfg(), store_provider=lambda: CompressionStore()), msgs)
        assert res.messages == msgs
