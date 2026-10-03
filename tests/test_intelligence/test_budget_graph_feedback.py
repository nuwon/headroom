"""Budget allocator (8), code graph (11), retention learning (12), speculative prep (15)."""

from __future__ import annotations

import time

from headroom.cache import compression_store as cs
from headroom.intelligence.budget import MAX_BIAS, MIN_BIAS, allocate
from headroom.intelligence.feedback import NEUTRAL, RetentionLearner, feature_key
from headroom.intelligence.graph import CodeGraph, GraphRegistry, parse_source
from headroom.intelligence.resources import resource_for
from headroom.intelligence.speculative import SpeculativePreparer
from headroom.intelligence.task_context import build_task_context


def tok(text: str) -> int:
    return max(1, len(text) // 4)


def turn(i: int, name: str, inp: dict, result: str, *, is_error: bool = False) -> list[dict]:
    block = {"type": "tool_result", "tool_use_id": f"c{i}", "content": result}
    if is_error:
        block["is_error"] = True
    return [
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": f"c{i}", "name": name, "input": inp}],
        },
        {"role": "user", "content": [block]},
    ]


# --------------------------------------------------------------------- budget
class TestBudget:
    def _mixed(self):
        logs = "\n".join(f"INFO request {i} served in 3ms" for i in range(4000))
        grep = "\n".join(f"src/m{i}.py:{i}: value = {i}" for i in range(2000))
        json_rows = "[" + ",".join(f'{{"id": {i}, "ok": true}}' for i in range(3000)) + "]"
        error = 'Traceback (most recent call last):\n  File "billing.py", line 9\nValueError: invoice INV-7781 rejected'
        msgs = [
            {"role": "user", "content": "why was invoice INV-7781 rejected?"},
            *turn(1, "Bash", {"command": "tail app.log"}, logs),
            *turn(2, "Grep", {"pattern": "value"}, grep),
            *turn(3, "Api", {"q": "rows"}, json_rows),
            *turn(4, "Bash", {"command": "python billing.py"}, error, is_error=True),
            *turn(5, "Bash", {"command": "cat notes.txt"}, "misc notes " * 50),
        ]
        return msgs

    def test_pressure_tightens_bulk_but_protects_evidence(self):
        msgs = self._mixed()
        task = build_task_context(msgs)
        alloc = allocate(
            msgs, count_tokens=tok, task=task, context_limit=40_000, pressure_threshold=0.5
        )
        assert alloc.pressure >= 0.5
        error_idx = 8  # the error tool result message
        assert alloc.biases.get(error_idx, 1.0) >= 1.3
        bulk = [alloc.biases.get(i, 1.0) for i in (2, 4, 6)]
        assert any(b < 1.0 for b in bulk)
        assert all(MIN_BIAS <= b <= MAX_BIAS for b in alloc.biases.values())

    def test_no_pressure_never_tightens(self):
        msgs = self._mixed()
        task = build_task_context(msgs)
        alloc = allocate(msgs, count_tokens=tok, task=task, context_limit=10_000_000)
        assert all(b >= 1.0 for b in alloc.biases.values())

    def test_diversity_cap(self):
        msgs = self._mixed()
        task = build_task_context(msgs)
        alloc = allocate(msgs, count_tokens=tok, task=task, context_limit=40_000, diversity_cap=0.3)
        target = alloc.available_tokens * 0.5
        for b in alloc.blocks:
            assert b.keep_fraction * b.tokens <= 0.3 * target + 1

    def test_frozen_prefix_ignored(self):
        msgs = self._mixed()
        task = build_task_context(msgs)
        alloc = allocate(
            msgs, count_tokens=tok, task=task, context_limit=40_000, frozen_message_count=len(msgs)
        )
        assert alloc.biases == {}


# ---------------------------------------------------------------------- graph
class TestGraph:
    def test_parse_multiple_languages(self):
        assert {"charge", "Gateway"} <= parse_source(
            "a.py", "class Gateway:\n    def charge(self):\n        pass\n"
        ).defines
        assert (
            "handler"
            in parse_source(
                "a.ts", "export async function handler(req) {}\nimport x from './billing'"
            ).defines
        )
        assert "Run" in parse_source("a.go", "func Run() {}\n").defines
        assert "parse" in parse_source("a.rs", "pub fn parse() {}\nuse crate::util;").defines
        assert (
            "Charge"
            in parse_source("A.cs", "public class Billing {\n  public void Charge() {}\n}").defines
        )

    def test_neighborhood_bfs(self):
        g = CodeGraph("w")
        g.update_file(
            "src/billing.py",
            "from src.gateway import Gateway\n\ndef charge():\n    return Gateway().send()\n",
        )
        g.update_file(
            "src/gateway.py", "class Gateway:\n    def send(self):\n        return log_event()\n"
        )
        g.update_file("src/log.py", "def log_event():\n    pass\n")
        g.update_file("src/unrelated.py", "def zebra():\n    pass\n")
        files, syms = g.neighborhood(files=["src/billing.py"])
        assert "src/gateway.py" in files and "src/log.py" in files
        assert "src/unrelated.py" not in files
        assert "Gateway" in syms
        assert g.proximity("error in gateway.py line 3") > 0

    def test_incremental_reparse_only_on_change(self):
        g = CodeGraph("w")
        assert g.update_file("a.py", "def alpha():\n    pass\n")
        assert not g.update_file("a.py", "def alpha():\n    pass\n")
        assert g.update_file("a.py", "def beta():\n    pass\n")
        _, syms = g.neighborhood(files=["a.py"])
        assert "beta" in syms and "alpha" not in syms

    def test_ingest_from_messages_and_windows_paths(self):
        msgs = [
            {"role": "user", "content": "fix charge"},
            *turn(
                1,
                "Read",
                {"file_path": "C:\\Repo\\src\\billing.py"},
                "     1→def charge():\n     2→    return 1\n",
            ),
        ]
        g = CodeGraph("w")
        assert g.ingest_messages(msgs) == 1
        files, syms = g.neighborhood(files=["c:/repo/src/billing.py"])
        assert "charge" in syms

    def test_vendor_ignored_and_registry_scoped(self):
        g = CodeGraph("w")
        assert not g.update_file("node_modules/x/index.js", "function f(){}")
        reg = GraphRegistry()
        assert reg.for_workspace("a") is not reg.for_workspace("b")
        assert reg.for_workspace("a") is reg.for_workspace("a")


# ------------------------------------------------------------------- feedback
class TestFeedback:
    def test_retrieval_lowers_prior_after_min_obs(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HEADROOM_INTELLIGENCE_DIR", str(tmp_path))
        learner = RetentionLearner(min_observations=3)
        key = feature_key("Bash", "indexed_preview")
        assert learner.prior(key) == NEUTRAL
        for i in range(5):
            learner.note_compressed(f"h{i}", key)
            learner.on_retrieval({"hash": f"h{i}", "retrieval_type": "full"})
        assert learner.prior(key) < 0.35
        assert learner.retention_bias("Bash", "indexed_preview") > 1.1
        learner.save()
        reloaded = RetentionLearner(min_observations=3)
        assert abs(reloaded.prior(key) - learner.prior(key)) < 1e-5

    def test_bounded_and_rewarded_when_never_retrieved(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HEADROOM_INTELLIGENCE_DIR", str(tmp_path))
        learner = RetentionLearner(min_observations=1, persist=False)
        key = feature_key("Api", "smart_crusher")
        for i in range(50):
            learner.note_compressed(f"x{i}", key)
        with learner._lock:  # age them
            for h, (k, _c, a) in list(learner._tracked.items()):
                learner._tracked[h] = (k, time.time() - 10_000, a)
        assert learner.sweep(ttl_s=1800) == 50
        assert learner.prior(key) <= 0.95 and learner.prior(key) > 0.9
        assert 0.8 <= learner.retention_bias("Api", "smart_crusher") <= 1.0

    def test_reread_penalizes_only_reads_searches(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HEADROOM_INTELLIGENCE_DIR", str(tmp_path))
        learner = RetentionLearner(min_observations=1, persist=False)
        read = resource_for("Read", {"file_path": "a.py"})
        cmd = resource_for("Bash", {"command": "pytest"})
        learner.observe_rewrite(read, "intel:delta")
        learner.observe_access(read)
        learner.observe_rewrite(cmd, "intel:delta")
        learner.observe_access(cmd)
        assert learner.stats()["keys"] == 1

    def test_store_listener_receives_no_payload(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HEADROOM_INTELLIGENCE_DIR", str(tmp_path))
        seen = []
        cs.add_retrieval_listener(seen.append)
        try:
            store = cs.CompressionStore()
            h = store.store("secret original", "c", tool_name="Bash", compression_strategy="x")
            store.retrieve(h)
        finally:
            cs.remove_retrieval_listener(seen.append)
        assert seen and seen[0]["hash"] == h
        assert "secret original" not in repr(seen[0])

    def test_advisor_told_about_bad_advised_outcome(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HEADROOM_INTELLIGENCE_DIR", str(tmp_path))
        calls = []

        class Adv:
            def record_bad_outcome(self, family, retrieval=True):
                calls.append(family)

        learner = RetentionLearner(persist=False, advisor_provider=lambda: Adv())
        learner.note_compressed("hh", "k", advised=True)
        learner.on_retrieval({"hash": "hh"})
        assert calls


# ---------------------------------------------------------------- speculative
class TestSpeculative:
    def test_prepare_is_bounded_and_cached(self):
        prep = SpeculativePreparer(workers=1, max_pending=2)
        msgs = []
        for i in range(6):
            msgs += turn(i, "Bash", {"command": f"cmd{i}"}, f"line {i}\n" * 1000)
        queued = prep.prepare(msgs)
        assert queued <= 2 + prep.stats()["completed"] + 6
        deadline = time.time() + 10
        while prep.stats()["pending"] and time.time() < deadline:
            time.sleep(0.05)
        st = prep.stats()
        assert st["pending"] == 0 and st["failed"] == 0
        assert st["submitted"] + st["rejected"] >= 6
        text = msgs[1]["content"][0]["content"]
        assert prep.invariants_for(text) is not None
        prep.shutdown()

    def test_never_blocks_after_shutdown(self):
        prep = SpeculativePreparer()
        prep.shutdown()
        assert prep.prepare(turn(1, "Bash", {"command": "x"}, "y" * 5000)) == 0
