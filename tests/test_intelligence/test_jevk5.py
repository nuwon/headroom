"""JevK5 + llama.cpp integration tests.

Subprocess-level: a fake ``llama-server`` (tests/test_intelligence/fixtures)
is launched as a real child process, so discovery, port reservation,
process-group start/stop, readiness polling and the HTTP protocol are
exercised for real on both POSIX and Windows (a ``.bat`` shim on Windows).
"""

from __future__ import annotations

import json
import os
import socket
import stat
import sys
import threading
import time
import urllib.request
from dataclasses import replace
from pathlib import Path

import pytest

from headroom._subprocess import pid_alive
from headroom.intelligence import jevk5_protocol as proto
from headroom.intelligence.advisor import DecisionAdvisor
from headroom.intelligence.bootstrap import run_setup, verify_protocol
from headroom.intelligence.config import IntelligenceConfig
from headroom.intelligence.decision_gateway import answer_batch, serve
from headroom.intelligence.jevk5_client import (
    DecisionClientError,
    GGUFDecisionClient,
    SystemOneClient,
)
from headroom.intelligence.jevk5_service import JevK5Service, reserve_port
from headroom.intelligence.llama_discovery import discover, probe
from headroom.intelligence.models import AdvisoryRequest, DecisionFamily, QuestionType
from headroom.intelligence.state import models_dir, read_runtime, runtime_path

FIXTURE = Path(__file__).parent / "fixtures" / "fake_llama_server.py"
IS_WINDOWS = sys.platform == "win32"


def make_fake_llama(directory: Path, name: str = "llama-server") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    if IS_WINDOWS:
        exe = directory / f"{name}.bat"
        exe.write_text(f'@"{sys.executable}" "{FIXTURE}" %*\r\n', encoding="utf-8")
        return exe
    exe = directory / name
    exe.write_text(f"#!{sys.executable}\n" + FIXTURE.read_text(encoding="utf-8"), encoding="utf-8")
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return exe


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.setenv("HEADROOM_INTELLIGENCE_DIR", str(tmp_path / "intel"))
    monkeypatch.setenv("LLAMA_CACHE", str(tmp_path / "llama-cache"))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hf-cache"))
    for var in ("FAKE_LLAMA_OOM", "FAKE_LLAMA_LOAD_S", "FAKE_LLAMA_OLD", "LLAMA_CPP_HOME"):
        monkeypatch.delenv(var, raising=False)
    yield


def settings(**overrides):  # type: ignore[no-untyped-def]
    base = IntelligenceConfig.from_env({"HEADROOM_JEVK5": "on"}).jevk5
    return replace(base, **overrides)


def put_cached_model(s) -> Path:  # type: ignore[no-untyped-def]
    models_dir().mkdir(parents=True, exist_ok=True)
    path = models_dir() / s.model_file
    path.write_bytes(b"GGUF-fake")
    return path


# ------------------------------------------------------------------ protocol
class TestProtocolParity:
    def test_upstream_installed_or_mirror(self):
        assert proto.protocol_source().startswith(("upstream:", "mirror:"))

    def test_mirror_matches_upstream(self):
        upstream = pytest.importorskip("jevk5.prompt")
        for state, crit, opts in [
            ("evidence ü", "pick", ["a: one", "b: two"]),
            ({"k": [1, 2]}, "which?", [f"{i}: x{i}" for i in range(16)]),
        ]:
            assert proto._mirror_prompt_text(state, crit, opts) == upstream.prompt_text(
                state, crit, opts
            )
        for q in (
            {"type": "noul", "instructions": "x"},
            {"type": "choice", "instructions": "x", "criteria": ["a", "b", "c"]},
            {"type": "choice", "instructions": "x", "criteria": {"a": "A", "b": None}},
            {"type": "score", "instructions": "x", "criteria": ["lo", "mid", "hi"]},
        ):
            assert proto._mirror_decision_options(q) == upstream.decision_options(q)

        def reader(texts):
            raw = [1.0 / (1 + (len(t) * 7 + i * 13) % 17) for i, t in enumerate(texts)]
            s = sum(raw)
            return [r / s for r in raw]

        for n in (3, 16, 17, 40, 300):
            texts = [f"opt{i}: description {i * 31 % 97}" for i in range(n)]
            for method in ("knockout", "tree"):
                ours = proto._mirror_spread(
                    reader, texts, method, 0.93 if method == "knockout" else None
                )
                theirs = upstream.spread(
                    reader, texts, method, 0.93 if method == "knockout" else None
                )
                assert ours == pytest.approx(theirs, abs=1e-12)
                assert sum(ours) == pytest.approx(1.0)

    def test_letter_distribution_missing_floor(self):
        probs, missing = proto.letter_distribution({"A": -0.1, "C": -3.0}, 3, 1.22)
        assert missing and abs(sum(probs) - 1) < 1e-12 and probs[0] > probs[2] > probs[1]

    def test_validate_question(self):
        with pytest.raises(ValueError):
            proto.validate_question({"type": "choice", "instructions": "x", "criteria": ["only"]})
        with pytest.raises(ValueError):
            proto.validate_question({"type": "maybe", "instructions": "x"})


# ------------------------------------------------------------------ discovery
class TestDiscovery:
    def test_path_discovery_and_capabilities(self, tmp_path, monkeypatch):
        exe = make_fake_llama(tmp_path / "bin")
        monkeypatch.setenv("PATH", str(exe.parent) + os.pathsep + os.environ.get("PATH", ""))
        result = discover(environ=dict(os.environ))
        assert result.selected is not None, [p.reason for p in result.probed]
        caps = result.selected
        assert caps.source == "path" and caps.has_hf_repo and caps.has_hf_file and caps.has_ngl
        assert caps.ngl_auto and caps.build_number == 9999
        assert caps.devices and caps.devices[0].startswith("CPU")

    def test_explicit_path_with_spaces(self, tmp_path):
        exe = make_fake_llama(tmp_path / "dir with spaces" / "llama cpp")
        result = discover(explicit=str(exe), environ={"PATH": ""})
        assert result.selected is not None and result.selected.source == "explicit"

    def test_llama_cpp_home_layout(self, tmp_path):
        home = tmp_path / "llama.cpp"
        rel = Path("build/bin/Release") if IS_WINDOWS else Path("build/bin")
        make_fake_llama(home / rel)
        result = discover(environ={"PATH": "", "LLAMA_CPP_HOME": str(home)})
        assert result.selected is not None and result.selected.source == "llama_cpp_home"

    def test_old_binary_rejected_with_reason(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FAKE_LLAMA_OLD", "1")
        caps = probe(make_fake_llama(tmp_path / "old"))
        assert not caps.ok and "--hf-repo" in caps.reason

    def test_missing_explicit_reported(self, tmp_path):
        result = discover(explicit=str(tmp_path / "nope"), environ={"PATH": ""})
        assert result.selected is None and result.probed[0].reason == "not found"


# ------------------------------------------------------------------ service
class TestService:
    def test_reserve_port_skips_busy(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as busy:
            busy.bind(("127.0.0.1", 0))
            busy.listen(1)
            taken = busy.getsockname()[1]
            assert reserve_port(taken, span=3) != taken

    def test_start_reuse_stop(self, tmp_path):
        exe = make_fake_llama(tmp_path / "bin")
        s = settings(llama_server=str(exe))
        put_cached_model(s)
        svc = JevK5Service(s)
        status = svc.start(wait=True, wait_timeout_s=30)
        try:
            assert status.state == "running", status.reason
            assert status.url.startswith("http://127.0.0.1:")
            record = read_runtime()
            assert record and record["pid"] == status.pid and record["owned"] is True
            # A second service object reuses the live instance instead of starting another,
            # and does not claim ownership while the owner process is alive.
            other = JevK5Service(s)
            reused = other.start(wait=True, wait_timeout_s=10)
            assert reused.url == status.url and reused.reason == "reused" and reused.owned is False
            assert other.stop() is False  # non-owner never kills it
        finally:
            assert svc.stop() is True
        deadline = time.monotonic() + 10
        while pid_alive(status.pid) and time.monotonic() < deadline:
            time.sleep(0.1)
        assert not pid_alive(status.pid)
        assert not runtime_path().exists()

    def test_auto_mode_never_downloads(self, tmp_path):
        exe = make_fake_llama(tmp_path / "bin")
        s = replace(settings(llama_server=str(exe)), mode="auto", allow_download=False)
        status = JevK5Service(s).start(wait=True, wait_timeout_s=5)
        assert status.state == "unavailable" and "not downloaded" in status.reason

    def test_no_llama_server(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PATH", "")
        s = settings(llama_server=str(tmp_path / "missing"), allow_build=False)
        status = JevK5Service(s).start(wait=True, wait_timeout_s=5, allow_build=False)
        assert status.state == "unavailable"

    def test_oom_retries_on_cpu(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FAKE_LLAMA_OOM", "1")
        exe = make_fake_llama(tmp_path / "bin")
        s = settings(llama_server=str(exe))
        put_cached_model(s)
        svc = JevK5Service(s)
        status = svc.start(wait=True, wait_timeout_s=30)
        try:
            assert status.state == "running", status.reason
        finally:
            svc.stop()

    def test_waits_for_model_load(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FAKE_LLAMA_LOAD_S", "1.5")
        exe = make_fake_llama(tmp_path / "bin")
        s = settings(llama_server=str(exe))
        put_cached_model(s)
        svc = JevK5Service(s)
        try:
            status = svc.start(wait=True, wait_timeout_s=30)
            assert status.state == "running"
        finally:
            svc.stop()

    def test_background_start(self, tmp_path):
        exe = make_fake_llama(tmp_path / "bin")
        s = settings(llama_server=str(exe))
        put_cached_model(s)
        svc = JevK5Service(s)
        try:
            first = svc.start(wait=False)
            assert first.state in ("starting", "running")
            deadline = time.monotonic() + 30
            while svc.status.state != "running" and time.monotonic() < deadline:
                time.sleep(0.1)
            assert svc.status.state == "running"
        finally:
            svc.stop()

    def test_stale_runtime_cleaned(self):
        runtime_path().parent.mkdir(parents=True, exist_ok=True)
        runtime_path().write_text(json.dumps({"pid": 2**22 + 12345, "url": "http://127.0.0.1:1"}))
        assert read_runtime() is None and not runtime_path().exists()


# ------------------------------------------------------------- live protocol
@pytest.fixture
def running(tmp_path):
    exe = make_fake_llama(tmp_path / "bin")
    s = settings(llama_server=str(exe))
    put_cached_model(s)
    svc = JevK5Service(s)
    status = svc.start(wait=True, wait_timeout_s=30)
    assert status.usable, status.reason
    yield s, status.url, svc
    svc.stop()


class TestClientAndGateway:
    def test_client_decisions_normalized(self, running):
        s, url, _ = running
        client = GGUFDecisionClient(
            url,
            temperature=s.temperature,
            knockout_temperature=s.knockout_temperature,
            timeout_s=10,
        )
        for q in (
            {"type": "noul", "instructions": "Is it true?"},
            {"type": "choice", "instructions": "Pick", "criteria": {"a": "A", "b": "B", "c": "C"}},
            {"type": "score", "instructions": "Rate", "criteria": ["0", "1", "2", "3", "4"]},
            {"type": "choice", "instructions": "Many", "criteria": [f"o{i}" for i in range(40)]},
        ):
            ans = client.decide("state", q)
            probs = ans.get("probabilities") or {"true": ans["noul"], "false": 1 - ans["noul"]}
            assert abs(sum(probs.values()) - 1.0) < 1e-6 and 0 < ans["confidence"] <= 1

    def test_client_matches_upstream_jevk5gguf(self, running):
        jevk5 = pytest.importorskip("jevk5")
        s, url, _ = running
        ours = GGUFDecisionClient(url, temperature=1.22, knockout_temperature=0.93, timeout_s=10)
        theirs = jevk5.JevK5GGUF(url, temperature=1.22, knockout_temperature=0.93, timeout_s=10)
        for q in (
            {"type": "choice", "instructions": "Pick one", "criteria": ["x", "y", "z"]},
            {"type": "choice", "instructions": "Many", "criteria": [f"o{i}" for i in range(33)]},
        ):
            a, _ = ours.probabilities("st", q)
            b, _ = theirs.probabilities("st", q)
            assert a == pytest.approx(b, abs=1e-12)

    def test_verify_protocol(self, running):
        from headroom.intelligence.bootstrap import SetupReport

        s, url, _ = running
        report = SetupReport()
        assert verify_protocol(url, s, report), [c.__dict__ for c in report.checks if not c.ok]

    def test_gateway_roundtrip_and_systemone_client(self, running):
        s, url, _ = running
        client = GGUFDecisionClient(url, temperature=1.22, knockout_temperature=0.93, timeout_s=10)
        server = serve(client)
        try:
            base = f"http://127.0.0.1:{server.server_address[1]}"
            with urllib.request.urlopen(f"{base}/health", timeout=5) as r:
                assert json.loads(r.read())["ok"] is True
            remote = SystemOneClient(base, timeout_s=10)
            ans = remote.decide("st", {"type": "noul", "instructions": "ok?"})
            assert 0 <= ans["noul"] <= 1 and "input_tokens" in ans
            status, payload = answer_batch(
                client, {"state": "x", "questions": {"q": {"type": "bogus"}}}
            )
            assert status == 400
        finally:
            server.shutdown()

    def test_run_setup_end_to_end(self, tmp_path):
        exe = make_fake_llama(tmp_path / "bin2")
        s = settings(llama_server=str(exe))
        put_cached_model(s)
        report = run_setup(
            s, install_package=False, allow_build=False, allow_download=False, wait_timeout_s=30
        )
        assert report.ok, [c.__dict__ for c in report.checks if not c.ok]
        assert read_runtime() is None  # stopped after verification (no --keep-running)


# -------------------------------------------------------------------- advisor
class _FakeClient:
    def __init__(self, answer=None, delay=0.0, fail=False):
        self.answer = answer or {
            "type": "choice",
            "confidence": 0.9,
            "choice": "a",
            "probabilities": {"a": 0.9, "b": 0.1},
            "input_tokens": 10,
        }
        self.delay = delay
        self.fail = fail
        self.calls = 0

    def decide(self, state, question):
        self.calls += 1
        if self.delay:
            time.sleep(self.delay)
        if self.fail:
            raise DecisionClientError("boom")
        return dict(self.answer)


def _req(state="the state", options=None):
    return AdvisoryRequest(
        DecisionFamily.TRANSFORM_CANDIDATE,
        QuestionType.CHOICE,
        state,
        "pick",
        options=options or {"a": "A", "b": "B"},
    )


def _advisor(client, **env):
    cfg = IntelligenceConfig.from_env(
        {"HEADROOM_INTELLIGENCE": "safe", "HEADROOM_JEVK5": "on", **env}
    )
    return DecisionAdvisor(cfg, client=client)


class TestAdvisor:
    def test_basic_and_cache(self):
        client = _FakeClient()
        adv = _advisor(client)
        with adv.request_scope():
            a = adv.advise(_req())
            b = adv.advise(_req())
        assert a is not None and a.weight == pytest.approx(0.25) and b is not None and b.cached
        assert client.calls == 1

    def test_timeout_fails_open_fast(self):
        adv = _advisor(_FakeClient(delay=2.0), HEADROOM_JEVK5_TIMEOUT_MS="100")
        started = time.perf_counter()
        with adv.request_scope():
            assert adv.advise(_req()) is None
        assert time.perf_counter() - started < 1.0
        assert adv.stats()["families"]["transform_candidate"]["timeouts"] == 1

    def test_budget(self):
        client = _FakeClient()
        adv = _advisor(client, HEADROOM_JEVK5_MAX_CALLS_PER_REQUEST="2")
        with adv.request_scope():
            results = [adv.advise(_req(state=f"s{i}")) for i in range(4)]
        assert sum(r is not None for r in results) == 2 and client.calls == 2

    def test_low_confidence_zero_weight(self):
        client = _FakeClient(
            answer={
                "type": "choice",
                "confidence": 0.4,
                "probabilities": {"a": 0.4, "b": 0.35, "c": 0.25},
                "input_tokens": 1,
            }
        )
        adv = _advisor(client)
        with adv.request_scope():
            r = adv.advise(_req(options={"a": "A", "b": "B", "c": "C"}))
        assert r is not None and r.weight == 0.0

    def test_non_english_skipped(self):
        client = _FakeClient()
        adv = _advisor(client)
        with adv.request_scope():
            assert adv.advise(_req(state="これは日本語の状態です。" * 20)) is None
        assert client.calls == 0

    def test_trust_lowers_weight_after_failures(self):
        failing = _FakeClient(fail=True)
        adv = _advisor(failing)
        for i in range(12):
            with adv.request_scope():
                adv.advise(_req(state=f"x{i}"))
        adv._client = _FakeClient()  # recovered
        with adv.request_scope():
            r = adv.advise(_req(state="fresh"))
        assert r is not None and r.weight < 0.25

    def test_state_bounded(self):
        seen = {}

        class C(_FakeClient):
            def decide(self, state, question):
                seen["len"] = len(state)
                return super().decide(state, question)

        adv = _advisor(C(), HEADROOM_JEVK5_MAX_STATE_TOKENS="300")
        with adv.request_scope():
            adv.advise(_req(state="word " * 10_000))
        assert seen["len"] <= 300 * 4 + 64

    def test_off_mode(self):
        cfg = IntelligenceConfig.from_env(
            {"HEADROOM_INTELLIGENCE": "safe", "HEADROOM_JEVK5": "off"}
        )
        adv = DecisionAdvisor(cfg, client=_FakeClient())
        assert adv.advise(_req()) is None

    def test_concurrent_requests_bounded(self):
        client = _FakeClient(delay=0.3)
        adv = _advisor(client, HEADROOM_JEVK5_TIMEOUT_MS="2000")
        results = []

        def go(i):
            with adv.request_scope():
                results.append(adv.advise(_req(state=f"c{i}")))

        threads = [threading.Thread(target=go, args=(i,)) for i in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert client.calls <= 6 and len(results) == 6
