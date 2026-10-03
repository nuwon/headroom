"""Quality gate over the context-intelligence eval corpus.

Runs ``benchmarks/intelligence_corpus.py`` through the proxy's real Claude
Code and Codex compression paths (configured like ``headroom proxy``) and
asserts, for every scenario:

* every fact the task depends on stays visible or exactly recoverable
  (retrieval marker whose stored original verifies byte-exact) in every
  posture;
* ``safe`` and ``full`` never send more tokens than ``off``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

BENCH = Path(__file__).resolve().parents[2] / "benchmarks"
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))

import intelligence_benchmark as bench  # noqa: E402
from intelligence_corpus import build_corpus  # noqa: E402

CORPUS = build_corpus()


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    from headroom.cache.compression_store import get_compression_store, reset_compression_store
    from headroom.ccr.tool_injection import set_ccr_search_enabled
    from headroom.intelligence.runtime import install_runtime
    from headroom.proxy.interceptors import enable_rich_interception

    workdir = str(tmp_path_factory.mktemp("intel-corpus"))
    rows: dict[tuple[str, str, str], tuple[int, int, int]] = {}
    try:
        for posture in bench.POSTURES:
            with bench._posture_env(posture, workdir):
                proxy = bench._build_proxy()
                for path, runner in (
                    ("claude_code", bench._run_anthropic),
                    ("codex", bench._run_codex),
                ):
                    for scenario in CORPUS:
                        reset_compression_store()
                        text, _before, after = runner(proxy, scenario)
                        _visible, recoverable = bench._facts(
                            scenario, text, get_compression_store()
                        )
                        rows[(posture, path, scenario.name)] = (after, recoverable, 0)
                if getattr(proxy, "intelligence", None) is not None:
                    proxy.intelligence.shutdown()
    finally:
        reset_compression_store()
        set_ccr_search_enabled(False)
        enable_rich_interception(False)
        install_runtime(None)
    return rows


@pytest.mark.parametrize("path", ["claude_code", "codex"])
@pytest.mark.parametrize("scenario", CORPUS, ids=lambda s: s.name)
def test_every_fact_survives_every_posture(results, path, scenario):
    for posture in bench.POSTURES:
        _after, recoverable, _ = results[(posture, path, scenario.name)]
        assert recoverable == len(scenario.facts), (posture, path, scenario.name)


@pytest.mark.parametrize("path", ["claude_code", "codex"])
@pytest.mark.parametrize("scenario", CORPUS, ids=lambda s: s.name)
def test_intelligence_never_sends_more_tokens(results, path, scenario):
    off = results[("off", path, scenario.name)][0]
    for posture in ("safe", "full"):
        assert results[(posture, path, scenario.name)][0] <= off, (posture, path, scenario.name)


def test_intelligence_saves_overall(results):
    for path in ("claude_code", "codex"):
        off = sum(v[0] for k, v in results.items() if k[0] == "off" and k[1] == path)
        safe = sum(v[0] for k, v in results.items() if k[0] == "safe" and k[1] == path)
        assert safe < off * 0.75, (path, off, safe)  # well over 25% fewer tokens
