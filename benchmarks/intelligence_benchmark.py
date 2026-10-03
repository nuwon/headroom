"""Context-intelligence benchmark: Headroom with intelligence off / safe / full.

Runs the deterministic eval corpus (``benchmarks/intelligence_corpus.py``)
through the proxy's real compression paths, configured exactly like
``headroom proxy`` (coding savings profile seeded, then env):

* Claude Code — ``proxy.anthropic_pipeline.apply(...)`` with the handler's
  kwargs (cold first turn: ``frozen_message_count=0``);
* Codex — ``proxy._compress_openai_responses_payload(...)`` (the HTTP and
  WebSocket Responses paths share it).

Per posture and scenario it reports:

* ``tokens_in`` / ``tokens_out`` / ``saved_pct`` of the whole request;
* ``facts_visible``: facts present verbatim in what the model sees;
* ``facts_recoverable``: visible, or retrievable through a retrieval marker
  whose stored original verifies byte-exact (``CompressionStore.verify_exact``);
* ``ms``: median wall time of the compression call.

Usage::

    python benchmarks/intelligence_benchmark.py                 # table
    python benchmarks/intelligence_benchmark.py --json out.json  # + raw rows
    python benchmarks/intelligence_benchmark.py --repeat 5       # steadier ms

ML compressors (Kompress) are used only when their model is installed; the
report states whether it was.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from intelligence_corpus import Scenario, build_corpus  # noqa: E402

POSTURES = ("off", "safe", "full")
MODEL_ANTHROPIC = "claude-sonnet-4-5"
MODEL_OPENAI = "gpt-5"


@contextlib.contextmanager
def _posture_env(posture: str, workdir: str):
    """Process env for one posture, restored afterwards."""
    saved = dict(os.environ)
    try:
        for key in list(os.environ):
            if key.startswith("HEADROOM_"):
                del os.environ[key]
        os.environ["HEADROOM_INTELLIGENCE"] = posture
        os.environ["HEADROOM_JEVK5"] = "off"
        os.environ["HEADROOM_INTELLIGENCE_DIR"] = os.path.join(workdir, f"intel-{posture}")
        os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
        if not KOMPRESS["enabled"]:
            # Deterministic, offline: never start a background model download
            # mid-run (that would change results between postures).
            os.environ["HEADROOM_DISABLE_KOMPRESS"] = "1"
        from headroom.agent_savings import seed_proxy_env_defaults

        seed_proxy_env_defaults()
        yield
    finally:
        os.environ.clear()
        os.environ.update(saved)


def _build_proxy() -> Any:
    from headroom.proxy.server import _proxy_config_from_env, create_app

    config = _proxy_config_from_env()
    config.cache_enabled = False
    config.rate_limit_enabled = False
    config.cost_tracking_enabled = False
    config.log_requests = False
    config.image_optimize = False
    return create_app(config).state.proxy


def _text_of(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _facts(scenario: Scenario, visible_text: str, store: Any) -> tuple[int, int]:
    from headroom.intelligence.invariants import ccr_hashes_in

    hashes = ccr_hashes_in(visible_text)
    originals = [t.output for t in scenario.turns]
    recover_pool: list[str] = []
    for h in hashes:
        for original in originals:
            try:
                if store.verify_exact(h, original):
                    recover_pool.append(original)
            except Exception:  # noqa: BLE001
                continue
    visible = sum(1 for f in scenario.facts if f in visible_text)
    recoverable = sum(
        1 for f in scenario.facts if f in visible_text or any(f in o for o in recover_pool)
    )
    return visible, recoverable


def _run_anthropic(proxy: Any, scenario: Scenario) -> tuple[str, int, int]:
    from headroom.agent_savings import proxy_pipeline_kwargs
    from headroom.utils import extract_user_query

    messages = scenario.anthropic_messages()
    result = proxy.anthropic_pipeline.apply(
        messages=copy.deepcopy(messages),
        model=MODEL_ANTHROPIC,
        model_limit=200_000,
        context=extract_user_query(messages),
        frozen_message_count=0,
        prefix_replay_guaranteed=True,
        request_id=f"bench-{scenario.name}",
        **proxy_pipeline_kwargs(proxy.config),
    )
    return _text_of(result.messages), result.tokens_before, result.tokens_after


def _run_codex(proxy: Any, scenario: Scenario) -> tuple[str, int, int]:
    payload = {"model": MODEL_OPENAI, "input": scenario.responses_input(), "stream": True}
    counter = proxy.openai_provider.get_token_counter(MODEL_OPENAI)
    before = counter.count_text(json.dumps(payload["input"]))
    out, *_ = proxy._compress_openai_responses_payload(
        copy.deepcopy(payload), model=MODEL_OPENAI, request_id=f"bench-{scenario.name}"
    )
    after = counter.count_text(json.dumps(out.get("input")))
    return _text_of(out.get("input")), before, after


def run(repeat: int = 3) -> list[dict[str, Any]]:
    from headroom.cache.compression_store import get_compression_store, reset_compression_store

    rows: list[dict[str, Any]] = []
    corpus = build_corpus()
    with tempfile.TemporaryDirectory(prefix="headroom-intel-bench-") as workdir:
        for posture in POSTURES:
            with _posture_env(posture, workdir):
                reset_compression_store()
                proxy = _build_proxy()
                for path, runner in (("claude_code", _run_anthropic), ("codex", _run_codex)):
                    for scenario in corpus:
                        timings: list[float] = []
                        text, before, after = "", 0, 0
                        for _ in range(max(1, repeat)):
                            reset_compression_store()
                            started = time.perf_counter()
                            text, before, after = runner(proxy, scenario)
                            timings.append((time.perf_counter() - started) * 1000.0)
                        visible, recoverable = _facts(scenario, text, get_compression_store())
                        rows.append(
                            {
                                "posture": posture,
                                "path": path,
                                "scenario": scenario.name,
                                "tokens_in": before,
                                "tokens_out": after,
                                "saved_pct": round(100.0 * (before - after) / before, 1)
                                if before
                                else 0.0,
                                "facts": len(scenario.facts),
                                "facts_visible": visible,
                                "facts_recoverable": recoverable,
                                "ms": round(statistics.median(timings), 1),
                            }
                        )
                with contextlib.suppress(Exception):
                    if getattr(proxy, "intelligence", None) is not None:
                        proxy.intelligence.shutdown()
    return rows


KOMPRESS = {"enabled": False}


def summarize(rows: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    lines.append(
        "Kompress (ML text compressor): "
        + (
            "enabled (if its model is cached)"
            if KOMPRESS["enabled"]
            else "disabled (--kompress to enable)"
        )
    )
    lines.append("")
    header = "| path | posture | tokens in | tokens out | saved | facts visible | facts recoverable | median ms |"
    lines.append(header)
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for path in ("claude_code", "codex"):
        for posture in POSTURES:
            sel = [r for r in rows if r["path"] == path and r["posture"] == posture]
            if not sel:
                continue
            t_in = sum(r["tokens_in"] for r in sel)
            t_out = sum(r["tokens_out"] for r in sel)
            facts = sum(r["facts"] for r in sel)
            vis = sum(r["facts_visible"] for r in sel)
            rec = sum(r["facts_recoverable"] for r in sel)
            ms = statistics.median(r["ms"] for r in sel)
            lines.append(
                f"| {path} | {posture} | {t_in} | {t_out} | {100.0 * (t_in - t_out) / max(1, t_in):.1f}% "
                f"| {vis}/{facts} | {rec}/{facts} | {ms:.1f} |"
            )
    lines.append("")
    lines.append("Per scenario (tokens out · facts visible/recoverable):")
    lines.append("")
    lines.append("| path | scenario | " + " | ".join(POSTURES) + " |")
    lines.append("|---|---|" + "---|" * len(POSTURES))
    for path in ("claude_code", "codex"):
        for name in dict.fromkeys(r["scenario"] for r in rows):
            cells = []
            for posture in POSTURES:
                r = next(
                    (
                        x
                        for x in rows
                        if x["path"] == path and x["scenario"] == name and x["posture"] == posture
                    ),
                    None,
                )
                cells.append(
                    "-"
                    if r is None
                    else f"{r['tokens_out']} · {r['facts_visible']}/{r['facts_recoverable']}/{r['facts']}"
                )
            lines.append(f"| {path} | {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument(
        "--kompress", action="store_true", help="allow the ML text compressor (needs its model)"
    )
    parser.add_argument("--json", type=Path, default=None, help="write raw rows here")
    parser.add_argument("--markdown", type=Path, default=None, help="write the table here")
    args = parser.parse_args()
    KOMPRESS["enabled"] = bool(args.kompress)
    rows = run(repeat=args.repeat)
    table = summarize(rows)
    print(table)
    if args.json:
        args.json.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    if args.markdown:
        args.markdown.write_text(table + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
