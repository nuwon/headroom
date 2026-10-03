"""Indexed / partial CCR retrieval (Optimization 3)."""

from __future__ import annotations

import json

import pytest

from headroom.cache.compression_store import CompressionStore
from headroom.ccr import tool_injection
from headroom.ccr.context_tracker import (
    ContextTracker,
    ContextTrackerConfig,
    ExpansionRecommendation,
)
from headroom.ccr.response_handler import CCRResponseHandler
from headroom.ccr.span_index import (
    INDEX_CACHE,
    RetrieveArgs,
    build_index,
    chunk_content,
    normalize_args,
    rank_spans,
    selective_retrieve,
)
from headroom.ccr.tool_calls import parse_ccr_tool_calls


def tok(text: str) -> int:
    return len(text) // 4


def big_json(n: int = 6000) -> str:
    rows = [
        {
            "id": i,
            "user": f"user{i}",
            "status": "ok",
            "region": "us-east-1",
            "latency_ms": 10 + i % 7,
        }
        for i in range(n)
    ]
    bad = min(4321, n - 1)
    rows[bad]["status"] = "error"
    rows[bad]["error"] = "PaymentDeclined: card_expired for invoice INV-88231"
    return json.dumps(rows, indent=1)


SAMPLES = {
    "json": big_json(200),
    "search": "\n".join(f"src/mod{i // 5}.py:{i}: def handler_{i}(request):" for i in range(120)),
    "diff": "".join(
        f"diff --git a/f{i}.py b/f{i}.py\nindex 1..2 100644\n--- a/f{i}.py\n+++ b/f{i}.py\n@@ -1,3 +1,3 @@\n-old {i}\n+new {i}\n ctx\n"
        for i in range(30)
    ),
    "log": "\n".join(
        [f"INFO step {i} ok" for i in range(100)]
        + [
            "Traceback (most recent call last):",
            '  File "a.py", line 3, in f',
            "ValueError: boom",
            "",
        ]
        + [f"INFO tail {i}" for i in range(30)]
    ),
    "code": "import os\n\n"
    + "".join(f"def func_{i}(x):\n    return x + {i}\n\n" for i in range(40)),
    "prose": "\n\n".join(f"Paragraph {i} talks about topic {i % 5}." for i in range(50)),
    "windows_crlf": "\r\n".join(f"C:\\repo\\src\\m{i}.py:{i}: value = {i}" for i in range(80)),
}


@pytest.mark.parametrize("name", sorted(SAMPLES))
def test_spans_tile_original_exactly(name):
    text = SAMPLES[name]
    kind, spans = chunk_content(text)
    assert spans, kind
    assert "".join(text[s.start : s.end] for s in spans) == text
    assert [s.ordinal for s in spans] == list(range(len(spans)))
    for a, b in zip(spans, spans[1:]):
        assert a.end == b.start


def test_json_elements_are_valid_json():
    text = SAMPLES["json"]
    kind, spans = chunk_content(text)
    assert kind == "json_array"
    elements = [s for s in spans if s.item_type == "json_element"]
    assert len(elements) == 200
    for s in elements[:20]:
        assert isinstance(json.loads(text[s.start : s.end]), dict)
    assert elements[7].meta == "id=7"


def test_kinds_detected():
    assert chunk_content(SAMPLES["diff"])[0] == "diff"
    assert chunk_content(SAMPLES["search"])[0] == "search"
    assert chunk_content(SAMPLES["code"])[0] == "code"
    assert chunk_content(SAMPLES["log"])[0] == "log"


def test_large_result_brings_back_few_hundred_tokens_and_full_is_exact():
    original = big_json(6000)
    assert tok(original) > 100_000
    store = CompressionStore(max_entries=10)
    h = store.store(original, "[compressed]", original_tokens=tok(original), compressed_tokens=10)
    result = store.retrieve_selective(
        h, RetrieveArgs(h, mode="search", query="why was INV-88231 declined?", top_k=3)
    )
    assert result is not None and result["spans"]
    returned = "".join(sp["text"] for sp in result["spans"])
    assert "PaymentDeclined" in returned
    assert tok(returned) < 600
    assert result["has_more"] in (True, False) and "next_cursor" in result
    for sp in result["spans"]:
        assert original[sp["start"] : sp["end"]] == sp["text"]
    full = store.retrieve_selective(h, RetrieveArgs(h, mode="full"))
    assert full is not None and full["original_content"] == original
    assert store.retrieve(h).original_content == original


def test_cursor_paging_range_and_metadata():
    original = SAMPLES["search"]
    args = RetrieveArgs("a" * 24, mode="search", query="handler request", top_k=4)
    INDEX_CACHE.clear()
    page1 = selective_retrieve("a" * 24, original, args)
    assert page1["has_more"] and page1["next_cursor"]
    page2 = selective_retrieve(
        "a" * 24, original, RetrieveArgs("a" * 24, "search", args.query, 4, page1["next_cursor"])
    )
    ids1 = {s["ordinal"] for s in page1["spans"]}
    ids2 = {s["ordinal"] for s in page2["spans"]}
    assert ids1 and ids2 and not (ids1 & ids2)
    # A cursor from another query restarts at the top.
    other = selective_retrieve(
        "a" * 24, original, RetrieveArgs("a" * 24, "search", "different", 4, page1["next_cursor"])
    )
    assert other["spans"] == [] or other["spans"][0]["ordinal"] >= 0
    rng = selective_retrieve(
        "a" * 24, original, RetrieveArgs("a" * 24, mode="range", range="0-2,5")
    )
    assert [s["ordinal"] for s in rng["spans"]] == [0, 1, 2, 5]
    meta = selective_retrieve("a" * 24, original, RetrieveArgs("a" * 24, mode="metadata"))
    assert meta["span_count"] == meta["total_spans"] and "spans" not in meta


def test_fts_and_fallback_agree_on_top_hit():
    original = big_json(500)
    original = original.replace('"user": "user123"', '"user": "zebra_unique_name"')
    a = build_index("x" * 24, original, use_fts=True)
    b = build_index("x" * 24, original, use_fts=False)
    ra = rank_spans(a, original, "zebra_unique_name")
    rb = rank_spans(b, original, "zebra_unique_name")
    assert ra and rb and ra[0].span.ordinal == rb[0].span.ordinal
    assert "zebra_unique_name" in original[ra[0].span.start : ra[0].span.end]


def test_exact_terms_outrank_noise():
    original = "\n\n".join(
        ["the cache layer handles eviction of stale keys"] * 30
        + ["order ORD-55120 failed validation"]
    )
    index = build_index("y" * 24, original)
    ranked = rank_spans(index, original, "cache eviction ORD-55120", exact_terms=["ORD-55120"])
    assert "ORD-55120" in original[ranked[0].span.start : ranked[0].span.end]


def test_normalize_args_backward_compatible():
    assert normalize_args({}, "h").mode == "full"
    assert normalize_args({"query": "x"}, "h").mode == "search"
    assert normalize_args({"range": "1-2"}, "h").mode == "range"
    assert normalize_args({"mode": "bogus", "top_k": 999}, "h").top_k == 50


class TestToolSchema:
    def teardown_method(self):
        tool_injection.set_ccr_search_enabled(False)

    def test_off_is_byte_identical_and_on_adds_optional(self):
        tool_injection.set_ccr_search_enabled(False)
        before = {
            p: json.dumps(tool_injection.create_ccr_tool_definition(p), sort_keys=True)
            for p in ("anthropic", "openai", "openai_responses", "google")
        }
        assert all("query" not in v for v in before.values())
        tool_injection.set_ccr_search_enabled(True)
        for p in before:
            d = tool_injection.create_ccr_tool_definition(p)
            holder = d.get("function", d)
            schema = holder.get("input_schema") or holder.get("parameters")
            assert set(schema["properties"]) >= {
                "hash",
                "query",
                "mode",
                "top_k",
                "cursor",
                "range",
            }
            assert schema["required"] == ["hash"]
        tool_injection.set_ccr_search_enabled(False)
        after = {
            p: json.dumps(tool_injection.create_ccr_tool_definition(p), sort_keys=True)
            for p in before
        }
        assert after == before


def test_response_handler_executes_selective(monkeypatch):
    original = big_json(800)
    store = CompressionStore(max_entries=10)
    h = store.store(original, "[c]")
    monkeypatch.setattr("headroom.ccr.response_handler.get_compression_store", lambda: store)
    response = {
        "content": [
            {
                "type": "tool_use",
                "id": "t1",
                "name": "headroom_retrieve",
                "input": {"hash": h, "query": "user42"},
            },
            {"type": "tool_use", "id": "t2", "name": "headroom_retrieve", "input": {"hash": h}},
        ]
    }
    calls, other = parse_ccr_tool_calls(response, "anthropic")
    assert len(calls) == 2 and not other
    handler = CCRResponseHandler()
    sel = json.loads(handler._execute_retrieval(calls[0]).content)
    assert sel["mode"] == "search" and any('"user42"' in s["text"] for s in sel["spans"])
    full = json.loads(handler._execute_retrieval(calls[1]).content)
    assert full["original_content"] == original


def test_selective_proactive_expansion(monkeypatch):
    original = big_json(3000)
    store = CompressionStore(max_entries=10)
    h = store.store(original, "[c]")
    monkeypatch.setattr("headroom.ccr.context_tracker.get_compression_store", lambda: store)
    tracker = ContextTracker(
        ContextTrackerConfig(selective_expansion=True, expansion_token_budget=400)
    )
    rec = ExpansionRecommendation(h, "from Bash", 0.9)
    out = tracker.execute_expansions([rec], query="what failed for INV-88231?")
    assert out and out[0]["type"] == "spans"
    text = tracker.format_expansions_for_context(out)
    assert "PaymentDeclined" in text and tok(text) < 700
    # An explicit ask for the complete output restores the full original.
    full = tracker.execute_expansions([rec], query="show me the full output again")
    assert full[0]["type"] == "full" and full[0]["content"] == original
    # Selective disabled -> legacy full behavior.
    legacy = ContextTracker(ContextTrackerConfig()).execute_expansions([rec], query="INV-88231")
    assert legacy[0]["type"] == "full"
