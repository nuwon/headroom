"""Indexed, partial CCR retrieval (Optimization 3, plan §10).

CCR stores exact originals by hash. This module adds a *span index* over an
entry so a later turn can pull back the few hundred relevant tokens of a
100k-token result instead of the whole blob, while ``mode="full"`` still
returns the exact original byte-for-byte.

Spans are cut along semantic structure:

* JSON arrays — one element per span (offsets of each element in the
  original string, found with ``JSONDecoder.raw_decode``);
* search/grep output — one span per file group;
* diffs — one span per file, split per hunk when a file is large;
* logs / test output — stack traces and failure blocks kept whole, other
  lines grouped into fixed windows;
* source code — one span per top-level definition (plus a header span);
* prose — paragraphs.

Every span is an exact ``original[start:end]`` slice, so retrieved spans are
always verbatim. Ranking uses SQLite FTS5 (``bm25()``) when the interpreter's
SQLite has it, else a deterministic in-Python BM25; exact task-entity matches
and an optional external prior (graph proximity, learned retention) are fused
on top.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import sqlite3
import threading
from collections import Counter, OrderedDict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

MAX_SPAN_CHARS = 4000
LOG_WINDOW_LINES = 12
_INDEX_CACHE_MAX_ENTRIES = 64
_INDEX_CACHE_MAX_CHARS = 64 * 1024 * 1024

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+(?:[.\-/:][A-Za-z0-9_]+)*")
_STOP = frozenset(
    "the a an and or of to in on for is are was were be by with at from as it this that what "
    "which how why when where who do does did show me find get all any".split()
)


@dataclass(frozen=True)
class Span:
    ordinal: int
    start: int
    end: int
    item_type: str
    meta: str = ""

    def text(self, original: str) -> str:
        return original[self.start : self.end]


@dataclass
class SpanIndex:
    entry_hash: str
    kind: str
    spans: list[Span]
    content_hash: str
    _fts: sqlite3.Connection | None = field(default=None, repr=False)
    _postings: dict[str, dict[int, int]] | None = field(default=None, repr=False)
    _lengths: list[int] | None = field(default=None, repr=False)

    def span_id(self, span: Span) -> str:
        return f"{self.entry_hash}:{span.ordinal}"

    def summary(self) -> dict[str, Any]:
        types = Counter(s.item_type for s in self.spans)
        metas = [s.meta for s in self.spans if s.meta]
        return {
            "hash": self.entry_hash,
            "kind": self.kind,
            "span_count": len(self.spans),
            "item_types": dict(types),
            "sample_keys": metas[:20],
        }


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

_DIFF_FILE_RE = re.compile(r"^diff --git .*$", re.M)
_HUNK_RE = re.compile(r"^@@ .*$", re.M)
_GREP_LINE_RE = re.compile(r"^([^\s:][^:\n]{0,300}?):(\d+)[:-]", re.M)
_DEF_RE = re.compile(
    r"^(?:(?:pub(?:\([^)]*\))?|export|async|static|public|private|protected|default|abstract|final)\s+)*"
    r"(?:def|class|fn|func|function|interface|struct|enum|trait|impl|type|const|module)\b[^\n]*$",
    re.M,
)
_TRACE_START_RE = re.compile(
    r"^(?:Traceback \(most recent call last\)|Exception in thread|panicked at|thread '.*' panicked|"
    r"_{3,} .* _{3,}|={3,} FAILURES ={3,}|FAIL[: ]|ERROR[: ]|Caused by:)",
    re.M,
)


def detect_kind(text: str) -> str:
    stripped = text.lstrip()
    if stripped.startswith("["):
        try:
            value, _ = json.JSONDecoder().raw_decode(stripped)
            if isinstance(value, list):
                return "json_array"
        except (ValueError, json.JSONDecodeError):
            pass
    if stripped.startswith("{"):
        try:
            value, _ = json.JSONDecoder().raw_decode(stripped)
            if isinstance(value, dict) and any(
                isinstance(v, list) and len(v) > 3 for v in value.values()
            ):
                return "json_object"
        except (ValueError, json.JSONDecodeError):
            pass
    head = text[:20_000]
    if _DIFF_FILE_RE.search(head) or (head.startswith("--- ") and "\n+++ " in head):
        return "diff"
    lines = head.splitlines()[:200]
    if lines and sum(1 for ln in lines if _GREP_LINE_RE.match(ln)) >= max(3, len(lines) // 2):
        return "search"
    if len(_DEF_RE.findall(head)) >= 3:
        return "code"
    if _TRACE_START_RE.search(head) or re.search(
        r"\b(?:INFO|WARN|DEBUG|ERROR|PASSED|FAILED)\b", head
    ):
        return "log"
    return "prose"


def _json_element_spans(text: str, array_start: int) -> list[Span]:
    """Offsets of each element of the JSON array beginning at ``array_start``."""
    decoder = json.JSONDecoder()
    spans: list[Span] = []
    i = array_start + 1
    n = len(text)
    ordinal = 0
    while i < n:
        while i < n and text[i] in " \t\r\n,":
            i += 1
        if i >= n or text[i] == "]":
            break
        try:
            value, end = decoder.raw_decode(text, i)
        except json.JSONDecodeError:
            return []
        meta = ""
        if isinstance(value, dict):
            for key in ("id", "name", "path", "key", "title", "url", "file"):
                if key in value and isinstance(value[key], (str, int)):
                    meta = f"{key}={value[key]}"
                    break
        spans.append(Span(ordinal, i, end, "json_element", meta[:120]))
        ordinal += 1
        i = end
    return spans


def _split_large(spans: list[Span], text: str) -> list[Span]:
    """Split spans larger than MAX_SPAN_CHARS on line boundaries (still exact)."""
    out: list[Span] = []
    for span in spans:
        if span.end - span.start <= MAX_SPAN_CHARS:
            out.append(span)
            continue
        cursor = span.start
        part = 0
        while cursor < span.end:
            limit = min(span.end, cursor + MAX_SPAN_CHARS)
            if limit < span.end:
                nl = text.rfind("\n", cursor, limit)
                if nl > cursor:
                    limit = nl + 1
            out.append(
                Span(
                    0,
                    cursor,
                    limit,
                    span.item_type,
                    f"{span.meta}#{part}" if span.meta else f"part{part}",
                )
            )
            cursor = limit
            part += 1
    return [Span(i, s.start, s.end, s.item_type, s.meta) for i, s in enumerate(out)]


def _line_offsets(text: str) -> list[int]:
    offsets = [0]
    for m in re.finditer("\n", text):
        offsets.append(m.end())
    if offsets[-1] != len(text):
        offsets.append(len(text))
    return offsets


def _chunk_by_markers(
    text: str, marker_re: re.Pattern[str], item_type: str, meta_fn: Callable[[str], str]
) -> list[Span]:
    starts = [m.start() for m in marker_re.finditer(text)]
    if not starts:
        return []
    bounds = ([0] if starts[0] > 0 else []) + starts + [len(text)]
    spans: list[Span] = []
    for a, b in zip(bounds, bounds[1:]):
        if b <= a:
            continue
        chunk = text[a:b]
        first = chunk.split("\n", 1)[0]
        is_header = a == 0 and (not starts or starts[0] > 0)
        spans.append(Span(len(spans), a, b, "header" if is_header else item_type, meta_fn(first)))
    return spans


def chunk_content(text: str, kind: str | None = None) -> tuple[str, list[Span]]:
    """Return ``(kind, spans)``; spans tile ``text`` exactly (no gaps, no overlap)."""
    if not text:
        return "empty", []
    kind = kind or detect_kind(text)
    spans: list[Span] = []
    if kind == "json_array":
        start = text.index("[")
        spans = _json_element_spans(text, start)
        if spans:
            spans = _tile(text, spans, "json_syntax")
    elif kind == "json_object":
        # Index the largest array value's elements; the rest is one span.
        try:
            obj, _ = json.JSONDecoder().raw_decode(text.lstrip())
            biggest = max(
                (k for k, v in obj.items() if isinstance(v, list)), key=lambda k: len(obj[k])
            )
            m = re.search(r'"' + re.escape(biggest) + r'"\s*:\s*\[', text)
            if m:
                spans = _tile(text, _json_element_spans(text, m.end() - 1), "json_syntax")
        except (ValueError, json.JSONDecodeError):
            spans = []
    elif kind == "diff":
        spans = _chunk_by_markers(
            text, _DIFF_FILE_RE, "diff_file", lambda first: first[11:].split(" b/")[0][:160]
        )
        if spans:
            refined: list[Span] = []
            for s in spans:
                if s.end - s.start > MAX_SPAN_CHARS:
                    sub = text[s.start : s.end]
                    hunks = [m.start() for m in _HUNK_RE.finditer(sub)]
                    if hunks:
                        pts = [0, *hunks, len(sub)]
                        for a, b in zip(pts, pts[1:]):
                            if b > a:
                                refined.append(
                                    Span(
                                        0,
                                        s.start + a,
                                        s.start + b,
                                        "diff_hunk" if a else "diff_header",
                                        s.meta,
                                    )
                                )
                        continue
                refined.append(s)
            spans = refined
    elif kind == "search":
        spans = _search_spans(text)
    elif kind == "code":
        spans = _chunk_by_markers(text, _DEF_RE, "definition", lambda first: first.strip()[:160])
    elif kind == "log":
        spans = _log_spans(text)
    if not spans:
        spans = _prose_spans(text)
        if kind not in ("prose", "log"):
            kind = f"{kind}+prose"
    spans = _split_large(spans, text)
    return kind, spans


def _tile(text: str, spans: list[Span], filler_type: str) -> list[Span]:
    """Fill gaps between spans so the spans tile the whole text."""
    out: list[Span] = []
    cursor = 0
    for s in spans:
        if s.start > cursor:
            out.append(Span(0, cursor, s.start, filler_type, ""))
        out.append(s)
        cursor = s.end
    if cursor < len(text):
        out.append(Span(0, cursor, len(text), filler_type, ""))
    return [Span(i, s.start, s.end, s.item_type, s.meta) for i, s in enumerate(out)]


def _search_spans(text: str) -> list[Span]:
    offsets = _line_offsets(text)
    spans: list[Span] = []
    group_start = 0
    current = None
    for i in range(len(offsets) - 1):
        line = text[offsets[i] : offsets[i + 1]]
        m = _GREP_LINE_RE.match(line)
        path = m.group(1) if m else current
        if path != current and i > 0 and offsets[i] > group_start:
            spans.append(Span(len(spans), group_start, offsets[i], "search_group", current or ""))
            group_start = offsets[i]
        current = path
    if group_start < len(text):
        spans.append(Span(len(spans), group_start, len(text), "search_group", current or ""))
    return spans


def _log_spans(text: str) -> list[Span]:
    offsets = _line_offsets(text)
    n = len(offsets) - 1
    spans: list[Span] = []
    i = 0
    while i < n:
        line = text[offsets[i] : offsets[i + 1]]
        if _TRACE_START_RE.match(line):
            # Keep a failure block whole: until a blank line or the next block start.
            j = i + 1
            while j < n:
                nxt = text[offsets[j] : offsets[j + 1]]
                if not nxt.strip() or _TRACE_START_RE.match(nxt):
                    break
                j += 1
            spans.append(
                Span(len(spans), offsets[i], offsets[j], "failure_block", line.strip()[:160])
            )
            i = j
            continue
        j = min(n, i + LOG_WINDOW_LINES)
        k = i + 1
        while k < j and not _TRACE_START_RE.match(text[offsets[k] : offsets[k + 1]]):
            k += 1
        spans.append(Span(len(spans), offsets[i], offsets[k], "log_lines", ""))
        i = k
    return spans


def _prose_spans(text: str) -> list[Span]:
    spans: list[Span] = []
    cursor = 0
    for m in re.finditer(r"\n\s*\n", text):
        if m.end() - cursor > 0:
            spans.append(Span(len(spans), cursor, m.end(), "paragraph", ""))
            cursor = m.end()
    if cursor < len(text):
        spans.append(Span(len(spans), cursor, len(text), "paragraph", ""))
    return spans


# ---------------------------------------------------------------------------
# Index + ranking
# ---------------------------------------------------------------------------


def _tokens(text: str) -> list[str]:
    out: list[str] = []
    for tok in _TOKEN_RE.findall(text.lower()):
        if tok in _STOP or len(tok) < 2:
            continue
        out.append(tok)
        if any(c in tok for c in "./-:"):
            out.extend(p for p in re.split(r"[./\-:]", tok) if len(p) >= 2 and p not in _STOP)
    return out


_FTS5: bool | None = None


def fts5_available() -> bool:
    global _FTS5
    if _FTS5 is None:
        try:
            con = sqlite3.connect(":memory:")
            con.execute("CREATE VIRTUAL TABLE t USING fts5(x)")
            con.close()
            _FTS5 = True
        except sqlite3.Error:
            _FTS5 = False
    return _FTS5


def build_index(entry_hash: str, original: str, *, use_fts: bool | None = None) -> SpanIndex:
    kind, spans = chunk_content(original)
    content_hash = hashlib.sha256(original.encode("utf-8", "surrogatepass")).hexdigest()[:24]
    index = SpanIndex(entry_hash, kind, spans, content_hash)
    if (fts5_available() if use_fts is None else use_fts) and spans:
        try:
            con = sqlite3.connect(":memory:", check_same_thread=False)
            con.execute(
                "CREATE VIRTUAL TABLE spans USING fts5(body, meta, tokenize='unicode61 remove_diacritics 2')"
            )
            con.executemany(
                "INSERT INTO spans(rowid, body, meta) VALUES (?, ?, ?)",
                (
                    (
                        s.ordinal,
                        " ".join(_tokens(original[s.start : s.end][:MAX_SPAN_CHARS])),
                        s.meta,
                    )
                    for s in spans
                ),
            )
            index._fts = con
            return index
        except sqlite3.Error as exc:
            logger.debug("FTS5 index build failed (%s); using lexical fallback", exc)
    postings: dict[str, dict[int, int]] = {}
    lengths: list[int] = []
    for s in spans:
        toks = _tokens(original[s.start : s.end][:MAX_SPAN_CHARS]) + _tokens(s.meta)
        lengths.append(len(toks))
        for tok, count in Counter(toks).items():
            postings.setdefault(tok, {})[s.ordinal] = count
    index._postings = postings
    index._lengths = lengths
    return index


def _lexical_scores(index: SpanIndex, terms: list[str]) -> dict[int, float]:
    if not terms or not index.spans:
        return {}
    if index._fts is not None:
        safe = [t.replace('"', "") for t in dict.fromkeys(terms) if t]
        if not safe:
            return {}
        query = " OR ".join(f'"{t}"' for t in safe[:64])
        try:
            rows = index._fts.execute(
                "SELECT rowid, bm25(spans) FROM spans WHERE spans MATCH ? ORDER BY rowid",
                (query,),
            ).fetchall()
        except sqlite3.Error:
            rows = []
        # bm25() is lower-is-better (negative); flip so higher is better.
        return {int(r): -float(score) for r, score in rows}
    postings = index._postings or {}
    lengths = index._lengths or []
    n = len(index.spans)
    avg = (sum(lengths) / n) if n else 1.0
    k1, b = 1.5, 0.75
    scores: dict[int, float] = {}
    for term in dict.fromkeys(terms):
        plist = postings.get(term)
        if not plist:
            continue
        idf = math.log(1 + (n - len(plist) + 0.5) / (len(plist) + 0.5))
        for ordinal, tf in plist.items():
            dl = lengths[ordinal] if ordinal < len(lengths) else avg
            scores[ordinal] = scores.get(ordinal, 0.0) + idf * (tf * (k1 + 1)) / (
                tf + k1 * (1 - b + b * dl / max(avg, 1e-9))
            )
    return scores


@dataclass(frozen=True)
class RankedSpan:
    span: Span
    score: float


def rank_spans(
    index: SpanIndex,
    original: str,
    query: str,
    *,
    exact_terms: Iterable[str] = (),
    prior: Callable[[Span], float] | None = None,
    semantic: Callable[[list[str], str], list[float]] | None = None,
) -> list[RankedSpan]:
    """Rank spans for ``query`` (deterministic; ties broken by ordinal)."""
    terms = _tokens(query)
    lex = _lexical_scores(index, terms)
    max_lex = max(lex.values(), default=0.0) or 1.0
    exact = [e.lower() for e in exact_terms if e and len(e) >= 2]
    # Quoted phrases / identifiers in the query itself count as exact terms.
    for groups in re.findall(r"`([^`]{2,80})`|\"([^\"]{2,80})\"", query):
        exact.extend(g.lower() for g in groups if g)
    sem_scores: dict[int, float] = {}
    if semantic is not None and lex:
        shortlist = sorted(lex, key=lambda o: (-lex[o], o))[:48]
        try:
            vals = semantic(
                [original[index.spans[o].start : index.spans[o].end][:2000] for o in shortlist],
                query,
            )
            sem_scores = dict(zip(shortlist, vals))
        except Exception:  # noqa: BLE001 - embeddings are optional
            sem_scores = {}
    ranked: list[RankedSpan] = []
    for span in index.spans:
        score = 0.6 * (lex.get(span.ordinal, 0.0) / max_lex)
        if sem_scores:
            score += 0.25 * sem_scores.get(span.ordinal, 0.0)
        if exact:
            body = original[span.start : span.end].lower()
            hits = sum(1 for e in exact if e in body or e in span.meta.lower())
            score += 0.5 * min(1.0, hits / max(1, len(exact)))
        if prior is not None:
            try:
                score += 0.15 * max(0.0, min(1.0, float(prior(span))))
            except Exception:  # noqa: BLE001
                pass
        if span.item_type == "failure_block" and score > 0:
            score += 0.05
        if score > 0:
            ranked.append(RankedSpan(span, round(score, 9)))
    ranked.sort(key=lambda r: (-r.score, r.span.ordinal))
    return ranked


# ---------------------------------------------------------------------------
# Process-wide index cache (bounded by entries and total indexed chars)
# ---------------------------------------------------------------------------


class _IndexCache:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._items: OrderedDict[tuple[str, int], tuple[SpanIndex, int]] = OrderedDict()
        self._chars = 0

    def get_or_build(self, entry_hash: str, original: str) -> SpanIndex:
        key = (entry_hash, len(original))
        with self._lock:
            hit = self._items.get(key)
            if hit is not None:
                self._items.move_to_end(key)
                return hit[0]
        index = build_index(entry_hash, original)
        with self._lock:
            self._items[key] = (index, len(original))
            self._chars += len(original)
            while self._items and (
                len(self._items) > _INDEX_CACHE_MAX_ENTRIES or self._chars > _INDEX_CACHE_MAX_CHARS
            ):
                _, (old, size) = self._items.popitem(last=False)
                self._chars -= size
                if old._fts is not None:
                    try:
                        old._fts.close()
                    except sqlite3.Error:
                        pass
        return index

    def prime(self, entry_hash: str, original: str) -> None:
        self.get_or_build(entry_hash, original)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
            self._chars = 0


INDEX_CACHE = _IndexCache()


# ---------------------------------------------------------------------------
# Retrieval API
# ---------------------------------------------------------------------------

MODES = ("full", "search", "range", "metadata")
DEFAULT_TOP_K = 5
MAX_TOP_K = 50


@dataclass(frozen=True)
class RetrieveArgs:
    hash: str
    mode: str = "full"
    query: str = ""
    top_k: int = DEFAULT_TOP_K
    cursor: str = ""
    range: str = ""

    @property
    def selective(self) -> bool:
        return self.mode != "full"


def normalize_args(raw: dict[str, Any], hash_key: str) -> RetrieveArgs:
    """Parse optional retrieve arguments; old ``{hash}`` calls stay ``full``."""
    query = str(raw.get("query") or "").strip()[:2000]
    mode = str(raw.get("mode") or "").strip().lower()
    rng = str(raw.get("range") or "").strip()[:200]
    if mode not in MODES:
        mode = "search" if query else ("range" if rng else "full")
    try:
        top_k = int(raw.get("top_k") or DEFAULT_TOP_K)
    except (TypeError, ValueError):
        top_k = DEFAULT_TOP_K
    top_k = max(1, min(MAX_TOP_K, top_k))
    return RetrieveArgs(hash_key, mode, query, top_k, str(raw.get("cursor") or "")[:200], rng)


def _parse_range(spec: str, n: int) -> list[int]:
    ordinals: list[int] = []
    for part in re.split(r"[,\s]+", spec.strip()):
        if not part:
            continue
        m = re.fullmatch(r"(\d+)\s*-\s*(\d+)", part)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            ordinals.extend(range(min(a, b), min(max(a, b), n - 1) + 1))
        elif part.isdigit():
            ordinals.append(int(part))
    return [o for o in dict.fromkeys(ordinals) if 0 <= o < n][:MAX_TOP_K]


def _cursor_offset(cursor: str, query: str) -> int:
    if not cursor:
        return 0
    qh = hashlib.sha256(query.encode()).hexdigest()[:8]
    m = re.fullmatch(r"(?:([0-9a-f]{8}):)?(\d+)", cursor.strip())
    if not m:
        return 0
    if m.group(1) and m.group(1) != qh:
        return 0  # cursor from a different query: restart
    return int(m.group(2))


def selective_retrieve(
    entry_hash: str,
    original: str,
    args: RetrieveArgs,
    *,
    exact_terms: Iterable[str] = (),
    prior: Callable[[Span], float] | None = None,
    semantic: Callable[[list[str], str], list[float]] | None = None,
    max_chars: int = 24_000,
) -> dict[str, Any]:
    """Answer a non-full retrieval from the span index (exact span text)."""
    index = INDEX_CACHE.get_or_build(entry_hash, original)
    base: dict[str, Any] = {
        "hash": entry_hash,
        "mode": args.mode,
        "total_spans": len(index.spans),
        "kind": index.kind,
    }
    if args.mode == "metadata":
        return {**base, **index.summary(), "original_chars": len(original)}
    if args.mode == "range":
        ordinals = _parse_range(args.range, len(index.spans))
        chosen = [(index.spans[o], None) for o in ordinals]
        has_more = False
        next_cursor = ""
    else:
        ranked = rank_spans(
            index, original, args.query, exact_terms=exact_terms, prior=prior, semantic=semantic
        )
        offset = _cursor_offset(args.cursor, args.query)
        page = ranked[offset : offset + args.top_k]
        chosen = [(r.span, r.score) for r in page]
        has_more = offset + args.top_k < len(ranked)
        qh = hashlib.sha256(args.query.encode()).hexdigest()[:8]
        next_cursor = f"{qh}:{offset + args.top_k}" if has_more else ""
        base["matched_spans"] = len(ranked)
    spans_out: list[dict[str, Any]] = []
    used = 0
    for span, score in chosen:
        text = span.text(original)
        if used + len(text) > max_chars and spans_out:
            has_more = True
            break
        used += len(text)
        item: dict[str, Any] = {
            "span_id": index.span_id(span),
            "ordinal": span.ordinal,
            "start": span.start,
            "end": span.end,
            "type": span.item_type,
            "text": text,
        }
        if span.meta:
            item["key"] = span.meta
        if score is not None:
            item["score"] = round(score, 4)
        spans_out.append(item)
    return {**base, "spans": spans_out, "has_more": has_more, "next_cursor": next_cursor}
