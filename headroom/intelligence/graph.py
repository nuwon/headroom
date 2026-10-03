"""Workspace-local code graph for graph-scoped relevance (Optimization 11).

Nodes are files, modules and definitions (functions, classes, types);
edges are ``imports``, ``defines``, ``references`` (a file mentions a symbol
defined elsewhere) and ``read-with`` (files read in the same turn). The graph
is built incrementally from what the session actually touches:

* source text seen in tool results (Read output, ``cat``/``Get-Content``),
* files on disk referenced by the conversation, when they exist locally
  (the proxy runs on the agent's machine; vendor/generated dirs skipped).

Each file is re-parsed only when its content hash changes, and every graph is
scoped by workspace key (no cross-project edges). Parsing is a fast,
dependency-free extractor for the languages Headroom's code compressor
supports (Python, JS/TS, Go, Rust, Java, C/C++, C#, Ruby, PHP, shell);
unknown languages fall back to a lexical file node with no definitions.

The graph answers one question for the rest of the system: *given the active
files/symbols, what is structurally nearby?* (:meth:`CodeGraph.neighborhood`,
BFS up to distance 2), which feeds the relevance query, the budget allocator
and CCR span ranking via :meth:`CodeGraph.proximity`.
"""

from __future__ import annotations

import hashlib
import os
import re
import threading
from collections import OrderedDict, defaultdict, deque
from collections.abc import Iterable
from dataclasses import dataclass, field

from .messages import build_tool_call_index, iter_tool_results
from .resources import canonical_path, resource_for

_IGNORED_DIRS = (
    "/node_modules/",
    "/.git/",
    "/vendor/",
    "/dist/",
    "/build/",
    "/target/",
    "/.venv/",
    "/venv/",
    "/__pycache__/",
    "/.next/",
    "/out/",
    "/third_party/",
)
_MAX_FILE_BYTES = 1_000_000

_LANG_BY_EXT = {
    ".py": "python",
    ".pyi": "python",
    ".js": "js",
    ".jsx": "js",
    ".mjs": "js",
    ".cjs": "js",
    ".ts": "js",
    ".tsx": "js",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".kt": "java",
    ".c": "c",
    ".h": "c",
    ".cc": "c",
    ".cpp": "c",
    ".hpp": "c",
    ".cs": "csharp",
    ".rb": "ruby",
    ".php": "php",
    ".sh": "shell",
    ".ps1": "shell",
}

_DEF_PATTERNS: dict[str, re.Pattern[str]] = {
    "python": re.compile(r"^\s*(?:async\s+)?(?:def|class)\s+([A-Za-z_]\w*)", re.M),
    "js": re.compile(
        r"^\s*(?:export\s+(?:default\s+)?)?(?:async\s+)?(?:function\*?|class|interface|type|enum)\s+([A-Za-z_$][\w$]*)"
        r"|^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?(?:\([^)]*\)|[A-Za-z_$][\w$]*)\s*=>",
        re.M,
    ),
    "go": re.compile(r"^\s*func\s+(?:\([^)]*\)\s*)?([A-Za-z_]\w*)|^\s*type\s+([A-Za-z_]\w*)", re.M),
    "rust": re.compile(
        r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?(?:fn|struct|enum|trait|type|mod)\s+([A-Za-z_]\w*)",
        re.M,
    ),
    "java": re.compile(
        r"^\s*(?:(?:public|private|protected|static|final|abstract|open|data)\s+)*(?:class|interface|enum|record|fun)\s+([A-Za-z_]\w*)"
        r"|^\s*(?:(?:public|private|protected|static|final|synchronized)\s+)+[\w<>\[\], ]+\s+([A-Za-z_]\w*)\s*\(",
        re.M,
    ),
    "c": re.compile(
        r"^[A-Za-z_][\w\s\*&:<>,]*?\b([A-Za-z_]\w*)\s*\([^;{]*\)\s*\{|^\s*(?:struct|class|enum)\s+([A-Za-z_]\w*)",
        re.M,
    ),
    "csharp": re.compile(
        r"^\s*(?:(?:public|private|protected|internal|static|sealed|abstract|partial|async)\s+)*(?:class|interface|struct|enum|record)\s+([A-Za-z_]\w*)"
        r"|^\s*(?:(?:public|private|protected|internal|static|virtual|override|async)\s+)+[\w<>\[\], ]+\s+([A-Za-z_]\w*)\s*\(",
        re.M,
    ),
    "ruby": re.compile(r"^\s*(?:def|class|module)\s+(?:self\.)?([A-Za-z_]\w*[?!]?)", re.M),
    "php": re.compile(
        r"^\s*(?:(?:public|private|protected|static|abstract|final)\s+)*(?:function|class|interface|trait)\s+([A-Za-z_]\w*)",
        re.M,
    ),
    "shell": re.compile(r"^\s*(?:function\s+)?([A-Za-z_][\w-]*)\s*\(\)\s*\{", re.M),
}
_IMPORT_PATTERNS: dict[str, re.Pattern[str]] = {
    "python": re.compile(r"^\s*(?:from\s+([\w.]+)\s+import|import\s+([\w.]+))", re.M),
    "js": re.compile(
        r"""(?:import\s+[^'";]*?from\s+|import\s*\(\s*|require\(\s*)['"]([^'"]+)['"]"""
    ),
    "go": re.compile(r"""^\s*(?:import\s+)?(?:\w+\s+)?"([\w./-]+)"\s*$""", re.M),
    "rust": re.compile(r"^\s*(?:pub\s+)?(?:use|mod)\s+([\w:]+)", re.M),
    "java": re.compile(r"^\s*import\s+(?:static\s+)?([\w.]+)", re.M),
    "c": re.compile(r"""^\s*#\s*include\s+[<"]([^>"]+)[>"]""", re.M),
    "csharp": re.compile(r"^\s*using\s+([\w.]+)\s*;", re.M),
    "ruby": re.compile(r"""^\s*require(?:_relative)?\s+['"]([^'"]+)['"]""", re.M),
    "php": re.compile(r"^\s*(?:use|require(?:_once)?|include(?:_once)?)\s+['\"]?([\w\\/.]+)", re.M),
    "shell": re.compile(r"^\s*(?:source|\.)\s+(\S+)", re.M),
}
_IDENT_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]{3,}\b")
_NUMBERED_PREFIX_RE = re.compile(r"^\s*\d+(?:\t|→)", re.M)


def language_for(path: str) -> str | None:
    return _LANG_BY_EXT.get(os.path.splitext(path)[1].lower())


@dataclass
class FileNode:
    path: str
    lang: str | None
    content_hash: str
    defines: set[str] = field(default_factory=set)
    imports: set[str] = field(default_factory=set)
    mentions: set[str] = field(default_factory=set)


def parse_source(path: str, text: str) -> FileNode:
    lang = language_for(path)
    text = _NUMBERED_PREFIX_RE.sub("", text) if _NUMBERED_PREFIX_RE.search(text[:2000]) else text
    h = hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:16]
    node = FileNode(path, lang, h)
    if lang is None:
        return node
    for m in _DEF_PATTERNS[lang].finditer(text):
        name = next((g for g in m.groups() if g), None)
        if name and len(name) >= 2:
            node.defines.add(name)
    for m in _IMPORT_PATTERNS[lang].finditer(text):
        name = next((g for g in m.groups() if g), None)
        if name:
            node.imports.add(name.strip())
    node.mentions = set(_IDENT_RE.findall(text[:400_000]))
    return node


class CodeGraph:
    """Graph for one workspace (thread-safe, bounded)."""

    def __init__(self, workspace_key: str = "", *, max_files: int = 2000) -> None:
        self.workspace_key = workspace_key
        self.max_files = max_files
        self._lock = threading.RLock()
        self._files: OrderedDict[str, FileNode] = OrderedDict()
        self._symbol_to_files: dict[str, set[str]] = defaultdict(set)
        self._mentioned_by: dict[str, set[str]] = defaultdict(set)
        self._read_with: dict[str, set[str]] = defaultdict(set)

    # ------------------------------------------------------------ building
    def update_file(self, path: str, text: str) -> bool:
        """Parse ``text`` for ``path`` if it changed. Returns True when updated."""
        key = canonical_path(path)
        if not key or any(d in f"/{key}/" for d in _IGNORED_DIRS):
            return False
        h = hashlib.sha256(
            _NUMBERED_PREFIX_RE.sub("", text).encode("utf-8", "replace")
        ).hexdigest()[:16]
        with self._lock:
            existing = self._files.get(key)
            if existing is not None and existing.content_hash == h:
                self._files.move_to_end(key)
                return False
        node = parse_source(key, text)
        with self._lock:
            if existing is not None:
                self._unindex(key, existing)
            self._files[key] = node
            self._files.move_to_end(key)
            for sym in node.defines:
                self._symbol_to_files[sym].add(key)
            for sym in node.mentions:
                self._mentioned_by[sym].add(key)
            while len(self._files) > self.max_files:
                old_key, old = self._files.popitem(last=False)
                self._unindex(old_key, old)
        return True

    def _unindex(self, key: str, node: FileNode) -> None:
        for sym in node.defines:
            self._symbol_to_files[sym].discard(key)
        for sym in node.mentions:
            refs = self._mentioned_by.get(sym)
            if refs is not None:
                refs.discard(key)
                if not refs:
                    del self._mentioned_by[sym]

    def update_from_disk(self, path: str, *, roots: Iterable[str] = ()) -> bool:
        p = path.strip().strip("'\"")
        candidates = [p] if os.path.isabs(p) else [os.path.join(r, p) for r in roots if r]
        for cand in candidates:
            try:
                if os.path.isfile(cand) and os.path.getsize(cand) <= _MAX_FILE_BYTES:
                    with open(cand, encoding="utf-8", errors="replace") as fh:
                        return self.update_file(cand if os.path.isabs(p) else p, fh.read())
            except OSError:
                continue
        return False

    def note_read_together(self, paths: Iterable[str]) -> None:
        keys = [canonical_path(p) for p in paths if p]
        with self._lock:
            for a in keys:
                for b in keys:
                    if a != b:
                        self._read_with[a].add(b)

    def ingest_messages(self, messages: list[dict], *, roots: Iterable[str] = ()) -> int:
        """Feed file contents seen in tool results; returns files updated."""
        updated = 0
        by_message: dict[int, list[str]] = defaultdict(list)
        for ref in iter_tool_results(messages, build_tool_call_index(messages)):
            res = resource_for(ref.tool_name, ref.tool_input)
            if res is None or res.kind != "file":
                continue
            path = res.label
            if language_for(path) is None:
                continue
            by_message[ref.message_index].append(path)
            if (
                res.is_read
                and ref.text
                and not any(m in ref.text for m in ("Retrieve original: hash=", "<<ccr:"))
            ):
                updated += int(self.update_file(path, ref.text))
            elif roots:
                updated += int(self.update_from_disk(path, roots=roots))
        for paths in by_message.values():
            if len(paths) > 1:
                self.note_read_together(paths)
        return updated

    # ------------------------------------------------------------- queries
    def _edges(self, key: str) -> set[str]:
        node = self._files.get(key)
        out: set[str] = set(self._read_with.get(key, ()))
        if node is None:
            return out
        for imp in node.imports:
            tail = imp.replace(".", "/").replace("::", "/").lstrip("./")
            for other in self._files:
                if other != key and (
                    other.endswith(tail + os.path.splitext(other)[1]) or f"/{tail}/" in f"/{other}"
                ):
                    out.add(other)
        for sym in node.mentions & set(self._symbol_to_files):
            out.update(f for f in self._symbol_to_files[sym] if f != key)
        for sym in node.defines:
            out.update(
                f for f in self._mentioned_by.get(sym, ()) if f != key
            )  # callers / referrers
        return out

    def neighborhood(
        self,
        *,
        files: Iterable[str] = (),
        symbols: Iterable[str] = (),
        max_distance: int = 2,
        limit: int = 24,
    ) -> tuple[list[str], list[str]]:
        """``(nearby_files, nearby_symbols)`` by BFS from the active nodes."""
        with self._lock:
            starts = {canonical_path(f) for f in files if f}
            for sym in symbols:
                starts.update(self._symbol_to_files.get(sym, ()))
            starts = {s for s in starts if s in self._files}
            if not starts:
                return [], []
            dist: dict[str, int] = dict.fromkeys(starts, 0)
            queue = deque(sorted(starts))
            while queue:
                cur = queue.popleft()
                if dist[cur] >= max_distance:
                    continue
                for nxt in sorted(self._edges(cur)):
                    if nxt not in dist:
                        dist[nxt] = dist[cur] + 1
                        queue.append(nxt)
            ordered = sorted(dist, key=lambda k: (dist[k], k))
            near_files = [f for f in ordered if f not in starts][:limit]
            near_syms: list[str] = []
            for f in ordered:
                for sym in sorted(self._files[f].defines):
                    if sym not in near_syms:
                        near_syms.append(sym)
                if len(near_syms) >= limit:
                    break
            self._last_distances = dist
            return near_files, near_syms[:limit]

    def proximity(self, text: str, distances: dict[str, int] | None = None) -> float:
        """1.0 when ``text`` mentions an active/near file or its symbols, decaying with distance."""
        dist = distances if distances is not None else getattr(self, "_last_distances", {})
        if not dist or not text:
            return 0.0
        lowered = text.lower()
        best = 0.0
        for f, d in dist.items():
            base = f.rsplit("/", 1)[-1]
            if base and base.lower() in lowered:
                best = max(best, 1.0 / (1 + d))
        return best

    @property
    def file_count(self) -> int:
        return len(self._files)


class GraphRegistry:
    """Per-workspace graphs (bounded LRU of workspaces)."""

    def __init__(self, *, max_workspaces: int = 16, max_files: int = 2000) -> None:
        self._graphs: OrderedDict[str, CodeGraph] = OrderedDict()
        self._lock = threading.Lock()
        self.max_workspaces = max_workspaces
        self.max_files = max_files

    def for_workspace(self, key: str) -> CodeGraph:
        with self._lock:
            g = self._graphs.get(key)
            if g is None:
                g = CodeGraph(key, max_files=self.max_files)
                self._graphs[key] = g
            self._graphs.move_to_end(key)
            while len(self._graphs) > self.max_workspaces:
                self._graphs.popitem(last=False)
            return g
