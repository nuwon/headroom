"""Workspace, agent-session and event identities (plan §4.3, §4.8).

* ``workspace_id`` is ``SHA256(canonical real project root)[:24]``. The root is
  canonicalized with :func:`headroom.intelligence.resources.canonical_path`
  (symlinks resolved, separators unified, case folded on Windows), the same
  identity Phase 1 uses for resources. The raw path never becomes a filename.
* An *agent session* is the wrapped client's own session: Claude Code's
  ``metadata.user_id`` session suffix or ``x-claude-code-session-id``, Codex's
  ``prompt_cache_key`` or ``session_id`` header. When the client sends none, it
  is a hash of the conversation root. Inside one agent session each
  conversation *lineage* (main thread, subagents) keeps its own task state.
* Event ids are deterministic hashes of their provenance, so a retried or
  replayed request maps to the same id and the store's uniqueness constraints
  make consumers idempotent.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from headroom.intelligence.resources import canonical_path

_CLAUDE_SESSION_RE = re.compile(r"_session_([0-9a-fA-F-]{8,64})")
_UUIDISH_RE = re.compile(r"^[A-Za-z0-9_.:-]{6,128}$")
_PROJECT_MARKERS = (".git", ".hg", ".svn")
_CONTINUATION_RE = re.compile(
    r"(?i)^\s*(?:this session is being continued from a previous conversation"
    r"|<summary>|summary of (?:the )?previous conversation)"
)


def _sha(text: str, n: int = 24) -> str:
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()[:n]


def identity_path(path: str) -> str:
    """Comparison form of a path: canonical, and case-folded on Windows."""
    canon = canonical_path(path)
    if os.name == "nt":
        canon = canon.casefold()
    return canon


def find_project_root(cwd: str) -> str:
    """Nearest ancestor holding a VCS marker, else ``cwd`` itself (resolved)."""
    if not cwd:
        return ""
    try:
        start = Path(cwd).expanduser().resolve(strict=False)
    except (OSError, RuntimeError):
        return cwd
    for candidate in (start, *start.parents):
        try:
            if any((candidate / marker).exists() for marker in _PROJECT_MARKERS):
                return str(candidate)
        except OSError:
            break
    return str(start)


def workspace_id(project_root: str) -> str:
    return _sha("ws:" + identity_path(project_root))


def new_session_id() -> str:
    """Random 128-bit session id (hex)."""
    return os.urandom(16).hex()


def stable_id(*parts: Any, n: int = 32) -> str:
    raw = json.dumps([str(p) for p in parts], ensure_ascii=False, separators=(",", ":"))
    return _sha(raw, n)


def is_continuation_summary(text: str) -> bool:
    """True when a conversation root is a compaction summary of an earlier one."""
    return bool(_CONTINUATION_RE.search(text[:400] if text else ""))


def lineage_key(root_text: str) -> str:
    return _sha("lineage:" + " ".join((root_text or "").split())[:4000], 16)


def _header(headers: Any, name: str) -> str:
    if not headers:
        return ""
    try:
        value = headers.get(name) or headers.get(name.lower()) or ""
    except Exception:  # noqa: BLE001
        return ""
    return str(value).strip()


def claude_session_id(body: dict[str, Any], headers: Any = None) -> str:
    """Claude Code session id, or ``""`` when the client did not send one."""
    explicit = _header(headers, "x-claude-code-session-id")
    if explicit and _UUIDISH_RE.match(explicit):
        return explicit
    meta = body.get("metadata") if isinstance(body, dict) else None
    user_id = meta.get("user_id") if isinstance(meta, dict) else None
    if isinstance(user_id, str) and user_id:
        if user_id.lstrip().startswith("{"):
            try:
                decoded = json.loads(user_id)
            except (ValueError, TypeError):
                decoded = None
            if isinstance(decoded, dict):
                sid = decoded.get("session_id") or decoded.get("sessionId")
                if isinstance(sid, str) and _UUIDISH_RE.match(sid):
                    return sid
        m = _CLAUDE_SESSION_RE.search(user_id)
        if m:
            return m.group(1)
    return ""


def codex_session_id(payload: dict[str, Any], headers: Any = None) -> str:
    """Codex session (conversation) id, or ``""``."""
    for name in ("session_id", "x-codex-session-id", "conversation_id"):
        value = _header(headers, name)
        if value and _UUIDISH_RE.match(value):
            return value
    key = payload.get("prompt_cache_key") if isinstance(payload, dict) else None
    if isinstance(key, str) and _UUIDISH_RE.match(key.strip()):
        return key.strip()
    return ""


def detect_agent(headers: Any, body: dict[str, Any] | None, provider: str) -> str:
    """``claude_code`` | ``codex`` | ``""`` (not a recognized coding agent)."""
    ua = _header(headers, "user-agent").lower()
    originator = _header(headers, "originator").lower()
    if provider == "anthropic":
        if (
            "claude-cli" in ua
            or "claude-code" in ua
            or _header(headers, "x-claude-code-session-id")
        ):
            return "claude_code"
        if body and claude_session_id(body):
            return "claude_code"
        return ""
    if "codex" in ua or "codex" in originator:
        return "codex"
    return ""
