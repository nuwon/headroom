"""Stable resource identities for tool results (plan §12, "Resource identity").

A re-read of the same file, a re-run of the same search or test command, or
a repeated listing should map to the same identity across turns — never the
tool-call id, which changes every call.

Identities:

* file read      -> ``file:<canonical path>#<range>``
* directory/glob -> ``list:<canonical dir>|<pattern>``
* grep/search    -> ``search:<pattern>|<scope>|<flags>``
* shell command  -> ``cmd:<normalized command>`` (file-reading commands map to
  ``file:`` so a ``cat`` and a ``Read`` of the same file share an identity)
* anything else  -> ``tool:<name>:<sha of canonical input>``

Paths are canonicalized for Windows and POSIX alike: backslashes become ``/``,
``..``/``.`` segments are folded, drive letters and (on case-insensitive
Windows paths) the whole path are lower-cased, and an existing local path is
resolved through symlinks (the proxy runs on the same machine as the agent).
"""

from __future__ import annotations

import hashlib
import json
import os
import posixpath
import re
from dataclasses import dataclass
from typing import Any

READ_TOOL_NAMES = frozenset({"read", "read_file", "view", "readfile", "open_file", "cat"})
LIST_TOOL_NAMES = frozenset(
    {"glob", "ls", "list", "list_dir", "list_directory", "find_files", "listdir"}
)
SEARCH_TOOL_NAMES = frozenset(
    {"grep", "search", "search_files", "codebase_search", "ripgrep", "find_in_files"}
)
SHELL_TOOL_NAMES = frozenset(
    {
        "bash",
        "shell",
        "local_shell",
        "shell_command",
        "powershell",
        "pwsh",
        "exec_command",
        "run_terminal_cmd",
    }
)
_WIN_DRIVE_RE = re.compile(r"^[A-Za-z]:")
_RANGE_KEYS = ("offset", "limit", "line_range", "start_line", "end_line", "ranges", "head_limit")


def canonical_path(raw: str, *, resolve_symlinks: bool = True) -> str:
    """Normalize a path for identity purposes (does not require existence)."""
    p = (raw or "").strip().strip("'\"")
    if not p:
        return ""
    windows_like = bool(_WIN_DRIVE_RE.match(p)) or ("\\" in p and "/" not in p)
    if resolve_symlinks and not windows_like and os.path.isabs(p):
        try:
            if os.path.exists(p):
                p = os.path.realpath(p)
        except OSError:
            pass
    p = p.replace("\\", "/")
    drive = ""
    if _WIN_DRIVE_RE.match(p):
        drive, p = p[:2].lower(), p[2:]
    p = posixpath.normpath(p) if p else p
    if p == ".":
        p = ""
    out = f"{drive}{p}"
    if windows_like:
        out = out.lower()  # NTFS/ReFS are case-insensitive by default
    return out.rstrip("/") or out


def _norm_ws(text: str) -> str:
    return " ".join(str(text).split())


def _first(tool_input: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = tool_input.get(key)
        if isinstance(value, str) and value.strip():
            return value
        if isinstance(value, list) and value and all(isinstance(v, str) for v in value):
            return " ".join(value)
    return ""


@dataclass(frozen=True)
class ResourceRef:
    identity: str
    kind: str  # file | list | search | cmd | tool
    is_read: bool  # output is raw file content the agent may patch
    label: str  # short human label for representations


def _shell_command(tool_input: dict[str, Any]) -> str:
    cmd = tool_input.get("command", tool_input.get("cmd", ""))
    if isinstance(cmd, list):
        cmd = " ".join(str(c) for c in cmd)
    return cmd if isinstance(cmd, str) else ""


def _read_target_from_command(command: str) -> str | None:
    """File path read by a shell read command (cat/head/Get-Content/type …)."""
    from headroom.transforms.content_router import _is_read_command

    if not _is_read_command(command):
        return None
    tokens = re.findall(r"\"[^\"]*\"|'[^']*'|\S+", command)
    candidates = [
        t.strip("'\"")
        for t in tokens
        if not t.startswith("-")
        and ("." in t or "/" in t or "\\" in t)
        and not t.endswith((".exe", "|"))
    ]
    return candidates[-1] if candidates else None


# Absolute on either OS: "C:\\x", "C:/x", "\\\\server\\share", "/x", "~/x".
_ABSOLUTE_PATH_RE = re.compile(r"^(?:[A-Za-z]:[\\/]|[\\/]|~)")


def resource_for(tool_name: str, tool_input: dict[str, Any] | None) -> ResourceRef | None:
    """Return the stable resource identity for a tool invocation, or None."""
    tool_input = tool_input or {}
    name = (tool_name or "").strip()
    low = name.lower().split("__")[-1]  # mcp__server__tool -> tool
    rng = ",".join(
        f"{k}={tool_input[k]}" for k in _RANGE_KEYS if tool_input.get(k) not in (None, "")
    )
    if low in READ_TOOL_NAMES:
        path = canonical_path(
            _first(tool_input, "file_path", "path", "filename", "file", "target_file")
        )
        if not path:
            return None
        return ResourceRef(f"file:{path}#{rng}", "file", True, path)
    if low in LIST_TOOL_NAMES:
        path = canonical_path(
            _first(tool_input, "path", "directory", "dir", "target_directory") or "."
        )
        pattern = _norm_ws(_first(tool_input, "pattern", "glob", "glob_pattern"))
        return ResourceRef(f"list:{path}|{pattern}", "list", False, f"{path} {pattern}".strip())
    if low in SEARCH_TOOL_NAMES:
        pattern = _first(tool_input, "pattern", "query", "regex", "search")
        if not pattern:
            return None
        scope = canonical_path(_first(tool_input, "path", "include", "glob", "directory"))
        flags = ",".join(
            f"{k}={tool_input[k]}"
            for k in sorted(tool_input)
            if k
            not in ("pattern", "query", "regex", "search", "path", "include", "glob", "directory")
            and isinstance(tool_input[k], (str, int, bool))
        )
        return ResourceRef(
            f"search:{pattern}|{scope}|{flags}", "search", False, f"search {pattern!r}"
        )
    if low in SHELL_TOOL_NAMES:
        command = _shell_command(tool_input)
        if not command:
            return None
        target = _read_target_from_command(command)
        if target:
            workdir = _first(tool_input, "workdir", "cwd", "working_directory")
            if workdir and not _ABSOLUTE_PATH_RE.match(target.strip("'\"")):
                # Codex passes relative paths plus a per-call workdir; the
                # same relative name in two checkouts is two resources.
                target = workdir.rstrip("\\/") + "/" + target.strip("'\"")
            path = canonical_path(target)
            return ResourceRef(f"file:{path}#cmd", "file", True, path)
        from headroom.transforms.content_router import _strip_cd_prefix

        normalized = _norm_ws(_strip_cd_prefix(command))
        workdir = canonical_path(_first(tool_input, "workdir", "cwd", "working_directory"))
        return ResourceRef(f"cmd:{workdir}|{normalized}", "cmd", False, normalized[:80])
    if not name:
        return None
    try:
        canon = json.dumps(tool_input, sort_keys=True, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        canon = repr(sorted(tool_input.items()))
    digest = hashlib.sha256(canon.encode("utf-8", "replace")).hexdigest()[:16]
    return ResourceRef(f"tool:{name}:{digest}", "tool", False, name)


def content_hash(text: str) -> str:
    """Line-ending-insensitive content hash (CRLF and LF copies match)."""
    normalized = text.replace("\r\n", "\n")
    return hashlib.sha256(normalized.encode("utf-8", "surrogatepass")).hexdigest()[:24]
