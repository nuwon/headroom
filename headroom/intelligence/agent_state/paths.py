"""Workspace path semantics for the validator and the scope firewall.

* Paths are resolved against the session ``cwd`` when relative.
* Symlinks and Windows junctions are resolved (``os.path.realpath``) before
  any workspace-bound decision, so a link pointing outside the project cannot
  smuggle a write out of it (plan §17.4).
* Comparisons are case-insensitive on Windows and case-sensitive elsewhere.
* Drive letters and UNC prefixes are kept intact. ``..`` segments are folded
  before the containment check.
"""

from __future__ import annotations

import ntpath
import os
import posixpath
import re

_WIN_DRIVE_RE = re.compile(r"^[A-Za-z]:")


def is_windows_path(path: str) -> bool:
    return bool(_WIN_DRIVE_RE.match(path or "")) or (path or "").startswith("\\\\")


def is_absolute(path: str) -> bool:
    p = path or ""
    return p.startswith("/") or is_windows_path(p) or p.startswith("~")


def join(base: str, path: str) -> str:
    p = (path or "").strip().strip("'\"")
    if not p:
        return base
    if p.startswith("~"):
        return os.path.expanduser(p)
    if is_absolute(p) or not base:
        return p
    if is_windows_path(base):
        return ntpath.join(base, p)
    return posixpath.join(base, p)


def resolve(path: str, *, cwd: str = "") -> str:
    """Absolute, ``..``-folded and (where it exists locally) symlink-resolved path."""
    full = join(cwd, path)
    if is_windows_path(full) and os.name != "nt":
        return ntpath.normpath(full)
    try:
        return os.path.realpath(full)
    except (OSError, ValueError):
        return os.path.normpath(full)


def _cmp(path: str) -> str:
    p = (path or "").replace("\\", "/").rstrip("/")
    if os.name == "nt" or is_windows_path(path):
        p = p.casefold()
    return p


def within(path: str, root: str) -> bool:
    """True when ``path`` (already resolved) lies inside ``root``."""
    if not path or not root:
        return False
    p, r = _cmp(path), _cmp(root)
    return p == r or p.startswith(r + "/")


def relative_to_root(path: str, root: str) -> str:
    """Project-relative ``/``-separated path, or the input when outside the root."""
    if not path:
        return ""
    if not root:
        return path.replace("\\", "/")
    p = path.replace("\\", "/")
    r = root.replace("\\", "/").rstrip("/")
    if _cmp(p) == _cmp(r):
        return "."
    if _cmp(p).startswith(_cmp(r) + "/"):
        return p[len(r) + 1 :]
    return p


def escapes_root(path: str, root: str, *, cwd: str = "") -> bool:
    """True when ``path`` resolves (through links, ``..``) outside ``root``."""
    if not path or not root:
        return False
    return not within(resolve(path, cwd=cwd or root), resolve(root))


def top_component(rel: str, depth: int = 1) -> str:
    parts = [p for p in (rel or "").replace("\\", "/").split("/") if p and p != "."]
    return "/".join(parts[:depth])
