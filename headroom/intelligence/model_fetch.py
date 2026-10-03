"""Locate or fetch the JevK5 GGUF (plan §4.6).

Primary path: ``llama-server --hf-repo … --hf-file …`` downloads/reuses the
model itself. This module covers the rest:

* :func:`find_cached_model` — look for an already-downloaded copy in
  llama.cpp's cache (``LLAMA_CACHE``; XDG cache on Linux,
  ``%LOCALAPPDATA%\\llama.cpp`` on Windows, ``~/Library/Caches/llama.cpp`` on
  macOS), the Hugging Face hub cache, and Headroom's own model dir, so a model
  is never downloaded twice.
* :func:`download_model` — resumable fallback downloader: writes
  ``<file>.part`` with HTTP Range resume, verifies size (and sha256 when the
  Hub publishes one), then atomically renames into place.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import ssl
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from pathlib import Path

from .state import models_dir

logger = logging.getLogger(__name__)

HF_BASE = "https://huggingface.co"
CHUNK = 1 << 20


def llama_cache_dirs(environ: Mapping[str, str] | None = None) -> list[Path]:
    env = os.environ if environ is None else environ
    dirs: list[Path] = []
    if env.get("LLAMA_CACHE"):
        dirs.append(Path(env["LLAMA_CACHE"]).expanduser())
    home = Path(env.get("USERPROFILE") or env.get("HOME") or str(Path.home()))
    if sys.platform == "win32":
        local = Path(env.get("LOCALAPPDATA") or home / "AppData" / "Local")
        dirs.append(local / "llama.cpp")
    elif sys.platform == "darwin":
        dirs.append(home / "Library" / "Caches" / "llama.cpp")
    else:
        xdg = env.get("XDG_CACHE_HOME")
        dirs.append((Path(xdg) if xdg else home / ".cache") / "llama.cpp")
    return dirs


def hf_hub_cache_dirs(environ: Mapping[str, str] | None = None) -> list[Path]:
    env = os.environ if environ is None else environ
    dirs: list[Path] = []
    if env.get("HF_HUB_CACHE"):
        dirs.append(Path(env["HF_HUB_CACHE"]).expanduser())
    if env.get("HF_HOME"):
        dirs.append(Path(env["HF_HOME"]).expanduser() / "hub")
    home = Path(env.get("USERPROFILE") or env.get("HOME") or str(Path.home()))
    dirs.append(home / ".cache" / "huggingface" / "hub")
    return dirs


def find_cached_model(
    repo: str, filename: str, environ: Mapping[str, str] | None = None
) -> Path | None:
    """Return a complete local copy of ``repo/filename`` if one exists."""
    org_repo = repo.replace("/", "_")
    for base in llama_cache_dirs(environ):
        for name in (f"{org_repo}_{filename}", filename):
            cand = base / name
            if cand.is_file() and cand.stat().st_size > 0:
                return cand
    hub_name = "models--" + repo.replace("/", "--")
    for base in hf_hub_cache_dirs(environ):
        snaps = base / hub_name / "snapshots"
        if snaps.is_dir():
            for snap in sorted(snaps.iterdir(), reverse=True):
                cand = snap / filename
                if cand.is_file():
                    return cand
    own = models_dir() / filename
    if own.is_file() and own.stat().st_size > 0:
        return own
    return None


def _ssl_context() -> ssl.SSLContext:
    # Corporate TLS inspection roots live in the OS trust store; Headroom
    # already depends on truststore for exactly this reason.
    try:
        import truststore

        return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    except Exception:  # noqa: BLE001
        return ssl.create_default_context()


def _headers(extra: Mapping[str, str] | None = None) -> dict[str, str]:
    headers = {"User-Agent": "headroom-intelligence/1"}
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if extra:
        headers.update(extra)
    return headers


def expected_sha256(
    repo: str, filename: str, timeout: float = 20.0
) -> tuple[str | None, int | None]:
    """Best-effort (sha256, size) from the Hub tree API; ``(None, None)`` on failure."""
    url = f"{HF_BASE}/api/models/{repo}/tree/main"
    try:
        req = urllib.request.Request(url, headers=_headers())
        with urllib.request.urlopen(req, timeout=timeout, context=_ssl_context()) as resp:
            items = json.loads(resp.read())
    except (urllib.error.URLError, OSError, ValueError):
        return None, None
    for item in items if isinstance(items, list) else []:
        if isinstance(item, dict) and item.get("path") == filename:
            lfs = item.get("lfs") or {}
            return lfs.get("oid") or lfs.get("sha256"), item.get("size") or lfs.get("size")
    return None, None


def download_model(
    repo: str,
    filename: str,
    *,
    dest_dir: Path | None = None,
    progress: Callable[[int, int | None], None] | None = None,
    timeout: float = 60.0,
    verify: bool = True,
    base_url: str = HF_BASE,
) -> Path:
    """Resumable download to ``dest_dir/filename`` via ``filename.part``."""
    target_dir = dest_dir or models_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    final = target_dir / filename
    if final.is_file() and final.stat().st_size > 0:
        return final
    part = target_dir / f"{filename}.part"
    url = f"{base_url}/{repo}/resolve/main/{filename}"
    sha, size_hint = (
        expected_sha256(repo, filename) if verify and base_url == HF_BASE else (None, None)
    )

    attempts = 0
    while True:
        attempts += 1
        have = part.stat().st_size if part.exists() else 0
        extra = {"Range": f"bytes={have}-"} if have else None
        req = urllib.request.Request(url, headers=_headers(extra))
        try:
            with urllib.request.urlopen(req, timeout=timeout, context=_ssl_context()) as resp:
                status = getattr(resp, "status", 200)
                if have and status == 200:
                    # Server ignored the Range header: restart cleanly.
                    have = 0
                    mode = "wb"
                else:
                    mode = "ab" if have else "wb"
                length = resp.headers.get("Content-Length")
                total = (have + int(length)) if length and length.isdigit() else size_hint
                with open(part, mode) as fh:
                    done = have
                    while True:
                        chunk = resp.read(CHUNK)
                        if not chunk:
                            break
                        fh.write(chunk)
                        done += len(chunk)
                        if progress is not None:
                            progress(done, total)
                    fh.flush()
                    os.fsync(fh.fileno())
            if size_hint and part.stat().st_size < int(size_hint) and attempts < 8:
                # Stream ended early without an error (proxy reset, sleep):
                # resume from where we are rather than failing the setup.
                time.sleep(min(16, 2**attempts))
                continue
            break
        except urllib.error.HTTPError as exc:
            if exc.code == 416 and part.exists():
                break  # already complete
            if attempts >= 4 or exc.code in (401, 403, 404):
                raise
        except (urllib.error.URLError, OSError, TimeoutError):
            if attempts >= 4:
                raise
        time.sleep(min(16, 2**attempts))

    size = part.stat().st_size
    if size_hint and size != int(size_hint):
        raise OSError(f"incomplete download: {size} of {size_hint} bytes (kept {part} for resume)")
    if sha:
        h = hashlib.sha256()
        with open(part, "rb") as fh:
            for block in iter(lambda: fh.read(CHUNK), b""):
                h.update(block)
        if h.hexdigest() != sha.lower():
            part.unlink(missing_ok=True)
            raise OSError("sha256 mismatch for downloaded model; removed partial file")
    os.replace(part, final)
    return final
