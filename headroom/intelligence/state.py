"""On-disk state for the intelligence layer.

Layout under ``<workspace>/intelligence`` (``~/.headroom/intelligence`` by
default, ``%USERPROFILE%\\.headroom\\intelligence`` on Windows):

* ``setup.json``   — last ``headroom intelligence setup`` report: discovered
  ``llama-server``, capabilities, model location, protocol check results.
* ``runtime.json`` — the live service: pid, executable, url, model,
  start time, ownership. Stale entries (dead pid) are cleaned on read.
* ``models/``      — GGUF files fetched by Headroom's fallback downloader.
* ``llama.cpp-managed/`` — Headroom-managed llama.cpp checkout + build.
* ``logs/``        — llama-server stdout/stderr of owned processes.

All writes are atomic (temp file + ``os.replace``) so a crash never leaves a
half-written record, on Windows or POSIX.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from headroom import paths
from headroom._subprocess import identity_mismatch, pid_alive, proc_identity

logger = logging.getLogger(__name__)

INTELLIGENCE_DIR_ENV = "HEADROOM_INTELLIGENCE_DIR"


def intelligence_dir() -> Path:
    override = os.environ.get(INTELLIGENCE_DIR_ENV, "").strip()
    if override:
        return Path(override).expanduser()
    return paths.workspace_dir() / "intelligence"


def setup_path() -> Path:
    return intelligence_dir() / "setup.json"


def runtime_path() -> Path:
    return intelligence_dir() / "runtime.json"


def models_dir() -> Path:
    return intelligence_dir() / "models"


def managed_llama_root() -> Path:
    return intelligence_dir() / "llama.cpp-managed"


def logs_dir() -> Path:
    return intelligence_dir() / "logs"


def write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        logger.debug("ignoring unreadable %s: %s", path, exc)
        return None
    return data if isinstance(data, dict) else None


def read_setup() -> dict[str, Any] | None:
    return read_json(setup_path())


def write_setup(report: dict[str, Any]) -> None:
    write_json_atomic(setup_path(), {**report, "written_at": time.time()})


def record_runtime(
    *,
    pid: int,
    executable: str,
    url: str,
    model_repo: str,
    model_file: str,
    owned: bool,
    proxy_pid: int | None = None,
) -> dict[str, Any]:
    ident = proc_identity(pid)
    record = {
        "pid": pid,
        "executable": executable,
        "url": url,
        "model_repo": model_repo,
        "model_file": model_file,
        "owned": owned,
        "started_at": time.time(),
        "proxy_pid": proxy_pid if proxy_pid is not None else os.getpid(),
        "identity_source": ident[0] if ident else None,
        "identity_start": ident[1] if ident else None,
    }
    write_json_atomic(runtime_path(), record)
    return record


def read_runtime(*, clean_stale: bool = True) -> dict[str, Any] | None:
    """Return the live runtime record, removing it when its process is gone."""
    record = read_json(runtime_path())
    if record is None:
        return None
    pid = record.get("pid")
    alive = isinstance(pid, int) and pid_alive(pid)
    if alive and isinstance(pid, int):
        if identity_mismatch(record.get("identity_source"), record.get("identity_start"), pid):
            alive = False  # PID was recycled by an unrelated process
    if not alive:
        if clean_stale:
            clear_runtime()
        return None
    return record


def clear_runtime() -> None:
    try:
        runtime_path().unlink()
    except FileNotFoundError:
        pass
    except OSError as exc:
        logger.debug("could not remove runtime record: %s", exc)
