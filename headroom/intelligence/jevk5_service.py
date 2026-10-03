"""JevK5 / llama-server process lifecycle (plan §4.6–4.8, §32).

* Binds **only** ``127.0.0.1`` (an operator-supplied ``HEADROOM_JEVK5_URL`` is
  used as-is and never started or stopped).
* Reserves a free loopback port (preferring 8091 upward), releases it right
  before launch and retries on an "address in use" collision.
* POSIX: the child gets its own session/process group so stop() can signal the
  whole group. Windows: ``CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW`` — no
  stray console, and termination uses ``taskkill /T`` for the tree.
* Records pid/executable/url/model/ownership in ``runtime.json``; stale
  records (dead pid, recycled pid) are cleaned on read.
* An instance already started by another Headroom process is reused, never
  restarted; an *owned* child is stopped when its owner shuts down.
* GPU out-of-memory at load time triggers one retry with
  ``--n-gpu-layers 0`` (CPU), else the deterministic path simply continues.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from headroom import _subprocess

from .config import JevK5Settings
from .llama_discovery import LlamaCapabilities, discover, no_window_flags
from .model_fetch import find_cached_model
from .state import (
    clear_runtime,
    logs_dir,
    managed_llama_root,
    read_runtime,
    read_setup,
    record_runtime,
)

logger = logging.getLogger(__name__)

IS_WINDOWS = sys.platform == "win32"
LOOPBACK = "127.0.0.1"
PREFERRED_PORT = 8091
_OOM_MARKERS = (
    "out of memory",
    "cudamalloc failed",
    "failed to allocate",
    "erroroutofdevicememory",
    "unable to allocate",
    "not enough memory",
)
_ADDR_IN_USE_MARKERS = (
    "address already in use",
    "couldn't bind",
    "bind failed",
    "only one usage of each socket",
)


def reserve_port(preferred: int = PREFERRED_PORT, *, span: int = 40) -> int:
    """Return a currently-free loopback port, trying ``preferred`` upward first."""
    candidates = list(range(preferred, preferred + span)) if preferred else []
    for port in [*candidates, 0]:
        with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
            if not IS_WINDOWS:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((LOOPBACK, port))
            except OSError:
                continue
            return int(sock.getsockname()[1])
    raise OSError("no free loopback port available")


def http_json(
    url: str, payload: dict[str, Any] | None = None, *, timeout: float = 5.0
) -> tuple[int, Any]:
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
            status = getattr(resp, "status", 200)
    except urllib.error.HTTPError as exc:
        try:
            body = exc.read()
        except Exception:  # noqa: BLE001
            body = b""
        status = exc.code
    try:
        return status, json.loads(body) if body else None
    except ValueError:
        return status, None


def health(url: str, *, timeout: float = 2.0) -> tuple[bool, str]:
    """``(ready, detail)`` from ``GET /health``."""
    try:
        status, body = http_json(f"{url.rstrip('/')}/health", timeout=timeout)
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        return False, f"unreachable: {type(exc).__name__}"
    if status == 200:
        detail = body.get("status", "ok") if isinstance(body, dict) else "ok"
        return detail in ("ok", "ready", True), str(detail)
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict):
            return False, str(err.get("message") or status)
    return False, f"http {status}"


@dataclass
class ServiceStatus:
    state: str  # disabled | external | running | starting | stopped | unavailable | failed
    url: str = ""
    owned: bool = False
    pid: int | None = None
    reason: str = ""
    executable: str = ""
    model: str = ""
    capabilities: dict[str, Any] = field(default_factory=dict)

    @property
    def usable(self) -> bool:
        return self.state in ("running", "external")

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "url": self.url,
            "owned": self.owned,
            "pid": self.pid,
            "reason": self.reason,
            "executable": self.executable,
            "model": self.model,
        }


class JevK5Service:
    """Owns at most one llama-server child per process."""

    def __init__(self, settings: JevK5Settings) -> None:
        self.settings = settings
        self._lock = threading.RLock()
        self._proc: subprocess.Popen[bytes] | None = None
        self._status = ServiceStatus("stopped")
        self._log_path: Path | None = None
        self._starter: threading.Thread | None = None

    # ------------------------------------------------------------------ query
    @property
    def status(self) -> ServiceStatus:
        with self._lock:
            if self._status.state in ("running", "starting") and self._proc is not None:
                if self._proc.poll() is not None:
                    self._status = ServiceStatus(
                        "failed", reason=f"llama-server exited with {self._proc.returncode}"
                    )
                    clear_runtime()
            return self._status

    def url(self) -> str | None:
        st = self.status
        return st.url if st.usable else None

    # -------------------------------------------------------------- discovery
    def resolve_llama(self, *, allow_build: bool) -> LlamaCapabilities | None:
        recorded = ""
        setup = read_setup() or {}
        llama = setup.get("llama_server")
        if isinstance(llama, dict) and isinstance(llama.get("path"), str):
            recorded = llama["path"]
        result = discover(
            explicit=self.settings.llama_server,
            recorded=recorded,
            managed_root=managed_llama_root() / "src",
        )
        if result.selected is not None:
            return result.selected
        if not allow_build:
            return None
        from .llama_build import build_managed
        from .llama_discovery import probe

        built = build_managed()
        if not built.ok:
            logger.warning("JevK5: managed llama.cpp build failed: %s", built.reason)
            return None
        caps = probe(built.executable, "managed")
        return caps if caps.ok else None

    def model_args(self, *, allow_download: bool) -> list[str] | None:
        local = find_cached_model(self.settings.model_repo, self.settings.model_file)
        if local is not None:
            return ["--model", str(local)]
        if allow_download:
            return ["--hf-repo", self.settings.model_repo, "--hf-file", self.settings.model_file]
        return None

    def launch_args(
        self, caps: LlamaCapabilities, port: int, model_args: list[str], *, cpu_only: bool = False
    ) -> list[str]:
        ngl = self.settings.gpu_layers or "auto"
        if cpu_only:
            ngl = "0"
        elif ngl == "auto" and not caps.ngl_auto:
            ngl = "99" if caps.gpu else "0"
        return [
            caps.path,
            *model_args,
            "--ctx-size",
            str(self.settings.ctx),
            "--n-gpu-layers",
            ngl,
            "--host",
            LOOPBACK,
            "--port",
            str(port),
            "--parallel",
            "1",
        ]

    # --------------------------------------------------------------- lifecycle
    def start(
        self,
        *,
        allow_download: bool | None = None,
        allow_build: bool | None = None,
        wait: bool = True,
        wait_timeout_s: float = 120.0,
    ) -> ServiceStatus:
        """Start (or adopt) the decision service. Idempotent, never raises."""
        with self._lock:
            if self.settings.mode == "off":
                self._status = ServiceStatus("disabled", reason="HEADROOM_JEVK5=off")
                return self._status
            if self.settings.external:
                ready, detail = health(self.settings.url)
                self._status = ServiceStatus(
                    "external" if ready else "unavailable",
                    url=self.settings.url,
                    reason="" if ready else detail,
                )
                return self._status
            current = self.status
            if current.state in ("running", "starting"):
                return current
            existing = read_runtime()
            if existing and isinstance(existing.get("url"), str):
                ready, _ = health(existing["url"])
                if ready and existing.get("model_file") == self.settings.model_file:
                    proxy_pid = existing.get("proxy_pid")
                    from headroom._subprocess import pid_alive

                    owner_alive = isinstance(proxy_pid, int) and pid_alive(proxy_pid)
                    # Reuse a live instance. Adopt ownership only if its owner
                    # is gone, so it can never be orphaned.
                    owned = not owner_alive
                    if owned:
                        record_runtime(
                            pid=int(existing["pid"]),
                            executable=str(existing.get("executable", "")),
                            url=existing["url"],
                            model_repo=self.settings.model_repo,
                            model_file=self.settings.model_file,
                            owned=True,
                        )
                    self._status = ServiceStatus(
                        "running",
                        url=existing["url"],
                        owned=owned,
                        pid=int(existing["pid"]),
                        executable=str(existing.get("executable", "")),
                        model=self.settings.model_file,
                        reason="reused",
                    )
                    return self._status

            dl = self.settings.allow_download if allow_download is None else allow_download
            build = self.settings.allow_build if allow_build is None else allow_build
            caps = self.resolve_llama(allow_build=build)
            if caps is None:
                self._status = ServiceStatus(
                    "unavailable",
                    reason="no compatible llama-server found"
                    + ("" if build else " (run `headroom intelligence setup` to build one)"),
                )
                return self._status
            margs = self.model_args(allow_download=dl)
            if margs is None:
                self._status = ServiceStatus(
                    "unavailable",
                    reason=(
                        f"model {self.settings.model_file} is not downloaded yet "
                        "(run `headroom intelligence setup` or set HEADROOM_JEVK5=on)"
                    ),
                    executable=caps.path,
                    capabilities=caps.to_dict(),
                )
                return self._status
            self._launch(caps, margs)
            if not wait:
                self._starter = threading.Thread(
                    target=self._await_ready,
                    args=(caps, margs, wait_timeout_s),
                    name="headroom-jevk5-ready",
                    daemon=True,
                )
                self._starter.start()
                return self._status
        return self._await_ready(caps, margs, wait_timeout_s)

    def _launch(self, caps: LlamaCapabilities, margs: list[str], *, cpu_only: bool = False) -> None:
        port = self.settings.port or reserve_port()
        args = self.launch_args(caps, port, margs, cpu_only=cpu_only)
        logs_dir().mkdir(parents=True, exist_ok=True)
        self._log_path = logs_dir() / f"llama-server-{port}.log"
        log = open(self._log_path, "ab")  # noqa: SIM115 - handed to the child
        kwargs: dict[str, Any] = {
            "stdout": log,
            "stderr": subprocess.STDOUT,
            "stdin": subprocess.DEVNULL,
        }
        if IS_WINDOWS:
            kwargs["creationflags"] = (
                getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200) | no_window_flags()
            )
        else:
            kwargs["start_new_session"] = True
        try:
            self._proc = _subprocess.Popen(args, **kwargs)
        except OSError as exc:
            log.close()
            self._status = ServiceStatus(
                "failed", reason=f"launch failed: {exc}", executable=caps.path
            )
            return
        finally:
            with contextlib.suppress(Exception):
                log.close()
        url = f"http://{LOOPBACK}:{port}"
        record_runtime(
            pid=self._proc.pid,
            executable=caps.path,
            url=url,
            model_repo=self.settings.model_repo,
            model_file=self.settings.model_file,
            owned=True,
        )
        self._status = ServiceStatus(
            "starting",
            url=url,
            owned=True,
            pid=self._proc.pid,
            executable=caps.path,
            model=self.settings.model_file,
            capabilities=caps.to_dict(),
        )

    def _log_tail(self, limit: int = 8192) -> str:
        if self._log_path is None:
            return ""
        try:
            with open(self._log_path, "rb") as fh:
                fh.seek(0, os.SEEK_END)
                size = fh.tell()
                fh.seek(max(0, size - limit))
                return fh.read().decode("utf-8", "replace")
        except OSError:
            return ""

    def _await_ready(
        self, caps: LlamaCapabilities, margs: list[str], timeout_s: float, *, retried: bool = False
    ) -> ServiceStatus:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            with self._lock:
                proc, st = self._proc, self._status
            if proc is None or st.state not in ("starting", "running"):
                return st
            code = proc.poll()
            if code is not None:
                tail = self._log_tail().lower()
                clear_runtime()
                if not retried and any(m in tail for m in _ADDR_IN_USE_MARKERS):
                    with self._lock:
                        self._launch(caps, margs)
                    return self._await_ready(caps, margs, timeout_s, retried=True)
                if not retried and any(m in tail for m in _OOM_MARKERS):
                    logger.warning("JevK5: GPU out of memory loading the model; retrying on CPU")
                    with self._lock:
                        self._launch(caps, margs, cpu_only=True)
                    return self._await_ready(caps, margs, timeout_s, retried=True)
                with self._lock:
                    self._status = ServiceStatus(
                        "failed",
                        reason=f"llama-server exited with {code} (log: {self._log_path})",
                        executable=caps.path,
                    )
                    return self._status
            ready, _detail = health(st.url, timeout=2.0)
            if ready:
                with self._lock:
                    self._status.state = "running"
                    return self._status
            time.sleep(0.5)
        with self._lock:
            if self._status.state == "starting":
                self._status.reason = "still loading (timed out waiting for /health)"
            return self._status

    def stop(self, *, force: bool = False) -> bool:
        """Stop an owned instance. External/adopted-by-others instances are left alone."""
        with self._lock:
            if self.settings.external:
                return False
            proc = self._proc
            st = self._status
            if proc is not None:
                _terminate_tree(proc.pid, proc)
                self._proc = None
                clear_runtime()
                self._status = ServiceStatus("stopped")
                return True
            if (st.owned or force) and st.pid:
                _terminate_tree(st.pid, None)
                clear_runtime()
                self._status = ServiceStatus("stopped")
                return True
            if force:
                record = read_runtime()
                if record and record.get("owned") and isinstance(record.get("pid"), int):
                    _terminate_tree(record["pid"], None)
                    clear_runtime()
                    return True
            return False


def _terminate_tree(
    pid: int, proc: subprocess.Popen[bytes] | None, *, grace_s: float = 5.0
) -> None:
    if IS_WINDOWS:
        with contextlib.suppress(Exception):
            _subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                capture_output=True,
                timeout=15,
                creationflags=no_window_flags(),
            )
        if proc is not None:
            with contextlib.suppress(Exception):
                proc.wait(timeout=grace_s)
        return
    try:
        pgid = os.getpgid(pid)
    except (ProcessLookupError, PermissionError, OSError):
        pgid = None
    sent = False
    if pgid is not None and pgid != os.getpgid(0):
        with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
            os.killpg(pgid, signal.SIGTERM)
            sent = True
    if not sent:
        with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
            os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + grace_s
    while time.monotonic() < deadline:
        if proc is not None:
            if proc.poll() is not None:
                return
        else:
            from headroom._subprocess import pid_alive

            if not pid_alive(pid):
                return
        time.sleep(0.1)
    if pgid is not None and pgid != os.getpgid(0):
        with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
            os.killpg(pgid, signal.SIGKILL)
    with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
        os.kill(pid, signal.SIGKILL)
    if proc is not None:
        with contextlib.suppress(Exception):
            proc.wait(timeout=2)


_SERVICES: dict[tuple[Any, ...], JevK5Service] = {}
_SERVICES_LOCK = threading.Lock()


def get_service(settings: JevK5Settings) -> JevK5Service:
    key = (
        settings.url,
        settings.llama_server,
        settings.model_repo,
        settings.model_file,
        settings.port,
    )
    with _SERVICES_LOCK:
        svc = _SERVICES.get(key)
        if svc is None:
            svc = JevK5Service(settings)
            _SERVICES[key] = svc
        return svc


def stop_all_owned() -> int:
    """Stop every owned service of this process (proxy shutdown hook)."""
    stopped = 0
    with _SERVICES_LOCK:
        services = list(_SERVICES.values())
    for svc in services:
        try:
            if svc.status.owned and svc.stop():
                stopped += 1
        except Exception:  # noqa: BLE001
            logger.debug("JevK5 stop failed", exc_info=True)
    return stopped
