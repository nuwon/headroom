"""Direct-argv subprocess execution for Headroom-owned steps (plan §18).

There is never a shell string. Commands run as argv lists, with the same
privileges as ``headroom wrap`` and never elevated. On Windows the child gets
``CREATE_NO_WINDOW`` and its own process group, and a timeout kills the whole
tree with ``taskkill /T /F``. On POSIX the child gets its own session, and a
timeout kills the process group. Output is decoded as UTF-8 with replacement.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass

from headroom import _subprocess

MAX_CAPTURE = 4 * 1024 * 1024


@dataclass(frozen=True)
class ProcResult:
    argv: tuple[str, ...]
    returncode: int | None  # None: could not start or timed out
    stdout: str
    stderr: str
    duration_ms: float
    timed_out: bool = False
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    @property
    def output(self) -> str:
        if self.stderr and self.stdout:
            return self.stdout + "\n" + self.stderr
        return self.stdout or self.stderr


def _kill_tree(proc: subprocess.Popen) -> None:
    try:
        if sys.platform.startswith("win"):
            _subprocess.run(
                ["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                capture_output=True,
                timeout=10,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (OSError, subprocess.SubprocessError):
        try:
            proc.kill()
        except OSError:
            pass


def run_argv(
    argv: list[str] | tuple[str, ...],
    *,
    cwd: str,
    timeout: float = 120.0,
    env: dict[str, str] | None = None,
) -> ProcResult:
    started = time.perf_counter()
    kwargs: dict = {
        "cwd": cwd or None,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "stdin": subprocess.DEVNULL,
        "text": True,
        "env": env,
    }
    if sys.platform.startswith("win"):
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(
            subprocess, "CREATE_NEW_PROCESS_GROUP", 0
        )
    else:
        kwargs["start_new_session"] = True
    try:
        proc = _subprocess.Popen(list(argv), **kwargs)
    except (OSError, ValueError) as exc:
        return ProcResult(tuple(argv), None, "", "", 0.0, error=str(exc))
    try:
        out, err = proc.communicate(timeout=timeout)
        timed_out = False
    except subprocess.TimeoutExpired:
        _kill_tree(proc)
        try:
            out, err = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            out, err = "", ""
        timed_out = True
    ms = (time.perf_counter() - started) * 1000.0
    return ProcResult(
        tuple(argv),
        None if timed_out else proc.returncode,
        (out or "")[-MAX_CAPTURE:],
        (err or "")[-MAX_CAPTURE:],
        ms,
        timed_out=timed_out,
    )


def git(root: str, *args: str, timeout: float = 3.0) -> ProcResult:
    return run_argv(["git", "-C", root, *args], cwd=root, timeout=timeout)
