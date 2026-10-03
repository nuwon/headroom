"""Headroom-managed llama.cpp build — the fallback when discovery finds nothing.

Never touches the user's own llama.cpp. Clones (or fast-forwards)
``https://github.com/ggml-org/llama.cpp`` into
``<workspace>/intelligence/llama.cpp-managed/src`` and builds only the
``llama-server`` target with CMake, following upstream ``docs/build.md``:

* NVIDIA CUDA toolchain detected (``nvcc`` on PATH or ``CUDA_PATH``) →
  ``-DGGML_CUDA=ON``; otherwise the plain CPU backend. A missing CUDA
  toolchain is never an error.
* Windows: Ninja when available (from a VS developer prompt), otherwise
  CMake's default Visual Studio generator (multi-config, so the binary lands
  in ``build/bin/Release``). Linux/macOS: ordinary CMake.

No administrator/root privileges are needed; everything lives in the
per-user workspace.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from headroom import _subprocess

from .llama_discovery import layouts, no_window_flags
from .state import logs_dir, managed_llama_root

logger = logging.getLogger(__name__)

LLAMA_CPP_REPO = "https://github.com/ggml-org/llama.cpp.git"


@dataclass
class BuildPlan:
    src: Path
    build: Path
    cuda: bool
    generator: str | None
    commands: list[list[str]] = field(default_factory=list)


@dataclass
class BuildResult:
    ok: bool
    executable: str = ""
    reason: str = ""
    cuda: bool = False
    log_path: str = ""


def missing_tools(environ: Mapping[str, str] | None = None) -> list[str]:
    env = os.environ if environ is None else environ
    path = env.get("PATH")
    missing = [t for t in ("git", "cmake") if shutil.which(t, path=path) is None]
    if sys.platform != "win32":
        if not any(shutil.which(c, path=path) for c in ("c++", "g++", "clang++")):
            missing.append("c++ compiler")
    return missing


def cuda_available(environ: Mapping[str, str] | None = None) -> bool:
    env = os.environ if environ is None else environ
    if shutil.which("nvcc", path=env.get("PATH")):
        return True
    cuda_path = env.get("CUDA_PATH") or env.get("CUDA_HOME")
    if cuda_path:
        nvcc = Path(cuda_path) / "bin" / ("nvcc.exe" if sys.platform == "win32" else "nvcc")
        return nvcc.is_file()
    return False


def plan_build(
    root: Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    jobs: int | None = None,
    force_cpu: bool = False,
) -> BuildPlan:
    env = os.environ if environ is None else environ
    base = root or managed_llama_root()
    src = base / "src"
    build = src / "build"
    cuda = cuda_available(env) and not force_cpu
    generator: str | None = None
    if sys.platform == "win32" and shutil.which("ninja", path=env.get("PATH")):
        generator = "Ninja"
    configure = ["cmake", "-S", str(src), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release"]
    if generator:
        configure += ["-G", generator]
    configure += [
        "-DLLAMA_BUILD_TESTS=OFF",
        "-DLLAMA_BUILD_EXAMPLES=OFF",
        "-DLLAMA_BUILD_SERVER=ON",
        "-DBUILD_SHARED_LIBS=OFF",
    ]
    configure.append("-DGGML_CUDA=ON" if cuda else "-DGGML_CUDA=OFF")
    n = jobs or max(1, (os.cpu_count() or 2) - 1)
    build_cmd = [
        "cmake",
        "--build",
        str(build),
        "--config",
        "Release",
        "--target",
        "llama-server",
        "-j",
        str(n),
    ]
    if (src / ".git").is_dir():
        fetch = [
            ["git", "-C", str(src), "fetch", "--depth", "1", "origin"],
            ["git", "-C", str(src), "reset", "--hard", "FETCH_HEAD"],
        ]
    else:
        fetch = [["git", "clone", "--depth", "1", LLAMA_CPP_REPO, str(src)]]
    return BuildPlan(src, build, cuda, generator, [*fetch, configure, build_cmd])


def built_executable(src: Path) -> Path | None:
    for rel in layouts():
        cand = src / rel
        if cand.is_file():
            return cand
    return None


def build_managed(
    root: Path | None = None,
    *,
    progress: Callable[[str], None] | None = None,
    timeout_s: float = 3600.0,
    force_cpu: bool = False,
) -> BuildResult:
    missing = missing_tools()
    if missing:
        return BuildResult(False, reason=f"missing build tools: {', '.join(missing)}")
    plan = plan_build(root, force_cpu=force_cpu)
    plan.src.parent.mkdir(parents=True, exist_ok=True)
    logs_dir().mkdir(parents=True, exist_ok=True)
    log_path = logs_dir() / "llama-build.log"
    with open(log_path, "a", encoding="utf-8", errors="replace") as log:
        for cmd in plan.commands:
            if progress is not None:
                progress(" ".join(cmd[:4]) + (" …" if len(cmd) > 4 else ""))
            log.write(f"\n$ {' '.join(cmd)}\n")
            log.flush()
            try:
                proc = _subprocess.run(
                    cmd,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=timeout_s,
                    creationflags=no_window_flags(),
                    stdin=subprocess.DEVNULL,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                return BuildResult(False, reason=f"{cmd[0]} failed: {exc}", log_path=str(log_path))
            if proc.returncode != 0:
                if plan.cuda and not force_cpu and "cmake" in cmd[0]:
                    log.write("\nCUDA build failed; retrying with the CPU backend\n")
                    log.flush()
                    return build_managed(
                        root, progress=progress, timeout_s=timeout_s, force_cpu=True
                    )
                return BuildResult(
                    False,
                    reason=f"`{' '.join(cmd[:3])}` exited {proc.returncode} (see {log_path})",
                    log_path=str(log_path),
                )
    exe = built_executable(plan.src)
    if exe is None:
        return BuildResult(
            False, reason="build finished but llama-server was not found", log_path=str(log_path)
        )
    return BuildResult(True, executable=str(exe), cuda=plan.cuda, log_path=str(log_path))
