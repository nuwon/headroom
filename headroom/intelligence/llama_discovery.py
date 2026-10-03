"""Find and capability-test an existing ``llama-server`` (plan §4.3).

Discovery order is deterministic:

1. ``HEADROOM_JEVK5_LLAMA_SERVER`` (explicit executable);
2. the path recorded by a previous successful ``headroom intelligence setup``;
3. ``PATH`` (``llama-server`` and, on Windows, ``llama-server.exe``);
4. ``LLAMA_CPP_HOME`` build layouts;
5. a small documented list of common install roots (never a disk scan);
6. the Headroom-managed build (``<workspace>/intelligence/llama.cpp-managed``).

Each candidate is probed with ``--version``, ``--help`` (must advertise
``--hf-repo``, ``--hf-file`` and ``--n-gpu-layers``/``-ngl``) and
``--list-devices`` (short timeout). A candidate is rejected only on an actual
incompatibility, never because its version string looks unfamiliar.

The user's own llama.cpp installation is never modified.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from headroom import _subprocess

logger = logging.getLogger(__name__)

IS_WINDOWS = sys.platform == "win32"
EXE_NAME = "llama-server.exe" if IS_WINDOWS else "llama-server"

PROBE_TIMEOUT_S = 15.0
LIST_DEVICES_TIMEOUT_S = 20.0

# Build-tree layouts under a llama.cpp checkout (LLAMA_CPP_HOME / managed).
_LINUX_LAYOUTS = ("build/bin/llama-server", "bin/llama-server", "llama-server")
_WINDOWS_LAYOUTS = (
    "build/bin/Release/llama-server.exe",
    "build/Release/bin/llama-server.exe",
    "build/bin/llama-server.exe",
    "bin/llama-server.exe",
    "llama-server.exe",
)


def no_window_flags() -> int:
    """``creationflags`` that keep a child from opening a console on Windows."""
    if not IS_WINDOWS:
        return 0
    return getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)


def layouts() -> tuple[str, ...]:
    return _WINDOWS_LAYOUTS if IS_WINDOWS else _LINUX_LAYOUTS


def common_install_roots(environ: Mapping[str, str] | None = None) -> list[Path]:
    """The documented, bounded list of extra places to look."""
    env = os.environ if environ is None else environ
    home = Path(env.get("USERPROFILE") or env.get("HOME") or str(Path.home()))
    roots: list[Path] = []
    if IS_WINDOWS:
        local = Path(env.get("LOCALAPPDATA") or home / "AppData" / "Local")
        program_files = Path(env.get("ProgramFiles") or "C:/Program Files")
        roots += [
            local / "Microsoft" / "WinGet" / "Links",  # winget shim dir
            home / "scoop" / "shims",
            home / "scoop" / "apps" / "llama.cpp" / "current",
            Path(env.get("ChocolateyInstall") or "C:/ProgramData/chocolatey") / "bin",
            local / "llama.cpp",
            local / "Programs" / "llama.cpp",
            program_files / "llama.cpp",
            home / "llama.cpp",
            Path("C:/llama.cpp"),
        ]
    else:
        roots += [
            home / ".local" / "bin",
            Path("/usr/local/bin"),
            Path("/usr/bin"),
            Path("/opt/homebrew/bin"),
            Path("/opt/llama.cpp"),
            home / "llama.cpp",
            home / "src" / "llama.cpp",
        ]
    return roots


@dataclass(frozen=True)
class LlamaCapabilities:
    path: str
    source: str
    ok: bool
    reason: str = ""
    version: str = ""
    build_number: int | None = None
    has_hf_repo: bool = False
    has_hf_file: bool = False
    has_ngl: bool = False
    ngl_auto: bool = False
    has_list_devices: bool = False
    devices: tuple[str, ...] = ()
    gpu: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "source": self.source,
            "ok": self.ok,
            "reason": self.reason,
            "version": self.version,
            "build_number": self.build_number,
            "has_hf_repo": self.has_hf_repo,
            "has_hf_file": self.has_hf_file,
            "has_ngl": self.has_ngl,
            "ngl_auto": self.ngl_auto,
            "has_list_devices": self.has_list_devices,
            "devices": list(self.devices),
            "gpu": self.gpu,
        }


@dataclass
class DiscoveryResult:
    selected: LlamaCapabilities | None
    probed: list[LlamaCapabilities] = field(default_factory=list)


def _is_executable(path: Path) -> bool:
    try:
        if not path.is_file():
            return False
    except OSError:
        return False
    if IS_WINDOWS:
        return path.suffix.lower() in (".exe", ".bat", ".cmd")
    return os.access(path, os.X_OK)


def candidate_paths(
    *,
    explicit: str = "",
    recorded: str = "",
    environ: Mapping[str, str] | None = None,
    managed_root: Path | None = None,
) -> list[tuple[Path, str]]:
    """Ordered, de-duplicated (path, source) candidates. Never scans disks."""
    env = os.environ if environ is None else environ
    out: list[tuple[Path, str]] = []
    seen: set[str] = set()

    def add(p: Path | str | None, source: str) -> None:
        if not p:
            return
        path = Path(str(p).strip().strip('"')).expanduser()
        try:
            key = os.path.normcase(str(path.resolve()))
        except OSError:
            key = os.path.normcase(str(path))
        if key in seen:
            return
        seen.add(key)
        out.append((path, source))

    if explicit:
        add(explicit, "explicit")
    if recorded:
        add(recorded, "recorded")
    for name in ("llama-server", "llama-server.exe") if IS_WINDOWS else ("llama-server",):
        found = shutil.which(name, path=env.get("PATH"))
        if found:
            add(found, "path")
    home = env.get("LLAMA_CPP_HOME")
    if home:
        for rel in layouts():
            add(Path(home) / rel, "llama_cpp_home")
    for root in common_install_roots(env):
        for rel in (EXE_NAME, *layouts()):
            candidate = root / rel
            if _is_executable(candidate):
                add(candidate, "common_root")
    if managed_root is not None:
        for rel in layouts():
            add(managed_root / rel, "managed")
    return out


_BUILD_RE = re.compile(r"version:\s*(\d+)\s*\(([0-9a-f]+)\)", re.I)
_DEVICE_RE = re.compile(r"^\s*([A-Za-z]+\d*)\s*:\s*(.+)$")


def _run(cmd: list[str], timeout: float) -> tuple[int, str]:
    try:
        proc = _subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            creationflags=no_window_flags(),
            stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired:
        return -1, "timeout"
    except (OSError, ValueError) as exc:
        return -2, f"{type(exc).__name__}: {exc}"
    return proc.returncode, (proc.stdout or "") + "\n" + (proc.stderr or "")


def probe(path: Path | str, source: str = "explicit") -> LlamaCapabilities:
    """Capability-test one ``llama-server`` executable."""
    p = Path(path)
    if not _is_executable(p):
        return LlamaCapabilities(str(p), source, False, "not an executable file")
    code, version_out = _run([str(p), "--version"], PROBE_TIMEOUT_S)
    if code == -2:
        return LlamaCapabilities(str(p), source, False, f"cannot execute: {version_out}")
    version = ""
    build_number: int | None = None
    m = _BUILD_RE.search(version_out)
    if m:
        build_number = int(m.group(1))
        version = f"{m.group(1)} ({m.group(2)})"
    else:
        first = next((ln.strip() for ln in version_out.splitlines() if ln.strip()), "")
        version = first[:120]
    code, help_out = _run([str(p), "--help"], PROBE_TIMEOUT_S)
    if code == -2:
        return LlamaCapabilities(str(p), source, False, f"--help failed: {help_out}", version)
    has_hf_repo = "--hf-repo" in help_out or "-hfr" in help_out
    has_hf_file = "--hf-file" in help_out or "-hff" in help_out
    has_ngl = "--n-gpu-layers" in help_out or "-ngl" in help_out
    ngl_auto = bool(re.search(r"n-gpu-layers[^\n]*auto|-ngl[^\n]*auto", help_out))
    has_list = "--list-devices" in help_out
    devices: list[str] = []
    if has_list:
        code, dev_out = _run([str(p), "--list-devices"], LIST_DEVICES_TIMEOUT_S)
        if code >= 0:
            in_list = False
            for line in dev_out.splitlines():
                if "available devices" in line.lower():
                    in_list = True
                    continue
                if in_list:
                    dm = _DEVICE_RE.match(line)
                    if dm:
                        devices.append(f"{dm.group(1)}: {dm.group(2).strip()}"[:160])
    gpu = any(
        d.split(":", 1)[0]
        .upper()
        .startswith(("CUDA", "VULKAN", "METAL", "ROCM", "HIP", "SYCL", "MUSA", "OPENCL"))
        for d in devices
    )
    missing = [
        flag
        for flag, ok in (
            ("--hf-repo", has_hf_repo),
            ("--hf-file", has_hf_file),
            ("--n-gpu-layers", has_ngl),
        )
        if not ok
    ]
    if missing:
        return LlamaCapabilities(
            str(p),
            source,
            False,
            f"missing required option(s): {', '.join(missing)} (llama.cpp too old)",
            version,
            build_number,
            has_hf_repo,
            has_hf_file,
            has_ngl,
            ngl_auto,
            has_list,
            tuple(devices),
            gpu,
        )
    return LlamaCapabilities(
        str(p),
        source,
        True,
        "",
        version,
        build_number,
        has_hf_repo,
        has_hf_file,
        has_ngl,
        ngl_auto,
        has_list,
        tuple(devices),
        gpu,
    )


def discover(
    *,
    explicit: str = "",
    recorded: str = "",
    environ: Mapping[str, str] | None = None,
    managed_root: Path | None = None,
    candidates: Iterable[tuple[Path, str]] | None = None,
) -> DiscoveryResult:
    """Return the first compatible ``llama-server`` in discovery order."""
    probed: list[LlamaCapabilities] = []
    pool = (
        list(candidates)
        if candidates is not None
        else candidate_paths(
            explicit=explicit, recorded=recorded, environ=environ, managed_root=managed_root
        )
    )
    for path, source in pool:
        if not _is_executable(path):
            if source in ("explicit", "recorded"):
                probed.append(LlamaCapabilities(str(path), source, False, "not found"))
            continue
        caps = probe(path, source)
        probed.append(caps)
        if caps.ok:
            return DiscoveryResult(caps, probed)
        logger.info("llama-server candidate rejected: %s (%s)", path, caps.reason)
    return DiscoveryResult(None, probed)
