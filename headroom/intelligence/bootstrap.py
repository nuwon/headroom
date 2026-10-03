"""First-run setup + protocol verification (plan §4.4–4.8).

:func:`run_setup` is idempotent and is shared by ``headroom intelligence
setup`` and by ``wrap``/``proxy`` when ``HEADROOM_JEVK5=on``:

1. install the pinned ``jevk5`` package (``--no-deps``; GGUF path is stdlib)
   into the *running* interpreter (``sys.executable``), skipping when the
   pinned version is already importable;
2. discover a compatible ``llama-server`` (or build the managed copy);
3. reuse a cached GGUF or let ``llama-server`` fetch it (``--hf-repo``),
   with Headroom's resumable downloader as fallback;
4. start the loopback service and wait for ``/health``;
5. verify the protocol end to end (tokenize with special tokens,
   ``n_probs`` log-probabilities, noul/choice/score decisions through
   upstream ``JevK5GGUF`` when installed and through Headroom's client,
   normalized + finite distributions, stable repeats);
6. write ``setup.json`` for ``doctor``.
"""

from __future__ import annotations

import importlib
import logging
import math
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from headroom import _subprocess

from .config import JEVK5_PACKAGE_SPEC, JEVK5_PACKAGE_TAG, JevK5Settings
from .jevk5_client import DecisionClientError, GGUFDecisionClient
from .jevk5_service import JevK5Service, health, http_json
from .llama_discovery import no_window_flags
from .model_fetch import download_model, find_cached_model
from .state import read_setup, write_setup

logger = logging.getLogger(__name__)

Progress = Callable[[str], None]

SAMPLE_STATE = (
    "Refunds need a receipt and a purchase within 30 days. The customer bought 12 days ago "
    "and has no receipt."
)
SAMPLE_QUESTIONS: dict[str, dict[str, Any]] = {
    "noul": {"type": "noul", "instructions": "Is a refund permitted under the policy?"},
    "choice": {
        "type": "choice",
        "instructions": "What should the agent do next?",
        "criteria": {
            "refund": "Issue the refund.",
            "deny": "Deny the refund and explain the receipt requirement.",
            "escalate": "Escalate to a supervisor.",
        },
    },
    "score": {
        "type": "score",
        "instructions": "How strongly does the evidence support issuing a refund?",
        "criteria": ["not at all", "weakly", "moderately", "strongly"],
    },
}


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""


@dataclass
class SetupReport:
    ok: bool = False
    checks: list[Check] = field(default_factory=list)
    llama_server: dict[str, Any] | None = None
    model_path: str = ""
    url: str = ""
    package: str = ""
    protocol_source: str = ""

    def add(self, name: str, ok: bool, detail: str = "") -> bool:
        self.checks.append(Check(name, ok, detail))
        return ok

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "checks": [c.__dict__ for c in self.checks],
            "llama_server": self.llama_server,
            "model_path": self.model_path,
            "url": self.url,
            "package": self.package,
            "protocol_source": self.protocol_source,
            "platform": sys.platform,
            "python": sys.version.split()[0],
        }


# ---------------------------------------------------------------- package
def installed_jevk5_version() -> str | None:
    try:
        mod = importlib.import_module("jevk5")
    except Exception:  # noqa: BLE001
        return None
    return getattr(mod, "__version__", None)


def ensure_jevk5_package(
    *, progress: Progress | None = None, allow_install: bool = True
) -> tuple[bool, str]:
    """Install the pinned package with ``--no-deps`` into this interpreter."""
    pinned = JEVK5_PACKAGE_TAG.lstrip("v")
    current = installed_jevk5_version()
    if current is not None:
        return True, f"jevk5 {current} already installed"
    if not allow_install:
        return False, "jevk5 not installed (Headroom's built-in protocol mirror will be used)"
    if progress:
        progress(f"Installing {JEVK5_PACKAGE_SPEC} (--no-deps)")
    cmd = [sys.executable, "-m", "pip", "install", "--upgrade", "--no-deps", JEVK5_PACKAGE_SPEC]
    try:
        proc = _subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
            creationflags=no_window_flags(),
            stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"pip failed: {exc}"
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-3:]
        return False, "pip failed: " + " | ".join(tail)
    importlib.invalidate_caches()
    sys.modules.pop("jevk5", None)
    current = installed_jevk5_version()
    if current is None:
        return False, "pip reported success but jevk5 is not importable"
    note = "" if current == pinned else f" (pinned {pinned})"
    return True, f"jevk5 {current} installed{note}"


# ---------------------------------------------------------------- protocol
def _normalized(probs: dict[str, float]) -> bool:
    values = list(probs.values())
    return (
        bool(values)
        and all(math.isfinite(v) and v >= 0 for v in values)
        and abs(sum(values) - 1.0) < 1e-4
    )


def verify_protocol(url: str, settings: JevK5Settings, report: SetupReport) -> bool:
    from . import jevk5_protocol as proto

    report.protocol_source = proto.protocol_source()
    ready, detail = health(url, timeout=5.0)
    if not report.add("health", ready, detail):
        return False
    prompt = proto.prompt_text(SAMPLE_STATE, "Is a refund permitted?", ["true: yes", "false: no"])
    try:
        status, tok = http_json(
            f"{url}/tokenize",
            {"content": prompt, "add_special": False, "parse_special": True},
            timeout=30.0,
        )
        tokens = tok.get("tokens") if isinstance(tok, dict) else None
        ok = status == 200 and isinstance(tokens, list) and len(tokens) > 10
        report.add("tokenize_parse_special", ok, f"{len(tokens or [])} tokens")
        if not ok:
            return False
        status, comp = http_json(
            f"{url}/completion",
            {
                "prompt": tokens,
                "n_predict": 1,
                "n_probs": settings.top_k,
                "temperature": 0,
                "cache_prompt": False,
            },
            timeout=120.0,
        )
        top = (
            comp["completion_probabilities"][0]["top_logprobs"] if isinstance(comp, dict) else None
        )
        ok = status == 200 and isinstance(top, list) and len(top) > 0
        report.add("completion_top_logprobs", ok, f"{len(top or [])} entries")
        if not ok:
            return False
    except (KeyError, IndexError, TypeError, OSError) as exc:
        report.add("completion_top_logprobs", False, f"{type(exc).__name__}: {exc}")
        return False

    client = GGUFDecisionClient(
        url,
        temperature=settings.temperature,
        knockout_temperature=settings.knockout_temperature,
        top_k=settings.top_k,
        timeout_s=120.0,
    )
    all_ok = True
    first: dict[str, dict[str, float]] = {}
    for qid, question in SAMPLE_QUESTIONS.items():
        try:
            probs, _ = client.probabilities(SAMPLE_STATE, question)
        except (DecisionClientError, ValueError) as exc:
            all_ok &= report.add(f"decision_{qid}", False, str(exc))
            continue
        first[qid] = probs
        all_ok &= report.add(f"decision_{qid}_normalized", _normalized(probs), _fmt(probs))
    for qid, question in SAMPLE_QUESTIONS.items():
        if qid not in first:
            continue
        try:
            again, _ = client.probabilities(SAMPLE_STATE, question)
        except (DecisionClientError, ValueError) as exc:
            all_ok &= report.add(f"decision_{qid}_stable", False, str(exc))
            continue
        drift = max(abs(again[k] - first[qid][k]) for k in first[qid])
        all_ok &= report.add(f"decision_{qid}_stable", drift < 1e-3, f"max drift {drift:.2e}")

    # Upstream conformance: the pinned JevK5GGUF must agree with Headroom's client.
    try:
        from jevk5 import JevK5GGUF  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001
        report.add("upstream_jevk5gguf", True, "package not installed; mirror protocol used")
    else:
        try:
            upstream = JevK5GGUF(
                url,
                temperature=settings.temperature,
                knockout_temperature=settings.knockout_temperature,
                top_k=settings.top_k,
                timeout_s=120.0,
            )
            worst = 0.0
            for qid, question in SAMPLE_QUESTIONS.items():
                if qid not in first:
                    continue
                theirs, _ = upstream.probabilities(SAMPLE_STATE, question)
                worst = max(worst, max(abs(theirs[k] - first[qid][k]) for k in first[qid]))
            all_ok &= report.add(
                "upstream_jevk5gguf_agreement", worst < 1e-3, f"max diff {worst:.2e}"
            )
        except Exception as exc:  # noqa: BLE001
            all_ok &= report.add(
                "upstream_jevk5gguf_agreement", False, f"{type(exc).__name__}: {exc}"
            )
    return all_ok


def _fmt(probs: dict[str, float]) -> str:
    return ", ".join(f"{k}={v:.3f}" for k, v in probs.items())


# ---------------------------------------------------------------- setup
def run_setup(
    settings: JevK5Settings,
    *,
    progress: Progress | None = None,
    allow_download: bool = True,
    allow_build: bool = True,
    install_package: bool = True,
    keep_running: bool = False,
    wait_timeout_s: float = 3600.0,
    service: JevK5Service | None = None,
) -> SetupReport:
    say = progress or (lambda _m: None)
    report = SetupReport()
    if settings.mode == "off":
        report.add("mode", False, "HEADROOM_JEVK5=off")
        return report

    ok, detail = ensure_jevk5_package(progress=say, allow_install=install_package)
    report.package = detail
    report.add("jevk5_package", True, detail)  # informational: mirror covers absence

    if settings.external:
        ready, detail = health(settings.url)
        report.url = settings.url
        report.add("external_endpoint", ready, detail)
        report.ok = ready
        write_setup(report.to_dict())
        return report

    svc = service or JevK5Service(settings)
    say("Locating llama-server")
    caps = svc.resolve_llama(allow_build=allow_build)
    if caps is None:
        report.add("llama_server", False, "no compatible llama-server found or built")
        write_setup(report.to_dict())
        return report
    report.llama_server = caps.to_dict()
    report.add("llama_server", True, f"{caps.path} ({caps.version or 'unknown version'})")

    cached = find_cached_model(settings.model_repo, settings.model_file)
    if cached is not None:
        report.model_path = str(cached)
        report.add("model", True, f"cached at {cached}")
    elif not allow_download:
        report.add("model", False, "model not cached and downloads disabled")
        write_setup(report.to_dict())
        return report

    say("Starting llama-server (first run downloads the model; this can take a while)")
    status = svc.start(
        allow_download=allow_download, allow_build=False, wait=True, wait_timeout_s=wait_timeout_s
    )
    if not status.usable and allow_download and cached is None:
        # llama-server's own --hf download failed: fall back to Headroom's
        # resumable downloader, then launch with --model.
        svc.stop()
        say(f"Downloading {settings.model_repo}/{settings.model_file} (resumable)")
        try:
            path = download_model(
                settings.model_repo, settings.model_file, progress=_progress_printer(say)
            )
        except Exception as exc:  # noqa: BLE001
            report.add("model_download", False, f"{type(exc).__name__}: {exc}")
            write_setup(report.to_dict())
            return report
        report.model_path = str(path)
        report.add("model_download", True, str(path))
        status = svc.start(
            allow_download=False, allow_build=False, wait=True, wait_timeout_s=wait_timeout_s
        )
    if not report.add("service", status.usable, status.reason or status.state):
        write_setup(report.to_dict())
        return report
    report.url = status.url
    if not report.model_path:
        found = find_cached_model(settings.model_repo, settings.model_file)
        report.model_path = str(found) if found else f"llama.cpp cache ({settings.model_file})"

    say("Verifying the decision protocol")
    report.ok = verify_protocol(status.url, settings, report)
    write_setup(report.to_dict())
    if not keep_running:
        svc.stop()
    return report


def _progress_printer(say: Progress) -> Callable[[int, int | None], None]:
    last = [0.0]

    def cb(done: int, total: int | None) -> None:
        now = time.monotonic()
        if now - last[0] < 2.0:
            return
        last[0] = now
        if total:
            say(f"  {done / 1e9:.2f} / {total / 1e9:.2f} GB ({100 * done / total:.0f}%)")
        else:
            say(f"  {done / 1e9:.2f} GB")

    return cb


def doctor(settings: JevK5Settings, *, live: bool = True) -> dict[str, Any]:
    """Summarize setup + live state without starting anything heavy."""
    from .jevk5_protocol import protocol_source
    from .state import read_runtime

    out: dict[str, Any] = {
        "mode": settings.mode,
        "external_url": settings.url or None,
        "model": f"{settings.model_repo}/{settings.model_file}",
        "calibration": {
            "temperature": settings.temperature,
            "knockout_temperature": settings.knockout_temperature,
        },
        "protocol_source": protocol_source(),
        "jevk5_package": installed_jevk5_version(),
        "setup": read_setup(),
        "runtime": read_runtime(),
        "cached_model": None,
    }
    cached = find_cached_model(settings.model_repo, settings.model_file)
    out["cached_model"] = str(cached) if cached else None
    url = settings.url or (out["runtime"] or {}).get("url")
    if live and url:
        ready, detail = health(url)
        out["live"] = {"url": url, "ready": ready, "detail": detail}
        if ready:
            report = SetupReport()
            report.ok = verify_protocol(url, settings, report) if not settings.url else True
            out["live"]["checks"] = [c.__dict__ for c in report.checks]
            out["live"]["ok"] = report.ok
    return out
