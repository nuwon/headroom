"""``headroom intelligence`` — set up and inspect the context-intelligence layer.

headroom intelligence setup     # install jevk5, find/build llama-server, fetch model, verify
headroom intelligence status    # resolved feature flags + service state
headroom intelligence doctor    # live protocol checks against the running service
headroom intelligence stop      # stop a Headroom-owned llama-server
headroom intelligence gateway   # serve /v1/systemone on 127.0.0.1 (for Rust/other tools)
"""

from __future__ import annotations

import json
import os
import sys
import time

import click

from .main import main


def _settings(model_file: str | None = None, llama_server: str | None = None):  # type: ignore[no-untyped-def]
    from dataclasses import replace

    from headroom.intelligence.config import IntelligenceConfig, manifest_for

    cfg = IntelligenceConfig.from_env()
    settings = cfg.jevk5
    if model_file:
        manifest = manifest_for(model_file)
        settings = replace(
            settings,
            model_file=model_file,
            temperature=manifest.temperature if manifest else settings.temperature,
            knockout_temperature=(
                manifest.knockout_temperature if manifest else settings.knockout_temperature
            ),
        )
    if llama_server:
        settings = replace(settings, llama_server=llama_server)
    if settings.mode == "off":
        # `setup`/`doctor` are explicit requests; run them even when the proxy
        # posture is off so a user can prepare the model ahead of enabling it.
        settings = replace(settings, mode="on")
    return cfg, settings


@main.group("intelligence")
def intelligence_group() -> None:
    """Context intelligence: JevK5 decision advisor + token-efficiency features."""


@intelligence_group.command("setup")
@click.option(
    "--llama-server",
    "llama_server",
    default=None,
    help="Path to an existing llama-server executable.",
)
@click.option(
    "--model-file",
    default=None,
    help="GGUF file in alibiserikbay/JevK5-GGUF (default: pinned 4B Q8_0).",
)
@click.option("--no-download", is_flag=True, help="Fail instead of downloading the model.")
@click.option("--no-build", is_flag=True, help="Never build a managed llama.cpp copy.")
@click.option("--no-install", is_flag=True, help="Do not pip-install the jevk5 package.")
@click.option("--keep-running", is_flag=True, help="Leave the verified llama-server running.")
@click.option("--json", "json_output", is_flag=True)
def intelligence_setup(
    llama_server: str | None,
    model_file: str | None,
    no_download: bool,
    no_build: bool,
    no_install: bool,
    keep_running: bool,
    json_output: bool,
) -> None:
    """Install, locate, download and verify everything JevK5 needs. Idempotent."""
    from headroom.intelligence.bootstrap import run_setup

    _cfg, settings = _settings(model_file, llama_server)
    say = (lambda m: None) if json_output else (lambda m: click.echo(f"• {m}"))
    report = run_setup(
        settings,
        progress=say,
        allow_download=not no_download,
        allow_build=not no_build,
        install_package=not no_install,
        keep_running=keep_running,
    )
    if json_output:
        click.echo(json.dumps(report.to_dict(), indent=2))
    else:
        for check in report.checks:
            mark = "✓" if check.ok else "✗"
            click.echo(f"  {mark} {check.name}: {check.detail}")
        click.echo("")
        if report.ok:
            click.echo(
                "JevK5 is ready. Headroom will start it automatically (HEADROOM_JEVK5=auto)."
            )
        else:
            click.echo("JevK5 setup did not complete. Headroom keeps working deterministically.")
    if not report.ok:
        sys.exit(1)


@intelligence_group.command("status")
@click.option("--json", "json_output", is_flag=True)
def intelligence_status(json_output: bool) -> None:
    """Show resolved intelligence features and the decision-service state."""
    from headroom.intelligence.config import FEATURE_ENV_VARS, IntelligenceConfig
    from headroom.intelligence.state import read_runtime, read_setup

    cfg = IntelligenceConfig.from_env()
    payload = {
        "config": cfg.to_dict(),
        "runtime": read_runtime(),
        "setup_ok": (read_setup() or {}).get("ok"),
    }
    if json_output:
        click.echo(json.dumps(payload, indent=2, default=str))
        return
    click.echo(f"Intelligence posture: {cfg.level}  (HEADROOM_INTELLIGENCE)")
    for name, env in FEATURE_ENV_VARS.items():
        state = "on " if getattr(cfg, name) else "off"
        click.echo(f"  [{state}] {name:<26} {env}")
    click.echo(f"JevK5: mode={cfg.jevk5.mode} model={cfg.jevk5.model_file}")
    rt = payload["runtime"]
    if rt:
        click.echo(
            f"  service: running at {rt.get('url')} (pid {rt.get('pid')}, owned={rt.get('owned')})"
        )
    else:
        click.echo("  service: not running")
    click.echo(f"  last setup ok: {payload['setup_ok']}")


@intelligence_group.command("doctor")
@click.option("--json", "json_output", is_flag=True)
@click.option("--no-live", is_flag=True, help="Skip live protocol checks.")
def intelligence_doctor(json_output: bool, no_live: bool) -> None:
    """Diagnose llama-server, model cache and the live decision protocol."""
    from headroom.intelligence.bootstrap import doctor

    _cfg, settings = _settings()
    out = doctor(settings, live=not no_live)
    if json_output:
        click.echo(json.dumps(out, indent=2, default=str))
        return
    click.echo(
        f"mode: {out['mode']}   protocol: {out['protocol_source']}   jevk5 package: {out['jevk5_package']}"
    )
    click.echo(f"model: {out['model']}   cached: {out['cached_model'] or 'no'}")
    setup = out.get("setup") or {}
    llama = setup.get("llama_server") or {}
    click.echo(f"llama-server: {llama.get('path', 'not set up')} {llama.get('version', '')}")
    if llama.get("devices"):
        click.echo(f"  devices: {', '.join(llama['devices'])}")
    live = out.get("live")
    if live is None:
        click.echo(
            "service: not running (start with `headroom intelligence setup --keep-running` or `headroom wrap`)"
        )
        return
    click.echo(f"service: {live['url']} ready={live['ready']} ({live['detail']})")
    for check in live.get("checks", []):
        click.echo(f"  {'✓' if check['ok'] else '✗'} {check['name']}: {check['detail']}")
    if live.get("ok") is False:
        sys.exit(1)


@intelligence_group.command("stop")
@click.option(
    "--force", is_flag=True, help="Also stop an owned instance recorded by another process."
)
def intelligence_stop(force: bool) -> None:
    """Stop the Headroom-owned llama-server (never an operator-managed one)."""
    from headroom.intelligence.jevk5_service import JevK5Service
    from headroom.intelligence.state import read_runtime

    _cfg, settings = _settings()
    record = read_runtime()
    if not record:
        click.echo("No Headroom-owned JevK5 service is running.")
        return
    svc = JevK5Service(settings)
    if svc.stop(force=True if force or record.get("owned") else False):
        click.echo(f"Stopped llama-server (pid {record.get('pid')}).")
    else:
        click.echo("Service is not owned by Headroom; left running.")


@intelligence_group.command("gateway")
@click.option(
    "--port", type=int, default=int(os.environ.get("HEADROOM_JEVK5_GATEWAY_PORT", "0") or 0)
)
def intelligence_gateway(port: int) -> None:
    """Serve the loopback /v1/systemone decision gateway in the foreground."""
    from headroom.intelligence.decision_gateway import serve
    from headroom.intelligence.jevk5_client import GGUFDecisionClient, SystemOneClient
    from headroom.intelligence.jevk5_service import get_service

    _cfg, settings = _settings()
    if settings.external:
        client = SystemOneClient(settings.url, timeout_s=settings.timeout_ms / 1000)
    else:
        svc = get_service(settings)
        status = svc.start(wait=True, wait_timeout_s=600)
        if not status.usable:
            raise click.ClickException(f"decision service unavailable: {status.reason}")
        client = GGUFDecisionClient(
            status.url,
            temperature=settings.temperature,
            knockout_temperature=settings.knockout_temperature,
            top_k=settings.top_k,
            timeout_s=max(5.0, settings.timeout_ms / 1000),
        )
    server = serve(client, port=port)
    click.echo(
        f"Decision gateway on http://127.0.0.1:{server.server_address[1]}/v1/systemone (Ctrl+C to stop)"
    )
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        from headroom.intelligence.jevk5_service import stop_all_owned

        stop_all_owned()
