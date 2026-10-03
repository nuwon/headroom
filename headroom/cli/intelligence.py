"""``headroom intelligence`` — set up and inspect the context-intelligence layer.

headroom intelligence setup     # install jevk5, find/build llama-server, fetch model, verify
headroom intelligence status    # resolved feature flags + service state
headroom intelligence doctor    # live protocol checks against the running service
headroom intelligence stop      # stop a Headroom-owned llama-server
headroom intelligence gateway   # serve /v1/systemone on 127.0.0.1 (for Rust/other tools)

Phase 2 agent-state diagnostics (read-only; ``verify --run`` executes on request):

headroom intelligence state      # task id, revision, compact task state
headroom intelligence evidence   # claim heads, sources, status, confidence (--claim KEY)
headroom intelligence contracts  # tool families, learned rules, capability matrix
headroom intelligence scope      # change contract, task-owned changes, warnings
headroom intelligence verify     # risk, verification tiers, reasons (--run to execute)
headroom intelligence workflows  # candidate / promoted / disabled macros
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any

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
    client: Any
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


# ------------------------------------------------------------ agent state
# Phase 2 diagnostics. Read-only unless ``verify --run`` is given explicitly.


def _agent_state_rt(project: str | None, session: str | None):  # type: ignore[no-untyped-def]
    from headroom.intelligence.agent_state import diagnostics
    from headroom.intelligence.agent_state.config import AgentStateConfigError

    try:
        rt = diagnostics.runtime(project, session)
    except AgentStateConfigError as exc:
        raise click.ClickException(str(exc)) from exc
    if rt is None:
        raise click.ClickException("agent-state store is unavailable for this project")
    return rt


def _emit(data: dict[str, Any], json_output: bool, render) -> None:  # type: ignore[no-untyped-def]
    if json_output:
        click.echo(json.dumps(data, indent=2, default=str))
        return
    for key in ("project", "session", "database"):
        click.echo(f"{key}: {data.get(key)}")
    render(data)


_project_opt = click.option("--project", default=None, help="Project directory (default: cwd).")
_session_opt = click.option(
    "--session", default=None, help="Agent session id (default: most recent)."
)
_json_opt = click.option("--json", "json_output", is_flag=True)


@intelligence_group.command("state")
@_project_opt
@_session_opt
@_json_opt
def intelligence_state(project: str | None, session: str | None, json_output: bool) -> None:
    """Current task id, revision and compact task state."""
    from headroom.intelligence.agent_state import diagnostics

    def render(d: dict[str, Any]) -> None:
        task = d.get("task")
        if not task:
            click.echo("task: none observed yet")
            return
        click.echo(f"task: {task['task_id']} revision={task['revision']} status={task['status']}")
        for sec in d.get("sections", []):
            click.echo(f"{sec['heading']}:")
            for line in sec["lines"]:
                click.echo(f"  {line}")

    _emit(diagnostics.state(_agent_state_rt(project, session)), json_output, render)


@intelligence_group.command("evidence")
@_project_opt
@_session_opt
@click.option("--claim", default=None, help="Exact claim key, e.g. 'tests_passing|latest'.")
@click.option("--limit", default=20, show_default=True, type=int)
@_json_opt
def intelligence_evidence(
    project: str | None, session: str | None, claim: str | None, limit: int, json_output: bool
) -> None:
    """Current claim heads with source, status and confidence."""
    from headroom.intelligence.agent_state import diagnostics

    def render(d: dict[str, Any]) -> None:
        for r in d.get("records", []):
            click.echo(
                f"  {r['id']} [{r['status']}/{r['source']} {r['confidence']:.2f}] {r['claim']}: {r['text']}"
            )
        for c in d.get("conflicts", []):
            click.echo(f"  conflict: {c['low']} contradicted by {c['high']}")

    _emit(
        diagnostics.evidence(_agent_state_rt(project, session), claim=claim, limit=limit),
        json_output,
        render,
    )


@intelligence_group.command("contracts")
@_project_opt
@_session_opt
@_json_opt
def intelligence_contracts(project: str | None, session: str | None, json_output: bool) -> None:
    """Tool families, learned rules and the enforcement capability matrix."""
    from headroom.intelligence.agent_state import diagnostics

    def render(d: dict[str, Any]) -> None:
        click.echo(f"modes: {d['modes']}")
        click.echo("capabilities:")
        for k, v in d["capabilities"].items():
            click.echo(f"  {k}: {v}")
        click.echo("learned rules:")
        for r in d["learned_rules"] or [
            {
                "rule_id": "-",
                "executable": "",
                "subcommand": "",
                "reason": "none",
                "failures": 0,
                "disabled_reason": None,
            }
        ]:
            click.echo(
                f"  {r['rule_id']} {r['executable']} {r['subcommand']} {r['reason']} x{r['failures']} {r['disabled_reason'] or ''}"
            )
        for v in d["recent_validations"]:
            click.echo(
                f"  {v['tool_name']}: {v['outcome']} ({v['enforced']}, {v['source']}) {v['reason']}"
            )

    _emit(diagnostics.contracts(_agent_state_rt(project, session)), json_output, render)


@intelligence_group.command("scope")
@_project_opt
@_session_opt
@_json_opt
def intelligence_scope(project: str | None, session: str | None, json_output: bool) -> None:
    """Change contract, task-owned changes, warnings and expansions."""
    from headroom.intelligence.agent_state import diagnostics

    def render(d: dict[str, Any]) -> None:
        c = d.get("contract")
        if not c:
            click.echo("contract: none")
            return
        click.echo(f"mode: {c['mode']}  class: {c['task_class']}  budget: {c['max_change_budget']}")
        click.echo(f"in scope: {', '.join(c['explicit_in_scope_paths']) or '-'}")
        click.echo(f"excluded: {', '.join(c['explicit_out_of_scope_paths']) or '-'}")
        click.echo(f"subsystems: {', '.join(c['expected_subsystems']) or '-'}")
        for path, cls in d.get("changes", []):
            click.echo(f"  changed {path}: {cls}")
        for e in d.get("expansions", []):
            click.echo(f"  expansion {e['path']}: {e['reason_code']} ({e['confidence']})")
        for w in d.get("warnings", []):
            click.echo(f"  warning {w['path']}: {w['reason']}")

    _emit(diagnostics.scope(_agent_state_rt(project, session)), json_output, render)


@intelligence_group.command("verify")
@_project_opt
@_session_opt
@click.option(
    "--run", "run_plan", is_flag=True, help="Execute the staged plan (default: inspect only)."
)
@click.option("--max-tier", type=click.IntRange(1, 3), default=None)
@_json_opt
def intelligence_verify(
    project: str | None,
    session: str | None,
    run_plan: bool,
    max_tier: int | None,
    json_output: bool,
) -> None:
    """Risk score, selected verification tiers and reason codes."""
    from headroom.intelligence.agent_state import diagnostics

    def render(d: dict[str, Any]) -> None:
        plan = d.get("plan")
        if not plan:
            click.echo(f"plan: none ({d.get('reason', 'planner disabled')})")
            return
        click.echo(
            f"risk: {plan['risk_score']}  max tier: {plan['max_tier']}  changed: {', '.join(plan['changed'])}"
        )
        for tier, cmds in sorted(plan["commands"].items()):
            for c in cmds:
                click.echo(f"  tier {tier}: {' '.join(c['argv'])}")
        click.echo(f"reasons: {', '.join(plan['rationale_codes']) or '-'}")
        if "result" in d:
            click.echo(
                f"result: {d['result'].get('status')} tiers={[t['tier'] for t in d['result'].get('tiers', [])]}"
            )

    _emit(
        diagnostics.verify(_agent_state_rt(project, session), run=run_plan, max_tier=max_tier),
        json_output,
        render,
    )


@intelligence_group.command("workflows")
@_project_opt
@_session_opt
@_json_opt
def intelligence_workflows(project: str | None, session: str | None, json_output: bool) -> None:
    """Candidate, promoted and disabled workflow macros, and why."""
    from headroom.intelligence.agent_state import diagnostics

    def render(d: dict[str, Any]) -> None:
        for m in d["macros"]:
            state = "enabled" if m["enabled"] else f"disabled ({m['disabled_reason']})"
            click.echo(f"  {m['name']} [{m['origin']}/{m['safety']}] {state}: {m['description']}")
        for c in d["candidates"]:
            click.echo(
                f"  candidate {c['signature']}: {c['observations']} obs, {c['sessions']} sessions, {c['safety']} :: {' -> '.join(c['steps'])}"
            )

    _emit(diagnostics.workflows(_agent_state_rt(project, session)), json_output, render)
