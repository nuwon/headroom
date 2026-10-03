"""Read-only diagnostics behind ``headroom intelligence state|evidence|...`` (plan §14.1).

Every function returns plain, already-redacted dictionaries built from the
workspace database. Output never includes raw tool payloads or stored
content beyond short display strings that were redacted at write time.
``verify --run`` is the only path that executes anything, and it runs only on
explicit request.
"""

from __future__ import annotations

import os
from typing import Any

from .store import loads


def runtime(project: str | None = None, session: str | None = None) -> Any | None:
    """A runtime for ``project`` bound to ``session`` or the most recent session."""
    from headroom.rollout import resolve_rollout

    from .config import AgentStateConfig
    from .runtime import AgentStateService

    cfg = AgentStateConfig.from_env(rollout=resolve_rollout())
    service = AgentStateService(cfg)
    ws = service.workspace_for(os.path.abspath(project or os.getcwd()))
    if ws is None:
        return None
    if session:
        row = ws.store.query_one(
            "SELECT agent, agent_session, lineage FROM sessions WHERE session_key = ? OR agent_session = ? "
            "ORDER BY last_seen DESC LIMIT 1",
            (session, session),
        )
    else:
        row = ws.store.query_one(
            "SELECT agent, agent_session, lineage FROM sessions WHERE agent_session NOT LIKE 'mcp-%' "
            "ORDER BY last_seen DESC LIMIT 1"
        )
    if row is None:
        return service.runtime_for(ws, agent="cli", agent_session="cli", lineage="main")
    return service.runtime_for(
        ws, agent=row["agent"], agent_session=row["agent_session"], lineage=row["lineage"] or "main"
    )


def _header(rt: Any) -> dict[str, Any]:
    return {
        "workspace": rt.workspace.workspace_id,
        "project": rt.workspace.root,
        "session": rt.session_key,
        "database": str(rt.store.path),
        "schema_version": rt.store.schema_version(),
    }


def state(rt: Any) -> dict[str, Any]:
    ts = rt.task_state
    st = ts.state if ts is not None else None
    out = _header(rt)
    if st is None:
        out["task"] = None
        return out
    out["task"] = {"task_id": st.task_id, "revision": st.revision, "status": st.status.value}
    out["sections"] = [{"heading": h, "lines": lines} for _, h, lines in sorted(ts.sections())]
    out["atoms"] = [
        {
            "label": a.label,
            "kind": a.kind.value,
            "state": a.state.value,
            "origin": a.origin.value,
            "confidence": a.confidence,
            "text": a.text,
            "evidence": list(a.evidence_ids),
        }
        for a in st.atoms
    ]
    return out


def evidence(rt: Any, *, claim: str | None = None, limit: int = 20) -> dict[str, Any]:
    out = _header(rt)
    led = rt.evidence
    if led is None:
        out["records"] = []
        return out
    if claim:
        rec = led.get_claim(claim)
        recs = [rec] if rec is not None else []
    else:
        rows = rt.store.query(
            "SELECT e.evidence_id FROM claim_heads h JOIN evidence e ON e.evidence_id = h.evidence_id "
            "ORDER BY h.updated_at DESC LIMIT ?",
            (int(limit),),
        )
        recs = [r for r in (led.get(row["evidence_id"]) for row in rows) if r is not None]
    out["records"] = [
        {
            "id": r.evidence_id,
            "claim": r.claim_key,
            "status": r.status,
            "source": r.source_kind,
            "confidence": r.confidence,
            "text": r.display_text,
        }
        for r in recs
    ]
    out["conflicts"] = [
        {
            "low": a.evidence_id if a.authority < b.authority else b.evidence_id,
            "high": b.evidence_id if a.authority < b.authority else a.evidence_id,
        }
        for a, b in led.contradictions(limit=10)
    ]
    return out


def contracts(rt: Any) -> dict[str, Any]:
    from .contracts import BUILTIN_CONTRACTS

    out = _header(rt)
    out["capabilities"] = rt.capabilities.to_dict()
    out["modes"] = {
        "tool_contract_mode": rt.config.contract_mode.value,
        "scope_mode": rt.config.scope_mode.value,
    }
    out["families"] = {f.value: list(c.deterministic_rules) for f, c in BUILTIN_CONTRACTS.items()}
    out["learned_rules"] = [
        {
            k: r[k]
            for k in (
                "rule_id",
                "family",
                "executable",
                "subcommand",
                "reason",
                "failures",
                "hits",
                "disabled_reason",
            )
        }
        for r in (rt.contracts.learned_rules() if rt.contracts is not None else [])
    ]
    out["recent_validations"] = [
        dict(r)
        for r in rt.store.query(
            "SELECT tool_name, outcome, rule, enforced, source, reason FROM validations ORDER BY ts DESC LIMIT 15"
        )
    ]
    return out


def scope(rt: Any) -> dict[str, Any]:
    out = _header(rt)
    out.update(rt.scope.describe() if rt.scope is not None else {"contract": None})
    return out


def verify(rt: Any, *, run: bool = False, max_tier: int | None = None) -> dict[str, Any]:
    out = _header(rt)
    planner = rt.test_impact
    if planner is None:
        out["plan"] = None
        return out
    plan = planner.plan(force=True)
    if plan is None:
        out["plan"] = None
        out["reason"] = "no task-owned changes"
        return out
    out["plan"] = plan.to_json()
    out["adapters"] = [{"framework": a.framework, "confidence": c} for a, c in planner.project()[1]]
    if run:
        out["result"] = planner.run_plan(plan, max_tier=max_tier)
    return out


def workflows(rt: Any) -> dict[str, Any]:
    from .workflows import all_macros, candidate_summary

    out = _header(rt)
    out["macros"] = [
        {
            "name": m.name,
            "origin": m.origin,
            "safety": m.safety_class,
            "enabled": m.enabled,
            "disabled_reason": m.disabled_reason,
            "support": m.support_count,
            "success_rate": round(m.success_rate, 3),
            "turns_saved": m.estimated_turns_saved,
            "description": m.description,
            "inputs": (m.input_schema or {}).get("required") or [],
        }
        for m in all_macros(rt.store)
    ]
    out["candidates"] = candidate_summary(rt.store)
    runs = rt.store.query(
        "SELECT macro_id, status, steps_done, steps_total, duration_ms FROM workflow_runs ORDER BY ts DESC LIMIT 10"
    )
    out["recent_runs"] = [dict(r) for r in runs]
    return out


def status_for(rt: Any) -> dict[str, Any]:
    out = _header(rt)
    row = rt.store.query_one(
        "SELECT capabilities FROM sessions WHERE session_key = ?", (rt.session_key,)
    )
    out["capabilities"] = loads(row["capabilities"], {}) if row is not None else {}
    return out
