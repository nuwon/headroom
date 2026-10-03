"""``headroom_workflow``: the one Headroom-owned macro tool (plan §10.8).

This is served by the existing Headroom MCP server (``headroom mcp serve``),
which ``headroom wrap`` registers for Claude Code and Codex. Each agent
integration therefore uses the same transport and needs nothing of its own.
The server runs in the agent's project directory and environment, so macro
steps run with the agent's PATH, venv and privileges, never elevated ones.

* The tool is listed only when the workspace has at least one eligible macro.
  Its description names those macros (bounded to about 250 tokens) and stays
  fixed for the MCP session, so the tools prefix stays cache-stable. Macros
  promoted later are announced in the live-turn agent state.
* Execution shares the workspace database with the proxy, so every step's
  events and evidence land where the proxy's systems read them.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

TOOL_NAME = "headroom_workflow"
DESCRIPTION_BUDGET_CHARS = 1000  # ~250 tokens


def _service() -> Any | None:
    from headroom.rollout import resolve_rollout

    from . import build_service
    from .config import AgentStateConfig, AgentStateConfigError

    try:
        cfg = AgentStateConfig.from_env(rollout=resolve_rollout())
    except AgentStateConfigError:
        return None
    if not cfg.workflows.enabled:
        return None
    return build_service(cfg)


def runtime_for_cwd(cwd: str | None = None) -> Any | None:
    """Runtime bound to the workspace's most recently active agent session."""
    service = _service()
    if service is None:
        return None
    ws = service.workspace_for(cwd or os.getcwd())
    if ws is None:
        return None
    row = ws.store.query_one(
        "SELECT agent, agent_session, lineage FROM sessions ORDER BY last_seen DESC LIMIT 1"
    )
    if row is not None and row["agent_session"]:
        return service.runtime_for(
            ws,
            agent=row["agent"],
            agent_session=row["agent_session"],
            lineage=row["lineage"] or "main",
        )
    return service.runtime_for(ws, agent="mcp", agent_session=f"mcp-{os.getpid()}", lineage="main")


def eligible(cwd: str | None = None) -> list[Any]:
    from .workflows import eligible_macros

    rt = runtime_for_cwd(cwd)
    if rt is None or rt.workflows is None:
        return []
    return eligible_macros(rt.store, rt.workspace.workspace_id)


def tool_spec(macros: list[Any]) -> dict[str, Any] | None:
    """MCP tool definition for the eligible macros, or None (tool not listed)."""
    if not macros:
        return None
    lines = []
    used = 0
    for m in macros:
        params = ", ".join((m.input_schema or {}).get("required") or [])
        line = f"- {m.name}{f'({params})' if params else ''}: {m.description}"[:200]
        if used + len(line) > DESCRIPTION_BUDGET_CHARS:
            break
        lines.append(line)
        used += len(line)
    names = [m.name for m in macros][: len(lines)]
    return {
        "name": TOOL_NAME,
        "description": (
            "Run a learned or built-in Headroom workflow macro in one call instead of several "
            "tool turns. Only read-only/verification macros exist; every step is validated "
            "and recorded. Returns a compact summary. Macros:\n" + "\n".join(lines)
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "macro": {"type": "string", "enum": names, "description": "macro name"},
                "inputs": {
                    "type": "object",
                    "description": "macro inputs (project-relative paths only)",
                    "additionalProperties": {"type": "string"},
                },
            },
            "required": ["macro"],
            "additionalProperties": False,
        },
    }


def run(arguments: dict[str, Any], cwd: str | None = None) -> str:
    """Execute a macro; returns the compact model-facing result text."""
    from .workflows import MacroError, WorkflowExecutor, format_result

    name = str((arguments or {}).get("macro") or "")
    inputs = (arguments or {}).get("inputs") or {}
    if not name:
        return "error: missing 'macro'"
    if not isinstance(inputs, dict):
        return "error: 'inputs' must be an object"
    rt = runtime_for_cwd(cwd)
    if rt is None or rt.workflows is None:
        return "error: workflow macros are disabled (HEADROOM_WORKFLOW_MACROS=off)"
    try:
        result = WorkflowExecutor(rt).run(name, inputs)
    except MacroError as exc:
        return f"error: {exc}"
    return format_result(result)
