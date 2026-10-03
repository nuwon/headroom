"""Host pre-tool hook handling: the real pre-execution control point (plan §7.9).

``headroom wrap`` installs a PreToolUse hook for Claude Code
(``.claude/settings.local.json``) and, when hooks are enabled, for Codex
(``$CODEX_HOME/hooks.json``). The hook is the stdlib script
:mod:`headroom.intelligence.agent_state.hook_client`. It POSTs the host's
hook payload to the local proxy and maps the answer to the host protocol:

* ``deny``: exit code 2 with the reason on stderr. The host blocks the call and
  shows the reason to the model.
* ``allow`` with context: Claude Code receives
  ``hookSpecificOutput.additionalContext``.
* anything else, including an unreachable proxy: exit 0, and the host proceeds
  exactly as it would without Headroom.

Capabilities are recorded truthfully:

* **Claude Code**: its documented exit-2 semantics make
  ``can_block_before_execution`` true once its hook is seen.
* **Codex**: hook semantics are not publicly documented, so blocks are only
  *requested* until history proves one was honored.
* **Argument rewriting**: never done through a hook. Claude Code documents
  ``updatedInput`` only together with a permission decision, and answering
  ``allow`` would bypass the user's permission prompt.
"""

from __future__ import annotations

import time
from typing import Any

from .events import AgentEvent, EventType
from .families import normalize_tool_call
from .ids import stable_id

AGENTS = {"claude": "claude_code", "claude_code": "claude_code", "codex": "codex"}


def handle_pretool(service: Any, payload: dict[str, Any]) -> dict[str, Any]:
    agent = AGENTS.get(str(payload.get("agent") or "").lower(), "")
    hook = payload.get("hook") if isinstance(payload.get("hook"), dict) else {}
    which = payload.get("which") if isinstance(payload.get("which"), dict) else None
    cwd = str(hook.get("cwd") or payload.get("cwd") or "")
    tool_name = str(hook.get("tool_name") or "")
    tool_input = hook.get("tool_input")
    if not agent or not cwd or not tool_name:
        return {"decision": "allow"}
    workspace = service.workspace_for(cwd)
    if workspace is None:
        return {"decision": "allow"}
    agent_session = str(hook.get("session_id") or "") or "hook"
    runtimes = service.runtimes_for_agent_session(workspace, agent_session)
    if runtimes:
        rt = max(runtimes, key=lambda r: (r.user_turns, -len(r.session_key)))
    else:
        rt = service.runtime_for(
            workspace, agent=agent, agent_session=agent_session, lineage="main"
        )
    caps = rt.capabilities
    first_hook = caps.hook_seen_at is None
    caps.hook_seen_at = time.time()
    if agent == "claude_code":
        caps.can_block_before_execution = not caps.downgraded
    elif not caps.can_block_before_execution and not caps.downgraded:
        caps.block_unverified = True
    caps.can_rewrite_safe_args = False
    if first_hook:
        service.metrics.bump("hooks_registered")
    call_id = str(hook.get("tool_use_id") or "") or stable_id(
        tool_name, repr(tool_input), time.time()
    )
    inv = normalize_tool_call(tool_name, tool_input, default_cwd=cwd)
    ev = AgentEvent(
        event_id=stable_id(rt.session_key, "proposed", call_id),
        workspace_id=workspace.workspace_id,
        session_id=rt.session_key,
        task_id=rt.task_id,
        event_type=EventType.TOOL_CALL_PROPOSED,
        tool_name=tool_name,
        operation=inv.family.value,
        metadata={"source": "hook", "safety": inv.safety.value},
        transient={
            "invocation": inv,
            "input": tool_input if isinstance(tool_input, dict) else {},
            "call_id": call_id,
        },
    )
    with rt.lock:
        if rt.contracts is not None:
            outcome = rt.contracts.validate(ev, source="hook", which=which)
        else:
            outcome = rt.check_proposed(ev, source="hook")
        rt.persist_events([ev])
        enforced = getattr(outcome, "enforced", "allowed") if outcome is not None else "allowed"
        if enforced not in ("blocked", "block_requested"):
            started = AgentEvent(
                event_id=stable_id(rt.session_key, "started", call_id),
                workspace_id=workspace.workspace_id,
                session_id=rt.session_key,
                task_id=rt.task_id,
                event_type=EventType.TOOL_CALL_STARTED,
                tool_name=tool_name,
                operation=inv.family.value,
                metadata={"source": "hook"},
            )
            rt.persist_events([started])
        rt.save_session()
    service.metrics.bump(f"hook_{enforced}")
    if enforced in ("blocked", "block_requested"):
        service.note_block(rt, call_id)
        reason = outcome.message if outcome is not None else "blocked"
        return {
            "decision": "deny",
            "enforced": enforced,
            "reason": f"Headroom blocked this {tool_name} call: {reason}",
        }
    if enforced == "warned" and outcome is not None and outcome.findings:
        context = "; ".join(f.message for f in outcome.findings[:2])
        return {
            "decision": "allow",
            "enforced": "warned",
            "context": f"Headroom tool-call check: {context}",
        }
    return {"decision": "allow", "enforced": enforced}
