"""Complexity-aware verbosity and effort for outbound turns (Optimization 10).

Thin adapters the provider handlers call at their output-shaping sites. They
read the process's installed :class:`IntelligenceRuntime` and are no-ops when
it is absent or the relevant feature is off:

* :func:`adjust_verbosity` — with ``HEADROOM_COMPLEXITY`` (safe posture),
  never steers a turn that asked "why"/"explain" to conclusions-only, and
  honours an explicit request for brevity. Only consulted where the output
  shaper already runs (``HEADROOM_OUTPUT_SHAPER``, treatment arm).
* :func:`route_effort` — with ``HEADROOM_EFFORT_ROUTING=1`` only (never part
  of a posture), moves an effort field the client already sent, with
  per-conversation hysteresis. Raising is immediate; lowering needs two
  consecutive agreeing mechanical turns.

Both fail open: any exception leaves the body and level untouched.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

PROVIDERS = ("anthropic", "openai_chat", "openai_responses")


def _runtime() -> Any | None:
    from .runtime import current_runtime

    return current_runtime()


def turn_messages(body: dict[str, Any], provider: str) -> list[dict[str, Any]]:
    """Chat-shaped messages of a provider body (Responses items adapted)."""
    if provider == "openai_responses":
        from .messages import responses_items_to_messages

        return responses_items_to_messages(body.get("input"))
    messages = body.get("messages")
    return messages if isinstance(messages, list) else []


def _assess(body: dict[str, Any], provider: str, runtime: Any) -> Any:
    from headroom.proxy.output_complexity import advise_complexity, assess_turn

    messages = turn_messages(body, provider)
    tc = assess_turn(messages)
    return advise_complexity(tc, messages, runtime.advisor_or_none())


def adjust_verbosity(body: dict[str, Any], provider: str, level: int) -> tuple[int, list[str]]:
    """Return ``(level, labels)`` for the output shaper's verbosity level."""
    runtime = _runtime()
    if runtime is None or not runtime.config.complexity:
        return level, []
    try:
        from headroom.proxy.output_complexity import decide_verbosity

        tc = _assess(body, provider, runtime)
        runtime.metrics.bump("semantic_turn_kinds", tc.semantic_turn_kind.value)
        new_level = decide_verbosity(tc, int(level))
        if new_level == level:
            return level, []
        return new_level, [f"intel:verbosity:{level}->{new_level}"]
    except Exception:  # noqa: BLE001 - never break forwarding
        logger.debug("intelligence verbosity adjustment skipped", exc_info=True)
        return level, []


def route_effort(body: dict[str, Any], provider: str, conversation_key: str | None) -> list[str]:
    """Route the client's effort field in place; return labels (empty = untouched)."""
    runtime = _runtime()
    if runtime is None or provider not in PROVIDERS:
        return []
    router = getattr(runtime, "effort_router", None)
    if router is None:
        return []
    try:
        from headroom.proxy.output_complexity import apply_effort, client_effort, decide_effort

        requested = client_effort(body, provider)
        if not requested:
            return []
        tc = _assess(body, provider, runtime)
        decision = decide_effort(tc, requested)
        routed = router.route(conversation_key or "_", decision, requested)
        if not routed or not apply_effort(body, provider, routed):
            return []
        runtime.metrics.bump("effort_changes", f"{requested}->{routed}")
        return [f"intel:effort:{requested}->{routed}:{decision.reason}"]
    except Exception:  # noqa: BLE001
        logger.debug("intelligence effort routing skipped", exc_info=True)
        return []


# Codex / Responses coding tools kept full alongside the shared core set.
_CATALOG_EXTRA_CORE = frozenset(
    {
        "shell",
        "shell_command",
        "local_shell",
        "exec_command",
        "write_stdin",
        "apply_patch",
        "update_plan",
        "view_image",
        "headroom_retrieve",
    }
)


def _natively_deferred(tools: list[Any]) -> bool:
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        if tool.get("defer_loading") or str(tool.get("type", "")).startswith("tool_search"):
            return True
    return False


def apply_tool_catalog(body: dict[str, Any], provider: str, *, cache_cold: bool) -> list[str]:
    """Progressive tool catalog (``HEADROOM_TOOL_CATALOG``); returns labels.

    Only where native tool search did not apply (the request carries no
    ``defer_loading`` / ``tool_search`` entry). The materialized set is sticky
    per conversation and grows only when ``cache_cold``.
    """
    runtime = _runtime()
    if runtime is None:
        return []
    catalog = getattr(runtime, "catalog", None)
    if catalog is None:
        return []
    tools = body.get("tools")
    if not isinstance(tools, list) or not tools or _natively_deferred(tools):
        return []
    try:
        from headroom.proxy.helpers import resolved_core_tools

        from .task_context import build_task_context

        messages = turn_messages(body, provider)
        task = build_task_context(messages, provider=provider)
        result = catalog.apply(
            tools,
            messages=messages,
            query=task.relevance_query(),
            core=resolved_core_tools(_CATALOG_EXTRA_CORE),
            session_key=None,
            cache_cold=cache_cold,
            user_text=task.current_user_text,
            advisor=runtime.advisor_or_none(),
            goal=task.current_user_text,
        )
        if not result.changed:
            return []
        body["tools"] = result.tools
        runtime.metrics.bump("catalog_applied")
        runtime.metrics.bump("catalog_bytes_saved", amount=result.bytes_saved)
        return [f"intel:tool_catalog:{result.compacted}compacted:{result.bytes_saved}B"]
    except Exception:  # noqa: BLE001
        logger.debug("intelligence tool catalog skipped", exc_info=True)
        return []
