"""Phase 2 agent-state layer: task state, evidence, contracts, scope, test impact, workflows.

Entry points:

* :class:`~headroom.intelligence.agent_state.runtime.AgentStateService`: the
  proxy composition root, built from
  :class:`~headroom.intelligence.agent_state.config.AgentStateConfig`.
* :func:`build_service`: builds a service from the environment and rollout
  snapshot, or returns None when every feature is off.
"""

from __future__ import annotations

from typing import Any

from .config import AgentStateConfig, AgentStateConfigError

__all__ = ["AgentStateConfig", "AgentStateConfigError", "build_service"]


def build_service(config: AgentStateConfig, *, intelligence: Any | None = None) -> Any | None:
    """An :class:`AgentStateService`, or None when every Phase 2 feature is off."""
    if not config.any_enabled:
        return None
    from .runtime import AgentStateService

    return AgentStateService(config, intelligence=intelligence)
