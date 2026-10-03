"""Headroom context intelligence.

A deterministic "context compiler" layer (task-conditioned relevance,
invariant guard, policy risk budgets, multi-candidate arbiter, indexed CCR,
delta encoding, budget allocation, graph relevance, retention learning,
speculative preparation) plus an optional local decision advisor (JevK5 served
by llama.cpp's ``llama-server``).

Everything is configured through :class:`IntelligenceConfig` (env vars /
CLI flags / savings profile — no rollout channel). Every feature fails open to
Headroom's existing deterministic behavior.
"""

from .config import (
    ArbiterWeights,
    IntelligenceConfig,
    JevK5Settings,
    disabled_config,
)
from .task_context import TaskContext, build_task_context

__all__ = [
    "ArbiterWeights",
    "IntelligenceConfig",
    "JevK5Settings",
    "TaskContext",
    "build_task_context",
    "disabled_config",
]
