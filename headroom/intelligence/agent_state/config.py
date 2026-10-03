"""Configuration for the agent-state layer (Phase 2 features 16, 17, 19, 23, 24, 20).

Each feature is a stable rollout flag in :data:`headroom.rollout.FEATURES`,
enabled by default. The emergency override variables are the flags'
``legacy_env`` aliases:

=========================  ===========================  ======================
Feature                    Rollout flag                 Override
=========================  ===========================  ======================
Task State Compiler        ``task_state_compiler``      ``HEADROOM_TASK_STATE``
Evidence Ledger            ``evidence_ledger``          ``HEADROOM_EVIDENCE_LEDGER``
Tool Contract Validator    ``tool_contract_validator``  ``HEADROOM_TOOL_CONTRACTS``
Scope / Drift Firewall     ``scope_firewall``           ``HEADROOM_SCOPE_FIREWALL``
Test Impact Planner        ``test_impact_planner``      ``HEADROOM_TEST_IMPACT``
Workflow Macro Compiler    ``workflow_macro_compiler``  ``HEADROOM_WORKFLOW_MACROS``
=========================  ===========================  ======================

Each override takes ``auto|on|off``. ``auto`` is the default: the feature runs
for coding-agent sessions (Claude Code, Codex) that call tools. ``on`` runs it
for any tool-using session, and ``off`` bypasses it.

Knobs are validated at startup, and a bad value raises
:class:`AgentStateConfigError` instead of being ignored.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

FEATURE_ENV: dict[str, str] = {
    "task_state_compiler": "HEADROOM_TASK_STATE",
    "evidence_ledger": "HEADROOM_EVIDENCE_LEDGER",
    "tool_contract_validator": "HEADROOM_TOOL_CONTRACTS",
    "scope_firewall": "HEADROOM_SCOPE_FIREWALL",
    "test_impact_planner": "HEADROOM_TEST_IMPACT",
    "workflow_macro_compiler": "HEADROOM_WORKFLOW_MACROS",
}
STATE_DIR_ENV = "HEADROOM_AGENT_STATE_DIR"

_ON = {"on", "1", "true", "yes", "enabled"}
_OFF = {"off", "0", "false", "no", "disabled"}
_AUTO = {"auto", ""}

AUTO_CLASS_CHOICES = frozenset({"read_only", "verification"})
_FORBIDDEN_AUTO_CLASSES = frozenset(
    {"local_reversible", "mutating", "external_side_effect", "external", "unknown"}
)


class AgentStateConfigError(ValueError):
    """An agent-state configuration value is invalid."""


class FeatureMode(str, Enum):
    OFF = "off"
    AUTO = "auto"
    ON = "on"

    @property
    def enabled(self) -> bool:
        return self is not FeatureMode.OFF


class EnforcementMode(str, Enum):
    """Validator and firewall enforcement (plan §7.9 / §8)."""

    OBSERVE = "observe"  # record only
    WARN = "warn"  # surface a compact warning, still allow
    PROTECT = "protect"  # block deterministic violations where a real pre-exec hook exists

    @classmethod
    def parse(cls, raw: str | None, *, name: str, default: EnforcementMode) -> EnforcementMode:
        value = (raw or "").strip().lower()
        if not value:
            return default
        try:
            return cls(value)
        except ValueError:
            raise AgentStateConfigError(
                f"{name}={raw!r} is invalid; expected one of protect, warn, observe"
            ) from None


def parse_feature_mode(raw: str | None, *, name: str) -> FeatureMode:
    value = (raw or "").strip().lower()
    if value in _AUTO:
        return FeatureMode.AUTO
    if value in _ON:
        return FeatureMode.ON
    if value in _OFF:
        return FeatureMode.OFF
    raise AgentStateConfigError(f"{name}={raw!r} is invalid; expected auto, on or off")


def _int(env: Mapping[str, str], name: str, default: int, *, lo: int, hi: int) -> int:
    raw = env.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw.strip())
    except ValueError:
        raise AgentStateConfigError(f"{name}={raw!r} is not an integer") from None
    if not lo <= value <= hi:
        raise AgentStateConfigError(f"{name}={value} is out of range [{lo}, {hi}]")
    return value


def _float(env: Mapping[str, str], name: str, default: float) -> float:
    raw = env.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw.strip())
    except ValueError:
        raise AgentStateConfigError(f"{name}={raw!r} is not a number") from None
    if not 0.0 < value < 1.0:
        raise AgentStateConfigError(f"{name}={value} must be strictly between 0 and 1")
    return value


def default_state_dir() -> Path:
    override = os.environ.get(STATE_DIR_ENV, "").strip()
    if override:
        return Path(override).expanduser()
    from headroom.intelligence.state import intelligence_dir

    return intelligence_dir() / "agent_state"


@dataclass(frozen=True)
class AgentStateConfig:
    task_state: FeatureMode = FeatureMode.AUTO
    evidence: FeatureMode = FeatureMode.AUTO
    contracts: FeatureMode = FeatureMode.AUTO
    scope: FeatureMode = FeatureMode.AUTO
    test_impact: FeatureMode = FeatureMode.AUTO
    workflows: FeatureMode = FeatureMode.AUTO
    max_tokens: int = 1200
    scope_mode: EnforcementMode = EnforcementMode.PROTECT
    contract_mode: EnforcementMode = EnforcementMode.PROTECT
    contract_auto_repair: bool = True
    workflow_min_observations: int = 3
    workflow_auto_classes: frozenset[str] = AUTO_CLASS_CHOICES
    test_risk_tier2: float = 0.35
    test_risk_tier3: float = 0.70
    state_dir: str = ""
    # Budgets (plan §12.1). The combined cap is ``max_tokens``.
    task_state_tokens: int = 900
    task_state_target_tokens: int = 600
    evidence_tokens: int = 350
    scope_tokens: int = 180
    verification_tokens: int = 250
    workflow_tokens: int = 250
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def any_enabled(self) -> bool:
        return any(
            m.enabled
            for m in (
                self.task_state,
                self.evidence,
                self.contracts,
                self.scope,
                self.test_impact,
                self.workflows,
            )
        )

    def mode_for(self, feature: str) -> FeatureMode:
        return {
            "task_state_compiler": self.task_state,
            "evidence_ledger": self.evidence,
            "tool_contract_validator": self.contracts,
            "scope_firewall": self.scope,
            "test_impact_planner": self.test_impact,
            "workflow_macro_compiler": self.workflows,
        }[feature]

    def state_path(self) -> Path:
        return Path(self.state_dir) if self.state_dir else default_state_dir()

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        for key, value in list(out.items()):
            if isinstance(value, Enum):
                out[key] = value.value
            elif isinstance(value, frozenset):
                out[key] = sorted(value)
        out.pop("extra", None)
        out["state_dir"] = str(self.state_path())
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AgentStateConfig:
        kwargs: dict[str, Any] = {}
        for key in (
            "task_state",
            "evidence",
            "contracts",
            "scope",
            "test_impact",
            "workflows",
        ):
            if key in data:
                kwargs[key] = FeatureMode(str(data[key]))
        for key in ("scope_mode", "contract_mode"):
            if key in data:
                kwargs[key] = EnforcementMode(str(data[key]))
        for key in (
            "max_tokens",
            "workflow_min_observations",
            "task_state_tokens",
            "task_state_target_tokens",
            "evidence_tokens",
            "scope_tokens",
            "verification_tokens",
            "workflow_tokens",
        ):
            if key in data:
                kwargs[key] = int(data[key])
        for key in ("test_risk_tier2", "test_risk_tier3"):
            if key in data:
                kwargs[key] = float(data[key])
        if "contract_auto_repair" in data:
            kwargs["contract_auto_repair"] = bool(data["contract_auto_repair"])
        if "workflow_auto_classes" in data:
            kwargs["workflow_auto_classes"] = frozenset(data["workflow_auto_classes"])
        if "state_dir" in data:
            kwargs["state_dir"] = str(data["state_dir"] or "")
        return cls(**kwargs)

    @classmethod
    def from_env(
        cls, environ: Mapping[str, str] | None = None, *, rollout: Any = None
    ) -> AgentStateConfig:
        """Resolve and validate. Raises :class:`AgentStateConfigError` on a bad value."""
        env = os.environ if environ is None else environ
        modes: dict[str, FeatureMode] = {}
        for feature, var in FEATURE_ENV.items():
            mode = parse_feature_mode(env.get(var), name=var)
            if rollout is not None:
                try:
                    decision = rollout.decision(feature)
                except KeyError:
                    decision = None
                if decision is not None:
                    if not decision.enabled:
                        # HEADROOM_DISABLE_FEATURES or the alias set to off.
                        mode = FeatureMode.OFF
                    elif decision.reason.value == "explicit" and mode is FeatureMode.AUTO:
                        # Named in HEADROOM_FEATURES: run for every tool session.
                        mode = FeatureMode.ON
            modes[feature] = mode
        max_tokens = _int(env, "HEADROOM_AGENT_STATE_MAX_TOKENS", 1200, lo=200, hi=8000)
        scope_mode = EnforcementMode.parse(
            env.get("HEADROOM_SCOPE_MODE"),
            name="HEADROOM_SCOPE_MODE",
            default=EnforcementMode.PROTECT,
        )
        contract_mode = EnforcementMode.parse(
            env.get("HEADROOM_TOOL_CONTRACT_MODE"),
            name="HEADROOM_TOOL_CONTRACT_MODE",
            default=EnforcementMode.PROTECT,
        )
        repair_raw = (env.get("HEADROOM_TOOL_CONTRACT_REPAIR") or "on").strip().lower()
        if repair_raw not in _ON | _OFF | {"auto"}:
            raise AgentStateConfigError(
                f"HEADROOM_TOOL_CONTRACT_REPAIR={repair_raw!r} is invalid; expected on or off"
            )
        min_obs = _int(env, "HEADROOM_WORKFLOW_MIN_OBSERVATIONS", 3, lo=3, hi=100)
        classes_raw = env.get("HEADROOM_WORKFLOW_AUTO_CLASSES")
        classes = AUTO_CLASS_CHOICES
        if classes_raw is not None and classes_raw.strip():
            parsed = {c.strip().lower() for c in classes_raw.replace(";", ",").split(",") if c}
            parsed.discard("")
            unsafe = parsed & _FORBIDDEN_AUTO_CLASSES
            if unsafe:
                raise AgentStateConfigError(
                    "HEADROOM_WORKFLOW_AUTO_CLASSES may only contain read_only and verification; "
                    f"{', '.join(sorted(unsafe))} can never auto-execute as learned macros"
                )
            unknown = parsed - AUTO_CLASS_CHOICES
            if unknown:
                raise AgentStateConfigError(
                    f"HEADROOM_WORKFLOW_AUTO_CLASSES has unknown class(es): {', '.join(sorted(unknown))}"
                )
            classes = frozenset(parsed)
        tier2 = _float(env, "HEADROOM_TEST_RISK_TIER2", 0.35)
        tier3 = _float(env, "HEADROOM_TEST_RISK_TIER3", 0.70)
        if tier2 >= tier3:
            raise AgentStateConfigError(
                f"HEADROOM_TEST_RISK_TIER2 ({tier2}) must be below HEADROOM_TEST_RISK_TIER3 ({tier3})"
            )
        state_dir = (env.get(STATE_DIR_ENV) or "").strip()
        return cls(
            task_state=modes["task_state_compiler"],
            evidence=modes["evidence_ledger"],
            contracts=modes["tool_contract_validator"],
            scope=modes["scope_firewall"],
            test_impact=modes["test_impact_planner"],
            workflows=modes["workflow_macro_compiler"],
            max_tokens=max_tokens,
            scope_mode=scope_mode,
            contract_mode=contract_mode,
            contract_auto_repair=repair_raw not in _OFF,
            workflow_min_observations=min_obs,
            workflow_auto_classes=classes,
            test_risk_tier2=tier2,
            test_risk_tier3=tier3,
            state_dir=str(Path(state_dir).expanduser()) if state_dir else "",
        )


def disabled_agent_state_config() -> AgentStateConfig:
    off = FeatureMode.OFF
    return AgentStateConfig(
        task_state=off, evidence=off, contracts=off, scope=off, test_impact=off, workflows=off
    )
