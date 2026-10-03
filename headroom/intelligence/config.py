"""Typed configuration for Headroom's context-intelligence layer.

Every knob introduced by the intelligence layer (the 15 token-efficiency
optimizations plus the optional JevK5 decision advisor) is resolved here,
exactly once, into an immutable :class:`IntelligenceConfig`. Downstream code
receives that object; it never re-reads ``os.environ``. Environment variables
are the compatibility/operator input, not an internal message bus.

Enablement model (no rollout channel involved — every feature is available on
the stable channel and is switched with a plain env var or CLI flag):

``HEADROOM_INTELLIGENCE`` selects a posture:

* ``off``  — every intelligence feature disabled (library default).
* ``safe`` — deterministic, quality-preserving features: task-conditioned
  relevance query, invariant guard, policy risk budgets, transform arbiter,
  indexed CCR search, selective proactive expansion, retention learning,
  semantic turn complexity, graph relevance, speculative preparation.
  The ``coding`` savings profile seeds this posture.
* ``full`` — ``safe`` plus the features that rewrite more aggressively:
  pre-context admission/externalization, cross-turn delta encoding, the global
  budget allocator and the progressive tool catalog.

Each feature also has its own env var (``1``/``0``) that overrides the
posture default in either direction, e.g. ``HEADROOM_DELTA=1`` with
``HEADROOM_INTELLIGENCE=safe``.

JevK5 (the local decision model) is controlled independently by
``HEADROOM_JEVK5=auto|on|off`` and is only ever advisory.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, replace
from typing import Any, Literal

logger = logging.getLogger(__name__)

IntelligenceLevel = Literal["off", "safe", "full"]
JevK5Mode = Literal["auto", "on", "off"]

_TRUE = {"1", "true", "yes", "on", "enabled"}
_FALSE = {"0", "false", "no", "off", "disabled"}

#: Hard ceiling on the JevK5 advisory weight. JevK5 never receives majority
#: authority over deterministic utility (plan §4.2).
MAX_ADVISORY_WEIGHT = 0.40

# ---------------------------------------------------------------------------
# Versioned JevK5 model manifest. The calibration temperatures are properties
# of the exact GGUF file, published on the upstream model card (README table
# "temperature / knockout_temperature"). Keep them here, in one place, and
# test them — never trust a stale docstring default.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class JevK5ModelManifest:
    package_tag: str
    model_repo: str
    model_file: str
    temperature: float
    knockout_temperature: float
    approx_size_bytes: int
    max_input_tokens: int = 16384


JEVK5_MANIFESTS: dict[str, JevK5ModelManifest] = {
    "jevk5-4b-v0.3-Q8_0.gguf": JevK5ModelManifest(
        package_tag="v0.3.0",
        model_repo="alibiserikbay/JevK5-GGUF",
        model_file="jevk5-4b-v0.3-Q8_0.gguf",
        temperature=1.22,
        knockout_temperature=0.93,
        approx_size_bytes=4_480_000_000,
    ),
    "jevk5-4b-v0.3-Q5_K_M.gguf": JevK5ModelManifest(
        package_tag="v0.3.0",
        model_repo="alibiserikbay/JevK5-GGUF",
        model_file="jevk5-4b-v0.3-Q5_K_M.gguf",
        temperature=1.22,
        knockout_temperature=0.93,
        approx_size_bytes=3_070_000_000,
    ),
    "jevk5-4b-v0.3-Q4_K_M.gguf": JevK5ModelManifest(
        package_tag="v0.3.0",
        model_repo="alibiserikbay/JevK5-GGUF",
        model_file="jevk5-4b-v0.3-Q4_K_M.gguf",
        temperature=1.22,
        knockout_temperature=0.93,
        approx_size_bytes=2_710_000_000,
    ),
    "jevk5-9b-v0.3-Q8_0.gguf": JevK5ModelManifest(
        package_tag="v0.3.0",
        model_repo="alibiserikbay/JevK5-GGUF",
        model_file="jevk5-9b-v0.3-Q8_0.gguf",
        temperature=1.049,
        knockout_temperature=1.2,
        approx_size_bytes=9_530_000_000,
    ),
}

DEFAULT_JEVK5_MODEL_FILE = "jevk5-4b-v0.3-Q8_0.gguf"
DEFAULT_JEVK5_MODEL_REPO = "alibiserikbay/JevK5-GGUF"
JEVK5_PACKAGE_TAG = "v0.3.0"
JEVK5_PACKAGE_SPEC = f"jevk5 @ git+https://github.com/allebee/jevk5@{JEVK5_PACKAGE_TAG}"


def manifest_for(model_file: str) -> JevK5ModelManifest | None:
    return JEVK5_MANIFESTS.get(model_file)


# ---------------------------------------------------------------------------
# Env parsing helpers (tolerant: a typo degrades to the default with a warning)
# ---------------------------------------------------------------------------


def _parse_bool(env: Mapping[str, str], name: str) -> bool | None:
    raw = env.get(name)
    if raw is None or not raw.strip():
        return None
    value = raw.strip().lower()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    logger.warning("Ignoring %s=%r (expected 1/0/true/false)", name, raw)
    return None


def _parse_int(env: Mapping[str, str], name: str, default: int, *, minimum: int = 0) -> int:
    raw = env.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw.strip())
    except ValueError:
        logger.warning("Ignoring %s=%r (expected an integer); using %s", name, raw, default)
        return default
    if value < minimum:
        logger.warning("%s=%s is below the minimum %s; using %s", name, value, minimum, minimum)
        return minimum
    return value


def _parse_float(
    env: Mapping[str, str],
    name: str,
    default: float,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    raw = env.get(name)
    if raw is None or not raw.strip():
        value = default
    else:
        try:
            value = float(raw.strip())
        except ValueError:
            logger.warning("Ignoring %s=%r (expected a number); using %s", name, raw, default)
            value = default
    if value != value:  # NaN
        value = default
    if minimum is not None and value < minimum:
        value = minimum
    if maximum is not None and value > maximum:
        value = maximum
    return value


def parse_level(raw: str | None) -> IntelligenceLevel:
    if raw is None or not raw.strip():
        return "off"
    value = raw.strip().lower()
    if value in ("off", "0", "false", "no", "disabled", "none"):
        return "off"
    if value in ("safe", "on", "1", "true", "yes", "default", "standard"):
        return "safe"
    if value in ("full", "max", "aggressive", "all"):
        return "full"
    logger.warning("Unknown HEADROOM_INTELLIGENCE=%r; using 'off'", raw)
    return "off"


def parse_jevk5_mode(raw: str | None) -> JevK5Mode:
    if raw is None or not raw.strip():
        return "auto"
    value = raw.strip().lower()
    if value in ("auto",):
        return "auto"
    if value in _TRUE or value == "on":
        return "on"
    if value in _FALSE or value == "off":
        return "off"
    logger.warning("Unknown HEADROOM_JEVK5=%r; using 'auto'", raw)
    return "auto"


# ---------------------------------------------------------------------------
# Config objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArbiterWeights:
    """Deterministic-utility weights for the Transform Arbiter (plan §13).

    They live in exactly one place and must sum to 1.0 (asserted by tests and
    by :meth:`validate`).
    """

    relevance_retention: float = 0.28
    soft_invariant_recall: float = 0.20
    recoverability: float = 0.16
    token_savings: float = 0.14
    cache_cost: float = 0.10
    learned_prior: float = 0.07
    latency: float = 0.05

    def total(self) -> float:
        return sum(getattr(self, f.name) for f in fields(self))

    def validate(self) -> None:
        if abs(self.total() - 1.0) > 1e-9:
            raise ValueError(f"ArbiterWeights must sum to 1.0, got {self.total()}")


@dataclass(frozen=True)
class JevK5Settings:
    """Settings for the optional JevK5 decision advisor (plan §4.2)."""

    mode: JevK5Mode = "auto"
    autostart: bool = True
    llama_server: str = ""
    url: str = ""
    model_repo: str = DEFAULT_JEVK5_MODEL_REPO
    model_file: str = DEFAULT_JEVK5_MODEL_FILE
    ctx: int = 8192
    gpu_layers: str = "auto"
    top_k: int = 40
    timeout_ms: int = 1200
    min_confidence: float = 0.60
    advisory_weight: float = 0.25
    max_calls_per_request: int = 4
    decision_cache_ttl_seconds: int = 30
    max_state_tokens: int = 6000
    port: int = 0  # 0 = pick a free loopback port, preferring 8091 upward
    # Explicit opt-ins for the heavyweight first-run steps. `setup` (and
    # HEADROOM_JEVK5=on) always allow them; `auto` never triggers a multi-GB
    # download or a source build on its own.
    allow_download: bool = False
    allow_build: bool = False
    temperature: float = 1.22
    knockout_temperature: float = 0.93

    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    @property
    def external(self) -> bool:
        """True when the operator manages the endpoint (never start/stop it)."""
        return bool(self.url)


@dataclass(frozen=True)
class IntelligenceConfig:
    """Immutable, resolved intelligence configuration."""

    level: IntelligenceLevel = "off"
    # Optimization 1 — task-conditioned relevance query.
    task_query: bool = False
    # Optimization 7 — invariant guard with automatic backoff.
    invariant_guard: bool = False
    # Optimization 13 — real max_lossy_ratio / volatile_token_threshold gates.
    policy_budget: bool = False
    # Optimization 6 — multi-candidate Transform Arbiter.
    arbiter: bool = False
    # Optimization 3 — indexed/partial CCR retrieval.
    ccr_search: bool = False
    ccr_selective_expansion: bool = False
    ccr_expansion_token_budget: int = 1500
    # Optimization 4 — pre-context admission/externalization.
    admission: bool = False
    admission_min_tokens: int = 2000
    admission_preview_tokens: int = 400
    # Optimization 5 — cross-turn semantic delta encoding.
    delta: bool = False
    delta_reads: bool = False
    delta_min_tokens: int = 200
    # Optimization 8 — global information-per-token budget allocator.
    budget_allocator: bool = False
    budget_pressure_threshold: float = 0.5
    budget_diversity_cap: float = 0.6
    # Optimization 9 — semantic progressive tool catalog.
    tool_catalog: bool = False
    tool_catalog_top_k: int = 5
    tool_catalog_min_tools: int = 12
    # Optimization 10 — complexity-aware turn classification / effort routing.
    complexity: bool = False
    effort_routing: bool = False
    # Optimization 11 — graph/symbol-scoped coding context.
    graph: bool = False
    graph_max_files: int = 2000
    # Optimization 12 — retrieval-feedback retention learning.
    feedback: bool = False
    feedback_alpha: float = 0.15
    feedback_min_observations: int = 5
    # Optimization 15 — speculative/background candidate preparation.
    speculative: bool = False
    speculative_workers: int = 2
    speculative_max_pending: int = 64
    # Optimization 14 — rich tool-result interception (rollout feature is
    # promoted to stable; this flag turns on the richer interceptors).
    rich_interceptors: bool = False

    arbiter_weights: ArbiterWeights = field(default_factory=ArbiterWeights)
    jevk5: JevK5Settings = field(default_factory=JevK5Settings)

    @property
    def any_enabled(self) -> bool:
        return any(
            getattr(self, name) for name in _FEATURE_ENV if isinstance(getattr(self, name), bool)
        )

    @property
    def advisor_enabled(self) -> bool:
        return self.jevk5.enabled and self.level != "off"

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"level": self.level}
        for name in _FEATURE_ENV:
            out[name] = getattr(self, name)
        out["jevk5"] = {
            "mode": self.jevk5.mode,
            "external": self.jevk5.external,
            "model_file": self.jevk5.model_file,
            "timeout_ms": self.jevk5.timeout_ms,
            "advisory_weight": self.jevk5.advisory_weight,
        }
        return out

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> IntelligenceConfig:
        env = os.environ if environ is None else environ
        level = parse_level(env.get("HEADROOM_INTELLIGENCE"))
        defaults = _LEVEL_DEFAULTS[level]
        values: dict[str, Any] = {"level": level}
        for name, env_name in _FEATURE_ENV.items():
            override = _parse_bool(env, env_name)
            values[name] = defaults.get(name, False) if override is None else override
        values["ccr_expansion_token_budget"] = _parse_int(
            env, "HEADROOM_CCR_EXPANSION_TOKEN_BUDGET", 1500, minimum=0
        )
        values["admission_min_tokens"] = _parse_int(
            env, "HEADROOM_ADMISSION_MIN_TOKENS", 2000, minimum=1
        )
        values["admission_preview_tokens"] = _parse_int(
            env, "HEADROOM_ADMISSION_PREVIEW_TOKENS", 400, minimum=32
        )
        values["delta_min_tokens"] = _parse_int(env, "HEADROOM_DELTA_MIN_TOKENS", 200, minimum=1)
        values["budget_pressure_threshold"] = _parse_float(
            env, "HEADROOM_BUDGET_PRESSURE_THRESHOLD", 0.5, minimum=0.0, maximum=1.0
        )
        values["budget_diversity_cap"] = _parse_float(
            env, "HEADROOM_BUDGET_DIVERSITY_CAP", 0.6, minimum=0.1, maximum=1.0
        )
        values["tool_catalog_top_k"] = _parse_int(env, "HEADROOM_TOOL_CATALOG_TOP_K", 5, minimum=1)
        values["tool_catalog_min_tools"] = _parse_int(
            env, "HEADROOM_TOOL_CATALOG_MIN_TOOLS", 12, minimum=2
        )
        values["graph_max_files"] = _parse_int(env, "HEADROOM_GRAPH_MAX_FILES", 2000, minimum=1)
        values["feedback_alpha"] = _parse_float(
            env, "HEADROOM_RETENTION_LEARNING_ALPHA", 0.15, minimum=0.01, maximum=0.5
        )
        values["feedback_min_observations"] = _parse_int(
            env, "HEADROOM_RETENTION_LEARNING_MIN_OBS", 5, minimum=1
        )
        values["speculative_workers"] = _parse_int(
            env, "HEADROOM_SPECULATIVE_WORKERS", 2, minimum=1
        )
        values["speculative_max_pending"] = _parse_int(
            env, "HEADROOM_SPECULATIVE_MAX_PENDING", 64, minimum=1
        )
        values["jevk5"] = jevk5_settings_from_env(env)
        config = cls(**values)
        config.arbiter_weights.validate()
        return config

    def with_overrides(self, **overrides: Any) -> IntelligenceConfig:
        return replace(self, **overrides)


def jevk5_settings_from_env(env: Mapping[str, str]) -> JevK5Settings:
    mode = parse_jevk5_mode(env.get("HEADROOM_JEVK5"))
    model_file = (env.get("HEADROOM_JEVK5_MODEL_FILE") or DEFAULT_JEVK5_MODEL_FILE).strip()
    manifest = manifest_for(model_file)
    default_temp = manifest.temperature if manifest else 1.22
    default_ko = manifest.knockout_temperature if manifest else 0.93
    weight = _parse_float(
        env,
        "HEADROOM_JEVK5_ADVISORY_WEIGHT",
        0.25,
        minimum=0.0,
        maximum=MAX_ADVISORY_WEIGHT,
    )
    autostart = _parse_bool(env, "HEADROOM_JEVK5_AUTOSTART")
    allow_download = _parse_bool(env, "HEADROOM_JEVK5_ALLOW_DOWNLOAD")
    allow_build = _parse_bool(env, "HEADROOM_JEVK5_ALLOW_BUILD")
    return JevK5Settings(
        mode=mode,
        autostart=True if autostart is None else autostart,
        llama_server=(env.get("HEADROOM_JEVK5_LLAMA_SERVER") or "").strip(),
        url=(env.get("HEADROOM_JEVK5_URL") or "").strip().rstrip("/"),
        model_repo=(env.get("HEADROOM_JEVK5_MODEL_REPO") or DEFAULT_JEVK5_MODEL_REPO).strip(),
        model_file=model_file,
        ctx=_parse_int(env, "HEADROOM_JEVK5_CTX", 8192, minimum=512),
        gpu_layers=(env.get("HEADROOM_JEVK5_GPU_LAYERS") or "auto").strip() or "auto",
        top_k=_parse_int(env, "HEADROOM_JEVK5_TOP_K", 40, minimum=16),
        timeout_ms=_parse_int(env, "HEADROOM_JEVK5_TIMEOUT_MS", 1200, minimum=50),
        min_confidence=_parse_float(
            env, "HEADROOM_JEVK5_MIN_CONFIDENCE", 0.60, minimum=0.0, maximum=1.0
        ),
        advisory_weight=weight,
        max_calls_per_request=_parse_int(env, "HEADROOM_JEVK5_MAX_CALLS_PER_REQUEST", 4, minimum=0),
        decision_cache_ttl_seconds=_parse_int(
            env, "HEADROOM_JEVK5_DECISION_CACHE_TTL_SECONDS", 30, minimum=0
        ),
        max_state_tokens=_parse_int(env, "HEADROOM_JEVK5_MAX_STATE_TOKENS", 6000, minimum=256),
        port=_parse_int(env, "HEADROOM_JEVK5_PORT", 0, minimum=0),
        allow_download=(mode == "on") if allow_download is None else allow_download,
        allow_build=(mode == "on") if allow_build is None else allow_build,
        temperature=_parse_float(env, "HEADROOM_JEVK5_TEMPERATURE", default_temp, minimum=0.05),
        knockout_temperature=_parse_float(
            env, "HEADROOM_JEVK5_KNOCKOUT_TEMPERATURE", default_ko, minimum=0.05
        ),
    )


# Feature name -> env var. Order is the documentation order.
_FEATURE_ENV: dict[str, str] = {
    "task_query": "HEADROOM_TASK_QUERY",
    "invariant_guard": "HEADROOM_INVARIANT_GUARD",
    "policy_budget": "HEADROOM_POLICY_RISK_BUDGET",
    "arbiter": "HEADROOM_ARBITER",
    "ccr_search": "HEADROOM_CCR_SEARCH",
    "ccr_selective_expansion": "HEADROOM_CCR_SELECTIVE_EXPANSION",
    "admission": "HEADROOM_ADMISSION",
    "delta": "HEADROOM_DELTA",
    "delta_reads": "HEADROOM_DELTA_READS",
    "budget_allocator": "HEADROOM_BUDGET_ALLOCATOR",
    "tool_catalog": "HEADROOM_TOOL_CATALOG",
    "complexity": "HEADROOM_COMPLEXITY_ROUTING",
    "effort_routing": "HEADROOM_EFFORT_ROUTING",
    "graph": "HEADROOM_GRAPH_RELEVANCE",
    "feedback": "HEADROOM_RETENTION_LEARNING",
    "speculative": "HEADROOM_SPECULATIVE_PREP",
    "rich_interceptors": "HEADROOM_RICH_INTERCEPTORS",
}

FEATURE_ENV_VARS: Mapping[str, str] = _FEATURE_ENV

_SAFE_FEATURES = {
    "task_query",
    "invariant_guard",
    "policy_budget",
    "arbiter",
    "ccr_search",
    "ccr_selective_expansion",
    "complexity",
    "graph",
    "feedback",
    "speculative",
}
# Effort routing is deliberately absent from every posture: changing a
# request's reasoning parameters costs prompt-cache re-writes that Headroom's
# own measurements found larger than the output savings. It stays a separate,
# explicit opt-in (HEADROOM_EFFORT_ROUTING=1) and is cache-gated even then.
# delta_reads likewise stays opt-in: file reads are read-protected by default.
_FULL_FEATURES = _SAFE_FEATURES | {
    "admission",
    "delta",
    "budget_allocator",
    "tool_catalog",
    "rich_interceptors",
}

_LEVEL_DEFAULTS: dict[str, dict[str, bool]] = {
    "off": {},
    "safe": dict.fromkeys(_SAFE_FEATURES, True),
    "full": dict.fromkeys(_FULL_FEATURES, True),
}


def level_features(level: IntelligenceLevel) -> frozenset[str]:
    return frozenset(_LEVEL_DEFAULTS[level])


_DISABLED = IntelligenceConfig()


def disabled_config() -> IntelligenceConfig:
    """The all-off configuration (deterministic legacy behavior)."""
    return _DISABLED
