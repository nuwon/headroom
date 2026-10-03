# Phase 2 — Agent State, Evidence, Validation, Scope, Test and Workflow Intelligence

Implementation report for features 16, 17, 19, 23, 24 and 20. Phase 2 builds on
the Phase 1 context-intelligence work ([`IMPLEMENTATION_REPORT.md`](IMPLEMENTATION_REPORT.md)).

## 1. Baseline

| Item | Value |
|---|---|
| Starting SHA | `90f1f6f6c9128d184300e31aa1aa1c9c7a01e585` |
| Branch | `claude/confident-ramanujan-i7yqsi` |
| `git status --short` | clean |
| Python | 3.11.15 |
| Rust | rustc 1.95.0 |
| Headroom | 1.0.0-dev |
| OS / arch | Linux x86_64 (cloud container) |
| `llama-server` | not installed here. JevK5 runs against the protocol-faithful fake from Phase 1 |

Phase 1 rollout state: the intelligence posture `HEADROOM_INTELLIGENCE=off|safe|full`
defaults to `safe` under the `coding` savings profile, and JevK5 defaults to `auto`.
Every Phase 1 module lives under `headroom/intelligence/`.

### Baseline test results (before any Phase 2 edit)

| Suite | Result |
|---|---|
| `tests/test_intelligence/` | 293 passed |
| Proxy and handlers (Anthropic, OpenAI, Responses, shaper) | 1,687 passed |
| Router, pipeline and CCR | 640 passed |
| CLI and wrap | 1,192 passed, with 2 failures that predate Phase 2: `test_mcp_reconcile::test_ordinary_install_does_not_adopt_serena`, and a timing-sensitive Codex WebSocket scheduler test that passes on its own |
| Rust workspace | 1,560 passed. Four binaries that load Kompress ONNX hang in this container's ORT dynamic load, an environment issue that predates Phase 2 |

## 2. Phase 1 integration map

| Concept | Actual local module / type |
|---|---|
| TaskContext | `headroom/intelligence/task_context.py:TaskContext` (request-scoped) |
| DecisionGateway | `headroom/intelligence/advisor.py:DecisionAdvisor` (`choose()`, finite options, budget and deadline); `decision_gateway.py` (loopback `/v1/intelligence/systemone`) |
| Rollout snapshot | `headroom/rollout.py:FEATURES` / `RolloutSnapshot` (`ProxyConfig.rollout`) |
| Workspace identity | `headroom/memory/storage_router.py:ProjectResolver` via `AnthropicHandlerMixin._resolve_ccr_workspace`; `headroom/intelligence/responses.py:codex_workspace` (Codex `<cwd>`) |
| Session identity | `headroom/cache/prefix_tracker.py:SessionTrackerStore.compute_session_id`, a coarse model and system-prompt hash. It had no agent-session extraction, so Phase 2 adds one |
| SQLite helper | none general (`cache/backends/sqlite.py` is CCR-specific). Phase 2 adds the agent-state store |
| Redaction | key-based `proxy/wire_debug_redaction_policy.py`, plus the text regexes in `cache/compression_store.py`. Phase 2 unifies them in `headroom/redaction.py` |
| Normalized tool event | none. Phase 2 adds `AgentEvent` |
| Resource identity | `headroom/intelligence/resources.py:resource_for` / `canonical_path` |
| Injected local tool path | `headroom/ccr/mcp_server.py:HeadroomMCPServer`, which `wrap` registers for Claude Code and Codex, plus CCR proxy tool injection |
| Host hooks | `headroom/cli/init.py` (`_ensure_claude_hooks`, `_ensure_codex_hooks`, `features.hooks`), plus the `wrap` marker-dedup pattern (`_ensure_claude_wrap_selfheal_hook`) |
| Graph service | `headroom/intelligence/graph.py:GraphRegistry.for_workspace().neighborhood()` |
| Feedback store | `headroom/intelligence/feedback.py:RetentionLearner` |
| Telemetry | `IntelligenceRuntime.metrics`, `/v1/intelligence/status`, `/stats` |
| Local state dir | `headroom/intelligence/state.py:intelligence_dir()` (`~/.headroom/intelligence`) |
| Request / response handlers | `proxy/handlers/anthropic.py:handle_anthropic_messages`; `proxy/handlers/openai.py:_compress_openai_responses_payload_in_executor` (Codex HTTP and WebSocket) |
