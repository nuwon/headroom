# Phase 2 — Agent State, Evidence, Validation, Scope, Test and Workflow Intelligence

Implementation report for features 16, 17, 19, 23, 24 and 20. Phase 2 builds on
the Phase 1 context-intelligence work ([`IMPLEMENTATION_REPORT.md`](IMPLEMENTATION_REPORT.md)).

## 1. Starting SHA / ending SHA

| Item | Value |
|---|---|
| Starting SHA | `90f1f6f6c9128d184300e31aa1aa1c9c7a01e585` |
| Ending SHA (implementation) | `4b3e5d4`. This report is committed on top of it |
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

Phase 2 reuses every row above. It creates no second TaskContext, gateway,
graph, CCR, rollout system or redactor. The two places that had no shared
implementation (redaction and agent-session identity) now have one.

## 3. Files added

**Runtime package `headroom/intelligence/agent_state/`**

| File | Role |
|---|---|
| `__init__.py` | Package entry point |
| `config.py` | `AgentStateConfig`: the six feature modes, knobs and budgets. Validated at startup (`AgentStateConfigError`) and round-tripped through the worker JSON |
| `ids.py` | Workspace id (`sha256(canonical root)[:24]`), agent-session extraction (Claude `metadata.user_id` / `x-claude-code-session-id`, Codex `prompt_cache_key` / `session_id`), conversation lineage, agent detection |
| `events.py` | `AgentEvent`, the normalized event stream with stable ids |
| `store.py` | The workspace-scoped SQLite store: WAL, migrations v1–v7, quarantine, retention |
| `families.py` | Tool-call normalization into families and safety classes. Covers Claude, Codex, POSIX shells, PowerShell and cmd |
| `normalizer.py` | Provider history → `AgentEvent`s |
| `results.py` | Test and build output parsers (pytest, Cargo, CTest JSON, Jest/Vitest, Go) |
| `paths.py` | Path canonicalization: symlinks and junctions resolved, Windows case folding, drive letters |
| `injection.py` | `StickyInjector`: cache-safe live-turn insertion with byte-identical replay |
| `serialization.py` | The one `<headroom_agent_state>` block: budgets, hashes, full and delta rendering |
| `runtime.py` | `AgentStateRuntime` (per session) and `AgentStateService` (the proxy composition root) |
| `task_state.py` | Feature 16 |
| `evidence.py` | Feature 17 |
| `contracts.py` | Feature 19 |
| `scope.py` | Feature 23 |
| `proc.py` | Cross-platform argv execution (process groups, `CREATE_NO_WINDOW`, `taskkill /T`) |
| `test_impact.py` | Feature 24 |
| `workflows.py` | Feature 20 |
| `hooks.py` | PreToolUse decision logic and truthful capability recording |
| `hook_client.py` | The stdlib-only hook script the hosts execute |
| `install.py` | Install and remove the hooks for Claude Code and Codex, idempotently and marker-tagged |
| `mcp.py` | The `headroom_workflow` MCP tool spec and execution |
| `advice.py` | JevK5 adapters, advisory and optional |
| `diagnostics.py` | Data behind the `headroom intelligence state/evidence/contracts/scope/verify/workflows` commands |

**Elsewhere**

| File | Role |
|---|---|
| `headroom/redaction.py` | The one shared redactor (keys, headers, bearer/basic values, provider tokens, JWTs, private keys, credential URLs, `.env` values) |
| `docs/content/docs/agent-state.mdx` | User documentation |
| `benchmarks/agent_state_benchmark.py` | Phase 2 benchmark (§18–§19) |
| `tests/test_intelligence/agent_state/*` | 267 tests across 10 test modules (§17) |
| `IMPLEMENTATION_REPORT_PHASE2.md` | This report |

## 4. Files modified

| File | Change |
|---|---|
| `headroom/rollout.py` | Six stable, default-on `FeatureSpec`s with the `HEADROOM_TASK_STATE`-style `auto/on/off` aliases |
| `headroom/intelligence/models.py` | `DecisionFamily` gains TASK_STATE, EVIDENCE_CONFLICT, TOOL_CONTRACT, SCOPE_NECESSITY and TEST_IMPACT for the existing gateway |
| `headroom/intelligence/runtime.py` | `prepare_request` feeds compact task-state terms into the Phase 1 `TaskContext.explicit_entities` (capped at 48) |
| `headroom/cache/compression_store.py` | Retrieval-log redaction delegates to `headroom.redaction`. The duplicate regexes are gone |
| `headroom/proxy/wire_debug_redaction_policy.py` | Re-exports the shared redactor |
| `headroom/proxy/models.py` | `ProxyConfig.agent_state` |
| `headroom/proxy/server.py` | Builds the service and shuts it down. Adds `GET /v1/agent-state/status` and `POST /v1/agent-state/hook` (loopback only, executor, fail-open allow). Adds agent state to `/v1/intelligence/status`, the worker JSON and `_proxy_config_from_env` |
| `headroom/proxy/handlers/anthropic.py` | Ingest before the wire-contract guard. Apply the sticky block, revert on a client-bytes guard, commit, or discard on the signed-thinking lock |
| `headroom/proxy/handlers/openai.py` | Responses (HTTP and WebSocket) ingest and insert inside the compression executor. With `optimize` off, HTTP and WebSocket frames are ingested only, never changed |
| `headroom/ccr/mcp_server.py` | `headroom_workflow` tool (spec cached per MCP session, run in a worker thread) |
| `headroom/cli/proxy.py` | Resolves `AgentStateConfig.from_env` against the rollout snapshot. A bad value exits 1 |
| `headroom/cli/wrap.py` | Installs and removes the Claude Code hook and the Codex hook (`hooks.json` plus a per-process `features.hooks=true`). Unwrap removes them |
| `headroom/cli/intelligence.py` | `state`, `evidence`, `contracts`, `scope`, `verify` (`--run`, `--max-tier`) and `workflows` commands |
| `docs/content/docs/meta.json` | Adds the Agent State page |
| `tests/conftest.py` | Autouse isolation of `HEADROOM_AGENT_STATE_DIR`, and closes stores |
| `tests/test_rollout.py` | Expects the six new default-on features |
| `tests/test_cli/test_wrap_codex.py` | Expects the per-session `--config features.hooks=true` and the hook file |

## 5. Schema / migration versions

There is one database per workspace, at
`<state dir>/<workspace id>/agent_state.sqlite3`.

| Version | Migration | Tables |
|---|---|---|
| 1 | agent-state runtime | `meta`, `sessions`, `events`, `processed` (idempotency) |
| 2 | feature 16: task state | `tasks`, `task_transitions` |
| 3 | feature 17: evidence ledger | `evidence`, `claim_heads`, `evidence_links` |
| 4 | feature 19: tool contracts | `tool_outcomes`, `learned_rules`, `validations` |
| 5 | feature 23: scope firewall | `change_contracts`, `git_baselines`, `task_changes`, `scope_expansions`, `scope_warnings` |
| 6 | feature 24: test impact | `test_cases`, `impact_edges`, `test_runs`, `verification_plans` |
| 7 | feature 20: workflow macros | `workflow_observations`, `workflow_macros`, `workflow_runs` |

- **Migrations.** They are forward-only, and each runs in its own `BEGIN IMMEDIATE` transaction recorded in `schema_version`.
- **Tested from every version.** `test_migrations_from_every_version` starts from each of v0–v6 and migrates to v7.
- **Newer and corrupt databases.** A database from a newer Headroom disables persistence and is never touched. A corrupt one is renamed to `*.corrupt-<time>` and a fresh one is created.
- **Session memo.** The per-session injection memo, a JSON column in `sessions`, gained three fields for delta blocks (section digests, full or delta, revision). No migration is needed: the reader accepts the old four-field entries and treats them as full blocks.

## 6. Feature 16: Task State Compiler

- **State.** A typed, durable `TaskState` per task, holding goal, constraints (hard or soft), acceptance criteria (each tagged test, build or deliverable), decisions, subgoals, blockers, unresolved questions and completed items. Every change is a `StateTransition`. The types are ADD_ATOM, SET_ATOM_STATE, SUPERSEDE, ATTACH_EVIDENCE, SET_STATUS and DECLARE_COMPLETE. `apply_transition` enforces the authority rules and persists revision N+1 in one transaction.
- **Extraction tiers.**
  - Tier A is deterministic: it uses the user's own words, label prefixes ("Constraints: …"), `;`-split units, polarity-aware dedupe, and corrections that supersede.
  - Tier B is JevK5 and is limited to finite questions: does a new message refine, replace or add to the goal, and do two constraints conflict. It never writes text.
  - Tier C is conservative: ambiguous input leaves the state unchanged.
- **New tasks.** A new task is detected, and a compaction summary keeps the prior lineage.
- **Completion needs evidence.** A completion claim satisfies a test or build criterion only when current passing evidence covers its targets. A task-owned edit re-opens satisfied criteria. A failure that predates the task is reported but does not block.
- **Injection.**
  - The block goes into the live turn under the 900/600-token task budget, through the sticky injector.
  - Major changes inject at once. Progress churn uses hysteresis, and the hysteresis backs off as state accumulates in history.
  - Later blocks are deltas (§19). The goal line omits sentences that are already their own atoms.
- **No frontier-model calls** are made.

## 7. Feature 17: Evidence Ledger

- **Records.** Atomic records with `source_kind`, `source_event_id`, `source_hash`, confidence, status and `valid_until`, fed by deterministic extractors on live events. They cover:
  - exit codes;
  - test and build results with failing ids;
  - compiler errors;
  - file hashes after writes;
  - git HEAD, branch and porcelain status;
  - tool versions;
  - user assertions;
  - agent assertions.
- **Supersession.** For single-valued claims a newer, equal or higher authority replaces the head, and the old record becomes `SUPERSEDED`. A lower authority never replaces the head: it is linked `SUPPORTS` when it agrees, and `CONTRADICTS` with the lower record marked `CONTRADICTED` when it disagrees. An agent's unsupported claim never overrides tool evidence.
- **Staleness.** Volatile claims expire on a TTL. File, HEAD and status claims go stale on the next mutation.
- **Links to Task State.** Evidence feeds Task State: blockers, satisfied criteria and pre-existing failures.
- **Privacy.**
  - Subjects are redacted *before* the claim key is computed, so a secret can never become a key.
  - Display text is clipped.
  - Raw outputs stay with CCR.
- **Restart and isolation** are tested (Scenario J; workspace isolation tests).

## 8. Feature 19: Tool Contract Validator

- **Built-in contracts by family:**
  - read/edit/write/delete;
  - search/list;
  - shell, build and test;
  - git;
  - MCP.

  They check for:
  - missing input paths;
  - a working directory that is not a directory;
  - unresolvable executables (`.cmd` shims on Windows);
  - wrong-platform commands (PowerShell cmdlets or drive paths on POSIX; POSIX-only tools on Windows);
  - malformed git refs (created refs are checked separately from paths);
  - missing pytest selectors and npm scripts;
  - schema-declared mutually exclusive arguments.
- **Safe repair.** It is whitelisted to semantics-preserving normalizations: a relative path made absolute against the project root, and separators normalized. It is applied only where Headroom executes the step itself (workflow macros).
- **Learned rules.** A call shape that failed `LEARN_MIN_FAILURES` (3) times under the same conditions becomes a narrow learned rule. The rule is keyed by family, executable, subcommand and the redacted call shape.
- **Truthful enforcement (`decide_enforcement`).**
  - A call seen only in history is `observed`.
  - It is `blocked` only when all of these hold: a live host hook, `protect` mode, a deterministic finding, and a host that can block.
  - Codex starts as `block_requested`. It is verified when history shows the blocked call never ran, and downgraded for the session if it did run.
  - Otherwise the finding is `warned`.
  - JevK5 can only warn.
- **Fast path.** Hook validation has a median of 2.1 ms and a p95 of 4.0 ms (§18).

## 9. Feature 23: Scope / Drift Firewall

- **ChangeContract.** It is compiled from task authority: named paths, exclusions, hard constraints, criteria, the dirty-tree baseline, graph neighbours, files proven by failures, and reads (which never authorize writes).
- **Classes.** IN_SCOPE, DEPENDENCY_SCOPE, GENERATED_EFFECT, UNRELATED, FORBIDDEN and AMBIGUOUS.
- **Dirty tree.** The git baseline taken at task start separates pre-existing user edits from task-owned changes by hash. Rename-aware porcelain parsing is used, with a 5 s status cache.
- **Scope expansion.** It needs evidence: a traceback or compiler error naming the file admits it as DEPENDENCY_SCOPE, recorded in `scope_expansions`.
- **Hard blocks.** FORBIDDEN is limited to deterministic cases, and only with a real hook in `protect` mode:
  - outside the workspace (symlinks and junctions resolved);
  - user exclusions;
  - `.git`;
  - installed dependencies;
  - credential files not named by the user;
  - git-ignored generated output.

  Unrelated and ambiguous edits warn, deduplicated per (contract revision, path, reason).

## 10. Feature 24: Test Impact Planner

- **Change set.** It comes from the Scope Firewall's task-owned set, never raw `git status`.
- **Adapters.** pytest (`--collect-only`), Cargo, CTest (`--show-only=json-v1`), Jest/Vitest (the project's package manager), and a generic project-command adapter. Python commands use the project's virtualenv (probed for pytest, cached per root).
- **Impact edges.** They are persisted with their weights, combined as `1-Π(1-w)`, and decay linearly over 90 days.
- **Risk and tiers.**
  - Risk uses the plan's 7-term formula.
  - Tier 1 runs below 0.35, tier 2 on pass up to 0.70, and tier 3 on pass at or above 0.70.
  - Mandatory tier-3 categories apply, including an explicit user request.
- **Failures and flaky tests.** The first failing tier stops the run. A known-flaky test gets one rerun, and differing outcomes are reported `FLAKY`. A pre-existing failure is reported and is non-blocking.
- **Evidence and task state.** Results are written to the Evidence Ledger and Task State.
- **Verify section.** It is injected only when actionable: the agent is verifying or declaring completion and current evidence is insufficient. Interpreters inside the project are shown relative to it.

## 11. Feature 20: Workflow Macro Compiler

- **Normalization.** Sequences are observed per turn, keyed by the call's message index, and normalized into argv templates.
- **Promotion thresholds:**
  - at least 3 observations;
  - a success rate of at least 0.90;
  - at least 2 sessions, or 3 uncorrected repetitions;
  - no contract or scope finding;
  - at least 2 turns saved.

  Only `read_only` and `verification` classes promote (`HEADROOM_WORKFLOW_AUTO_CLASSES` accepts nothing else). Network, external and mutating sequences never promote.
- **Slots.** Only project-relative paths that actually varied become slots, each with a validator.
- **Execution through `headroom_workflow`.** Promoted macros run through the one `headroom_workflow` MCP tool. Every step:
  - runs as argv;
  - passes the Tool Contract Validator and the Scope Firewall;
  - emits events and evidence.

  Verification steps call the Test Impact Planner. The run stops on the first failed step.
- **Disabling.** A macro is disabled after 2 failures in its last 5 runs, and invalidated by a change in config hash, executable or referenced path.
- **Seeded templates.** `run_test_impact_plan`, `show_task_owned_diff` and `verify_then_status` are enabled only where git or a test framework exists.

## 12. Runtime dataflow and hook points

```text
Claude Code ── POST /v1/messages ──► anthropic.py
   begin_anthropic(body, headers, client messages, cwd)       [executor]
     identity → runtime → history → AgentEvents (normative order:
       user → evidence(user) → task state → scope → workflows;
       result → evidence → task state → scope → contract learning
              → test impact → workflows)
     render → full block or delta (only when its base is live in history)
   apply_anthropic → sticky replay + new block on the last user message
   guard: client-bytes → revert | signed-thinking lock → discard | else commit

Codex ── POST /v1/responses / WS response.create ──► openai.py
   inside the compression executor: begin_responses → apply_responses
     (insert after anchor; no replay with previous_response_id)
   optimize off: begin_responses only (HTTP and WS); the frame is unchanged

Claude Code / Codex PreToolUse hook ── hook_client.py ──► POST /v1/agent-state/hook
   contracts.validate (incl. scope) → decide_enforcement → deny (exit 2) | allow(+context)

Model ── headroom_workflow (MCP) ──► WorkflowExecutor → contracts + scope per step
   → proc.run_argv → events → evidence → test impact
```

## 13. Agent capability / enforcement matrix

| Capability | Claude Code (`wrap claude`) | Codex (`wrap codex`) | Other proxy clients |
|---|---|---|---|
| Active in the default `auto` mode | yes, for tool-using sessions | yes, for tool-using sessions | no. Only when a feature is set to `on` |
| Observe calls | yes | yes | with `on`: yes, after execution |
| Block before execution | **yes**: PreToolUse exit 2 (documented) | **requested**: reported `blocked` only after history shows the block was honored; downgraded if the call ran anyway | no (`observed`) |
| Warn before execution | yes (`additionalContext`) | no: the hook allows the call, and the warning appears in the next live turn | no (next live turn) |
| Rewrite arguments | no: `updatedInput` requires an `allow` that would bypass the permission prompt | no | no |
| Execute `headroom_workflow` | yes (MCP) | yes (MCP) | no |
| Live-turn state block | yes | yes | with `on` only |

## 14. Windows test results

- **Not run on Windows.** No Windows host is available in this environment, so no test ran on Windows.
- **Simulated Windows semantics.** These are covered on Linux by tests that patch the platform or feed Windows-shaped input:
  - PowerShell `-Command` and `cmd /c` parsing (`test_runtime`, `test_contracts`);
  - drive-letter and backslash path semantics (`test_windows_path_semantics`);
  - Windows-only commands rejected on POSIX (`test_windows_only_commands_on_posix`);
  - case-insensitive scope matching (`test_windows_case_insensitive_matching`);
  - argv quoting with `list2cmdline` under `sys.platform = "win32"` (`test_windows_commands_are_argv_not_shell`);
  - backslash workflow slots, including `..\` traversal, drive paths and `.git\` (`test_slot_validator_handles_windows_shaped_values`).
- **Windows-only code paths not exercised here.** These are `CREATE_NO_WINDOW` with a new process group, `taskkill /T` on timeout (both guarded by `sys.platform.startswith("win")`), and junction resolution through `os.path.realpath`. All of them use only standard-library APIs.

## 15. Linux test results

All tests in this report ran on Linux x86_64 (Python 3.11.15).

## 16. Existing test-suite results

Run on the final implementation (Linux, `-p no:cacheprovider`).

| Suite | Result |
|---|---|
| `tests/test_intelligence/` (Phase 1 suites including the eval-corpus gate, plus agent state) | 558 passed |
| CLI and wrap (`tests/test_cli/`) | 921 passed, 1 skipped, 1 failed: `test_mcp_reconcile::test_ordinary_install_does_not_adopt_serena`, which predates Phase 2 (§1) |
| Proxy and handlers (`tests/test_proxy/`, `test_proxy*`, `test_anthropic*`, `test_openai*`, `test_codex*`, `test_responses*`) | 1,983 passed, 111 skipped, 4 failed in `test_codex_live.py` (see below) |
| Router, pipeline, CCR, compression store, rollout, redaction policy | 604 passed |
| `ruff check`, `ruff format --check` | clean |
| `mypy headroom` | 2 errors, both in files Phase 2 does not touch (`providers/model_metadata.py`, `integrations/langchain/chat_model.py`) |

**The four `test_codex_live.py` failures predate Phase 2.**

- **Symptom.** Each fails with "uvicorn proxy failed to start", and only when `tests/test_openai_codex_ws_lifecycle.py` runs earlier in the same process.
- **On their own they pass.** `test_codex_live.py` alone passes (10/10), and all nine `test_codex*` files together pass (96 passed).
- **Phase 2 is not involved.** The same pair fails at HEAD with all six Phase 2 features off. It also fails identically at the starting SHA `90f1f6f`, using that commit's own code.

It is a test-order interaction in the existing WebSocket lifecycle tests.

The agent-state tests added after this run (one in `test_integration.py`, one
in `test_workflows.py`) pass, and the agent-state suite is 267 passed in total.

## 17. New test-suite results

`tests/test_intelligence/agent_state/`: **267 passed**.

| File | Tests | Covers |
|---|---|---|
| `test_runtime.py` | 65 | Config and flags, identity, store pragmas, migrations from every version, quarantine, newer schema, retention, concurrency and isolation, sticky injection, redaction |
| `test_task_state.py` | 23 | Extraction, authority, supersession, completion, new task, compaction, injection placement |
| `test_evidence.py` | 18 | Extractors, supersession, contradiction, staleness, redaction, restart |
| `test_contracts.py` | 42 | Built-in contracts, Windows semantics, safe repair, learned rules, enforcement matrix, hook protocol |
| `test_scope.py` | 25 | Contract compilation, baseline, classes, evidenced expansion, hard blocks, Windows case folding |
| `test_test_impact.py` | 40 | Adapters, edges and decay, risk and tiers, mandatory tier 3, flaky reruns, pre-existing failures, actionable-only verify |
| `test_workflows.py` | 17 | Observation, promotion thresholds, safe classes, slots, execution through the validator and scope, disabling, invalidation |
| `test_injection_economy.py` | 12 | Delta blocks, `none` markers, re-basing after compaction, Codex incremental, event sections, goal de-duplication, history back-off, legacy memo |
| `test_integration.py` | 15 | The real proxy app (Anthropic and Responses): cache-safe replay, feature-off byte identity, `optimize` off ingest-only (HTTP and WS), hooks, wrap install and remove, worker handoff |
| `test_scenarios.py` | 10 | Plan scenarios A–J end to end |

## 18. Benchmarks before/after

Run with `python benchmarks/agent_state_benchmark.py`. It replays a
deterministic 43-request Claude Code session (a goal, 40 read, edit, test and
git turns, a drift edit and a dependency failure) through the real
`AgentStateService`, then runs 200 hook validations. **Before** is the
identical replay with every Phase 2 feature off, which is the Phase 1
baseline.

| Measure | Before (Phase 1) | After (Phase 2) |
|---|---|---|
| Per-request deterministic overhead | 0 | median 4.9 ms, p95 28 ms (first request 62 ms, which opens the database) |
| Hook validation latency | n/a | median 2.1 ms, p95 4.0 ms |
| Tokens sent over the session | 67,143 | 87,000 |
| Cache-prefix violations | 0 | 0 |
| Deterministically invalid calls that reached execution | 80 of 80 | **0 of 80** (blocked before execution) |
| Valid calls denied | 0 | **0** of 120 |
| Drift edit to an excluded path | not detected | detected, constraint marked VIOLATED, warning injected |
| Evidenced dependency (traceback names `mod4.py`) | n/a | admitted as DEPENDENCY_SCOPE |
| Tests selected for a low-risk leaf edit (R = 0.16) | full suite, 40 tests | tier 1, 1 test |
| Wall time, tier 1 vs full suite | 1.01 s (40 tests) | 0.95 s (1 test). In this tiny suite, interpreter start-up dominates |
| 4-step verify loop | 4 model turns | 1 `headroom_workflow` call (4/4 steps, status `success`) |

Phase 1 behaviour is unchanged when every Phase 2 feature is off.
`test_feature_off_is_byte_identical` checks this through the real proxy, and
the Phase 1 eval-corpus gate (`tests/test_intelligence/test_eval_corpus.py`)
still passes.

## 19. Token / tool-turn savings measurements

**Prompt additions.**

- **How the cost arises.** Every block stays in history and is re-sent on each later request, because removing or editing it would break the prompt cache. The cost of a block is therefore its size times the number of later requests.
- **Measured over the 43 requests above:**

  | Measure | Before the fixes below | After |
  |---|---|---|
  | Agent-state tokens re-sent across the session | 38,015 | **19,328** (−49%) |
  | State tokens present in the last request | 2,018 | **1,027** |
  | Injections | 11, all full | 10: 3 full, 7 deltas |
  | Tokens on first send | 1,884 | 920 |

- **What reduced it.** These fixes were found by this benchmark and made in this phase:
  - **Delta blocks.** A delta is used only when its base is provably present in the history. A full block is sent after four deltas, after compaction, on a task change, or with Codex `previous_response_id`.
  - **Event sections.** A delivered scope warning or a macro announcement no longer forces a second block when it disappears.
  - **Goal de-duplication.**
  - **Shorter verify commands.**
  - **History back-off for progress churn.**
- **Relative overhead.** In this synthetic replay the average request is only about 1,560 tokens (no system prompt, no tool schemas, tiny tool outputs), so the overhead is 29.6% of tokens sent. In a real session the absolute number is what carries over: about 1,000 state tokens in the history by request 43. For example, against a 30,000-token request that is about 3%. Those tokens sit in the cached prefix, so Anthropic bills them at the cache-read rate.

**Tool-turn and execution savings in the same replay.**

- 80 calls that would have failed or drifted were blocked before execution, with no false positives.
- One promoted macro collapsed a 4-step verify loop into one call. The metric `workflow_turns_saved_estimate` is 3.
- Verification selected 1 of 40 tests (97.5% avoided) for the low-risk edit. The full suite is still required at completion because the task states "All tests must pass" (a mandatory tier-3 category).

**Not measured.** These §20 metrics need a live model choosing its own actions, and this environment makes no frontier-model calls:
- input and output tokens per completed task;
- model turns;
- repeated reads;
- repeated environment discovery;
- reversions.

The replay holds the model's actions fixed, so it measures Headroom's side: overhead, blocks, selection and collapse.

## 20. Known limitations (platform / API facts only)

1. **Codex hook semantics are undocumented.** Codex does not publicly document whether a PreToolUse exit code 2 blocks the call. Headroom therefore reports `block_requested` until the history proves a block was honored, and turns blocking off for the session if it was not.
2. **Claude Code cannot rewrite arguments without allowing the call.** Claude Code accepts `updatedInput` only together with a permission decision. Answering `allow` would skip the user's permission prompt, so Headroom never rewrites arguments through the hook. Safe repairs apply only to steps Headroom executes itself.
3. **Other clients have no pre-execution control point.** Clients that are not wrapped give Headroom no pre-execution hook, so their calls are only ever `observed` after the fact.
4. **Codex `previous_response_id` hides earlier items.** With `previous_response_id` the earlier items live on the server, so Headroom cannot verify that earlier blocks are present. Those requests always get full blocks and never replay.

## 21. Confirmation

- All six selected features are implemented end to end:
  - wired into the Anthropic and Responses request paths, the hooks, the MCP tool and the CLI;
  - enabled by default through the normal rollout routes;
  - persisted in the versioned workspace store;
  - covered by the tests in §17.
- A search of the Phase 2 code finds no `TODO`, `FIXME`, `NotImplementedError`, stub adapters, disabled code or absolute developer paths.
- Features 18, 21, 22 and 25 were not implemented, as the plan specifies.
