# Context Intelligence + JevK5 — Implementation Report

This report covers the 15 token-efficiency/intelligence optimizations and the
JevK5 local decision model (llama.cpp `llama-server`). The target is local use
with **Claude Code** (Anthropic `/v1/messages`) and **Codex** (OpenAI Responses,
HTTP and WebSocket) on **Windows** and Linux.

User documentation: [`docs/content/docs/context-intelligence.mdx`](docs/content/docs/context-intelligence.mdx).

## Enablement (normal routes only)

There is no rollout channel. `tool_result_interceptors` moved from canary to
stable; the canary engine tests now use a dedicated `canary_probe` feature.

| Route | How |
|---|---|
| Environment | `HEADROOM_INTELLIGENCE=off\|safe\|full` plus one variable per feature (overrides the posture either way) |
| `headroom proxy` | `--intelligence off\|safe\|full`, `--jevk5 auto\|on\|off`, `--jevk5-url`, `--llama-server` |
| `headroom wrap <agent>` | `--intelligence` / `--jevk5` on every wrap subcommand. The proxy restarts automatically when its running posture differs |
| Savings profile | `coding` (the default) seeds `HEADROOM_INTELLIGENCE=safe` with setdefault, so an explicit value wins |
| Multi-worker | The resolved config travels in the worker JSON (`IntelligenceConfig.from_dict`) |
| Rust proxy | The same `HEADROOM_INTELLIGENCE` / `HEADROOM_TASK_QUERY` / `HEADROOM_INVARIANT_GUARD` / `HEADROOM_POLICY_RISK_BUDGET` variables |

`off` is byte-identical to a proxy without the layer. The process-wide switches
(CCR search schema, rich interceptors, installed runtime) are set on every
proxy construction, both on and off.

## The 15 optimizations

| # | Optimization | Where | Posture |
|---|---|---|---|
| 1 | Task-conditioned relevance query (TaskContext: paths, symbols, quoted terms, error codes; exact terms first) | `headroom/intelligence/task_context.py`, `runtime.prepare_request`; Rust `intelligence::relevance_query` (live zone no longer passes `EMPTY_QUERY`) | safe |
| 2 | Hybrid relevance correctness and shared embeddings: email TLD regex fixed on both sides (`[A-Z\|a-z]` → `[A-Za-z]`); one process-wide lazy `EmbeddingScorer` with an LRU embedding cache | `headroom/relevance/hybrid.py`, `crates/headroom-core/src/relevance/{embedding,embedding_cache,hybrid}.rs` | `HEADROOM_RUST_EMBEDDINGS=1` |
| 3 | Indexed/partial CCR retrieval: exact span index (FTS5/BM25, light stemming), `headroom_retrieve` query/mode/top_k/cursor/range, selective expansion | `headroom/ccr/span_index.py`, `tool_injection.py`, `mcp_server.py`, `context_tracker.py`, `/v1/retrieve` | safe |
| 4 | Pre-context admission: huge outputs become a task-aware preview plus a verified marker. Failure spans are always surfaced, and there is no preview when nothing matches | `headroom/intelligence/admission.py` | full |
| 5 | Cross-turn delta encoding (JSON-keyed/hashed, line-set, line-diff); the base must be complete; decisions memoized | `headroom/intelligence/delta.py`, `admission.py` | full |
| 6 | Transform Arbiter: router vs lossless fold vs indexed preview vs conservative re-compression; Pareto rule (keep the original when nothing is safe) | `headroom/intelligence/arbiter.py`, `ContentRouter._intelligence_review` | safe |
| 7 | Invariant guard with backoff ladder (entities, exit codes, numbers, errors, test summaries, truncation provenance, JSON/diff structure, marker validity) | `headroom/intelligence/invariants.py`; Rust `intelligence::InvariantSet` | safe |
| 8 | Global budget allocator (pressure-aware, diversity cap) | `headroom/intelligence/budget.py` | full |
| 9 | Progressive tool catalog, used only where native tool search did not apply; sticky per conversation | `headroom/proxy/tool_catalog.py`, `intelligence/turn_routing.py` (Anthropic, OpenAI chat, Responses) | full |
| 10 | Complexity-aware verbosity (all shaper sites); effort routing with hysteresis that never injects a field | `headroom/proxy/output_complexity.py`, `intelligence/turn_routing.py` | verbosity: safe; effort: `HEADROOM_EFFORT_ROUTING=1` only |
| 11 | Code-graph relevance (2-hop neighborhood) | `headroom/intelligence/graph.py` | safe |
| 12 | Retrieval-feedback retention learning (EWMA priors, persisted) | `headroom/intelligence/feedback.py` | safe |
| 13 | Policy admission: irreversible drop ≤ `max_lossy_ratio`; `volatile_token_threshold` consumed | `headroom/intelligence/policy.py`; Rust `compression_policy::{drop_ratios, admit_lossy, classify_change}` | safe |
| 14 | Rich interception: test-runner collapse, read outlines with exact CCR originals | `headroom/proxy/interceptors/{test_runner,astgrep}.py` | full |
| 15 | Speculative preparation (provenance/invariants/span index computed off the request path) | `headroom/intelligence/speculative.py` | safe |

## JevK5 + llama.cpp

- **Discovery** looks in this order: explicit path, recorded setup, `PATH` (including `.exe`), `LLAMA_CPP_HOME` layouts, WinGet/Scoop/Chocolatey/`LOCALAPPDATA`, then the managed build. Each candidate is capability-probed with `--version`, `--help` and `--list-devices`.
- **Managed build** uses a git clone plus `cmake --target llama-server`. It builds with CUDA when `nvcc` is present (retrying on CPU if that fails) and uses Ninja on Windows when available.
- **Model**: `alibiserikbay/JevK5-GGUF` / `jevk5-4b-v0.3-Q8_0.gguf`. It is taken from the HF/llama.cpp cache when present, otherwise downloaded with resume and SHA-256 verification.
- **Service**:
  - Binds to `127.0.0.1` only.
  - Takes a cross-process lock (`fcntl`/`msvcrt`).
  - Reuses a live service and adopts ownership only when the owner process is dead.
  - Retries once on address-in-use, and once on out-of-memory with `-ngl 0`.
  - On Windows it runs with `CREATE_NO_WINDOW` in its own process group, and stops with `taskkill /T`.
- **Protocol**: option-letter log-probabilities via `/tokenize` + `/completion` (`n_predict=1`, `n_probs`), with knockout for more than 16 options. Answers use the `/v1/systemone` shape, and setup verifies agreement with upstream `JevK5GGUF`.
- **Advisory only**:
  - The blend weight is at most 0.40 and drops to zero below the confidence floor.
  - Each request has a call budget and a deadline, and decisions are cached with a TTL.
  - Advice can never override a hard invariant or the policy budget.
- **CLI**: `headroom intelligence setup|status|doctor|stop|gateway`.

## Codex (Responses) path

Codex compression does not run `TransformPipeline`. It routes each Responses
item text as a `CompressionUnit`, with a unit-result cache keyed on `context`.
`headroom/intelligence/responses.py` adapts the layer to that path:

- builds the TaskContext from the `input` items;
- **pins** each item text's routing inputs on first sighting. Without this,
  every new question would recompress history and bust the prompt cache. A
  regression test shows that unpinned bytes drift and pinned bytes do not;
- runs delta/admission on a chat-shaped view and splices rewritten outputs
  back by `call_id`, preserving the output shape;
- runs the invariant/arbiter review per unit;
- applies the tool catalog after native tool-search deferral, with
  `cache_cold=False` (the set is chosen once per conversation).

## Windows

- **Command classification** covers PowerShell/cmd reads and searches: `Get-Content`/`gc`/`type`, `Select-String`/`sls`/`findstr`, `-Command`/`/c` inner commands, `.exe`/`.cmd` suffixes, the `&` call operator and `Set-Location`/`pushd` prefixes.
- **Paths**: drive letters, backslashes and case are normalized. Relative shell reads resolve against Codex's per-call `workdir`, so the same relative name in two checkouts is two resources. Content hashes ignore CRLF.
- **UTF-8**: subprocess output is decoded as UTF-8 explicitly (cp1252 would crash on non-ASCII output).
- **Per-project scoping**: the code graph is scoped per project, so one proxy serving several repos never mixes their symbols. Claude Code's project comes from the same resolver CCR uses (system-prompt `cwd`); Codex's comes from its `<environment_context><cwd>` block.
- **Docs**: the user docs include PowerShell and cmd examples.

## Measured impact

From `python benchmarks/intelligence_benchmark.py --repeat 5`, run on a deterministic corpus of 8 coding-agent scenarios with 23 task-critical facts, through the real proxy paths. The ML text compressor was disabled for determinism.

| Path | Posture | Tokens out / in | Saved | Facts visible | Facts recoverable | Median ms |
|---|---|---:|---:|---:|---:|---:|
| Claude Code | off | 35 934 / 45 483 | 21.0% | 23/23 | 23/23 | 2.4 |
| Claude Code | safe | 15 425 / 45 483 | 66.1% | 23/23 | 23/23 | 8.5 |
| Claude Code | full | 14 981 / 45 483 | 67.1% | 23/23 | 23/23 | 15.5 |
| Codex | off | 35 270 / 45 273 | 22.1% | 23/23 | 23/23 | 0.8 |
| Codex | safe | 9 178 / 45 273 | 79.7% | 23/23 | 23/23 | 7.5 |
| Codex | full | 9 032 / 45 273 | 80.0% | 23/23 | 23/23 | 9.3 |

`tests/test_intelligence/test_eval_corpus.py` turns the corpus into a gate. It fails if any posture loses a fact on either path, sends more tokens than `off`, or saves less than 25% overall.

## Verification

- **Python**:
  - `tests/test_intelligence/`: 293 tests. They cover foundation, JevK5 against a fake `llama-server` subprocess, indexed CCR, Windows commands, delta/admission, rich interceptors, budget/graph/feedback, catalog/complexity, request-path wiring, Rust parity and the eval corpus.
  - Existing suites run against the change:
    - router/pipeline/CCR: 640;
    - proxy/handler (Anthropic, OpenAI, Responses, shaper): 1,687;
    - Codex/Responses: 495;
    - CLI/wrap: 1,192;
    - rollout and interceptors.
  - Repo lint: `ruff check` and `ruff format` are clean. `mypy headroom` reports only the 2 pre-existing errors outside this change.
- **Rust**:
  - `cargo fmt --check` and `cargo clippy --workspace -D warnings` are clean.
  - New unit tests cover the policy admission parity fixture, the invariant guard, the env posture, relevance-query extraction, the live-zone intelligence hooks, the email regex and the embedding LRU cache.
  - The lexical-only build (`--no-default-features`) compiles.
- **Parity**: `tests/fixtures/intelligence_policy_parity.json` is asserted by both the Python (`test_rust_parity.py`) and Rust test suites.

## Known limitations and follow-ups

- The real JevK5 model (4 B, Q8, ~4.3 GB) was not downloaded in this environment. The service, protocol and advisor were exercised against a protocol-faithful fake `llama-server`. Run `headroom intelligence setup` then `headroom intelligence doctor` on the target machine to verify end to end.
- The Rust live zone implements optimizations 1, 7 and 13 (query, guard, policy). The arbiter, delta and admission stages are Python-proxy features, and `headroom wrap` launches the Python proxy.
- The Rust email-regex fix reaches Python's SmartCrusher anchors only after `headroom._core` is rebuilt (CI/release builds do this).
- Pre-existing and unrelated to this change:
  - `crates/headroom-core/tests/kompress_parity.rs` waits on a model download.
  - `tests/test_cli/test_mcp_reconcile.py::test_ordinary_install_does_not_adopt_serena` fails on the base commit in this environment.
  - `test_codex_ws_compression_scheduler::test_concurrent_compression_has_no_semaphore_tail` is timing-sensitive under load and passes in isolation.
