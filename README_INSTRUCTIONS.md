# Headroom setup and configuration: Context Intelligence, Agent State and JevK5

This guide covers everything this branch added to Headroom:

- **Part 1, Context Intelligence.** Fifteen task-aware compression features that decide *what the model sees*.
- **Part 2, Agent State.** Six features that keep the *operational state* of a coding session: task state, an evidence ledger, tool-call validation, a scope firewall, test-impact planning and workflow macros.
- **Part 3, JevK5.** An optional small local model that runs through llama.cpp. It breaks ties on a few multiple-choice decisions.

Everything works with Claude Code (`headroom wrap claude`) and Codex (`headroom wrap codex`) on Windows and Linux. JevK5 is never required: without it, every decision is deterministic.

Longer reference pages: [`docs/content/docs/context-intelligence.mdx`](docs/content/docs/context-intelligence.mdx) and [`docs/content/docs/agent-state.mdx`](docs/content/docs/agent-state.mdx).

---

## Quick start

You do not have to set anything. With the defaults, `headroom wrap claude` / `headroom wrap codex`:

- runs Context Intelligence at the `safe` posture (seeded by the default `coding` savings profile);
- runs all six Agent State features in `auto` mode, which means they act on Claude Code and Codex sessions that use tools;
- uses JevK5 only if you have already set it up. It never downloads or builds anything on its own in this mode.

Turn on more features, and the optional local model:

**PowerShell (Windows)**

```powershell
$env:HEADROOM_INTELLIGENCE = "full"     # safe + delta, admission, budget allocator, tool catalog, rich interceptors
headroom intelligence setup             # one time: install JevK5 (see Part 3)
headroom wrap claude                    # or: headroom wrap codex
```

**cmd (Windows)**

```bat
set HEADROOM_INTELLIGENCE=full
headroom intelligence setup
headroom wrap codex
```

**bash / zsh (Linux, macOS)**

```bash
export HEADROOM_INTELLIGENCE=full
headroom intelligence setup
headroom wrap claude
```

To make a variable permanent on Windows, use `setx HEADROOM_INTELLIGENCE full` (it applies to new terminals), or set it under *System Properties → Environment Variables*. On Linux, add the `export` line to `~/.bashrc` or `~/.zshrc`.

---

## What is on by default

| Layer | Default | Turn it off |
|---|---|---|
| Context Intelligence | `safe` under `headroom wrap` / `headroom proxy` (the `coding` profile). The library default is `off` | `HEADROOM_INTELLIGENCE=off` |
| Agent State, all 6 features | `auto`: Claude Code and Codex tool sessions only. Other API clients are untouched | Per feature (see Part 2), or `HEADROOM_DISABLE_FEATURES=...` |
| JevK5 local model | `auto`: used only when it is already installed | `HEADROOM_JEVK5=off` |
| `headroom_workflow` MCP tool | Registered by `wrap` along with the Headroom MCP server | `HEADROOM_WORKFLOW_MACROS=off` |
| PreToolUse hook (Claude Code / Codex) | Installed by `wrap` when Tool Contracts or the Scope Firewall is on | Turn off both `HEADROOM_TOOL_CONTRACTS` and `HEADROOM_SCOPE_FIREWALL` |

The layers are independent. `HEADROOM_INTELLIGENCE=off` does not turn off Agent State, and Agent State does not need JevK5.

Neither layer breaks a request. If a Phase 2 component fails, the request goes out exactly as it would without it. The one exception is an invalid Agent State setting: the proxy refuses to start and shows a clear error message.

---

# Part 1: Context Intelligence (15 optimizations)

## Enabling it

`HEADROOM_INTELLIGENCE` selects a **posture**:

| Value | Turns on |
|---|---|
| `off` (also `0`, `false`, `no`, `none`) | Nothing. Requests are byte-identical to a proxy without this layer |
| `safe` (also `on`, `1`, `true`, `yes`) | The ten "safe" features below |
| `full` (also `max`, `all`, `aggressive`) | `safe` plus the five "full" features below |

Other routes to the same setting:

- `headroom proxy --intelligence off|safe|full`
- `headroom wrap claude --intelligence full` (every `wrap` subcommand accepts it)

`wrap` restarts a running proxy whose posture differs from the one you asked for.

Every feature also has its own variable that overrides the posture **in either direction**. These variables accept `1/0`, `true/false`, `on/off` or `yes/no`. For example, `HEADROOM_INTELLIGENCE=safe` with `HEADROOM_DELTA=1` adds delta encoding to `safe`.

## The features

| Variable | In posture | What it does |
|---|---|---|
| `HEADROOM_TASK_QUERY` | safe | Builds a task context from your latest message (paths, symbols, quoted terms, error codes). Compression then keeps what *this task* needs instead of guessing from the raw text |
| `HEADROOM_INVARIANT_GUARD` | safe | Vetoes any rewrite that would drop a name you mentioned, an exit code, an error line or a test summary it cannot recover. It also vetoes changed numbers and partial output passed off as complete |
| `HEADROOM_POLICY_RISK_BUDGET` | safe | Caps the irreversible part of any drop: 0.45 for pay-as-you-go, 0.25 for subscription. Content stored exactly in CCR does not count as irreversible |
| `HEADROOM_ARBITER` | safe | Compares the router's result with a lossless fold, a task-aware indexed preview and a gentler re-compression. When none of them is safe, the original wins |
| `HEADROOM_CCR_SEARCH` | safe | Adds `query`, `mode`, `top_k`, `cursor` and `range` to `headroom_retrieve`, so the model fetches only the relevant parts of a stored original |
| `HEADROOM_CCR_SELECTIVE_EXPANSION` | safe | Proactive expansion returns task-relevant spans instead of whole originals |
| `HEADROOM_COMPLEXITY_ROUTING` | safe | Never steers a "why/explain" turn to terse output. Works with the output shaper |
| `HEADROOM_GRAPH_RELEVANCE` | safe | Adds files and symbols one or two hops away in a lightweight per-project code graph to the task context |
| `HEADROOM_RETENTION_LEARNING` | safe | Learns which tools' outputs the model keeps re-fetching, and compresses those more gently |
| `HEADROOM_SPECULATIVE_PREP` | safe | Indexes new tool outputs in the background while the request is in flight |
| `HEADROOM_DELTA` | full | Sends a repeated command, listing or search as a delta against its earlier output. The original is stored for exact recovery |
| `HEADROOM_ADMISSION` | full | A huge tool output enters context as a task-aware preview plus a verified retrieval marker |
| `HEADROOM_BUDGET_ALLOCATOR` | full | Under context pressure, gives the remaining budget to the blocks that matter most for the task |
| `HEADROOM_TOOL_CATALOG` | full | Where native tool search does not apply, keeps the relevant tool schemas in full and compacts the rest. The choice is sticky per conversation, so the prompt cache is preserved |
| `HEADROOM_RICH_INTERCEPTORS` | full | Collapses test-runner output and gives richer file-read outlines, with exact originals in CCR |
| `HEADROOM_DELTA_READS` | never (opt-in) | Also applies deltas to file reads. Off in every posture, because agents patch exact bytes |
| `HEADROOM_EFFORT_ROUTING` | never (opt-in) | Adjusts an effort field the client already sent; it never adds one. Opt-in only, because changing effort costs prompt-cache rewrites |

## Tunables

| Variable | Default | Meaning |
|---|---|---|
| `HEADROOM_CCR_EXPANSION_TOKEN_BUDGET` | `1500` | Token budget for proactive selective expansion |
| `HEADROOM_ADMISSION_MIN_TOKENS` | `2000` | Outputs at least this big go through admission |
| `HEADROOM_ADMISSION_PREVIEW_TOKENS` | `400` | Preview size for an admitted output |
| `HEADROOM_DELTA_MIN_TOKENS` | `200` | Smallest output considered for delta encoding |
| `HEADROOM_BUDGET_PRESSURE_THRESHOLD` | `0.5` | Context fill (0–1) at which the allocator starts |
| `HEADROOM_BUDGET_DIVERSITY_CAP` | `0.6` | Largest share of the budget one source may take |
| `HEADROOM_TOOL_CATALOG_TOP_K` | `5` | Tool schemas kept in full |
| `HEADROOM_TOOL_CATALOG_MIN_TOOLS` | `12` | The catalog applies only with at least this many tools |
| `HEADROOM_GRAPH_MAX_FILES` | `2000` | File cap for the code graph |
| `HEADROOM_RETENTION_LEARNING_ALPHA` | `0.15` | Learning rate, 0.01–0.5 |
| `HEADROOM_RETENTION_LEARNING_MIN_OBS` | `5` | Observations before a learned bias applies |
| `HEADROOM_SPECULATIVE_WORKERS` | `2` | Background indexing threads |
| `HEADROOM_SPECULATIVE_MAX_PENDING` | `64` | Queue cap for background indexing |
| `HEADROOM_INTELLIGENCE_DIR` | `~/.headroom/intelligence` (`%USERPROFILE%\.headroom\intelligence`) | Where intelligence state, JevK5 setup files and Agent State databases live |
| `HEADROOM_RUST_EMBEDDINGS` | unset | `1` loads one process-wide embedding model for semantic relevance in the Rust compressors |

A badly formed Phase 1 value, such as a word where a number is expected, is ignored with a warning and the default is used.

---

# Part 2: Agent State (six features)

## What each feature does

| Feature | What it does | What it is not |
|---|---|---|
| **Task State Compiler** | Keeps a compact, typed record of the task, taken from your own words and from tool results: the goal, binding constraints ("do not change the public API"), acceptance criteria ("all tests must pass"), decisions, pending subgoals, blockers and finished work. It marks a criterion done only when real evidence (a passing test run) proves it. A constraint that gets broken is shown as `VIOLATED` | Not a summary, and it never rewrites what you said |
| **Evidence Ledger** | Records short facts with their source: which command exited with which code, which tests failed, file hashes after edits, git HEAD and status, compiler errors, tool versions. A newer observation replaces an older one. When a tool and the agent disagree, the tool wins and the conflict is recorded | Not CCR or RAG. It never stores file contents or full outputs |
| **Tool Contract Validator** | Checks a tool call *before it runs*. It catches missing files, bad working directories, unknown executables, Windows-only commands on Linux (and the reverse), malformed git branch names, missing test selectors and npm scripts, and narrow rules learned from calls that failed 3 times | It does not replace the tool's own schema |
| **Scope / Drift Firewall** | Builds a "change contract" from the task: the files you named, your exclusions ("do not modify `src/billing/`"), dependencies proven by a failing build or test, and the git dirty-tree baseline. It flags edits unrelated to the task and blocks the clearly forbidden ones | Not a path allowlist. Reading a file never authorizes writing it, and real dependencies are admitted when there is evidence for them |
| **Test Impact Planner** | Picks the smallest useful set of tests for the files the task changed. It scores risk and escalates: tier 1 (direct tests), tier 2 (nearby tests), tier 3 (the full suite). It understands pytest, Cargo, CTest, Jest and Vitest, and uses your virtualenv and package manager | Not "run everything after every edit" |
| **Workflow Macro Compiler** | Learns repeated, successful **read-only or verification** sequences (for example: run tests, then `git status`, then `git diff`) and exposes them as one `headroom_workflow` tool call | Mutating, networked or external sequences are never learned |

## How the model sees it

When something important changes, Headroom adds one compact `<headroom_agent_state>` block to the **current** turn. Earlier blocks are re-sent byte-for-byte, so the prompt cache is never broken.

- **Deltas.** Later blocks carry only what changed. A cleared section is written as, for example, `blocked: none`.
- **Full blocks.** A full block is sent again after four deltas, after the client compacts its history, or when the task changes.
- **Size.** The block is capped at 1,200 tokens. Nothing is added for a one-turn chat.

## Enabling and disabling

All six features are **stable and on by default** in `auto` mode. Each one has a switch that takes `auto`, `on` or `off`:

| Variable | Feature | Rollout name |
|---|---|---|
| `HEADROOM_TASK_STATE` | Task State Compiler | `task_state_compiler` |
| `HEADROOM_EVIDENCE_LEDGER` | Evidence Ledger | `evidence_ledger` |
| `HEADROOM_TOOL_CONTRACTS` | Tool Contract Validator | `tool_contract_validator` |
| `HEADROOM_SCOPE_FIREWALL` | Scope / Drift Firewall | `scope_firewall` |
| `HEADROOM_TEST_IMPACT` | Test Impact Planner | `test_impact_planner` |
| `HEADROOM_WORKFLOW_MACROS` | Workflow Macro Compiler | `workflow_macro_compiler` |

- **`auto`** (default): active for Claude Code and Codex sessions that send tools.
- **`on`**: also active for any other client that sends tools through the proxy. Naming the feature in `HEADROOM_FEATURES` does the same, for example `HEADROOM_FEATURES=scope_firewall`.
- **`off`**: the feature does nothing. `HEADROOM_DISABLE_FEATURES` does the same and takes a comma-separated list of rollout names, for example `HEADROOM_DISABLE_FEATURES=scope_firewall,workflow_macro_compiler`.

Examples:

```powershell
# PowerShell: warn about drift instead of blocking it, and turn workflow macros off
$env:HEADROOM_SCOPE_MODE = "warn"
$env:HEADROOM_WORKFLOW_MACROS = "off"
headroom wrap claude
```

```bat
:: cmd: turn the whole Agent State layer off
set HEADROOM_DISABLE_FEATURES=task_state_compiler,evidence_ledger,tool_contract_validator,scope_firewall,test_impact_planner,workflow_macro_compiler
headroom wrap codex
```

## Knobs

These values are **validated at startup**. An invalid value stops the proxy with a clear message instead of being silently ignored.

| Variable | Default | Allowed | Meaning |
|---|---|---|---|
| `HEADROOM_AGENT_STATE_MAX_TOKENS` | `1200` | 200–8000 | Hard cap for the injected block |
| `HEADROOM_TOOL_CONTRACT_MODE` | `protect` | `protect`, `warn`, `observe` | `protect` blocks clearly invalid calls (where a hook exists), `warn` only warns, `observe` only records |
| `HEADROOM_SCOPE_MODE` | `protect` | `protect`, `warn`, `observe` | The same modes, for scope violations |
| `HEADROOM_TOOL_CONTRACT_REPAIR` | `on` | `on`, `off` | Safe, meaning-preserving repairs (such as making a relative path absolute). They apply only to steps Headroom runs itself, inside a workflow macro |
| `HEADROOM_WORKFLOW_MIN_OBSERVATIONS` | `3` | 3–100 | Observations needed before a macro is promoted |
| `HEADROOM_WORKFLOW_AUTO_CLASSES` | `read_only,verification` | Those two only | Macro classes allowed to promote. Any other class is rejected |
| `HEADROOM_TEST_RISK_TIER2` | `0.35` | 0–1, below tier 3 | Risk at which tier 2 also runs |
| `HEADROOM_TEST_RISK_TIER3` | `0.70` | 0–1 | Risk at which the full suite also runs |
| `HEADROOM_AGENT_STATE_DIR` | `~/.headroom/intelligence/agent_state` | A path | Where the per-project databases live |

## Blocking: what is really possible per agent

Headroom says a call was **blocked** only when it really stopped the call before it ran:

| | Claude Code (`wrap claude`) | Codex (`wrap codex`) | Other clients (only with `on`) |
|---|---|---|---|
| How it hooks in | PreToolUse hook in `.claude/settings.local.json` | PreToolUse hook in `hooks.json` in your Codex home (`CODEX_HOME`, by default `%USERPROFILE%\.codex` or `~/.codex`), with `features.hooks=true` for that session only | None |
| Block before execution | **Yes** | **Requested**. It is reported as blocked only after the history shows Codex honored the block, because Codex does not document this | No. Findings are recorded after the fact |
| Warning before execution | Yes | No. The warning appears in the next turn | No |
| Rewrite tool arguments | No. Doing so would bypass your permission prompt | No | No |
| `headroom_workflow` tool | Yes (MCP) | Yes (MCP) | No |

**Hooks.**

- `wrap` installs the hooks automatically, and `headroom unwrap claude` / `headroom unwrap codex` removes them. Your own hooks are left alone.
- The hook script uses only the Python standard library. If the proxy is not running, the hook exits at once and the agent carries on.

**Hard blocks.** These are limited to clear-cut cases:

- writes outside the project, with symlinks and junctions resolved;
- paths you excluded;
- `.git` internals;
- installed dependencies;
- credential files you did not name;
- git-ignored generated output.

Anything ambiguous gets a warning only. **Your words win.** Naming a file puts it in scope, and excluding one makes edits there a hard violation.

## Diagnostics

```bash
headroom intelligence state       # task id, revision and compact task state
headroom intelligence evidence    # current facts; --claim "tests_passing|latest" for one claim
headroom intelligence contracts   # capability matrix, learned rules, recent checks
headroom intelligence scope       # change contract, task-owned changes, warnings
headroom intelligence verify      # risk, tiers and reasons; --run executes the plan (--max-tier 1|2|3)
headroom intelligence workflows   # candidate, promoted and disabled macros
```

Each command accepts `--project PATH`, `--session ID` and `--json`. While the proxy runs, `GET http://127.0.0.1:<port>/v1/agent-state/status` serves the same data (loopback only).

## Storage and privacy

- **Location.** There is one SQLite database per project, at `<agent state dir>\<hash of the project path>\agent_state.sqlite3`. Data never leaves your machine, and one project cannot see another's data.
- **Redaction.** Everything is redacted before it is written: API keys, `Authorization` and `Cookie` values, tokens, passwords, private keys, credential URLs and `.env` values. File contents and full command outputs are never stored.
- **Retention.**
  - Completed tasks and durable evidence: 30 days.
  - Volatile facts: their TTL plus 7 days.
  - Test-impact edges: 90 days.
  - Unused macros: 90 days.
- **Damaged or newer databases.** A corrupt database is renamed to `*.corrupt-<time>` and kept. A database written by a newer Headroom version is left untouched, and persistence turns off.

---

# Part 3: JevK5, the optional local decision model

## What it is

[JevK5](https://huggingface.co/alibiserikbay/JevK5-GGUF) is a small (4B) model that answers multiple-choice, yes/no and score questions.

**Where it is consulted.** Headroom asks it only about a few ambiguous decisions:

- arbiter ties;
- whether to keep a tool schema in full;
- how complex a turn is;
- whether a new message refines or replaces the task goal;
- whether two constraints conflict.

**Limits.**

- Its advice is blended with the deterministic score at a bounded weight (25% by default, never more than 40%), and is ignored below 60% confidence.
- It can never override a safety check, block a tool call or write state text.
- It runs locally through llama.cpp's `llama-server`, bound to `127.0.0.1` only.
- No cloud or "frontier" model calls are made for it.

## Do I have to install it manually?

**No. Headroom installs it for you,** but only when you ask, because it means a download of 2.7–9.5 GB and possibly a compile. Headroom needs two things, and handles both:

1. **`llama-server`** (`llama-server.exe` on Windows). Headroom looks for one in this order:
   - a path you give it;
   - the one recorded by an earlier setup;
   - `PATH`;
   - `LLAMA_CPP_HOME`;
   - the usual WinGet, Scoop and Chocolatey locations (and `/usr/local/bin` or `/opt/homebrew/bin` on Linux and macOS);
   - its own managed build.

   **Building it.** If none is found, Headroom can **build one itself** from `github.com/ggml-org/llama.cpp`. The build:
   - goes into `~/.headroom/intelligence/llama.cpp-managed`;
   - needs no admin rights;
   - uses CUDA when `nvcc` is present, and the CPU backend otherwise.

   **What the build needs.**
   - **Windows:** `git`, `cmake` and the Visual Studio C++ Build Tools. Ninja is used when available. Running from a "Developer PowerShell for VS" is the easiest way.
   - **Linux and macOS:** `git`, `cmake` and a C++ compiler (`g++` or `clang++`).

   **Skipping the build.** Install a prebuilt `llama-server` instead (a release zip from github.com/ggml-org/llama.cpp/releases, WinGet, Scoop or Chocolatey) and either put it on `PATH` or pass `--llama-server C:\path\to\llama-server.exe`.

2. **The model file** (default `jevk5-4b-v0.3-Q8_0.gguf`, about 4.5 GB). Headroom reuses a copy already in your Hugging Face or llama.cpp cache (`HF_HUB_CACHE`, `HF_HOME`, `LLAMA_CACHE`, or the default cache folders). Otherwise it downloads the file with resume support and checks its SHA-256.

**Python package.** `setup` also pip-installs the pinned `jevk5` package (v0.3.0) into Headroom's own Python interpreter. It is used only to cross-check the protocol during setup; skip it with `--no-install`.

## Setting it up (recommended way)

**PowerShell (Windows)**

```powershell
headroom intelligence setup      # finds/builds llama-server, fetches the model, verifies end to end
headroom intelligence doctor     # re-checks llama-server, the model cache and a live decision
headroom wrap claude             # HEADROOM_JEVK5=auto (the default) now uses it automatically
```

**bash (Linux)**

```bash
headroom intelligence setup
headroom intelligence doctor
headroom wrap codex
```

`setup` is safe to run again; it skips anything already done. Its options:

| Option | Effect |
|---|---|
| `--llama-server PATH` | Use this `llama-server` executable |
| `--model-file FILE` | Pick another file from `alibiserikbay/JevK5-GGUF` (see below) |
| `--no-download` | Fail instead of downloading the model |
| `--no-build` | Never build llama.cpp |
| `--no-install` | Do not pip-install the `jevk5` package |
| `--keep-running` | Leave the verified `llama-server` running |
| `--json` | Machine-readable report |

Available model files (`HEADROOM_JEVK5_MODEL_FILE` or `--model-file`):

| File | Approximate size | Notes |
|---|---|---|
| `jevk5-4b-v0.3-Q8_0.gguf` | 4.5 GB | Default |
| `jevk5-4b-v0.3-Q5_K_M.gguf` | 3.1 GB | Smaller |
| `jevk5-4b-v0.3-Q4_K_M.gguf` | 2.7 GB | Smallest |
| `jevk5-9b-v0.3-Q8_0.gguf` | 9.5 GB | Larger model |

A GPU is optional. GPU layers are chosen automatically, and if the GPU runs out of memory Headroom retries once on the CPU.

## The three modes

| `HEADROOM_JEVK5` / `--jevk5` | Behaviour |
|---|---|
| `auto` (default) | Uses JevK5 if a `llama-server` and the model are already present, and starts the service with the proxy. It **never** downloads or builds anything. If something is missing, Headroom logs a warning and runs deterministically |
| `on` | The same, but the proxy may also build `llama-server` and download the model on first start (in the background). Running `headroom intelligence setup` first is still recommended, because it shows progress and verifies everything |
| `off` | Never used |

JevK5 is consulted only while Context Intelligence is on (`HEADROOM_INTELLIGENCE` is not `off`).

Other ways to set the mode:

- `headroom proxy --jevk5 auto|on|off --llama-server PATH --jevk5-url URL`
- `headroom wrap claude --jevk5 on`

## Using a decision service you run yourself

`HEADROOM_JEVK5_URL` (or `--jevk5-url`) points Headroom at a `/v1/systemone` decision endpoint that you manage. Headroom never starts or stops it. Note that this is a decision endpoint, not a bare `llama-server` URL. Two ways to provide one:

- the upstream JevK5 server; or
- Headroom's own gateway, which you run in a separate terminal:

  ```bash
  headroom intelligence gateway --port 8099      # serves http://127.0.0.1:8099/v1/systemone
  ```

  Then set `HEADROOM_JEVK5_URL=http://127.0.0.1:8099`.

## Managing the service

```bash
headroom intelligence status     # resolved features, JevK5 mode/model, service state, advisor stats
headroom intelligence doctor     # diagnostics (--no-live skips the live decision check)
headroom intelligence stop       # stop the Headroom-owned llama-server (--force for one started by another process)
```

Service behaviour:

- **Sharing.** Several Headroom processes share one service through a lock file. A service whose owner exited is adopted rather than left orphaned.
- **Windows.** The service runs without a console window and is stopped with `taskkill /T`.
- **Logs.** Logs go to `~/.headroom/intelligence/logs/llama-server-<port>.log`.

## JevK5 variables

| Variable | Default | Meaning |
|---|---|---|
| `HEADROOM_JEVK5` | `auto` | `auto`, `on` or `off` (see above) |
| `HEADROOM_JEVK5_URL` | unset | External `/v1/systemone` endpoint, never started or stopped by Headroom |
| `HEADROOM_JEVK5_LLAMA_SERVER` | unset | Path to `llama-server` or `llama-server.exe` |
| `HEADROOM_JEVK5_MODEL_REPO` | `alibiserikbay/JevK5-GGUF` | Hugging Face repository |
| `HEADROOM_JEVK5_MODEL_FILE` | `jevk5-4b-v0.3-Q8_0.gguf` | GGUF file to load |
| `HEADROOM_JEVK5_ALLOW_DOWNLOAD` | `true` only when the mode is `on` | Allow the proxy to fetch the model |
| `HEADROOM_JEVK5_ALLOW_BUILD` | `true` only when the mode is `on` | Allow the proxy to build llama.cpp |
| `HEADROOM_JEVK5_AUTOSTART` | `true` | Start the service together with the proxy |
| `HEADROOM_JEVK5_PORT` | `0` (pick a free loopback port, from 8091 up) | Fixed service port |
| `HEADROOM_JEVK5_GPU_LAYERS` | `auto` | `--n-gpu-layers`; `0` means CPU only |
| `HEADROOM_JEVK5_CTX` | `8192` | Context size (minimum 512) |
| `HEADROOM_JEVK5_TIMEOUT_MS` | `1200` | Per-decision deadline. Late answers are ignored |
| `HEADROOM_JEVK5_MAX_CALLS_PER_REQUEST` | `4` | Advisor calls allowed per proxied request |
| `HEADROOM_JEVK5_ADVISORY_WEIGHT` | `0.25` | Blend weight, clamped to at most 0.40 |
| `HEADROOM_JEVK5_MIN_CONFIDENCE` | `0.60` | Below this, the advice gets zero weight |
| `HEADROOM_JEVK5_DECISION_CACHE_TTL_SECONDS` | `30` | How long an identical decision is reused |
| `HEADROOM_JEVK5_MAX_STATE_TOKENS` | `6000` | Most context passed into one decision |
| `HEADROOM_JEVK5_TOP_K` | `40` | Log-probability candidates requested from llama.cpp (`n_probs`, minimum 16) |
| `HEADROOM_JEVK5_TEMPERATURE` / `HEADROOM_JEVK5_KNOCKOUT_TEMPERATURE` | Set per model (1.22 / 0.93 for the default) | Calibration. Leave these alone unless you know you need to change them |
| `HEADROOM_JEVK5_GATEWAY_PORT` | `0` | Default port for `headroom intelligence gateway` |
| `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` | unset | Hugging Face token for the download, if your network needs one |
| `LLAMA_CPP_HOME` | unset | Your own llama.cpp checkout, searched for a built `llama-server` |
| `HF_HUB_CACHE`, `HF_HOME`, `LLAMA_CACHE` | standard locations | Where Headroom looks for an already-downloaded model |

## If JevK5 does not start

- **`no compatible llama-server found`.** Run `headroom intelligence setup`, or install a prebuilt `llama-server` and pass `--llama-server`. On Windows, the managed build needs `git`, `cmake` and the Visual Studio C++ Build Tools.
- **`model ... is not downloaded yet`.** You are in `auto` mode, which never downloads. Run `headroom intelligence setup` once, or set `HEADROOM_JEVK5=on`.
- **Out of GPU memory.** Headroom retries once on the CPU by itself. To force the CPU, set `HEADROOM_JEVK5_GPU_LAYERS=0`. You can also choose a smaller model file.
- **Anything else.** Run `headroom intelligence doctor` and check `~/.headroom/intelligence/logs/`. Headroom keeps working without JevK5 in every case.

---

# Shared variables and paths

| Variable | Default | Meaning |
|---|---|---|
| `HEADROOM_FEATURES` | unset | Comma-separated rollout names to force on. For Agent State this means `on` mode |
| `HEADROOM_DISABLE_FEATURES` | unset | Comma-separated rollout names to turn off |
| `HEADROOM_WORKSPACE_DIR` | `~/.headroom` | Root of all Headroom state |
| `HEADROOM_INTELLIGENCE_DIR` | `<workspace>/intelligence` | Intelligence state, JevK5 setup (`setup.json`, `runtime.json`), managed llama.cpp, logs |
| `HEADROOM_AGENT_STATE_DIR` | `<intelligence dir>/agent_state` | Agent State databases |

On Windows, `~` means `%USERPROFILE%`, for example `C:\Users\you\.headroom`.

# Status endpoints

These are served on loopback only, while the proxy runs:

| Endpoint | Shows |
|---|---|
| `GET /v1/intelligence/status` | Resolved Context Intelligence features, counters, JevK5 advisor state, and the Agent State summary |
| `GET /v1/agent-state/status` | Agent State features, knobs, metrics and per-session capabilities |
| `GET /stats` | Gains an `intelligence` section (arbiter outcomes, tokens saved by source) |
| `GET /health` | Reports the active posture under `config.intelligence` |

# Turning everything off

```powershell
$env:HEADROOM_INTELLIGENCE = "off"
$env:HEADROOM_JEVK5 = "off"
$env:HEADROOM_DISABLE_FEATURES = "task_state_compiler,evidence_ledger,tool_contract_validator,scope_firewall,test_impact_planner,workflow_macro_compiler"
headroom wrap claude
```

With all three set, Headroom behaves exactly as it did before these features existed. Run `headroom unwrap claude` / `headroom unwrap codex` to remove the hooks as well.
