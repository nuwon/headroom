"""Agent-state runtime: one per wrapped workspace and session (plan §4.2).

:class:`AgentStateRuntime` owns a session's view of the six Phase 2 systems:
the Task State Compiler, Evidence Ledger, Tool Contract Validator, Scope
Firewall, Test Impact Planner and Workflow Macro Compiler. It holds no
process-global task state. Workspace knowledge (evidence, learned rules,
impact edges, macros) lives in the workspace database and is shared only
within that workspace, while task state is keyed by session and conversation
lineage.

:class:`AgentStateService` is the proxy-level composition root. It identifies
the agent session behind a request, finds or creates the runtime, ingests the
request's history as events in the normative order (plan §4.7), answers host
pre-tool hooks, and produces the one compact ``<headroom_agent_state>`` block
for the live turn.

Every entry point fails open (plan §16). Any exception leaves the request
exactly as it would be without Phase 2.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import AgentStateConfig, EnforcementMode, FeatureMode
from .events import AgentEvent, EventType
from .ids import (
    claude_session_id,
    codex_session_id,
    find_project_root,
    identity_path,
    is_continuation_summary,
    lineage_key,
    stable_id,
    workspace_id,
)
from .injection import PendingInjection, StickyInjector
from .normalizer import (
    HistoryItem,
    NormalizeContext,
    conversation_root,
    events_for_item,
    history_items,
    item_event_ids,
)
from .store import AgentStateStore, dumps, get_store, loads

logger = logging.getLogger(__name__)

MAX_RUNTIMES = 128
MINOR_REFRESH_REQUESTS = 6
# Delta blocks: at most this many deltas follow a full block, and a delta is
# used only when it is at most this share of the full block's size.
DELTA_CHAIN_MAX = 4
DELTA_MAX_SHARE = 0.75
# Every block stays in history and is re-sent on each later request. Minor
# progress churn backs off as that history grows: the refresh interval grows by
# MINOR_REFRESH_REQUESTS for every this many state tokens live in the history.
# Major changes (goal, constraints, blockers, conflicts...) are never delayed.
STATE_HISTORY_SOFT_TOKENS = 2000


# ------------------------------------------------------------------ metrics
@dataclass
class AgentStateMetrics:
    """Counters only: never content (plan §15)."""

    counters: Counter[str] = field(default_factory=Counter)
    latency_ms: dict[str, list[float]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def bump(self, name: str, amount: int = 1) -> None:
        with self._lock:
            self.counters[name] += amount

    def observe(self, name: str, ms: float) -> None:
        with self._lock:
            series = self.latency_ms.setdefault(name, [])
            series.append(ms)
            if len(series) > 512:
                del series[:256]

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            lat = {}
            for name, values in self.latency_ms.items():
                if values:
                    ordered = sorted(values)
                    lat[name] = {
                        "median": round(ordered[len(ordered) // 2], 3),
                        "p95": round(ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))], 3),
                        "n": len(ordered),
                    }
            return {"counters": dict(self.counters), "latency_ms": lat}


# ------------------------------------------------------------- capabilities
@dataclass
class Capabilities:
    """What Headroom can truthfully do for this agent session (plan §7.9)."""

    agent: str = ""
    can_observe_tool_call: bool = True
    can_rewrite_safe_args: bool = False
    can_block_before_execution: bool = False
    can_execute_local_headroom_tool: bool = False
    hook_seen_at: float | None = None
    # Host hook whose blocking semantics are undocumented (Codex): blocks are
    # *requested* until history shows a requested block was honored.
    block_unverified: bool = False
    downgraded: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "can_observe_tool_call": self.can_observe_tool_call,
            "can_rewrite_safe_args": self.can_rewrite_safe_args,
            "can_block_before_execution": self.can_block_before_execution,
            "can_execute_local_headroom_tool": self.can_execute_local_headroom_tool,
            "hook_seen_at": self.hook_seen_at,
            "block_unverified": self.block_unverified,
            "downgraded": self.downgraded,
        }

    @classmethod
    def from_dict(cls, data: Any) -> Capabilities:
        caps = cls()
        if isinstance(data, dict):
            for key in caps.to_dict():
                if key in data:
                    setattr(caps, key, data[key])
        return caps


# ---------------------------------------------------------------- workspace
@dataclass
class Workspace:
    workspace_id: str
    root: str  # canonical display path
    local: bool  # the root exists on this machine (filesystem-dependent checks allowed)
    store: AgentStateStore
    facts: dict[str, Any] = field(default_factory=dict)  # lazily computed project facts

    def contains(self, path: str) -> bool:
        if not path or not self.root:
            return False
        p = identity_path(path)
        r = identity_path(self.root)
        return p == r or p.startswith(r.rstrip("/") + "/")

    def relpath(self, path: str) -> str:
        from .paths import relative_to_root

        return relative_to_root(path, self.root)


# ------------------------------------------------------------ per-session
class AgentStateRuntime:
    """One wrapped agent session (lineage) inside one workspace."""

    def __init__(
        self,
        service: AgentStateService,
        workspace: Workspace,
        *,
        session_key: str,
        agent: str,
        agent_session: str,
        lineage: str,
    ) -> None:
        self.service = service
        self.config = service.config
        self.workspace = workspace
        self.store = workspace.store
        self.session_key = session_key
        self.agent = agent
        self.agent_session = agent_session
        self.lineage = lineage
        self.lock = threading.RLock()
        # Capabilities belong to the host session (shared by its lineages).
        self.capabilities = service.capabilities_for(workspace.workspace_id, agent_session, agent)
        self.metrics = service.metrics
        self._seq = 0
        self._known_ids: list[str] = []
        self._boundary: tuple[int, str, Counter] | None = None
        self.user_turns = 0
        self.history_tokens = 0
        self.pending_warnings: list[tuple[int, str, str]] = []  # (priority, key, text)
        self.last_tool_family = ""
        self.requests_since_injection = 0
        self.tool_schemas: dict[str, Any] = {}
        self.created_now = False
        self._load_session()
        cfg = self.config
        # Engines are created only for enabled features (disabled flags cost nothing).
        from .contracts import ToolContractValidator
        from .evidence import EvidenceLedger
        from .scope import ScopeFirewall
        from .task_state import TaskStateCompiler
        from .test_impact import TestImpactPlanner
        from .workflows import WorkflowCompiler

        self.evidence: EvidenceLedger | None = (
            EvidenceLedger(self) if cfg.evidence.enabled else None
        )
        self.task_state: TaskStateCompiler | None = (
            TaskStateCompiler(self) if cfg.task_state.enabled else None
        )
        self.scope: ScopeFirewall | None = ScopeFirewall(self) if cfg.scope.enabled else None
        self.contracts: ToolContractValidator | None = (
            ToolContractValidator(self) if cfg.contracts.enabled else None
        )
        self.test_impact: TestImpactPlanner | None = (
            TestImpactPlanner(self) if cfg.test_impact.enabled else None
        )
        self.workflows: WorkflowCompiler | None = (
            WorkflowCompiler(self) if cfg.workflows.enabled else None
        )
        if self.created_now:
            self._record_lifecycle(EventType.SESSION_START)
        if self.workflows is not None and not workspace.facts.get("seeded"):
            workspace.facts["seeded"] = True
            try:
                from .workflows import seed_macros

                seed_macros(self)
            except Exception:  # noqa: BLE001
                logger.debug("workflow seed registration failed", exc_info=True)

    # --------------------------------------------------------- persistence
    def _load_session(self) -> None:
        row = self.store.query_one(
            "SELECT * FROM sessions WHERE session_key = ?", (self.session_key,)
        )
        now = time.time()
        if row is None:
            self.created_now = True
            self.store.write(
                lambda c: c.execute(
                    "INSERT OR IGNORE INTO sessions(session_key, agent, agent_session, lineage, "
                    "created_at, last_seen, capabilities) VALUES (?,?,?,?,?,?,?)",
                    (
                        self.session_key,
                        self.agent,
                        self.agent_session,
                        self.lineage,
                        now,
                        now,
                        dumps(self.capabilities.to_dict()),
                    ),
                )
            )
        else:
            stored = Capabilities.from_dict(loads(row["capabilities"], {}))
            if stored.hook_seen_at and not self.capabilities.hook_seen_at:
                for key, value in stored.to_dict().items():
                    setattr(self.capabilities, key, value)
            self.capabilities.agent = self.agent or self.capabilities.agent
            cursor = loads(row["cursor"], {}) or {}
            self.user_turns = int(cursor.get("user_turns", 0) or 0)
            self.service.injector.load(self.session_key, loads(row["injections"], []))
        seq = self.store.query_one(
            "SELECT MAX(seq) AS s FROM events WHERE session_key = ?", (self.session_key,)
        )
        self._seq = int(seq["s"] or 0) if seq is not None else 0

    def save_session(self) -> None:
        caps = dumps(self.capabilities.to_dict())
        cursor = dumps({"user_turns": self.user_turns})
        memo = dumps([e.to_json() for e in self.service.injector.entries(self.session_key)])
        self.store.write(
            lambda c: c.execute(
                "UPDATE sessions SET last_seen = ?, capabilities = ?, cursor = ?, injections = ? "
                "WHERE session_key = ?",
                (time.time(), caps, cursor, memo, self.session_key),
            )
        )

    def _record_lifecycle(self, etype: EventType) -> None:
        ev = AgentEvent(
            event_id=stable_id(self.session_key, etype.value, int(time.time() // 60)),
            workspace_id=self.workspace.workspace_id,
            session_id=self.session_key,
            event_type=etype,
            metadata={"agent": self.agent},
        )
        self.persist_events([ev])

    def end(self) -> None:
        self._record_lifecycle(EventType.SESSION_END)
        self.store.write(
            lambda c: c.execute(
                "UPDATE sessions SET ended_at = ? WHERE session_key = ?",
                (time.time(), self.session_key),
            )
        )

    @property
    def task_id(self) -> str | None:
        return self.task_state.task_id if self.task_state is not None else None

    def persist_events(self, events: list[AgentEvent]) -> int:
        if not events:
            return 0
        task_id = self.task_id

        def run(c: Any) -> int:
            n = 0
            for ev in events:
                self._seq += 1
                row = ev.to_row()
                cur = c.execute(
                    "INSERT OR IGNORE INTO events(event_id, session_key, task_id, event_type, seq, "
                    "ts, tool_name, operation, resource_ids, path_refs, command_fingerprint, "
                    "exit_code, success, content_hash, metadata, source_ref) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        row["event_id"],
                        row["session_key"],
                        row["task_id"] or task_id,
                        row["event_type"],
                        self._seq,
                        row["ts"],
                        row["tool_name"],
                        row["operation"],
                        dumps(row["resource_ids"]),
                        dumps(row["path_refs"]),
                        row["command_fingerprint"],
                        row["exit_code"],
                        None if row["success"] is None else int(bool(row["success"])),
                        row["content_hash"],
                        dumps(row["metadata"]),
                        row["source_ref"],
                    ),
                )
                n += cur.rowcount or 0
            return n

        return int(self.store.write(run, default=0) or 0)

    def known_event_ids(self, ids: list[str]) -> set[str]:
        known: set[str] = set()
        for start in range(0, len(ids), 500):
            chunk = ids[start : start + 500]
            if not chunk:
                continue
            marks = ",".join("?" * len(chunk))
            rows = self.store.query(
                f"SELECT event_id FROM events WHERE event_id IN ({marks})",  # noqa: S608
                tuple(chunk),
            )
            known.update(r["event_id"] for r in rows)
        return known

    # -------------------------------------------------------------- ingest
    def ingest_history(
        self, items: list[HistoryItem], *, cwd: str, history_tokens: int = 0
    ) -> list[AgentEvent]:
        """Normalize ``items`` and dispatch the events not seen before, in order."""
        started = time.perf_counter()
        with self.lock:
            if self.task_state is not None:
                self.task_state.refresh()
            if self.scope is not None:
                self.scope.refresh()
            ids, start = self._ids_for(items)
            known = self.known_event_ids(ids[start:])
            ctx = NormalizeContext(
                workspace_id=self.workspace.workspace_id,
                session_key=self.session_key,
                project_root=self.workspace.root,
                cwd=cwd or self.workspace.root,
                now=time.time(),
            )
            fresh: list[AgentEvent] = []
            for item, eid in zip(items[start:], ids[start:]):
                if eid in known:
                    continue
                fresh.extend(events_for_item(item, eid, ctx))
            self.history_tokens = history_tokens or self.history_tokens
            if fresh:
                self.dispatch(fresh)
                self.persist_events(fresh)
            self.save_session()
        self.metrics.observe("ingest", (time.perf_counter() - started) * 1000.0)
        return fresh

    def _ids_for(self, items: list[HistoryItem]) -> tuple[list[str], int]:
        """Ids for every item, plus the index from which ids may be unseen.

        An append-only history (the common case) reuses the ids of the prefix
        identified last time and checks only the new suffix against the store.
        """
        if self._boundary is not None:
            n, h, seen = self._boundary
            if 0 < n <= len(items) and _item_hash(items[n - 1]) == h:
                seen = Counter(seen)
                ids = self._known_ids[:n] + item_event_ids(items[n:], self.session_key, seen)
                self._remember(items, ids, seen)
                return ids, n
        seen = Counter()
        ids = item_event_ids(items, self.session_key, seen)
        self._remember(items, ids, seen)
        return ids, 0

    def _remember(self, items: list[HistoryItem], ids: list[str], seen: Counter) -> None:
        self._known_ids = ids
        self._boundary = (len(items), _item_hash(items[-1]), seen) if items else None

    def dispatch(self, events: list[AgentEvent]) -> None:
        """Run consumers in the normative order of plan §4.7."""
        for ev in events:
            try:
                self._dispatch_one(ev)
            except Exception:  # noqa: BLE001 - one bad event never stops the rest
                logger.debug("agent-state consumer failed for %s", ev.event_type, exc_info=True)
                self.metrics.bump("consumer_errors")

    def _dispatch_one(self, ev: AgentEvent) -> None:
        t = ev.event_type
        if t is EventType.USER_MESSAGE:
            self.user_turns += 1
            if self.evidence is not None:
                self.evidence.on_user_assertion(ev)
            if self.task_state is not None:
                self.task_state.on_user_message(ev)
            if self.scope is not None:
                self.scope.on_task_update()
            if self.workflows is not None:
                self.workflows.on_user_message(ev)
        elif t is EventType.ASSISTANT_MESSAGE:
            if self.evidence is not None:
                self.evidence.on_assistant_message(ev)
            if self.task_state is not None:
                self.task_state.on_assistant_message(ev)
            if self.workflows is not None:
                self.workflows.on_assistant_message(ev)
        elif t is EventType.TOOL_CALL_PROPOSED:
            self.check_proposed(ev, source="history")
        elif t in (EventType.TOOL_CALL_FINISHED, EventType.TOOL_CALL_FAILED):
            self.last_tool_family = str(ev.operation or "")
            records = self.evidence.extract(ev) if self.evidence is not None else []
            if self.task_state is not None:
                self.task_state.on_tool_result(ev, records)
            if self.scope is not None:
                self.scope.on_tool_result(ev, records)
            if self.contracts is not None:
                self.contracts.on_tool_result(ev, records)
            if self.test_impact is not None:
                self.test_impact.on_tool_result(ev, records)
            if self.workflows is not None:
                self.workflows.on_tool_result(ev, records)
            self.service.check_block_honored(self, ev)

    def check_proposed(self, ev: AgentEvent, *, source: str) -> Any:
        """Validator then scope (plan §4.7 steps 2-3). Returns the validation outcome."""
        if self.contracts is not None:
            return self.contracts.validate(ev, source=source)
        if self.scope is None:
            return None
        from .contracts import _SEVERITY, RuleResult, ValidationOutcome, decide_enforcement

        inv = ev.transient.get("invocation")
        if inv is None or not (inv.paths_written or inv.paths_deleted):
            return None
        findings, cls = self.scope.classify_proposed(ev, source=source)
        result = max(
            (f.result for f in findings), key=lambda r: _SEVERITY[r], default=RuleResult.PASS
        )
        outcome = ValidationOutcome(result, findings, "allowed", "scope", scope_class=cls)
        decide_enforcement(self, outcome, source=source)
        return outcome

    # ----------------------------------------------------------- warnings
    def warn(self, key: str, text: str, *, priority: int = 0) -> None:
        with self.lock:
            if any(k == key for _, k, _ in self.pending_warnings):
                return
            self.pending_warnings.append((priority, key, text))
            if len(self.pending_warnings) > 16:
                self.pending_warnings.sort()
                del self.pending_warnings[16:]

    def take_warnings(self) -> list[str]:
        with self.lock:
            out = [t for _, _, t in sorted(self.pending_warnings)]
            self.pending_warnings.clear()
            return out

    # ------------------------------------------------------- collaborators
    def advisor(self) -> Any | None:
        return self.service.advisor()

    def graph(self) -> Any | None:
        return self.service.graph_for(self.workspace)

    def enforcement(self, feature: str) -> EnforcementMode:
        return self.config.scope_mode if feature == "scope" else self.config.contract_mode

    def can_block(self) -> bool:
        return bool(self.capabilities.can_block_before_execution)

    def block_unverified(self) -> bool:
        return bool(self.capabilities.block_unverified) and not self.capabilities.downgraded

    # ----------------------------------------------------------- rendering
    def render(self) -> tuple[str, str] | None:
        """``(block, material_hash)`` for the live turn, or None when not warranted."""
        from .serialization import render_agent_state

        return render_agent_state(self)

    def render_state(self) -> Any:
        """The composed state (:class:`~.serialization.ComposedState`) or None."""
        from .serialization import render_state

        return render_state(self)


def _item_hash(item: HistoryItem) -> str:
    import hashlib

    raw = f"{item.kind}|{item.call_id}|{item.name}|{item.text}|{item.interrupted}"
    return hashlib.sha256(raw.encode("utf-8", "surrogatepass")).hexdigest()[:24]


# ------------------------------------------------------------------ service
@dataclass
class RequestState:
    """Per-request handle between the provider handler and the service."""

    runtime: AgentStateRuntime
    block: str | None  # the full block
    state_hash: str
    changed: bool = True  # material state differs from the last committed insertion
    composed: Any = None  # serialization.ComposedState behind ``block``
    pending: PendingInjection | None = None
    labels: list[str] = field(default_factory=list)


class AgentStateService:
    """Proxy-level composition root for the agent-state layer."""

    def __init__(
        self,
        config: AgentStateConfig,
        *,
        intelligence: Any | None = None,
        state_dir: Path | None = None,
    ) -> None:
        self.config = config
        self.intelligence = intelligence
        self.state_dir = Path(state_dir) if state_dir else config.state_path()
        self.metrics = AgentStateMetrics()
        self.injector = StickyInjector()
        self._runtimes: OrderedDict[tuple[str, str], AgentStateRuntime] = OrderedDict()
        self._by_agent_session: dict[tuple[str, str], list[str]] = {}
        self._workspaces: dict[str, Workspace] = {}
        self._lock = threading.RLock()
        self._blocked_calls: dict[str, tuple[str, float]] = {}
        self._caps: dict[tuple[str, str], Capabilities] = {}
        # lineage key -> compact task-state terms for the Phase 1 TaskContext.
        self._views: OrderedDict[str, tuple[str, ...]] = OrderedDict()

    def capabilities_for(self, workspace_id: str, agent_session: str, agent: str) -> Capabilities:
        with self._lock:
            caps = self._caps.get((workspace_id, agent_session))
            if caps is None:
                caps = Capabilities(agent=agent)
                self._caps[(workspace_id, agent_session)] = caps
                if len(self._caps) > 4 * MAX_RUNTIMES:
                    self._caps.pop(next(iter(self._caps)))
            return caps

    # ------------------------------------------------------------ helpers
    def advisor(self) -> Any | None:
        runtime = self.intelligence
        if runtime is None:
            return None
        try:
            return runtime.advisor_or_none()
        except Exception:  # noqa: BLE001
            return None

    def graph_for(self, workspace: Workspace) -> Any | None:
        runtime = self.intelligence
        graphs = getattr(runtime, "graphs", None) if runtime is not None else None
        if graphs is None:
            return None
        try:
            return graphs.for_workspace(workspace.root or "_default")
        except Exception:  # noqa: BLE001
            return None

    def mode_allows(self, mode: FeatureMode, agent: str, has_tools: bool) -> bool:
        if mode is FeatureMode.OFF:
            return False
        if mode is FeatureMode.ON:
            return has_tools or bool(agent)
        return bool(agent) and has_tools

    def session_enabled(self, agent: str, has_tools: bool) -> bool:
        cfg = self.config
        return any(
            self.mode_allows(m, agent, has_tools)
            for m in (
                cfg.task_state,
                cfg.evidence,
                cfg.contracts,
                cfg.scope,
                cfg.test_impact,
                cfg.workflows,
            )
        )

    def workspace_for(self, cwd: str) -> Workspace | None:
        if not cwd:
            return None
        root = find_project_root(cwd) if os.path.isdir(cwd) else cwd
        wid = workspace_id(root)
        with self._lock:
            ws = self._workspaces.get(wid)
            if ws is not None:
                return ws
            store = get_store(self.state_dir, wid, root_display=root)
            if not store.available:
                self.metrics.bump("store_unavailable")
                logger.info(
                    "agent-state persistence disabled for %s: %s", wid, store.disabled_reason
                )
                return None
            ws = Workspace(wid, root, os.path.isdir(root), store)
            self._workspaces[wid] = ws
            return ws

    def runtime_for(
        self,
        workspace: Workspace,
        *,
        agent: str,
        agent_session: str,
        lineage: str,
    ) -> AgentStateRuntime:
        session_key = f"{agent or 'client'}:{agent_session or 'anon'}:{lineage}"
        key = (workspace.workspace_id, session_key)
        with self._lock:
            rt = self._runtimes.get(key)
            if rt is not None:
                self._runtimes.move_to_end(key)
                return rt
            rt = AgentStateRuntime(
                self,
                workspace,
                session_key=session_key,
                agent=agent,
                agent_session=agent_session,
                lineage=lineage,
            )
            self._runtimes[key] = rt
            lineages = self._by_agent_session.setdefault(
                (workspace.workspace_id, agent_session), []
            )
            if session_key not in lineages:
                lineages.append(session_key)
            while len(self._runtimes) > MAX_RUNTIMES:
                _, old = self._runtimes.popitem(last=False)
                try:
                    old.save_session()
                except Exception:  # noqa: BLE001
                    pass
            return rt

    def _resolve_lineage(
        self, workspace: Workspace, agent_session: str, items: list[HistoryItem], payload: Any
    ) -> str:
        root = conversation_root(items)
        if not root or is_continuation_summary(root):
            # Compaction summary / incremental continuation: keep the latest lineage
            # of this agent session so task state survives /compact.
            prior = self._by_agent_session.get((workspace.workspace_id, agent_session)) or []
            if prior:
                return prior[-1].rsplit(":", 1)[-1]
            row = workspace.store.query_one(
                "SELECT lineage FROM sessions WHERE agent_session = ? ORDER BY last_seen DESC LIMIT 1",
                (agent_session,),
            )
            if row is not None and row["lineage"]:
                return str(row["lineage"])
            return "main"
        return lineage_key(root)

    # ------------------------------------------------------- request paths
    def begin_anthropic(
        self,
        body: dict[str, Any],
        headers: Any,
        client_messages: list[dict[str, Any]],
        *,
        cwd: str,
    ) -> RequestState | None:
        """Ingest a Claude Code (or other Anthropic-wire) request; fail-open."""
        started = time.perf_counter()
        try:
            from .ids import detect_agent

            agent = detect_agent(headers, body, "anthropic")
            has_tools = bool(body.get("tools"))
            if not self.session_enabled(agent, has_tools):
                return None
            items = history_items(client_messages)
            if not items:
                return None
            workspace = self.workspace_for(cwd)
            if workspace is None:
                return None
            agent_session = claude_session_id(body, headers) or (
                "root-" + lineage_key(conversation_root(items))
            )
            lineage = self._resolve_lineage(workspace, agent_session, items, body)
            rt = self.runtime_for(
                workspace, agent=agent, agent_session=agent_session, lineage=lineage
            )
            self._note_tools(rt, body.get("tools"))
            tokens = _approx_history_tokens(client_messages)
            rt.ingest_history(items, cwd=cwd, history_tokens=tokens)
            return self._finish(rt, client_messages)
        except Exception:  # noqa: BLE001 - never break a request
            logger.debug("agent-state anthropic begin failed", exc_info=True)
            self.metrics.bump("errors")
            return None
        finally:
            self.metrics.observe("request", (time.perf_counter() - started) * 1000.0)

    def begin_responses(
        self,
        payload: dict[str, Any],
        *,
        client: str,
        headers: Any = None,
        cwd: str = "",
    ) -> RequestState | None:
        """Ingest a Codex (Responses) request; fail-open."""
        started = time.perf_counter()
        try:
            from headroom.intelligence.messages import responses_items_to_messages
            from headroom.intelligence.responses import codex_workspace

            agent = "codex" if (client or "").lower() == "codex" else ""
            if not agent:
                from .ids import detect_agent

                agent = detect_agent(headers, payload, "openai")
            has_tools = bool(payload.get("tools"))
            if not self.session_enabled(agent, has_tools):
                return None
            raw_items = payload.get("input")
            if not isinstance(raw_items, list):
                return None
            messages = responses_items_to_messages(raw_items)
            items = history_items(messages)
            if not items:
                return None
            cwd = cwd or codex_workspace(raw_items)
            workspace = self.workspace_for(cwd)
            if workspace is None:
                return None
            agent_session = codex_session_id(payload, headers) or (
                "root-" + lineage_key(conversation_root(items))
            )
            if payload.get("previous_response_id"):
                lineage = self._resolve_lineage(workspace, agent_session, [], payload)
            else:
                lineage = self._resolve_lineage(workspace, agent_session, items, payload)
            rt = self.runtime_for(
                workspace, agent=agent, agent_session=agent_session, lineage=lineage
            )
            self._note_tools(rt, payload.get("tools"))
            rt.ingest_history(items, cwd=cwd, history_tokens=_approx_history_tokens(messages))
            return self._finish(rt, raw_items)
        except Exception:  # noqa: BLE001
            logger.debug("agent-state responses begin failed", exc_info=True)
            self.metrics.bump("errors")
            return None
        finally:
            self.metrics.observe("request", (time.perf_counter() - started) * 1000.0)

    def _note_tools(self, rt: AgentStateRuntime, tools: Any) -> None:
        names: list[str] = []
        for tool in tools or []:
            if isinstance(tool, dict):
                name = tool.get("name") or (tool.get("function") or {}).get("name") or ""
                names.append(str(name))
        schemas: dict[str, Any] = {}
        for tool in tools or []:
            if isinstance(tool, dict):
                name = tool.get("name") or (tool.get("function") or {}).get("name") or ""
                schema = (
                    tool.get("input_schema")
                    or tool.get("parameters")
                    or (tool.get("function") or {}).get("parameters")
                )
                if name and isinstance(schema, dict):
                    schemas[str(name)] = {
                        **schema,
                        "description": str(tool.get("description") or "")[:400],
                    }
        rt.tool_schemas = schemas
        has_headroom_tool = any("headroom_" in n for n in names)
        if has_headroom_tool and not rt.capabilities.can_execute_local_headroom_tool:
            rt.capabilities.can_execute_local_headroom_tool = True

    def _remember_view(self, rt: AgentStateRuntime) -> None:
        """Compact TaskState view for TaskContext (plan §11.2): scope paths and changed files."""
        terms: list[str] = []
        if rt.scope is not None and rt.scope.contract is not None:
            terms.extend(rt.scope.contract.explicit_in_scope_paths[:8])
        if rt.task_id:
            rows = rt.store.query(
                "SELECT path FROM task_changes WHERE task_id = ? AND path != '*' ORDER BY updated_at DESC LIMIT 8",
                (rt.task_id,),
            )
            terms.extend(r["path"] for r in rows)
        with self._lock:
            self._views[rt.lineage] = tuple(dict.fromkeys(terms))[:12]
            self._views.move_to_end(rt.lineage)
            while len(self._views) > 256:
                self._views.popitem(last=False)

    def view_for(self, lineage: str) -> tuple[str, ...]:
        with self._lock:
            return self._views.get(lineage, ())

    def live_state_tokens(self, rt: AgentStateRuntime, history: list[Any] | None) -> int:
        """Tokens of this session's earlier blocks that are present in ``history``."""
        if not history:
            return 0
        from .serialization import tokens

        entries = self.injector.entries(rt.session_key)
        return sum(
            tokens(entries[p].text) for p in self.injector.live_positions(rt.session_key, history)
        )

    def _finish(
        self, rt: AgentStateRuntime, history: list[Any] | None = None
    ) -> RequestState | None:
        try:
            self._remember_view(rt)
        except Exception:  # noqa: BLE001
            logger.debug("task-state view update failed", exc_info=True)
        composed = rt.render_state()
        if composed is None:
            return RequestState(rt, None, self.injector.last_state_hash(rt.session_key))
        block, state_hash = composed.block, composed.state_hash
        last = self.injector.last_state_hash(rt.session_key)
        rt.requests_since_injection += 1
        if composed.has_events:
            changed = True  # a fresh scope warning or macro announcement
        elif state_hash == last:
            changed = False
        elif state_hash.split(":", 1)[0] != last.split(":", 1)[0]:
            changed = True  # goal, constraints, blockers, warnings, decisions...
        else:
            # Progress churn only (an edit/test loop flips criteria back and
            # forth): refresh with hysteresis, or at once on a completion claim.
            ts = rt.task_state
            declared = bool(ts is not None and ts.state is not None and ts.state.declared_complete)
            backoff = 1 + self.live_state_tokens(rt, history) // STATE_HISTORY_SOFT_TOKENS
            changed = declared or rt.requests_since_injection >= MINOR_REFRESH_REQUESTS * backoff
        return RequestState(rt, block, state_hash, changed=changed, composed=composed)

    def _encode(
        self, rs: RequestState, history: list[Any], *, incremental: bool = False
    ) -> tuple[str | None, bool]:
        """``(text, full)``: the full block, or a delta when its base is provably present.

        A delta is used only when the last full block and every delta after it
        are present in this request's history (so nothing it builds on was
        compacted away), the task is unchanged, the chain is short, and the
        delta is clearly smaller. With ``previous_response_id`` the history is
        server-side and cannot be checked, so the block is always full.
        """
        from .serialization import delta_block, parse_digests, tokens

        composed = rs.composed
        if rs.block is None or composed is None or not rs.changed or incremental:
            return rs.block, True
        if not composed.unique_headings:
            return rs.block, True
        key = rs.runtime.session_key
        entries = self.injector.entries(key)
        base = next((k for k in range(len(entries) - 1, -1, -1) if entries[k].full), None)
        if base is None or len(entries) - 1 - base >= DELTA_CHAIN_MAX:
            return rs.block, True
        live = set(self.injector.live_positions(key, history))
        if any(pos not in live for pos in range(base, len(entries))):
            return rs.block, True
        prev = entries[-1]
        if not prev.digests or not prev.revision:
            return rs.block, True
        if parse_digests(prev.digests).get("@task", "") != dict(composed.attrs).get("task", ""):
            return rs.block, True
        delta = delta_block(composed, prev.digests, prev.revision)
        if delta is None:
            return None, False  # nothing the model can see changed
        if tokens(delta) > DELTA_MAX_SHARE * tokens(rs.block):
            return rs.block, True
        return delta, False

    def _memo_meta(self, rs: RequestState, full: bool) -> dict[str, Any]:
        composed = rs.composed
        if composed is None:
            return {}
        return {
            "digests": composed.digests(),
            "full": full,
            "revision": dict(composed.attrs).get("revision", ""),
        }

    def apply_anthropic(
        self,
        rs: RequestState,
        client_messages: list[dict[str, Any]],
        outgoing: list[dict[str, Any]],
    ) -> list[dict[str, Any]] | None:
        try:
            text, full = self._encode(rs, client_messages)
            result = self.injector.apply_anthropic(
                rs.runtime.session_key,
                client_messages,
                outgoing,
                text,
                rs.state_hash,
                only_if_orphaned=not rs.changed,
                **self._memo_meta(rs, full),
            )
        except Exception:  # noqa: BLE001
            logger.debug("agent-state anthropic injection failed", exc_info=True)
            return None
        if result is None:
            return None
        messages, pending = result
        rs.pending = pending
        rs.labels = _labels(pending)
        return messages

    def apply_responses(
        self,
        rs: RequestState,
        client_items: list[Any],
        outgoing: list[Any],
        *,
        incremental: bool,
    ) -> list[Any] | None:
        try:
            text, full = self._encode(rs, client_items, incremental=incremental)
            result = self.injector.apply_responses(
                rs.runtime.session_key,
                client_items,
                outgoing,
                text,
                rs.state_hash,
                incremental=incremental,
                only_if_orphaned=not rs.changed,
                **self._memo_meta(rs, full),
            )
        except Exception:  # noqa: BLE001
            logger.debug("agent-state responses injection failed", exc_info=True)
            return None
        if result is None:
            return None
        items, pending = result
        rs.pending = pending
        rs.labels = _labels(pending)
        return items

    def discard(self, rs: RequestState | None) -> None:
        """The mutated body did not reach the wire: forget a committed insertion."""
        if rs is None or rs.pending is None or rs.pending.entry is None:
            return
        entry = rs.pending.entry
        with self.injector._lock:  # noqa: SLF001 - same module family
            entries = self.injector._memo.get(rs.runtime.session_key, [])  # noqa: SLF001
            if entry in entries:
                entries.remove(entry)
        rs.runtime.save_session()

    def commit(self, rs: RequestState | None) -> None:
        """The mutated body went on the wire: remember the insertion."""
        if rs is None or rs.pending is None:
            return
        try:
            self.injector.commit(rs.pending)
            if rs.pending.entry is not None:
                rs.runtime.requests_since_injection = 0
                tokens = max(1, len(rs.pending.entry.text) // 4)
                self.metrics.bump("injections")
                self.metrics.bump(
                    "injections_full" if rs.pending.entry.full else "injections_delta"
                )
                self.metrics.bump("injected_tokens", tokens)
                rt = rs.runtime
                if rt.task_state is not None:
                    rt.task_state.note_injected(tokens)
            rs.runtime.save_session()
        except Exception:  # noqa: BLE001
            logger.debug("agent-state commit failed", exc_info=True)

    # ------------------------------------------------------------- hooks
    def on_pretool_hook(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Pre-execution validation for a host PreToolUse hook (fail-open: allow)."""
        started = time.perf_counter()
        try:
            from .hooks import handle_pretool

            return handle_pretool(self, payload)
        except Exception:  # noqa: BLE001
            logger.debug("agent-state hook failed", exc_info=True)
            self.metrics.bump("errors")
            return {"decision": "allow"}
        finally:
            self.metrics.observe("hook", (time.perf_counter() - started) * 1000.0)

    def runtimes_for_agent_session(self, workspace: Workspace, agent_session: str) -> list[Any]:
        with self._lock:
            keys = list(self._by_agent_session.get((workspace.workspace_id, agent_session), []))
            return [
                self._runtimes[(workspace.workspace_id, k)]
                for k in keys
                if (workspace.workspace_id, k) in self._runtimes
            ]

    def note_block(self, rt: AgentStateRuntime, call_key: str) -> None:
        with self._lock:
            self._blocked_calls[call_key] = (rt.session_key, time.time())
            if len(self._blocked_calls) > 512:
                for k in list(self._blocked_calls)[:256]:
                    self._blocked_calls.pop(k, None)

    def check_block_honored(self, rt: AgentStateRuntime, ev: AgentEvent) -> None:
        """Downgrade blocking capability if a call we blocked shows up as executed."""
        call_id = str(ev.transient.get("call_id") or "")
        if not call_id:
            return
        with self._lock:
            hit = self._blocked_calls.pop(call_id, None)
        if hit is None:
            return
        text = str(ev.transient.get("text") or "")
        blocked_marker = "headroom" in text.lower() and ("block" in text.lower())
        caps = rt.capabilities
        if ev.success and not blocked_marker:
            caps.can_block_before_execution = False
            caps.block_unverified = False
            caps.downgraded = "a blocked call executed; enforcement not honored"
            self.metrics.bump("capability_downgrades")
        elif caps.block_unverified:
            # The requested block held: the host honors it.
            caps.can_block_before_execution = True
            caps.block_unverified = False
            self.metrics.bump("capability_block_verified")
        rt.save_session()

    # ------------------------------------------------------------ status
    def status(self) -> dict[str, Any]:
        with self._lock:
            sessions = [
                {
                    "session": rt.session_key,
                    "workspace": rt.workspace.workspace_id,
                    "task": rt.task_id,
                    "capabilities": rt.capabilities.to_dict(),
                }
                for rt in list(self._runtimes.values())[-16:]
            ]
        return {
            "config": self.config.to_dict(),
            "metrics": self.metrics.snapshot(),
            "sessions": sessions,
        }

    def shutdown(self) -> None:
        with self._lock:
            runtimes = list(self._runtimes.values())
        for rt in runtimes:
            try:
                rt.end()
                rt.save_session()
            except Exception:  # noqa: BLE001
                pass
        for ws in list(self._workspaces.values()):
            try:
                ws.store.maybe_maintain()
            except Exception:  # noqa: BLE001
                pass


def anthropic_cwd(body: dict[str, Any], headers: Any) -> str:
    """Project cwd for a Claude Code request: ``x-headroom-cwd`` or the system prompt."""
    try:
        header = headers.get("x-headroom-cwd") if headers is not None else None
    except Exception:  # noqa: BLE001
        header = None
    if header:
        return str(header).strip()
    from headroom.memory.storage_router import ProjectResolver, extract_system_prompt

    try:
        return ProjectResolver._extract_cwd_from_system_prompt(extract_system_prompt(body)) or ""  # noqa: SLF001
    except Exception:  # noqa: BLE001
        return ""


def _labels(pending: PendingInjection) -> list[str]:
    labels = []
    if pending.entry is not None:
        kind = "inject" if pending.entry.full else "inject_delta"
        labels.append(f"agent_state:{kind}:{max(1, len(pending.entry.text) // 4)}tok")
    if pending.replayed:
        labels.append(f"agent_state:replay:{pending.replayed}")
    return labels


def _approx_history_tokens(messages: list[dict[str, Any]]) -> int:
    total = 0
    for msg in messages:
        content = msg.get("content") if isinstance(msg, dict) else None
        if isinstance(content, str):
            total += len(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    for key in ("text", "content", "input"):
                        value = block.get(key)
                        if isinstance(value, str):
                            total += len(value)
                        elif value is not None:
                            total += len(str(value))
    return total // 4


# --------------------------------------------------------------- install
_SERVICE: AgentStateService | None = None


def install_service(service: AgentStateService | None) -> None:
    """Make ``service`` visible to the Phase 1 TaskContext builder (one per proxy)."""
    global _SERVICE
    _SERVICE = service


def current_service() -> AgentStateService | None:
    return _SERVICE


def task_state_terms(messages: list[dict[str, Any]]) -> tuple[str, ...]:
    """Exact terms from the conversation's TaskState view (empty when unavailable)."""
    service = _SERVICE
    if service is None or not messages:
        return ()
    from .normalizer import _flatten, clean_user_text

    for msg in messages[:12]:
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        text = clean_user_text(_flatten(msg.get("content")))
        if text:
            return service.view_for(lineage_key(text))
    return ()
