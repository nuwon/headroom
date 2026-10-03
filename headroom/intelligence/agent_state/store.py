"""Workspace-scoped SQLite store for agent state (plan §4.4, §12.3, §13, §16).

One database per workspace:
``<state-dir>/<workspace_id>/agent_state.sqlite3``. The state dir defaults to
``~/.headroom/intelligence/agent_state`` (``%USERPROFILE%`` on Windows) and
``HEADROOM_AGENT_STATE_DIR`` overrides it. The raw project path is never a
filename.

* WAL journal, foreign keys on, ``busy_timeout`` of 5000 ms.
* An explicit ``schema_version`` table. Migrations are forward-only, each in its
  own ``BEGIN IMMEDIATE`` transaction, one version per Phase 2 feature.
* A corrupt database, or one whose migration fails, is renamed to
  ``agent_state.sqlite3.corrupt-<ts>`` (with its WAL/SHM) and a fresh database
  is created. User data is never deleted. A database written by a *newer*
  Headroom is left untouched, and persistence is disabled for the process.
* Every text value is redacted before it is written. Retention runs at most
  once an hour per workspace and never on the hot path for every request.

Every public method fails open. The caller sees ``None``/empty and a debug
log, and the request path continues.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from headroom.redaction import redact_text, redact_value

logger = logging.getLogger(__name__)

DB_FILENAME = "agent_state.sqlite3"
BUSY_TIMEOUT_MS = 5000
MAINTENANCE_INTERVAL_S = 3600.0
DAY = 86400.0


class StoreUnavailable(RuntimeError):
    """Persistence is disabled for this workspace in this process."""


# ---------------------------------------------------------------- migrations
# Version N creates what feature N needs. Never edit a released migration;
# append a new one.
MIGRATIONS: list[tuple[int, str, str]] = [
    (
        1,
        "agent-state runtime: sessions, events, idempotency",
        """
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE sessions (
            session_key TEXT PRIMARY KEY,
            agent TEXT NOT NULL DEFAULT '',
            agent_session TEXT NOT NULL DEFAULT '',
            lineage TEXT NOT NULL DEFAULT '',
            created_at REAL NOT NULL,
            last_seen REAL NOT NULL,
            hook_seen_at REAL,
            capabilities TEXT NOT NULL DEFAULT '{}',
            cursor TEXT NOT NULL DEFAULT '{}',
            injections TEXT NOT NULL DEFAULT '[]',
            ended_at REAL
        );
        CREATE INDEX sessions_last_seen ON sessions(last_seen);
        CREATE TABLE events (
            event_id TEXT PRIMARY KEY,
            session_key TEXT NOT NULL,
            task_id TEXT,
            event_type TEXT NOT NULL,
            seq INTEGER NOT NULL,
            ts REAL NOT NULL,
            tool_name TEXT,
            operation TEXT,
            resource_ids TEXT NOT NULL DEFAULT '[]',
            path_refs TEXT NOT NULL DEFAULT '[]',
            command_fingerprint TEXT,
            exit_code INTEGER,
            success INTEGER,
            content_hash TEXT,
            metadata TEXT NOT NULL DEFAULT '{}',
            source_ref TEXT
        );
        CREATE INDEX events_session_seq ON events(session_key, seq);
        CREATE INDEX events_type_ts ON events(event_type, ts);
        CREATE TABLE processed (
            consumer TEXT NOT NULL,
            event_id TEXT NOT NULL,
            ts REAL NOT NULL,
            PRIMARY KEY (consumer, event_id)
        );
        """,
    ),
    (
        2,
        "feature 16: task state",
        """
        CREATE TABLE tasks (
            task_id TEXT PRIMARY KEY,
            session_key TEXT NOT NULL,
            revision INTEGER NOT NULL,
            status TEXT NOT NULL,
            state_json TEXT NOT NULL,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        );
        CREATE INDEX tasks_session ON tasks(session_key, updated_at);
        CREATE TABLE task_transitions (
            task_id TEXT NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
            revision INTEGER NOT NULL,
            transition_id TEXT NOT NULL,
            transition_json TEXT NOT NULL,
            created_at REAL NOT NULL,
            PRIMARY KEY (task_id, revision)
        );
        CREATE UNIQUE INDEX task_transitions_id ON task_transitions(task_id, transition_id);
        """,
    ),
    (
        3,
        "feature 17: evidence ledger",
        """
        CREATE TABLE evidence (
            evidence_id TEXT PRIMARY KEY,
            workspace_id TEXT NOT NULL,
            session_key TEXT,
            task_id TEXT,
            claim_key TEXT NOT NULL,
            claim_type TEXT NOT NULL,
            subject TEXT NOT NULL,
            predicate TEXT NOT NULL,
            value_json TEXT NOT NULL,
            display_text TEXT NOT NULL,
            polarity TEXT NOT NULL,
            source_kind TEXT NOT NULL,
            source_event_id TEXT NOT NULL,
            source_resource_ids TEXT NOT NULL DEFAULT '[]',
            source_hash TEXT,
            confidence REAL NOT NULL,
            status TEXT NOT NULL,
            valid_from REAL NOT NULL,
            valid_until REAL,
            supersedes_evidence_id TEXT,
            created_at REAL NOT NULL,
            pinned INTEGER NOT NULL DEFAULT 0,
            metadata TEXT NOT NULL DEFAULT '{}'
        );
        CREATE UNIQUE INDEX evidence_source ON evidence(source_event_id, claim_key);
        CREATE INDEX evidence_claim ON evidence(workspace_id, claim_key, status);
        CREATE INDEX evidence_task ON evidence(task_id, status);
        CREATE INDEX evidence_subject ON evidence(subject, predicate);
        CREATE INDEX evidence_created ON evidence(created_at);
        CREATE TABLE claim_heads (
            claim_key TEXT PRIMARY KEY,
            evidence_id TEXT NOT NULL REFERENCES evidence(evidence_id) ON DELETE CASCADE,
            updated_at REAL NOT NULL
        );
        CREATE TABLE evidence_links (
            src_id TEXT NOT NULL REFERENCES evidence(evidence_id) ON DELETE CASCADE,
            dst_id TEXT NOT NULL REFERENCES evidence(evidence_id) ON DELETE CASCADE,
            relation TEXT NOT NULL,
            created_at REAL NOT NULL,
            PRIMARY KEY (src_id, dst_id, relation)
        );
        CREATE INDEX evidence_links_dst ON evidence_links(dst_id);
        """,
    ),
    (
        4,
        "feature 19: tool contracts",
        """
        CREATE TABLE tool_outcomes (
            event_id TEXT PRIMARY KEY,
            family TEXT NOT NULL,
            executable TEXT NOT NULL DEFAULT '',
            subcommand TEXT NOT NULL DEFAULT '',
            shape TEXT NOT NULL,
            signature TEXT NOT NULL DEFAULT '',
            features TEXT NOT NULL DEFAULT '{}',
            success INTEGER NOT NULL,
            ts REAL NOT NULL
        );
        CREATE INDEX tool_outcomes_key ON tool_outcomes(family, executable, subcommand);
        CREATE TABLE learned_rules (
            rule_id TEXT PRIMARY KEY,
            family TEXT NOT NULL,
            executable TEXT NOT NULL,
            subcommand TEXT NOT NULL,
            shape TEXT NOT NULL,
            signature TEXT NOT NULL,
            predicate TEXT NOT NULL,
            reason TEXT NOT NULL,
            failures INTEGER NOT NULL,
            created_at REAL NOT NULL,
            last_reinforced REAL NOT NULL,
            last_hit REAL,
            hits INTEGER NOT NULL DEFAULT 0,
            disabled_reason TEXT
        );
        CREATE TABLE validations (
            validation_id TEXT PRIMARY KEY,
            session_key TEXT,
            tool_name TEXT,
            outcome TEXT NOT NULL,
            rule TEXT NOT NULL,
            reason TEXT NOT NULL,
            enforced TEXT NOT NULL,
            source TEXT NOT NULL,
            ts REAL NOT NULL
        );
        CREATE INDEX validations_session ON validations(session_key, ts);
        """,
    ),
    (
        5,
        "feature 23: scope firewall",
        """
        CREATE TABLE change_contracts (
            task_id TEXT PRIMARY KEY,
            revision INTEGER NOT NULL,
            contract_json TEXT NOT NULL,
            updated_at REAL NOT NULL
        );
        CREATE TABLE git_baselines (
            task_id TEXT PRIMARY KEY,
            head TEXT NOT NULL DEFAULT '',
            branch TEXT NOT NULL DEFAULT '',
            dirty_json TEXT NOT NULL DEFAULT '{}',
            untracked_json TEXT NOT NULL DEFAULT '[]',
            created_at REAL NOT NULL
        );
        CREATE TABLE task_changes (
            task_id TEXT NOT NULL,
            path TEXT NOT NULL,
            classification TEXT NOT NULL,
            reason TEXT NOT NULL DEFAULT '',
            first_event_id TEXT NOT NULL,
            last_event_id TEXT NOT NULL,
            lines_changed INTEGER NOT NULL DEFAULT 0,
            created INTEGER NOT NULL DEFAULT 0,
            updated_at REAL NOT NULL,
            PRIMARY KEY (task_id, path)
        );
        CREATE TABLE scope_expansions (
            task_id TEXT NOT NULL,
            path TEXT NOT NULL,
            reason_code TEXT NOT NULL,
            evidence_ids TEXT NOT NULL DEFAULT '[]',
            event_ids TEXT NOT NULL DEFAULT '[]',
            confidence REAL NOT NULL,
            created_at REAL NOT NULL,
            PRIMARY KEY (task_id, path)
        );
        CREATE TABLE scope_warnings (
            task_id TEXT NOT NULL,
            contract_revision INTEGER NOT NULL,
            path TEXT NOT NULL,
            reason TEXT NOT NULL,
            classification TEXT NOT NULL,
            message TEXT NOT NULL,
            delivered INTEGER NOT NULL DEFAULT 0,
            created_at REAL NOT NULL,
            PRIMARY KEY (task_id, contract_revision, path, reason)
        );
        """,
    ),
    (
        6,
        "feature 24: test impact",
        """
        CREATE TABLE test_cases (
            test_id TEXT PRIMARY KEY,
            framework TEXT NOT NULL,
            canonical_name TEXT NOT NULL,
            file_path TEXT NOT NULL DEFAULT '',
            tags TEXT NOT NULL DEFAULT '[]',
            estimated_runtime_ms REAL,
            flake_score REAL NOT NULL DEFAULT 0,
            last_status TEXT,
            last_run_at REAL,
            last_change_hash TEXT,
            history TEXT NOT NULL DEFAULT '[]'
        );
        CREATE INDEX test_cases_file ON test_cases(file_path);
        CREATE TABLE impact_edges (
            source_resource_id TEXT NOT NULL,
            test_id TEXT NOT NULL,
            edge_type TEXT NOT NULL,
            weight REAL NOT NULL,
            evidence_count INTEGER NOT NULL DEFAULT 1,
            last_observed_at REAL NOT NULL,
            PRIMARY KEY (source_resource_id, test_id, edge_type)
        );
        CREATE INDEX impact_edges_test ON impact_edges(test_id);
        CREATE TABLE test_runs (
            run_id TEXT PRIMARY KEY,
            task_id TEXT,
            framework TEXT NOT NULL,
            command_fingerprint TEXT NOT NULL,
            change_set_hash TEXT NOT NULL DEFAULT '',
            changed_resources TEXT NOT NULL DEFAULT '[]',
            passed INTEGER NOT NULL DEFAULT 0,
            failed INTEGER NOT NULL DEFAULT 0,
            skipped INTEGER NOT NULL DEFAULT 0,
            errors INTEGER NOT NULL DEFAULT 0,
            failed_ids TEXT NOT NULL DEFAULT '[]',
            passed_ids TEXT NOT NULL DEFAULT '[]',
            tier INTEGER,
            ts REAL NOT NULL
        );
        CREATE INDEX test_runs_task ON test_runs(task_id, ts);
        CREATE TABLE verification_plans (
            plan_id TEXT PRIMARY KEY,
            task_id TEXT,
            change_set_hash TEXT NOT NULL,
            risk REAL NOT NULL,
            plan_json TEXT NOT NULL,
            created_at REAL NOT NULL
        );
        CREATE INDEX verification_plans_task ON verification_plans(task_id, created_at);
        """,
    ),
    (
        7,
        "feature 20: workflow macros",
        """
        CREATE TABLE workflow_observations (
            signature TEXT NOT NULL,
            first_event_id TEXT NOT NULL,
            session_key TEXT NOT NULL,
            task_id TEXT,
            steps_json TEXT NOT NULL,
            values_json TEXT NOT NULL DEFAULT '[]',
            success INTEGER NOT NULL,
            turns INTEGER NOT NULL,
            corrected INTEGER NOT NULL DEFAULT 0,
            violated INTEGER NOT NULL DEFAULT 0,
            event_ids TEXT NOT NULL DEFAULT '[]',
            ts REAL NOT NULL,
            PRIMARY KEY (signature, first_event_id)
        );
        CREATE INDEX workflow_observations_sig ON workflow_observations(signature, ts);
        CREATE TABLE workflow_macros (
            macro_id TEXT PRIMARY KEY,
            workspace_id TEXT NOT NULL,
            name TEXT NOT NULL UNIQUE,
            description TEXT NOT NULL,
            version INTEGER NOT NULL,
            safety_class TEXT NOT NULL,
            origin TEXT NOT NULL,
            macro_json TEXT NOT NULL,
            support_count INTEGER NOT NULL DEFAULT 0,
            success_rate REAL NOT NULL DEFAULT 1.0,
            turns_saved REAL NOT NULL DEFAULT 0,
            tokens_saved REAL NOT NULL DEFAULT 0,
            config_hash TEXT NOT NULL DEFAULT '',
            created_at REAL NOT NULL,
            last_used_at REAL,
            disabled_reason TEXT
        );
        CREATE TABLE workflow_runs (
            run_id TEXT PRIMARY KEY,
            macro_id TEXT NOT NULL REFERENCES workflow_macros(macro_id) ON DELETE CASCADE,
            session_key TEXT,
            status TEXT NOT NULL,
            steps_done INTEGER NOT NULL,
            steps_total INTEGER NOT NULL,
            duration_ms REAL NOT NULL,
            failed_step TEXT,
            ts REAL NOT NULL
        );
        CREATE INDEX workflow_runs_macro ON workflow_runs(macro_id, ts);
        """,
    ),
]
SCHEMA_VERSION = MIGRATIONS[-1][0]


def dumps(value: Any) -> str:
    """Redacted, deterministic JSON."""
    return json.dumps(redact_value(value), sort_keys=True, ensure_ascii=False, default=str)


def loads(raw: Any, default: Any = None) -> Any:
    if raw is None or raw == "":
        return default
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return default


def red(text: Any, limit: int = 1024) -> str:
    return redact_text(str(text or ""))[:limit]


class AgentStateStore:
    """Thread-safe handle on one workspace database (fail-open)."""

    def __init__(self, path: Path, *, workspace_id: str, root_display: str = "") -> None:
        self.path = Path(path)
        self.workspace_id = workspace_id
        self.root_display = root_display
        self._lock = threading.RLock()
        self._conn: sqlite3.Connection | None = None
        self.disabled_reason = ""
        self.quarantined: list[str] = []
        self._last_maintenance = 0.0
        self._open()

    # ---------------------------------------------------------------- opening
    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(
            str(self.path),
            timeout=BUSY_TIMEOUT_MS / 1000.0,
            isolation_level=None,
            check_same_thread=False,
        )
        conn.row_factory = sqlite3.Row
        conn.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _open(self) -> None:
        try:
            conn = self._connect()
            check = conn.execute("PRAGMA quick_check").fetchone()
            if check is None or str(check[0]).lower() != "ok":
                conn.close()
                raise sqlite3.DatabaseError(f"quick_check failed: {check[0] if check else '?'}")
            self._conn = conn
            self._migrate()
        except StoreUnavailable as exc:
            self.disabled_reason = str(exc)
            self._close_quietly()
        except (sqlite3.DatabaseError, OSError) as exc:
            logger.warning("agent-state database unusable (%s); quarantining", exc)
            self._close_quietly()
            if isinstance(exc, OSError) and not self.path.exists():
                self.disabled_reason = f"unavailable: {exc}"
                return
            try:
                self._quarantine()
                self._conn = self._connect()
                self._migrate()
            except (sqlite3.DatabaseError, OSError, StoreUnavailable) as exc2:
                self.disabled_reason = f"unavailable after quarantine: {exc2}"
                self._close_quietly()

    def _close_quietly(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except sqlite3.Error:
                pass
        self._conn = None

    def _quarantine(self) -> None:
        stamp = time.strftime("%Y%m%dT%H%M%S")
        for suffix in ("", "-wal", "-shm"):
            src = Path(str(self.path) + suffix)
            if src.exists():
                dst = Path(f"{self.path}.corrupt-{stamp}{suffix}")
                n = 1
                while dst.exists():
                    dst = Path(f"{self.path}.corrupt-{stamp}-{n}{suffix}")
                    n += 1
                os.replace(src, dst)
                self.quarantined.append(str(dst))

    def _migrate(self) -> None:
        conn = self._conn
        assert conn is not None
        conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_version ("
            "version INTEGER PRIMARY KEY, description TEXT NOT NULL, applied_at REAL NOT NULL)"
        )
        row = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
        current = int(row[0] or 0)
        if current > SCHEMA_VERSION:
            raise StoreUnavailable(
                f"database schema v{current} is newer than this Headroom (v{SCHEMA_VERSION})"
            )
        for version, description, ddl in MIGRATIONS:
            if version <= current:
                continue
            conn.execute("BEGIN IMMEDIATE")
            try:
                again = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
                if int(again[0] or 0) >= version:  # another process migrated meanwhile
                    conn.execute("COMMIT")
                    continue
                for statement in [s.strip() for s in ddl.split(";") if s.strip()]:
                    conn.execute(statement)
                conn.execute(
                    "INSERT INTO schema_version(version, description, applied_at) VALUES (?,?,?)",
                    (version, description, time.time()),
                )
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
        with self.tx() as c:
            c.execute(
                "INSERT OR IGNORE INTO meta(key, value) VALUES ('workspace_id', ?)",
                (self.workspace_id,),
            )
            if self.root_display:
                c.execute(
                    "INSERT OR REPLACE INTO meta(key, value) VALUES ('root_display', ?)",
                    (self.root_display,),
                )

    # ----------------------------------------------------------------- access
    @property
    def available(self) -> bool:
        return self._conn is not None

    def schema_version(self) -> int:
        row = self.query_one("SELECT MAX(version) AS v FROM schema_version")
        return int(row["v"]) if row and row["v"] is not None else 0

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        """Atomic write transaction; raises :class:`StoreUnavailable` when disabled."""
        with self._lock:
            conn = self._conn
            if conn is None:
                raise StoreUnavailable(self.disabled_reason or "store closed")
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            else:
                conn.execute("COMMIT")

    def write(self, fn: Callable[[sqlite3.Connection], Any], default: Any = None) -> Any:
        """Run ``fn`` in a transaction; fail open with ``default``."""
        try:
            with self.tx() as conn:
                return fn(conn)
        except StoreUnavailable:
            return default
        except sqlite3.Error:
            logger.debug("agent-state write failed", exc_info=True)
            return default

    def query(self, sql: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        with self._lock:
            conn = self._conn
            if conn is None:
                return []
            try:
                return list(conn.execute(sql, params).fetchall())
            except sqlite3.Error:
                logger.debug("agent-state query failed", exc_info=True)
                return []

    def query_one(self, sql: str, params: tuple[Any, ...] = ()) -> sqlite3.Row | None:
        rows = self.query(sql, params)
        return rows[0] if rows else None

    def close(self) -> None:
        with self._lock:
            self._close_quietly()

    # ------------------------------------------------------------ maintenance
    def maybe_maintain(self, *, now: float | None = None, force: bool = False) -> bool:
        """Retention cleanup, at most once an hour per workspace (plan §12.3, §13.1)."""
        now = time.time() if now is None else now
        if not self.available:
            return False
        if not force and now - self._last_maintenance < MAINTENANCE_INTERVAL_S:
            return False
        row = self.query_one("SELECT value FROM meta WHERE key='last_maintenance'")
        if not force and row is not None:
            try:
                if now - float(row["value"]) < MAINTENANCE_INTERVAL_S:
                    self._last_maintenance = float(row["value"])
                    return False
            except ValueError:
                pass
        self._last_maintenance = now

        def run(c: sqlite3.Connection) -> None:
            month = now - 30 * DAY
            quarter = now - 90 * DAY
            c.execute("DELETE FROM events WHERE ts < ?", (month,))
            c.execute("DELETE FROM processed WHERE ts < ?", (month,))
            c.execute("DELETE FROM sessions WHERE last_seen < ?", (quarter,))
            c.execute(
                "DELETE FROM tasks WHERE status IN ('COMPLETED','ABANDONED') AND updated_at < ?",
                (month,),
            )
            c.execute("DELETE FROM tasks WHERE updated_at < ?", (quarter,))
            # Durable facts: 30 days unless pinned by a learned rule/macro.
            # Volatile facts: their TTL plus 7 days of history.
            c.execute(
                "DELETE FROM evidence WHERE pinned = 0 AND ("
                " (valid_until IS NULL AND created_at < ?)"
                " OR (valid_until IS NOT NULL AND valid_until < ?))",
                (month, now - 7 * DAY),
            )
            c.execute(
                "DELETE FROM claim_heads WHERE evidence_id NOT IN (SELECT evidence_id FROM evidence)"
            )
            c.execute("DELETE FROM tool_outcomes WHERE ts < ?", (month,))
            c.execute("DELETE FROM learned_rules WHERE last_reinforced < ?", (month,))
            c.execute("DELETE FROM validations WHERE ts < ?", (month,))
            for task_table in (
                "change_contracts",
                "git_baselines",
                "task_changes",
                "scope_expansions",
                "scope_warnings",
            ):
                c.execute(
                    f"DELETE FROM {task_table} WHERE task_id NOT IN (SELECT task_id FROM tasks)"  # noqa: S608
                )
            c.execute("DELETE FROM impact_edges WHERE last_observed_at < ?", (quarter,))
            c.execute("DELETE FROM test_runs WHERE ts < ?", (quarter,))
            c.execute("DELETE FROM verification_plans WHERE created_at < ?", (month,))
            c.execute("DELETE FROM workflow_observations WHERE ts < ?", (month,))
            c.execute(
                "DELETE FROM workflow_macros WHERE origin = 'learned' AND "
                "COALESCE(last_used_at, created_at) < ?",
                (quarter,),
            )
            c.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES ('last_maintenance', ?)",
                (str(now),),
            )

        self.write(run)
        return True


# ------------------------------------------------------------------ registry
_STORES: dict[str, AgentStateStore] = {}
_STORES_LOCK = threading.Lock()


def store_path(state_dir: Path, workspace_id: str) -> Path:
    return Path(state_dir) / workspace_id / DB_FILENAME


def get_store(state_dir: Path, workspace_id: str, *, root_display: str = "") -> AgentStateStore:
    """Process-wide store per workspace database (opened lazily, once)."""
    path = store_path(state_dir, workspace_id)
    key = str(path)
    with _STORES_LOCK:
        store = _STORES.get(key)
        if store is None or (not store.available and not store.disabled_reason):
            store = AgentStateStore(path, workspace_id=workspace_id, root_display=root_display)
            _STORES[key] = store
        return store


def close_all_stores() -> None:
    with _STORES_LOCK:
        for store in _STORES.values():
            store.close()
        _STORES.clear()
