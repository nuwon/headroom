"""Feature 17: Evidence Ledger with Provenance (plan §6).

The ledger holds atomic, source-linked observations, so later decisions can
use facts instead of raw logs or remembered prose. Each record says *who*
observed *what* (``source_kind``, ``source_event_id``, ``source_hash``), how
far it is trusted (``confidence``), and whether it is still current (``status``
plus ``valid_until``).

* **Supersession.** A newer observation of a single-valued ``claim_key`` from
  equal or higher authority replaces the head, and the old record becomes
  ``SUPERSEDED``. History is never deleted.
* **Contradiction.** A disagreement between *different* authorities keeps both
  records, marks the lower-authority one ``CONTRADICTED`` and links the pair
  with ``CONTRADICTS``. An unsupported agent assertion never replaces tool
  evidence.
* **Staleness.** Volatile claims expire (ports and free memory after 30 s,
  tool versions after 24 h). File hashes, git HEAD and working-tree status go
  stale on the next mutation event. User constraints never expire with time.

This is not CCR, RAG or semantic memory. Records hold short facts; raw outputs
stay with CCR. Text is redacted before it is persisted.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

from .events import AgentEvent
from .families import Family, ToolInvocation
from .ids import stable_id
from .results import (
    TestResult,
    failure_signature,
    first_error_line,
    parse_diagnostics,
    parse_test_output,
    tool_use_error,
)
from .store import dumps, loads, red

logger = logging.getLogger(__name__)

# Confidence defaults (plan §6.5).
CONF_TOOL_EXIT = 0.99
CONF_FILESYSTEM = 0.99
CONF_TEST = 0.99
CONF_BUILD = 0.98
CONF_GIT = 0.99
CONF_USER_INTENT = 0.95
CONF_USER_FACT = 0.75
CONF_AGENT = 0.55

AUTHORITY = {
    "TOOL": 3,
    "FILESYSTEM": 3,
    "TEST": 3,
    "BUILD": 3,
    "GIT": 3,
    "DERIVED": 2,
    "USER": 2,
    "AGENT": 1,
}
# Volatile TTLs in seconds (plan §6.8). None = until a mutation event.
TTL = {
    "port_state": 30.0,
    "free_memory": 30.0,
    "tool_version": 24 * 3600.0,
}
_MUTATION_STALES = ("file_hash", "file_exists")
_GIT_HEAD_CLAIMS = ("git_head", "git_dirty", "git_branch", "git_diffstat")
MAX_HASH_BYTES = 2 * 1024 * 1024


@dataclass(frozen=True)
class EvidenceRecord:
    evidence_id: str
    workspace_id: str
    claim_key: str
    claim_type: str
    subject: str
    predicate: str
    value: Any
    display_text: str
    polarity: str
    source_kind: str
    source_event_id: str
    confidence: float
    status: str = "ACTIVE"
    session_id: str | None = None
    task_id: str | None = None
    source_resource_ids: tuple[str, ...] = ()
    source_hash: str | None = None
    valid_from: float = 0.0
    valid_until: float | None = None
    supersedes_evidence_id: str | None = None
    created_at: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def authority(self) -> int:
        return AUTHORITY.get(self.source_kind, 1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "claim_key": self.claim_key,
            "claim_type": self.claim_type,
            "subject": self.subject,
            "predicate": self.predicate,
            "value": self.value,
            "display_text": self.display_text,
            "polarity": self.polarity,
            "source_kind": self.source_kind,
            "source_event_id": self.source_event_id,
            "confidence": self.confidence,
            "status": self.status,
            "task_id": self.task_id,
            "valid_until": self.valid_until,
            "supersedes_evidence_id": self.supersedes_evidence_id,
            "created_at": self.created_at,
        }


def _row_to_record(row: Any) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=row["evidence_id"],
        workspace_id=row["workspace_id"],
        claim_key=row["claim_key"],
        claim_type=row["claim_type"],
        subject=row["subject"],
        predicate=row["predicate"],
        value=loads(row["value_json"]),
        display_text=row["display_text"],
        polarity=row["polarity"],
        source_kind=row["source_kind"],
        source_event_id=row["source_event_id"],
        confidence=float(row["confidence"]),
        status=row["status"],
        session_id=row["session_key"],
        task_id=row["task_id"],
        source_resource_ids=tuple(loads(row["source_resource_ids"], []) or ()),
        source_hash=row["source_hash"],
        valid_from=float(row["valid_from"]),
        valid_until=row["valid_until"],
        supersedes_evidence_id=row["supersedes_evidence_id"],
        created_at=float(row["created_at"]),
        metadata=loads(row["metadata"], {}) or {},
    )


def _values_agree(a: Any, b: Any) -> bool:
    if isinstance(a, dict) and isinstance(b, dict):
        if "ok" in a and "ok" in b:
            return bool(a["ok"]) == bool(b["ok"])
        if "state" in a and "state" in b:
            return bool(a["state"] == b["state"])
        return bool(a == b)
    return bool(a == b)


class EvidenceLedger:
    """Ledger operations for one session (records are workspace-scoped)."""

    def __init__(self, runtime: Any) -> None:
        self.rt = runtime
        self.store = runtime.store
        self.workspace_id = runtime.workspace.workspace_id
        self.created = 0
        self.superseded = 0
        self.conflicts = 0
        self.lookup_hits = 0

    # ------------------------------------------------------------- writing
    def make(
        self,
        ev: AgentEvent | None,
        *,
        claim_type: str,
        subject: str,
        predicate: str,
        value: Any,
        display: str,
        source_kind: str,
        confidence: float,
        polarity: str = "NEUTRAL",
        key_suffix: str = "",
        valid_until: float | None = None,
        source_event_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> EvidenceRecord:
        subject = red(subject, 300)
        claim_key = f"{claim_type}|{subject}" + (f"|{key_suffix}" if key_suffix else "")
        sei = source_event_id or (ev.event_id if ev is not None else "")
        now = time.time()
        if valid_until is None and claim_type in TTL:
            valid_until = now + TTL[claim_type]
        return EvidenceRecord(
            evidence_id="E" + stable_id(sei, claim_key, n=15),
            workspace_id=self.workspace_id,
            claim_key=claim_key[:400],
            claim_type=claim_type,
            subject=red(subject, 300),
            predicate=predicate,
            value=value,
            display_text=red(display, 300),
            polarity=polarity,
            source_kind=source_kind,
            source_event_id=sei,
            confidence=confidence,
            session_id=self.rt.session_key,
            task_id=self.rt.task_id,
            source_resource_ids=tuple(ev.resource_ids) if ev is not None else (),
            source_hash=ev.content_hash if ev is not None else None,
            valid_from=now,
            valid_until=valid_until,
            created_at=now,
            metadata=metadata or {},
        )

    def record(self, rec: EvidenceRecord) -> EvidenceRecord | None:
        """Insert with supersession/contradiction handling; idempotent per (event, claim)."""

        def run(c: Any) -> EvidenceRecord | None:
            existing = c.execute(
                "SELECT * FROM evidence WHERE source_event_id = ? AND claim_key = ?",
                (rec.source_event_id, rec.claim_key),
            ).fetchone()
            if existing is not None:
                return _row_to_record(existing)
            head_row = c.execute(
                "SELECT e.* FROM claim_heads h JOIN evidence e ON e.evidence_id = h.evidence_id "
                "WHERE h.claim_key = ?",
                (rec.claim_key,),
            ).fetchone()
            head = _row_to_record(head_row) if head_row is not None else None
            status = "ACTIVE"
            supersedes = None
            make_head = True
            link: tuple[str, str, str] | None = None
            if head is not None and head.status == "ACTIVE":
                if head.authority > rec.authority:
                    # Lower authority never replaces a higher one. Disagreement keeps
                    # both and marks the newcomer contradicted; agreement corroborates.
                    make_head = False
                    if _values_agree(head.value, rec.value):
                        link = (rec.evidence_id, head.evidence_id, "SUPPORTS")
                    else:
                        status = "CONTRADICTED"
                        link = (head.evidence_id, rec.evidence_id, "CONTRADICTS")
                elif head.authority < rec.authority and not _values_agree(head.value, rec.value):
                    c.execute(
                        "UPDATE evidence SET status = 'CONTRADICTED' WHERE evidence_id = ?",
                        (head.evidence_id,),
                    )
                    link = (rec.evidence_id, head.evidence_id, "CONTRADICTS")
                else:
                    c.execute(
                        "UPDATE evidence SET status = 'SUPERSEDED' WHERE evidence_id = ?",
                        (head.evidence_id,),
                    )
                    supersedes = head.evidence_id
            elif head is not None and head.status == "STALE":
                supersedes = head.evidence_id
            new = EvidenceRecord(
                **{**rec.__dict__, "status": status, "supersedes_evidence_id": supersedes}
            )
            c.execute(
                "INSERT INTO evidence(evidence_id, workspace_id, session_key, task_id, claim_key, "
                "claim_type, subject, predicate, value_json, display_text, polarity, source_kind, "
                "source_event_id, source_resource_ids, source_hash, confidence, status, valid_from, "
                "valid_until, supersedes_evidence_id, created_at, metadata) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    new.evidence_id,
                    new.workspace_id,
                    new.session_id,
                    new.task_id,
                    new.claim_key,
                    new.claim_type,
                    new.subject,
                    new.predicate,
                    dumps(new.value),
                    new.display_text,
                    new.polarity,
                    new.source_kind,
                    new.source_event_id,
                    dumps(list(new.source_resource_ids)),
                    new.source_hash,
                    new.confidence,
                    new.status,
                    new.valid_from,
                    new.valid_until,
                    new.supersedes_evidence_id,
                    new.created_at,
                    dumps(new.metadata),
                ),
            )
            if make_head:
                c.execute(
                    "INSERT OR REPLACE INTO claim_heads(claim_key, evidence_id, updated_at) VALUES (?,?,?)",
                    (new.claim_key, new.evidence_id, new.created_at),
                )
            if link is not None:
                c.execute(
                    "INSERT OR IGNORE INTO evidence_links(src_id, dst_id, relation, created_at) "
                    "VALUES (?,?,?,?)",
                    (*link, new.created_at),
                )
            return new

        out: EvidenceRecord | None = self.store.write(run)
        if out is not None and out.created_at == rec.created_at:
            self.created += 1
            self.rt.metrics.bump("evidence_records_created")
            if out.supersedes_evidence_id:
                self.superseded += 1
                self.rt.metrics.bump("evidence_claims_superseded")
            if out.status == "CONTRADICTED" or self._has_contradiction(out.evidence_id):
                self.conflicts += 1
                self.rt.metrics.bump("evidence_conflicts_detected")
        return out

    def _has_contradiction(self, evidence_id: str) -> bool:
        row = self.store.query_one(
            "SELECT 1 FROM evidence_links WHERE (src_id = ? OR dst_id = ?) AND relation = 'CONTRADICTS'",
            (evidence_id, evidence_id),
        )
        return row is not None

    def link(self, src: str, dst: str, relation: str) -> None:
        self.store.write(
            lambda c: c.execute(
                "INSERT OR IGNORE INTO evidence_links(src_id, dst_id, relation, created_at) VALUES (?,?,?,?)",
                (src, dst, relation, time.time()),
            )
        )

    def mark_stale(self, claim_prefix: str) -> int:
        """Mark active heads whose claim key starts with ``claim_prefix`` STALE."""
        like = claim_prefix.replace("%", r"\%") + "%"

        def run(c: Any) -> int:
            cur = c.execute(
                "UPDATE evidence SET status = 'STALE' WHERE status = 'ACTIVE' AND evidence_id IN "
                "(SELECT evidence_id FROM claim_heads WHERE claim_key LIKE ? ESCAPE '\\')",
                (like,),
            )
            return cur.rowcount or 0

        return int(self.store.write(run, default=0) or 0)

    def pin(self, evidence_ids: list[str]) -> None:
        if not evidence_ids:
            return
        marks = ",".join("?" * len(evidence_ids))
        self.store.write(
            lambda c: c.execute(
                f"UPDATE evidence SET pinned = 1 WHERE evidence_id IN ({marks})",  # noqa: S608
                tuple(evidence_ids),
            )
        )

    # -------------------------------------------------------------- query
    def _fresh(self, rec: EvidenceRecord) -> EvidenceRecord:
        if rec.status == "ACTIVE" and rec.valid_until is not None and rec.valid_until < time.time():
            self.store.write(
                lambda c: c.execute(
                    "UPDATE evidence SET status = 'STALE' WHERE evidence_id = ?", (rec.evidence_id,)
                )
            )
            return EvidenceRecord(**{**rec.__dict__, "status": "STALE"})
        return rec

    def get_claim(self, claim_key: str) -> EvidenceRecord | None:
        row = self.store.query_one(
            "SELECT e.* FROM claim_heads h JOIN evidence e ON e.evidence_id = h.evidence_id "
            "WHERE h.claim_key = ?",
            (claim_key,),
        )
        if row is None:
            return None
        self.lookup_hits += 1
        self.rt.metrics.bump("evidence_lookup_hits")
        return self._fresh(_row_to_record(row))

    def get(self, evidence_id: str) -> EvidenceRecord | None:
        row = self.store.query_one("SELECT * FROM evidence WHERE evidence_id = ?", (evidence_id,))
        return self._fresh(_row_to_record(row)) if row is not None else None

    def get_active(
        self, subject: str | None = None, predicate: str | None = None, task_id: str | None = None
    ) -> list[EvidenceRecord]:
        sql = "SELECT * FROM evidence WHERE status = 'ACTIVE'"
        params: list[Any] = []
        if subject is not None:
            sql += " AND subject = ?"
            params.append(subject)
        if predicate is not None:
            sql += " AND predicate = ?"
            params.append(predicate)
        if task_id is not None:
            sql += " AND task_id = ?"
            params.append(task_id)
        sql += " ORDER BY created_at DESC LIMIT 500"
        out = [self._fresh(_row_to_record(r)) for r in self.store.query(sql, tuple(params))]
        return [r for r in out if r.status == "ACTIVE"]

    def get_support(self, evidence_id: str) -> list[EvidenceRecord]:
        rows = self.store.query(
            "SELECT e.* FROM evidence_links l JOIN evidence e ON e.evidence_id = l.src_id "
            "WHERE l.dst_id = ? AND l.relation IN ('SUPPORTS', 'DERIVED_FROM')",
            (evidence_id,),
        )
        return [_row_to_record(r) for r in rows]

    def get_conflicts(self, evidence_id: str) -> list[EvidenceRecord]:
        rows = self.store.query(
            "SELECT e.* FROM evidence_links l JOIN evidence e ON "
            "(e.evidence_id = CASE WHEN l.src_id = ? THEN l.dst_id ELSE l.src_id END) "
            "WHERE (l.src_id = ? OR l.dst_id = ?) AND l.relation = 'CONTRADICTS'",
            (evidence_id, evidence_id, evidence_id),
        )
        return [_row_to_record(r) for r in rows]

    def get_recent(self, limit: int = 20, types: tuple[str, ...] = ()) -> list[EvidenceRecord]:
        sql = "SELECT * FROM evidence"
        params: list[Any] = []
        if types:
            sql += f" WHERE claim_type IN ({','.join('?' * len(types))})"
            params.extend(types)
        sql += " ORDER BY created_at DESC, rowid DESC LIMIT ?"
        params.append(int(limit))
        return [self._fresh(_row_to_record(r)) for r in self.store.query(sql, tuple(params))]

    def query_exact(self, keys: list[str]) -> dict[str, EvidenceRecord]:
        out = {}
        for key in keys:
            rec = self.get_claim(key)
            if rec is not None:
                out[key] = rec
        return out

    def contradictions(self, limit: int = 5) -> list[tuple[EvidenceRecord, EvidenceRecord]]:
        rows = self.store.query(
            "SELECT l.src_id, l.dst_id FROM evidence_links l WHERE l.relation = 'CONTRADICTS' "
            "ORDER BY l.created_at DESC LIMIT ?",
            (limit,),
        )
        out = []
        for r in rows:
            src, dst = self.get(r["src_id"]), self.get(r["dst_id"])
            if src is not None and dst is not None:
                out.append((src, dst))
        return out

    # ---------------------------------------------------------- extractors
    def extract(self, ev: AgentEvent) -> list[EvidenceRecord]:
        """Deterministic extraction for one finished/failed tool event."""
        inv: ToolInvocation | None = ev.transient.get("invocation")
        if inv is None:
            return []
        text = str(ev.transient.get("text") or "")
        if inv.tool_name.lower().endswith("headroom_workflow"):
            return []  # the executor already recorded its steps' evidence
        out: list[EvidenceRecord] = []
        try:
            if inv.family in (
                Family.READ_FILE,
                Family.EDIT_FILE,
                Family.WRITE_FILE,
                Family.DELETE_FILE,
            ) or (inv.paths_written or inv.paths_deleted):
                out.extend(self._filesystem(ev, inv, text))
            if inv.command:
                out.extend(self._command(ev, inv, text))
                if inv.family is Family.TEST:
                    out.extend(self._tests(ev, inv, text))
                elif inv.family is Family.BUILD:
                    out.extend(self._build(ev, inv, text))
                if any(s.executable == "git" for s in inv.segments):
                    out.extend(self._git(ev, inv, text))
                out.extend(self._probes(ev, inv, text))
        except Exception:  # noqa: BLE001 - a parser bug yields no facts, never wrong ones
            logger.debug("evidence extraction failed", exc_info=True)
        return [r for r in (self.record(rec) for rec in out) if r is not None]

    def _rel(self, path: str) -> str:
        return str(self.rt.workspace.relpath(path))

    def _filesystem(self, ev: AgentEvent, inv: ToolInvocation, text: str) -> list[EvidenceRecord]:
        out = []
        missing = ev.success is False and bool(
            re.search(
                r"(?i)does not exist|no such file|not found|cannot find",
                tool_use_error(text) or text[:300],
            )
        )
        for path in inv.paths_read if inv.family is Family.READ_FILE else ():
            rel = self._rel(path)
            if missing:
                out.append(
                    self.make(
                        ev,
                        claim_type="file_exists",
                        subject=rel,
                        predicate="exists",
                        value=False,
                        display=f"{rel} does not exist",
                        source_kind="FILESYSTEM",
                        confidence=CONF_FILESYSTEM,
                        polarity="NEGATIVE",
                    )
                )
            elif ev.success:
                out.append(
                    self.make(
                        ev,
                        claim_type="file_exists",
                        subject=rel,
                        predicate="exists",
                        value=True,
                        display=f"{rel} exists",
                        source_kind="FILESYSTEM",
                        confidence=CONF_FILESYSTEM,
                        polarity="POSITIVE",
                    )
                )
                digest = self._disk_hash(path)
                if digest:
                    out.append(
                        self.make(
                            ev,
                            claim_type="file_hash",
                            subject=rel,
                            predicate="sha256",
                            value=digest,
                            display=f"{rel} sha256 {digest[:12]}",
                            source_kind="FILESYSTEM",
                            confidence=CONF_FILESYSTEM,
                        )
                    )
        if ev.success is not False:
            for path in inv.paths_written:
                rel = self._rel(path)
                self.mark_stale(f"file_hash|{rel}")
                digest = self._disk_hash(path)
                out.append(
                    self.make(
                        ev,
                        claim_type="file_modified",
                        subject=rel,
                        predicate="modified",
                        value={"sha256": digest, "tool": inv.tool_name},
                        display=f"{rel} modified",
                        source_kind="FILESYSTEM",
                        confidence=CONF_FILESYSTEM,
                        polarity="POSITIVE",
                    )
                )
                out.append(
                    self.make(
                        ev,
                        claim_type="file_exists",
                        subject=rel,
                        predicate="exists",
                        value=True,
                        display=f"{rel} exists",
                        source_kind="FILESYSTEM",
                        confidence=CONF_FILESYSTEM,
                        polarity="POSITIVE",
                    )
                )
                if digest:
                    out.append(
                        self.make(
                            ev,
                            claim_type="file_hash",
                            subject=rel,
                            predicate="sha256",
                            value=digest,
                            display=f"{rel} sha256 {digest[:12]}",
                            source_kind="FILESYSTEM",
                            confidence=CONF_FILESYSTEM,
                        )
                    )
            for path in inv.paths_deleted:
                rel = self._rel(path)
                self.mark_stale(f"file_hash|{rel}")
                out.append(
                    self.make(
                        ev,
                        claim_type="file_exists",
                        subject=rel,
                        predicate="exists",
                        value=False,
                        display=f"{rel} deleted",
                        source_kind="FILESYSTEM",
                        confidence=CONF_FILESYSTEM,
                        polarity="NEGATIVE",
                    )
                )
        return out

    def _disk_hash(self, path: str) -> str:
        if not self.rt.workspace.local or not path:
            return ""
        try:
            st = os.stat(path)
            if not os.path.isfile(path) or st.st_size > MAX_HASH_BYTES:
                return ""
            with open(path, "rb") as fh:
                data = fh.read()
            return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()[:24]
        except OSError:
            return ""

    def _command(self, ev: AgentEvent, inv: ToolInvocation, text: str) -> list[EvidenceRecord]:
        norm = " ".join(inv.command.split())[:200]
        cwd = self._rel(inv.cwd) if inv.cwd else "."
        dur = _duration(text)
        value = {
            "exit_code": ev.exit_code,
            "success": ev.success,
            "cwd": cwd,
            "duration_bucket": _bucket(dur),
            "signature": failure_signature(text) if ev.success is False else "",
        }
        status = "succeeded" if ev.success else ("failed" if ev.success is False else "ran")
        code = f" (exit {ev.exit_code})" if ev.exit_code is not None else ""
        out = [
            self.make(
                ev,
                claim_type="command_exit",
                subject=norm,
                predicate="exit",
                value=value,
                display=f"`{norm[:80]}` {status}{code}",
                source_kind="TOOL",
                confidence=CONF_TOOL_EXIT if ev.exit_code is not None else 0.9,
                polarity="POSITIVE" if ev.success else "NEGATIVE",
            )
        ]
        if (
            ev.success
            and len(inv.segments) == 1
            and any(a in ("--version", "-V", "version") for a in inv.segments[0].argv[1:2])
        ):
            line = next(
                (
                    ln.strip()
                    for ln in text.splitlines()
                    if ln.strip() and not ln.startswith(("Exit code", "Wall time", "Output"))
                ),
                "",
            )
            if line:
                exe = inv.segments[0].executable
                out.append(
                    self.make(
                        ev,
                        claim_type="tool_version",
                        subject=exe,
                        predicate="version",
                        value=line[:120],
                        display=f"{exe}: {line[:80]}",
                        source_kind="TOOL",
                        confidence=CONF_TOOL_EXIT,
                    )
                )
        return out

    def _tests(self, ev: AgentEvent, inv: ToolInvocation, text: str) -> list[EvidenceRecord]:
        parsed = parse_test_output(text, inv.test_framework)
        if parsed is None:
            return []  # unknown output: only the generic command fact stands
        return self.record_test_result(
            ev, parsed, command=inv.command, framework=inv.test_framework
        )

    def record_test_result(
        self,
        ev: AgentEvent | None,
        parsed: TestResult,
        *,
        command: str,
        framework: str,
        source_event_id: str = "",
    ) -> list[EvidenceRecord]:
        norm = " ".join(command.split())[:200]
        fw = parsed.framework or framework or "generic"
        full_suite = _is_full_suite(norm, fw)
        value = {
            "ok": parsed.ok,
            "passed": parsed.passed,
            "failed": parsed.failed,
            "errors": parsed.errors,
            "skipped": parsed.skipped,
            "failed_ids": parsed.failed_ids[:50],
            "passed_ids": parsed.passed_ids[:200],
            "command": norm,
            "framework": fw,
            "full_suite": full_suite,
            "duration_s": parsed.duration_s,
        }
        if parsed.failed_ids and self.rt.test_impact is not None:
            try:
                value["preexisting"] = self.rt.test_impact.preexisting_failures(
                    parsed.failed_ids, fw
                )
            except Exception:  # noqa: BLE001
                value["preexisting"] = []
        verdict = "passed" if parsed.ok else "FAILED"
        detail = f"{parsed.passed} passed, {parsed.failed} failed" + (
            f", {parsed.errors} errors" if parsed.errors else ""
        )
        recs = [
            self.make(
                ev,
                claim_type="test_aggregate",
                subject=f"{fw}|{norm}",
                predicate="result",
                value=value,
                display=f"{norm[:80]}: {verdict} ({detail})",
                source_kind="TEST",
                confidence=CONF_TEST,
                polarity="POSITIVE" if parsed.ok else "NEGATIVE",
                source_event_id=source_event_id,
            ),
            self.make(
                ev,
                claim_type="tests_passing",
                subject="latest",
                predicate="ok",
                value={"ok": parsed.ok},
                display=f"latest test run {verdict}",
                source_kind="TEST",
                confidence=CONF_TEST,
                polarity="POSITIVE" if parsed.ok else "NEGATIVE",
                source_event_id=source_event_id,
            ),
        ]
        for tid in parsed.failed_ids[:30]:
            recs.append(
                self.make(
                    ev,
                    claim_type="test_status",
                    subject=f"{fw}|{tid}",
                    predicate="status",
                    value={"state": "fail", "signature": parsed.failure_signatures.get(tid, "")},
                    display=f"{tid} failing",
                    source_kind="TEST",
                    confidence=CONF_TEST,
                    polarity="NEGATIVE",
                    source_event_id=source_event_id,
                )
            )
        for tid in parsed.passed_ids[:60]:
            recs.append(
                self.make(
                    ev,
                    claim_type="test_status",
                    subject=f"{fw}|{tid}",
                    predicate="status",
                    value={"state": "pass"},
                    display=f"{tid} passing",
                    source_kind="TEST",
                    confidence=CONF_TEST,
                    polarity="POSITIVE",
                    source_event_id=source_event_id,
                )
            )
        if ev is None:
            return [r for r in (self.record(x) for x in recs) if r is not None]
        return recs

    def _build(self, ev: AgentEvent, inv: ToolInvocation, text: str) -> list[EvidenceRecord]:
        norm = " ".join(inv.command.split())[:200]
        diags = parse_diagnostics(text) if ev.success is False or "error" in text.lower() else []
        ok = bool(ev.success) and not diags
        tool = inv.segments[0].executable if inv.segments else "build"
        value = {
            "ok": ok,
            "errors": len(diags),
            "command": norm,
            "diagnostics": [f"{self._rel(d.path)}:{d.line} {d.code}".strip() for d in diags[:10]],
        }
        recs = [
            self.make(
                ev,
                claim_type="build_status",
                subject=f"{tool}|{norm}",
                predicate="result",
                value=value,
                display=f"{norm[:80]}: {'ok' if ok else 'FAILED'}"
                + (f" ({len(diags)} errors)" if diags else ""),
                source_kind="BUILD",
                confidence=CONF_BUILD,
                polarity="POSITIVE" if ok else "NEGATIVE",
            ),
            self.make(
                ev,
                claim_type="build_passing",
                subject="latest",
                predicate="ok",
                value={"ok": ok},
                display=f"latest build {'ok' if ok else 'FAILED'}",
                source_kind="BUILD",
                confidence=CONF_BUILD,
                polarity="POSITIVE" if ok else "NEGATIVE",
            ),
        ]
        if ok:
            self.mark_stale("compiler_error|")
        for d in diags[:10]:
            rel = self._rel(d.path)
            recs.append(
                self.make(
                    ev,
                    claim_type="compiler_error",
                    subject=f"{rel}:{d.line}",
                    predicate="error",
                    value={"path": rel, "line": d.line, "code": d.code, "message": d.message[:160]},
                    display=f"{rel}:{d.line} {d.code} {d.message[:80]}".strip(),
                    source_kind="BUILD",
                    confidence=CONF_BUILD,
                    polarity="NEGATIVE",
                )
            )
        return recs

    def _git(self, ev: AgentEvent, inv: ToolInvocation, text: str) -> list[EvidenceRecord]:
        out: list[EvidenceRecord] = []
        git_segments = [s for s in inv.segments if s.executable == "git"]
        sub = git_segments[0].subcommand if git_segments else ""
        if ev.success is False:
            return out
        if sub in (
            "commit",
            "checkout",
            "switch",
            "reset",
            "merge",
            "rebase",
            "pull",
            "cherry-pick",
            "revert",
            "stash",
            "am",
            "restore",
            "add",
            "rm",
            "mv",
            "clean",
        ):
            for claim in _GIT_HEAD_CLAIMS:
                self.mark_stale(f"{claim}|repo")
            return out
        body = _strip_harness(text)
        if sub == "status":
            branch = re.search(r"(?m)^(?:On branch |## )([^\s.]+)", body)
            changed = re.findall(r"(?m)^\s*(?:modified|new file|deleted|renamed):\s+(.+)$", body)
            changed += [
                m[1] for m in re.findall(r"(?m)^([ MADRCU?!]{2}) (.+)$", body) if m[0].strip()
            ]
            clean = bool(re.search(r"(?i)working tree clean|nothing to commit", body))
            if branch:
                out.append(
                    self.make(
                        ev,
                        claim_type="git_branch",
                        subject="repo",
                        predicate="branch",
                        value=branch.group(1),
                        display=f"branch {branch.group(1)}",
                        source_kind="GIT",
                        confidence=CONF_GIT,
                    )
                )
            if clean or changed:
                files = sorted({c.strip() for c in changed})[:50]
                out.append(
                    self.make(
                        ev,
                        claim_type="git_dirty",
                        subject="repo",
                        predicate="dirty",
                        value={"dirty": not clean, "files": files},
                        display="working tree clean" if clean else f"{len(files)} changed files",
                        source_kind="GIT",
                        confidence=CONF_GIT,
                    )
                )
        elif sub in ("rev-parse", "log", "show"):
            m = re.search(r"\b([0-9a-f]{40})\b", body) or re.search(
                r"(?m)^commit ([0-9a-f]{7,40})", body
            )
            if m and (
                sub == "rev-parse"
                or "-1" in inv.command
                or "-n 1" in inv.command
                or "HEAD" in inv.command
            ):
                out.append(
                    self.make(
                        ev,
                        claim_type="git_head",
                        subject="repo",
                        predicate="head",
                        value=m.group(1),
                        display=f"HEAD {m.group(1)[:12]}",
                        source_kind="GIT",
                        confidence=CONF_GIT,
                    )
                )
        elif sub == "diff" and "--stat" in inv.command:
            m = re.search(
                r"(\d+) files? changed(?:, (\d+) insertions?\(\+\))?(?:, (\d+) deletions?\(-\))?",
                body,
            )
            if m:
                value = {
                    "files": int(m.group(1)),
                    "insertions": int(m.group(2) or 0),
                    "deletions": int(m.group(3) or 0),
                }
                out.append(
                    self.make(
                        ev,
                        claim_type="git_diffstat",
                        subject="repo",
                        predicate="diffstat",
                        value=value,
                        display=f"diff: {value['files']} files +{value['insertions']} -{value['deletions']}",
                        source_kind="GIT",
                        confidence=CONF_GIT,
                    )
                )
        return out

    def _probes(self, ev: AgentEvent, inv: ToolInvocation, text: str) -> list[EvidenceRecord]:
        out = []
        m = re.search(
            r"(?:https?://)?(localhost|127\.0\.0\.1|0\.0\.0\.0|\[::1\]):(\d{2,5})", inv.command
        )
        if m and any(
            s.executable in ("curl", "wget", "invoke-webrequest", "iwr", "nc") for s in inv.segments
        ):
            host = "127.0.0.1" if m.group(1) in ("localhost", "0.0.0.0", "[::1]") else m.group(1)
            refused = bool(
                re.search(
                    r"(?i)connection refused|failed to connect|couldn't connect|unable to connect|actively refused",
                    text,
                )
            )
            if refused or ev.success:
                state = "closed" if refused else "listening"
                out.append(
                    self.make(
                        ev,
                        claim_type="port_state",
                        subject=f"{host}:{m.group(2)}",
                        predicate="state",
                        value={"state": state},
                        display=f"{host}:{m.group(2)} {state}",
                        source_kind="TOOL",
                        confidence=CONF_TOOL_EXIT,
                    )
                )
        return out

    # ------------------------------------------------------- user / agent
    _AGENT_TESTS_PASS = re.compile(
        r"(?i)\b(?:all (?:the )?(?:\d+ )?tests? (?:now )?(?:pass|passed|are passing|green)|tests? (?:now )?(?:pass|passed|are passing)|test suite (?:passes|passed|is green))\b"
    )
    _AGENT_TESTS_FAIL = re.compile(
        r"(?i)\b(?:tests? (?:are )?(?:still )?failing|tests? fail(?:ed)?)\b"
    )
    _AGENT_BUILD_OK = re.compile(
        r"(?i)\b(?:build (?:now )?(?:passes|succeeds|succeeded|is green)|compiles? (?:cleanly|successfully|without errors))\b"
    )
    _USER_PORT = re.compile(
        r"(?i)\b(?:listening|running|serving|up) on (?:port )?(?:(localhost|127\.0\.0\.1):)?(\d{2,5})\b"
    )

    def on_assistant_message(self, ev: AgentEvent) -> list[EvidenceRecord]:
        text = str(ev.transient.get("text") or "")
        recs: list[EvidenceRecord] = []
        if self._AGENT_TESTS_PASS.search(text) and not self._AGENT_TESTS_FAIL.search(text):
            recs.append(
                self.make(
                    ev,
                    claim_type="tests_passing",
                    subject="latest",
                    predicate="ok",
                    value={"ok": True},
                    display="agent: tests pass",
                    source_kind="AGENT",
                    confidence=CONF_AGENT,
                    polarity="POSITIVE",
                )
            )
        if self._AGENT_BUILD_OK.search(text):
            recs.append(
                self.make(
                    ev,
                    claim_type="build_passing",
                    subject="latest",
                    predicate="ok",
                    value={"ok": True},
                    display="agent: build succeeds",
                    source_kind="AGENT",
                    confidence=CONF_AGENT,
                    polarity="POSITIVE",
                )
            )
        return [r for r in (self.record(x) for x in recs) if r is not None]

    def on_user_assertion(self, ev: AgentEvent) -> list[EvidenceRecord]:
        text = str(ev.transient.get("text") or "")
        recs = []
        for m in self._USER_PORT.finditer(text):
            recs.append(
                self.make(
                    ev,
                    claim_type="port_state",
                    subject=f"127.0.0.1:{m.group(2)}",
                    predicate="state",
                    value={"state": "listening"},
                    display=f"user: 127.0.0.1:{m.group(2)} listening",
                    source_kind="USER",
                    confidence=CONF_USER_FACT,
                    polarity="POSITIVE",
                )
            )
        return [r for r in (self.record(x) for x in recs) if r is not None]

    # ------------------------------------------------------------ render
    def sections(self) -> list[tuple[int, str, list[str]]]:
        """Compact, prioritized evidence lines (plan §6.10)."""
        out: list[tuple[int, str, list[str]]] = []
        contra = []
        for src, dst in self.contradictions(limit=3):
            low, high = (dst, src) if src.authority >= dst.authority else (src, dst)
            if time.time() - max(src.created_at, dst.created_at) > 3600:
                continue
            contra.append(
                f'- {low.evidence_id} "{low.display_text}" contradicted by {high.evidence_id} {high.display_text}'
            )
        if contra:
            out.append((5, "evidence-conflicts", contra))
        failing = []
        for rec in self.get_recent(limit=12, types=("test_aggregate", "build_status")):
            if rec.status != "ACTIVE" or not isinstance(rec.value, dict) or rec.value.get("ok"):
                continue
            ids = rec.value.get("failed_ids") or rec.value.get("diagnostics") or []
            extra = f" [{', '.join(str(i) for i in ids[:4])}]" if ids else ""
            failing.append(f"- {rec.evidence_id} {rec.display_text}{extra}")
            if len(failing) >= 3:
                break
        if failing:
            out.append((12, "failing", failing))
        errs = [
            f"- {r.display_text}"
            for r in self.get_recent(limit=8, types=("compiler_error",))
            if r.status == "ACTIVE"
        ][:5]
        if errs:
            out.append((14, "compiler-errors", errs))
        return out


def _strip_harness(text: str) -> str:
    m = re.search(r"(?m)^Output:\s*$", text or "")
    return text[m.end() :] if m else (text or "")


_DURATION_RE = re.compile(r"(?i)wall time:\s*([\d.]+)\s*s|\bin ([\d.]+)s\b|finished in ([\d.]+)s")


def _duration(text: str) -> float | None:
    m = _DURATION_RE.search(text or "")
    if not m:
        return None
    for g in m.groups():
        if g:
            try:
                return float(g)
            except ValueError:
                return None
    return None


def _bucket(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    if seconds < 1:
        return "<1s"
    if seconds < 10:
        return "<10s"
    if seconds < 60:
        return "<1m"
    if seconds < 600:
        return "<10m"
    return ">=10m"


_SELECTOR_RE = re.compile(
    r"(?:::|\s-k\s|\s-m\s|--test\s|\s-p\s|--package|\.py\b|\.rs\b|\.[jt]sx?\b|-R\s|--filter|-t\s)"
)


def _is_full_suite(command: str, framework: str) -> bool:
    """True when the run carries no test selector (whole project/package suite)."""
    tail = " " + command
    return not _SELECTOR_RE.search(tail)


def evidence_brief(rec: EvidenceRecord) -> dict[str, Any]:
    return {
        "id": rec.evidence_id,
        "claim": rec.claim_key,
        "status": rec.status,
        "source": rec.source_kind,
        "confidence": rec.confidence,
        "text": rec.display_text,
        "first_error": first_error_line(str(rec.value)) if rec.polarity == "NEGATIVE" else "",
    }
