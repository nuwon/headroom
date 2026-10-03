"""Feature 20: Workflow Macro Compiler (plan §10).

Repeated, successful, *safe* tool sequences are compiled into one
Headroom-owned operation (``headroom_workflow``), so the model does not spend
several turns rediscovering the same procedure.

The safety rules come first (§10.2, §17):

* Only ``READ_ONLY`` and ``VERIFICATION`` macros auto-promote or auto-execute.
  ``LOCAL_REVERSIBLE``, ``MUTATING``, ``EXTERNAL_SIDE_EFFECT`` and ``UNKNOWN``
  sequences are observed, never promoted. Network-capable steps are never
  promoted.
* A slot is generalized only where repeated observations differ and every
  observed value is a project-relative path. Executables, flags, URLs,
  absolute paths and refs are never parameters. Each slot has a validator: no
  ``..``, no leading ``-``, inside the project after symlink resolution, not
  protected.
* Steps run as direct argv (never a shell string), with the privileges of
  ``headroom wrap``. Every step passes the Tool Contract Validator and the
  Scope Firewall first, emits normal events and feeds the Evidence Ledger.
* A macro stops on its first failed step, auto-disables after 2 failures in
  its last 5 runs, and is invalidated when the build/test configuration,
  executable identity or a referenced path changes.

Promotion (§10.6) needs at least 3 observations, a success rate of at least
0.90, at least 2 sessions or 3 uncorrected repetitions, no contract or scope
violation, and at least 2 orchestration turns saved. Names are derived
deterministically from the steps; no model call is made.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import time
from dataclasses import dataclass, field
from typing import Any

from headroom.redaction import is_secret_path

from .events import AgentEvent, EventType, command_fingerprint
from .families import Family, SafetyClass, ToolInvocation, max_safety, normalize_tool_call
from .ids import stable_id
from .normalizer import INTERRUPT_RE
from .paths import is_absolute, relative_to_root, resolve, within
from .results import parse_test_output
from .store import dumps, loads, red

logger = logging.getLogger(__name__)

MIN_STEPS, MAX_STEPS = 2, 8
MAX_SEQUENCE_SECONDS = 15 * 60
CORRECTION_WINDOW_TURNS = 2
MIN_SUCCESS_RATE = 0.90
MIN_TURNS_SAVED = 2
DISABLE_FAILURES, DISABLE_WINDOW = 2, 5
SAFE_CLASSES = {"read_only": SafetyClass.READ_ONLY, "verification": SafetyClass.VERIFICATION}
_CORRECTION_RE = re.compile(
    r"(?i)^\s*(?:no[,.!]|nope|wrong|that'?s (?:wrong|not)|undo|revert|stop|don'?t do that|actually|wait|scratch that)"
)
_PATHISH_RE = re.compile(r"[\\/]|\.[A-Za-z0-9]{1,6}$")
_PROTECTED_PARTS = {".git", "node_modules", ".venv", "venv", "site-packages"}
_KIND_WORDS = {
    "test": "test",
    "build": "build",
    "lint": "lint",
    "read": "read",
    "search": "search",
    "list": "list",
    "info": "inspect",
}
_GIT_WORDS = {"diff": "diff", "status": "status", "log": "log", "show": "show"}


@dataclass(frozen=True)
class WorkflowStep:
    step_id: str
    tool_family: str
    original_tool_name: str
    operation: str
    argument_template: tuple[str, ...]  # argv template, or ("read", "{p1}") for native steps
    required_inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    success_predicate: str = "exit_zero"
    contract_id: str = ""
    safety_class: str = "UNKNOWN"
    timeout_policy: float = 600.0
    run_on_failure: bool = False
    native: str = ""  # "" (argv) | read | search | list | plan

    def to_json(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["argument_template"] = list(self.argument_template)
        d["required_inputs"] = list(self.required_inputs)
        d["outputs"] = list(self.outputs)
        return d

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> WorkflowStep:
        kw = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        for k in ("argument_template", "required_inputs", "outputs"):
            kw[k] = tuple(kw.get(k) or ())
        return cls(**kw)


@dataclass
class WorkflowMacro:
    macro_id: str
    workspace_id: str
    name: str
    description: str
    version: int
    safety_class: str
    steps: list[WorkflowStep]
    input_schema: dict[str, Any]
    output_schema: dict[str, Any] = field(default_factory=lambda: {"type": "object"})
    preconditions: list[str] = field(default_factory=list)
    success_predicate: str = "all_steps_exit_zero"
    failure_policy: str = "stop_on_first_failure"
    support_count: int = 0
    success_rate: float = 1.0
    estimated_turns_saved: float = 0.0
    estimated_tokens_saved: float = 0.0
    origin: str = "learned"
    config_hash: str = ""
    created_at: float = 0.0
    last_used_at: float | None = None
    disabled_reason: str | None = None
    supporting_event_ids: list[str] = field(default_factory=list)
    supporting_evidence_ids: list[str] = field(default_factory=list)
    literal_paths: list[str] = field(default_factory=list)
    executables: dict[str, str] = field(default_factory=dict)

    @property
    def enabled(self) -> bool:
        return not self.disabled_reason

    def to_json(self) -> dict[str, Any]:
        return {
            "steps": [s.to_json() for s in self.steps],
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "preconditions": self.preconditions,
            "success_predicate": self.success_predicate,
            "failure_policy": self.failure_policy,
            "supporting_event_ids": self.supporting_event_ids[:50],
            "supporting_evidence_ids": self.supporting_evidence_ids[:50],
            "literal_paths": self.literal_paths,
            "executables": self.executables,
        }


def _macro_from_row(row: Any) -> WorkflowMacro:
    body = loads(row["macro_json"], {}) or {}
    return WorkflowMacro(
        macro_id=row["macro_id"],
        workspace_id=row["workspace_id"],
        name=row["name"],
        description=row["description"],
        version=int(row["version"]),
        safety_class=row["safety_class"],
        steps=[WorkflowStep.from_json(s) for s in body.get("steps") or []],
        input_schema=body.get("input_schema") or {"type": "object", "properties": {}},
        output_schema=body.get("output_schema") or {"type": "object"},
        preconditions=body.get("preconditions") or [],
        success_predicate=body.get("success_predicate", "all_steps_exit_zero"),
        failure_policy=body.get("failure_policy", "stop_on_first_failure"),
        support_count=int(row["support_count"]),
        success_rate=float(row["success_rate"]),
        estimated_turns_saved=float(row["turns_saved"]),
        estimated_tokens_saved=float(row["tokens_saved"]),
        origin=row["origin"],
        config_hash=row["config_hash"],
        created_at=float(row["created_at"]),
        last_used_at=row["last_used_at"],
        disabled_reason=row["disabled_reason"],
        supporting_event_ids=body.get("supporting_event_ids") or [],
        supporting_evidence_ids=body.get("supporting_evidence_ids") or [],
        literal_paths=body.get("literal_paths") or [],
        executables=body.get("executables") or {},
    )


# -------------------------------------------------------------- templating
@dataclass(frozen=True)
class ObservedStep:
    event_id: str
    tool_name: str
    family: str
    safety: str
    template: tuple[str, ...]  # path-like args replaced by "{path}"
    values: tuple[str, ...]  # the concrete path-like args, in order
    native: str
    executable_ok: bool
    turn: int
    ts: float
    success: bool
    kind: str
    call_id: str = ""


def _step_from_invocation(
    inv: ToolInvocation, root: str
) -> tuple[tuple[str, ...], tuple[str, ...], str, bool, str]:
    """(template, values, native kind, executable, kind word)."""
    fam = inv.family
    if fam is Family.READ_FILE and not inv.command and inv.paths_read:
        rel = relative_to_root(inv.paths_read[0], root)
        return ("read", "{path}"), (rel,), "read", not is_absolute(rel), "read"
    if fam in (Family.SEARCH_TEXT, Family.LIST_FILES) and not inv.command:
        pattern = inv.patterns[0] if inv.patterns else str(inv.extra.get("glob") or "")
        scope = relative_to_root(inv.paths_read[0], root) if inv.paths_read else "."
        native = "search" if fam is Family.SEARCH_TEXT else "list"
        return (native, pattern, "{path}"), (scope,), native, not is_absolute(scope), native
    if inv.segments and len(inv.segments) == 1 and inv.argv_safe:
        seg = inv.segments[0]
        template: list[str] = []
        values: list[str] = []
        for i, a in enumerate(seg.argv):
            if i > 0 and not a.startswith("-") and _PATHISH_RE.search(a) and "://" not in a:
                template.append("{path}")
                values.append(
                    relative_to_root(resolve(a, cwd=inv.cwd or root), root)
                    if not is_absolute(a)
                    else a
                )
            else:
                template.append(a)
        kind = seg.kind
        word = (
            _GIT_WORDS.get(seg.subcommand, "git")
            if seg.executable == "git"
            else _KIND_WORDS.get(kind, seg.executable)
        )
        return tuple(template), tuple(values), "", True, word
    return (fam.value, inv.command[:120]), (), "", False, fam.value


def signature_of(steps: list[ObservedStep]) -> str:
    raw = json.dumps([[s.family, *s.template] for s in steps], ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()[:20]


def macro_name(
    steps: list[ObservedStep] | list[WorkflowStep], existing: set[str] | None = None
) -> str:
    words: list[str] = []
    for s in steps:
        w = s.kind if isinstance(s, ObservedStep) else s.operation
        if w and (not words or words[-1] != w):
            words.append(w)
    base = "_then_".join(words)[:48] or "workflow"
    base = re.sub(r"[^a-z0-9_]", "_", base.lower())
    if existing and base in existing:
        base = f"{base}_{hashlib.sha256(repr(steps).encode()).hexdigest()[:4]}"
    return base


# --------------------------------------------------------------- compiler
class WorkflowCompiler:
    """Observes sequences in one session and promotes eligible macros (workspace-wide)."""

    def __init__(self, runtime: Any) -> None:
        self.rt = runtime
        self.store = runtime.store
        self.config = runtime.config
        self.current: list[ObservedStep] = []
        self.pending: list[
            tuple[list[ObservedStep], int, bool]
        ] = []  # (steps, turns since close, corrected)
        self.turn = 0
        self._turn_keys: set[str] = set()
        self.announced: set[str] = set()

    def _flush_current(self, *, corrected: bool) -> None:
        if self.current:
            self.pending.append((self.current, 0, corrected))
        self.current = []

    def on_user_message(self, ev: AgentEvent) -> None:
        text = str(ev.transient.get("text") or "")
        correction = (
            bool(ev.transient.get("interrupted"))
            or bool(_CORRECTION_RE.match(text))
            or bool(INTERRUPT_RE.search(text))
        )
        if correction:
            self.pending = [(steps, n, True) for steps, n, _ in self.pending]
        self._flush_current(corrected=correction)
        self._finalize(force=False)

    def on_assistant_message(self, ev: AgentEvent) -> None:
        ref = str(ev.source_ref or "")
        self._advance_turn(ref if ref.startswith("msg:") else ev.event_id)

    def _advance_turn(self, key: str) -> None:
        """One agent turn per assistant message (text or tool calls).

        Pending sequences age toward their 2-turn correction window, and steps
        record which turn issued them (the turns a macro would save).
        """
        if key in self._turn_keys:
            return
        self._turn_keys.add(key)
        if len(self._turn_keys) > 4096:
            self._turn_keys = {key}
        self.turn += 1
        self.pending = [(steps, n + 1, c) for steps, n, c in self.pending]
        self._finalize(force=False)

    def on_tool_result(self, ev: AgentEvent, records: list[Any]) -> None:
        inv: ToolInvocation | None = ev.transient.get("invocation")
        if inv is None or ev.metadata.get("origin") == "headroom":
            return
        if inv.tool_name.lower().endswith("headroom_workflow"):
            self._flush_current(corrected=False)
            return
        turn = ev.transient.get("turn")
        if isinstance(turn, int) and turn >= 0:
            self._advance_turn(f"msg:{turn}")
        template, values, native, executable, kind = _step_from_invocation(
            inv, self.rt.workspace.root
        )
        step = ObservedStep(
            ev.event_id,
            inv.tool_name,
            inv.family.value,
            inv.safety.value,
            template,
            values,
            native,
            executable,
            self.turn,
            ev.timestamp,
            ev.success is not False,
            kind,
            str(ev.transient.get("call_id") or ""),
        )
        safe = inv.safety in (SafetyClass.READ_ONLY, SafetyClass.VERIFICATION) and not inv.network
        if not safe:
            self._flush_current(corrected=False)
            return
        if self.current and step.ts - self.current[0].ts > MAX_SEQUENCE_SECONDS:
            self._flush_current(corrected=False)
        self.current.append(step)
        if not step.success or len(self.current) >= MAX_STEPS:
            self._flush_current(corrected=False)

    def _finalize(self, *, force: bool) -> None:
        keep = []
        for steps, n, corrected in self.pending:
            if n >= CORRECTION_WINDOW_TURNS or force:
                self._observe(steps, corrected=corrected)
            else:
                keep.append((steps, n, corrected))
        self.pending = keep

    def _violations(self, steps: list[ObservedStep]) -> bool:
        """A contract finding or scope violation inside the sequence disqualifies it."""
        ids = [
            stable_id(
                stable_id(self.rt.session_key, "proposed", s.call_id), src, "validation", n=24
            )
            for s in steps
            if s.call_id
            for src in ("history", "hook")
        ]
        if ids:
            marks = ",".join("?" * len(ids))
            row = self.store.query_one(
                f"SELECT 1 FROM validations WHERE validation_id IN ({marks}) AND outcome IN ('BLOCK','WARN') LIMIT 1",  # noqa: S608
                tuple(ids),
            )
            if row is not None:
                return True
        row = self.store.query_one(
            "SELECT 1 FROM events WHERE session_key = ? AND event_type = 'SCOPE_VIOLATION' AND ts BETWEEN ? AND ? LIMIT 1",
            (self.rt.session_key, steps[0].ts - 1, steps[-1].ts + 1),
        )
        return row is not None

    def _observe(self, steps: list[ObservedStep], *, corrected: bool) -> None:
        if not (MIN_STEPS <= len(steps) <= MAX_STEPS):
            return
        if steps[-1].ts - steps[0].ts > MAX_SEQUENCE_SECONDS:
            return
        sig = signature_of(steps)
        event_ids = [s.event_id for s in steps]
        violated = self._violations(steps)
        success = all(s.success for s in steps)
        turns = len({s.turn for s in steps})
        payload = dumps(
            [
                {
                    "family": s.family,
                    "tool": s.tool_name,
                    "template": list(s.template),
                    "native": s.native,
                    "safety": s.safety,
                    "executable": s.executable_ok,
                    "kind": s.kind,
                }
                for s in steps
            ]
        )
        values = dumps([list(s.values) for s in steps])

        def run(c: Any) -> int:
            cur = c.execute(
                "INSERT OR IGNORE INTO workflow_observations(signature, first_event_id, session_key, task_id, steps_json, values_json, success, turns, corrected, violated, event_ids, ts) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    sig,
                    steps[0].event_id,
                    self.rt.agent_session or self.rt.session_key,
                    self.rt.task_id,
                    payload,
                    values,
                    int(success),
                    turns,
                    int(corrected),
                    int(violated),
                    dumps(event_ids),
                    time.time(),
                ),
            )
            return cur.rowcount or 0

        if self.store.write(run, default=0):
            self.rt.metrics.bump("workflow_observations")
            self.maybe_promote(sig)

    # ------------------------------------------------------------ promotion
    def maybe_promote(self, signature: str) -> WorkflowMacro | None:
        rows = self.store.query(
            "SELECT * FROM workflow_observations WHERE signature = ? ORDER BY ts", (signature,)
        )
        min_obs = self.config.workflow_min_observations
        if len(rows) < min_obs:
            return None
        if any(r["violated"] for r in rows):
            return None
        clean = [r for r in rows if not r["corrected"]]
        success_rate = sum(r["success"] for r in rows) / len(rows)
        if success_rate < MIN_SUCCESS_RATE:
            return None
        sessions = {r["session_key"] for r in clean if r["success"]}
        if not (len(sessions) >= 2 or len([r for r in clean if r["success"]]) >= max(3, min_obs)):
            return None
        steps_meta = loads(rows[0]["steps_json"], []) or []
        allowed = {SAFE_CLASSES[c] for c in self.config.workflow_auto_classes if c in SAFE_CLASSES}
        safeties = [SafetyClass(s["safety"]) for s in steps_meta]
        effective = max_safety(safeties)
        if effective not in allowed:
            return None
        if not all(s.get("executable") for s in steps_meta):
            return None
        turns = sorted(r["turns"] for r in rows if r["success"])
        if not turns or turns[len(turns) // 2] - 1 < MIN_TURNS_SAVED:
            return None
        values = [loads(r["values_json"], []) or [] for r in rows if r["success"]]
        built = self._parameterize(steps_meta, values)
        if built is None:
            return None
        wsteps, schema, literal_paths = built
        existing = {r["name"] for r in self.store.query("SELECT name FROM workflow_macros")}
        name = macro_name(wsteps, existing)
        macro_id = "M" + stable_id(self.rt.workspace.workspace_id, signature, n=14)
        if (
            self.store.query_one("SELECT 1 FROM workflow_macros WHERE macro_id = ?", (macro_id,))
            is not None
        ):
            return None
        planner = self.rt.test_impact
        config_hash = planner.config_hash() if planner is not None else ""
        executables = {
            s.argument_template[0]: (shutil.which(s.argument_template[0]) or "")
            for s in wsteps
            if not s.native and s.argument_template
        }
        event_ids = [e for r in rows for e in (loads(r["event_ids"], []) or [])]
        macro = WorkflowMacro(
            macro_id=macro_id,
            workspace_id=self.rt.workspace.workspace_id,
            name=name,
            description=_describe(wsteps),
            version=1,
            safety_class=effective.value,
            steps=wsteps,
            input_schema=schema,
            support_count=len(rows),
            success_rate=success_rate,
            estimated_turns_saved=float(turns[len(turns) // 2] - 1),
            origin="learned",
            config_hash=config_hash,
            created_at=time.time(),
            supporting_event_ids=event_ids[:50],
            literal_paths=literal_paths,
            executables=executables,
        )
        self.save(macro)
        self.rt.metrics.bump("workflow_promoted")
        return macro

    def _parameterize(
        self, steps_meta: list[dict[str, Any]], values: list[list[list[str]]]
    ) -> tuple[list[WorkflowStep], dict[str, Any], list[str]] | None:
        """Generalize only slots that differ across observations and are always project paths."""
        props: dict[str, Any] = {}
        required: list[str] = []
        literal: list[str] = []
        wsteps: list[WorkflowStep] = []
        slot_n = 0
        root = self.rt.workspace.root
        for i, meta in enumerate(steps_meta):
            template = list(meta["template"])
            observed = [v[i] if i < len(v) else [] for v in values]
            k = 0
            for j, tok in enumerate(template):
                if tok != "{path}":
                    continue
                column = [obs[k] if k < len(obs) else None for obs in observed]
                k += 1
                if any(c is None for c in column):
                    return None
                distinct = sorted({str(c) for c in column})
                if len(distinct) == 1:
                    if not _valid_project_path(distinct[0], root, must_exist=False):
                        return None
                    template[j] = distinct[0]
                    literal.append(distinct[0])
                    continue
                if not all(_valid_project_path(c, root, must_exist=False) for c in distinct):
                    return None  # forbidden generalization (absolute/outside/arbitrary)
                slot_n += 1
                name = f"path{slot_n}"
                validator = (
                    "test_path" if all(_looks_like_test(c) for c in distinct) else "project_path"
                )
                props[name] = {
                    "type": "string",
                    "validator": validator,
                    "description": f"project-relative path ({validator})",
                    "examples": distinct[:3],
                }
                required.append(name)
                template[j] = "{" + name + "}"
            native = meta.get("native", "")
            op = meta.get("kind") or meta["family"]
            wsteps.append(
                WorkflowStep(
                    step_id=f"s{i + 1}",
                    tool_family=meta["family"],
                    original_tool_name=meta["tool"],
                    operation=op,
                    argument_template=tuple(template),
                    required_inputs=tuple(
                        t[1:-1] for t in template if t.startswith("{") and t.endswith("}")
                    ),
                    contract_id=f"{meta['family']}@1",
                    safety_class=meta["safety"],
                    native=native,
                    timeout_policy=900.0 if meta["safety"] == "VERIFICATION" else 60.0,
                )
            )
        schema = {
            "type": "object",
            "properties": props,
            "required": required,
            "additionalProperties": False,
        }
        return wsteps, schema, literal

    def save(self, macro: WorkflowMacro) -> None:
        def run(c: Any) -> None:
            c.execute(
                "INSERT OR REPLACE INTO workflow_macros(macro_id, workspace_id, name, description, version, safety_class, origin, macro_json, support_count, success_rate, turns_saved, tokens_saved, config_hash, created_at, last_used_at, disabled_reason) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    macro.macro_id,
                    macro.workspace_id,
                    macro.name,
                    red(macro.description, 300),
                    macro.version,
                    macro.safety_class,
                    macro.origin,
                    dumps(macro.to_json()),
                    macro.support_count,
                    macro.success_rate,
                    macro.estimated_turns_saved,
                    macro.estimated_tokens_saved,
                    macro.config_hash,
                    macro.created_at,
                    macro.last_used_at,
                    macro.disabled_reason,
                ),
            )

        self.store.write(run)
        if macro.supporting_evidence_ids and self.rt.evidence is not None:
            self.rt.evidence.pin(macro.supporting_evidence_ids)

    # ------------------------------------------------------------- render
    def sections(self) -> list[tuple[int, str, list[str]]]:
        macros = eligible_macros(self.store, self.rt.workspace.workspace_id)
        fresh = [m for m in macros if m.origin == "learned" and m.name not in self.announced]
        if not fresh:
            return []
        lines = [f"- headroom_workflow macro={m.name}: {m.description}"[:200] for m in fresh[:3]]
        for m in fresh[:3]:
            self.announced.add(m.name)
        return [(60, "workflows", lines)]


def _describe(steps: list[WorkflowStep]) -> str:
    parts = []
    for s in steps:
        if s.native:
            parts.append(f"{s.native} {' '.join(s.argument_template[1:])}".strip())
        else:
            parts.append(" ".join(s.argument_template)[:60])
    return " -> ".join(parts)[:240]


def _looks_like_test(path: str) -> bool:
    return bool(
        re.search(
            r"(?:^|/)(?:tests?|spec|__tests__)/|(?:^|/)test_[^/]+$|_test\.\w+$|\.(?:test|spec)\.\w+$",
            path,
        )
    )


def _valid_project_path(value: str, root: str, *, must_exist: bool) -> bool:
    """Slot validator: project-relative, no traversal or flags, inside root, not protected."""
    if not isinstance(value, str) or not value or len(value) > 400:
        return False
    if value.startswith("-") or "\x00" in value or "\n" in value or "\r" in value:
        return False
    if is_absolute(value) or value.startswith("~"):
        return False
    norm = value.replace("\\", "/")
    if any(part == ".." for part in norm.split("/")):
        return False
    if any(part in _PROTECTED_PARTS for part in norm.split("/")) or is_secret_path(norm):
        return False
    if re.search(r"[;&|`$<>*?!{}]", value):
        return False
    if root:
        full = resolve(value, cwd=root)
        if not within(full, resolve(root)):
            return False
        if must_exist and not os.path.exists(full):
            return False
    return True


def eligible_macros(store: Any, workspace_id: str) -> list[WorkflowMacro]:
    rows = store.query(
        "SELECT * FROM workflow_macros WHERE workspace_id = ? AND disabled_reason IS NULL ORDER BY origin, name",
        (workspace_id,),
    )
    return [_macro_from_row(r) for r in rows]


def all_macros(store: Any) -> list[WorkflowMacro]:
    return [_macro_from_row(r) for r in store.query("SELECT * FROM workflow_macros ORDER BY name")]


def candidate_summary(store: Any) -> list[dict[str, Any]]:
    rows = store.query(
        "SELECT signature, COUNT(*) AS n, SUM(success) AS ok, SUM(corrected) AS corrected, SUM(violated) AS violated, "
        "COUNT(DISTINCT session_key) AS sessions, MAX(steps_json) AS steps FROM workflow_observations GROUP BY signature ORDER BY n DESC LIMIT 20"
    )
    out = []
    for r in rows:
        steps = loads(r["steps"], []) or []
        out.append(
            {
                "signature": r["signature"],
                "observations": r["n"],
                "successes": r["ok"],
                "corrected": r["corrected"],
                "violated": r["violated"],
                "sessions": r["sessions"],
                "steps": [" ".join(s.get("template") or [])[:60] for s in steps],
                "safety": max_safety([SafetyClass(s["safety"]) for s in steps]).value
                if steps
                else "UNKNOWN",
            }
        )
    return out


# ------------------------------------------------------------- seed macros
def seed_macros(runtime: Any) -> None:
    """Generic templates, enabled only when project capability detection proves them applicable."""
    ws = runtime.workspace
    store = runtime.store
    planner = runtime.test_impact
    git_repo = os.path.isdir(os.path.join(ws.root, ".git")) if ws.local else False
    has_tests = False
    if planner is not None and ws.local:
        try:
            has_tests = any(conf >= 0.6 for _, conf in planner.project()[1])
        except Exception:  # noqa: BLE001
            has_tests = False
    seeds = [
        (
            "run_test_impact_plan",
            "VERIFICATION",
            "run the staged verification plan for task-owned changes",
            [
                WorkflowStep(
                    "s1",
                    "test",
                    "headroom",
                    "verify",
                    ("plan",),
                    native="plan",
                    safety_class="VERIFICATION",
                    timeout_policy=1800.0,
                )
            ],
            has_tests,
        ),
        (
            "show_task_owned_diff",
            "READ_ONLY",
            "git diff --stat and short status of the working tree",
            [
                WorkflowStep(
                    "s1",
                    "git",
                    "headroom",
                    "diff",
                    ("git", "diff", "--stat"),
                    safety_class="READ_ONLY",
                    timeout_policy=30.0,
                ),
                WorkflowStep(
                    "s2",
                    "git",
                    "headroom",
                    "status",
                    ("git", "status", "--short"),
                    safety_class="READ_ONLY",
                    timeout_policy=30.0,
                ),
            ],
            git_repo,
        ),
        (
            "verify_then_status",
            "VERIFICATION",
            "tier-1 verification of task-owned changes, then git status",
            [
                WorkflowStep(
                    "s1",
                    "test",
                    "headroom",
                    "verify",
                    ("plan", "tier1"),
                    native="plan",
                    safety_class="VERIFICATION",
                    timeout_policy=1800.0,
                ),
                WorkflowStep(
                    "s2",
                    "git",
                    "headroom",
                    "status",
                    ("git", "status", "--short"),
                    safety_class="READ_ONLY",
                    timeout_policy=30.0,
                    run_on_failure=True,
                ),
            ],
            has_tests and git_repo,
        ),
    ]
    now = time.time()
    for name, safety, desc, steps, applicable in seeds:
        macro_id = "S" + stable_id(ws.workspace_id, name, n=14)
        row = store.query_one(
            "SELECT disabled_reason FROM workflow_macros WHERE macro_id = ?", (macro_id,)
        )
        reason = None if applicable else "not_applicable"
        if row is not None and row["disabled_reason"] not in (None, "not_applicable"):
            continue  # disabled by failure policy: keep until relearned
        macro = WorkflowMacro(
            macro_id,
            ws.workspace_id,
            name,
            desc,
            1,
            safety,
            steps,
            {"type": "object", "properties": {}, "additionalProperties": False},
            origin="seed",
            created_at=now,
            disabled_reason=reason,
        )
        store.write(
            lambda c, m=macro: c.execute(
                "INSERT INTO workflow_macros(macro_id, workspace_id, name, description, version, safety_class, origin, macro_json, created_at, disabled_reason) VALUES (?,?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(macro_id) DO UPDATE SET disabled_reason = excluded.disabled_reason",
                (
                    m.macro_id,
                    m.workspace_id,
                    m.name,
                    m.description,
                    m.version,
                    m.safety_class,
                    m.origin,
                    dumps(m.to_json()),
                    m.created_at,
                    m.disabled_reason,
                ),
            )
        )


# ---------------------------------------------------------------- executor
class MacroError(ValueError):
    pass


class WorkflowExecutor:
    """Runs one macro end to end (plan §10.9) inside an agent-state runtime."""

    def __init__(self, runtime: Any) -> None:
        self.rt = runtime
        self.store = runtime.store

    def resolve(self, name_or_id: str) -> WorkflowMacro:
        row = self.store.query_one(
            "SELECT * FROM workflow_macros WHERE macro_id = ? OR name = ?", (name_or_id, name_or_id)
        )
        if row is None:
            raise MacroError(f"unknown macro {name_or_id!r}")
        macro = _macro_from_row(row)
        if macro.workspace_id != self.rt.workspace.workspace_id:
            raise MacroError("macro belongs to a different workspace")
        if not macro.enabled:
            raise MacroError(f"macro {macro.name} is disabled ({macro.disabled_reason})")
        return macro

    def invalidation_reason(self, macro: WorkflowMacro) -> str:
        planner = self.rt.test_impact
        if macro.origin == "learned":
            if (
                planner is not None
                and macro.config_hash
                and planner.config_hash() != macro.config_hash
            ):
                return "config_changed"
            for exe, recorded in macro.executables.items():
                if (shutil.which(exe) or "") != recorded:
                    return "executable_changed"
            for p in macro.literal_paths:
                if not os.path.exists(resolve(p, cwd=self.rt.workspace.root)):
                    return "path_missing"
        return ""

    def disable(self, macro: WorkflowMacro, reason: str) -> None:
        self.store.write(
            lambda c: c.execute(
                "UPDATE workflow_macros SET disabled_reason = ? WHERE macro_id = ?",
                (reason, macro.macro_id),
            )
        )
        self.rt.metrics.bump("workflow_auto_disabled")

    def validate_inputs(self, macro: WorkflowMacro, inputs: dict[str, Any]) -> dict[str, str]:
        props = (macro.input_schema or {}).get("properties") or {}
        required = set((macro.input_schema or {}).get("required") or [])
        inputs = inputs or {}
        unknown = set(inputs) - set(props)
        if unknown:
            raise MacroError(f"unknown inputs: {', '.join(sorted(unknown))}")
        missing = required - set(inputs)
        if missing:
            raise MacroError(f"missing inputs: {', '.join(sorted(missing))}")
        out: dict[str, str] = {}
        for key, spec in props.items():
            if key not in inputs:
                continue
            value = inputs[key]
            if not isinstance(value, str) or not _valid_project_path(
                value, self.rt.workspace.root, must_exist=True
            ):
                raise MacroError(f"input {key!r} is not a valid project-relative path")
            if spec.get("validator") == "test_path" and not _looks_like_test(
                value.replace("\\", "/")
            ):
                raise MacroError(f"input {key!r} must be a test file")
            out[key] = value
        return out

    def run(self, name_or_id: str, inputs: dict[str, Any] | None = None) -> dict[str, Any]:
        started = time.perf_counter()
        macro = self.resolve(name_or_id)
        reason = self.invalidation_reason(macro)
        if reason:
            self.disable(macro, reason)
            raise MacroError(
                f"macro {macro.name} was invalidated ({reason}); use the ordinary tools"
            )
        values = self.validate_inputs(macro, dict(inputs or {}))
        run_id = "W" + stable_id(macro.macro_id, time.time(), os.getpid(), n=14)
        steps_done = 0
        failed_step = ""
        status = "success"
        evidence_refs: list[str] = []
        verification: dict[str, Any] = {}
        details: list[dict[str, Any]] = []
        raw_chars = 0
        for step in macro.steps:
            if status != "success" and not step.run_on_failure:
                break
            if step.native == "plan":
                outcome = self._run_plan(step)
                verification = outcome
                evidence_refs.extend(outcome.get("evidence_refs", []))
                ok = outcome.get("status") in ("passed", "nothing_to_verify")
                details.append(
                    {"step": step.step_id, "op": "verify", "status": outcome.get("status")}
                )
            else:
                ok, info, refs, chars = self._run_step(macro, step, values, run_id)
                raw_chars += chars
                evidence_refs.extend(refs)
                details.append({"step": step.step_id, **info})
                if "tests" in info:
                    verification = info["tests"]
            if ok:
                steps_done += 1
            elif status == "success":
                status = "failed"
                failed_step = step.step_id
        duration = (time.perf_counter() - started) * 1000.0
        self._record_run(macro, run_id, status, steps_done, failed_step, duration, raw_chars)
        result: dict[str, Any] = {
            "macro": macro.name,
            "status": status,
            "steps": f"{steps_done}/{len(macro.steps)}",
            "duration_ms": int(duration),
            "evidence_refs": evidence_refs[:12],
            "details": details,
        }
        if verification:
            result["verification"] = {
                k: verification[k]
                for k in ("passed", "failed", "status", "tiers", "failed_ids", "tests_run")
                if k in verification
            }
        if failed_step:
            result["failed_step"] = failed_step
        changes = self.rt.scope.task_owned_changes() if self.rt.scope is not None else []
        result["changed_files"] = len(changes)
        return result

    def _run_plan(self, step: WorkflowStep) -> dict[str, Any]:
        planner = self.rt.test_impact
        if planner is None:
            return {"status": "unavailable", "reason": "test impact planner disabled"}
        plan = planner.plan(force=True)
        if plan is None:
            return {"status": "nothing_to_verify", "reason": "no task-owned changes"}
        max_tier = 1 if "tier1" in step.argument_template else None
        out: dict[str, Any] = planner.run_plan(plan, max_tier=max_tier)
        passed = sum(t.get("passed", 0) for t in out.get("tiers", []))
        failed = sum(t.get("failed", 0) for t in out.get("tiers", []))
        out["passed"], out["failed"] = passed, failed
        out["tiers"] = [t["tier"] for t in out.get("tiers", [])]
        return out

    def _render(self, step: WorkflowStep, values: dict[str, str]) -> list[str]:
        out = []
        for tok in step.argument_template:
            m = re.fullmatch(r"\{(\w+)\}", tok)
            out.append(values[m.group(1)] if m else tok)
        return out

    def _invocation(
        self, step: WorkflowStep, argv: list[str]
    ) -> tuple[ToolInvocation, dict[str, Any]]:
        root = self.rt.workspace.root
        raw: dict[str, Any]
        if step.native == "read":
            raw = {"file_path": resolve(argv[1], cwd=root)}
            return normalize_tool_call("Read", raw, default_cwd=root), raw
        if step.native in ("search", "list"):
            raw = {"pattern": argv[1], "path": resolve(argv[2], cwd=root)}
            return normalize_tool_call(
                "Grep" if step.native == "search" else "Glob", raw, default_cwd=root
            ), raw
        raw = {"command": argv, "workdir": root}
        return normalize_tool_call("exec_command", raw, default_cwd=root), raw

    def _run_step(
        self, macro: WorkflowMacro, step: WorkflowStep, values: dict[str, str], run_id: str
    ) -> tuple[bool, dict[str, Any], list[str], int]:
        from .proc import run_argv

        argv = self._render(step, values)
        inv, raw = self._invocation(step, argv)
        if inv.safety not in (SafetyClass.READ_ONLY, SafetyClass.VERIFICATION) or inv.network:
            return (
                False,
                {
                    "op": step.operation,
                    "status": "refused",
                    "reason": f"step class {inv.safety.value}",
                },
                [],
                0,
            )
        call_id = stable_id(run_id, step.step_id, n=20)
        proposed = AgentEvent(
            event_id=stable_id(self.rt.session_key, "proposed", call_id),
            workspace_id=self.rt.workspace.workspace_id,
            session_id=self.rt.session_key,
            task_id=self.rt.task_id,
            event_type=EventType.TOOL_CALL_PROPOSED,
            tool_name=inv.tool_name,
            operation=inv.family.value,
            command_fingerprint=command_fingerprint(inv.command) if inv.command else None,
            metadata={"origin": "headroom", "macro": macro.name},
            transient={"invocation": inv, "input": raw, "call_id": call_id},
        )
        outcome = self.rt.check_proposed(proposed, source="workflow")
        self.rt.persist_events([proposed])
        if outcome is not None and getattr(outcome, "enforced", "") == "blocked":
            return (
                False,
                {"op": step.operation, "status": "blocked", "reason": outcome.message[:200]},
                [],
                0,
            )
        root = self.rt.workspace.root
        if step.native:
            text, success, code = self._native(step, argv, root)
            duration = 0.0
        else:
            res = run_argv(argv, cwd=root, timeout=step.timeout_policy)
            text = res.output if not res.error else res.error
            success = res.ok
            code = res.returncode
            duration = res.duration_ms
            text = f"Exit code: {code if code is not None else -1}\n{text}"
        finished = AgentEvent(
            event_id=stable_id(self.rt.session_key, "result", call_id),
            workspace_id=self.rt.workspace.workspace_id,
            session_id=self.rt.session_key,
            task_id=self.rt.task_id,
            event_type=EventType.TOOL_CALL_FINISHED if success else EventType.TOOL_CALL_FAILED,
            tool_name=inv.tool_name,
            operation=inv.family.value,
            exit_code=code,
            success=success,
            content_hash=hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:24],
            metadata={"origin": "headroom", "macro": macro.name, "duration_ms": int(duration)},
            transient={"invocation": inv, "input": raw, "text": text, "call_id": call_id},
        )
        self.rt.dispatch(
            [finished]
        )  # evidence, task state, scope, test impact (workflows skip origin=headroom)
        self.rt.persist_events([finished])
        refs = [
            r["evidence_id"]
            for r in self.store.query(
                "SELECT evidence_id FROM evidence WHERE source_event_id = ? ORDER BY created_at",
                (finished.event_id,),
            )
        ][:4]
        info: dict[str, Any] = {
            "op": step.operation,
            "status": "ok" if success else "failed",
            "exit": code,
        }
        if inv.family is Family.TEST:
            parsed = parse_test_output(text, inv.test_framework)
            if parsed is not None:
                info["tests"] = {
                    "passed": parsed.passed,
                    "failed": parsed.failed + parsed.errors,
                    "failed_ids": parsed.failed_ids[:10],
                }
        if not success:
            tail = [ln for ln in text.splitlines() if ln.strip()][-8:]
            info["tail"] = red("\n".join(tail), 800)
        if step.native in ("search", "list"):
            info["matches"] = text.count("\n")
            info["sample"] = red("\n".join(text.splitlines()[:10]), 800)
        return success, info, refs, len(text)

    def _native(
        self, step: WorkflowStep, argv: list[str], root: str
    ) -> tuple[str, bool, int | None]:
        if step.native == "read":
            path = resolve(argv[1], cwd=root)
            try:
                with open(path, encoding="utf-8", errors="replace") as fh:
                    body = fh.read(2_000_000)
                return f"{body.count(chr(10)) + 1} lines", True, 0
            except OSError as exc:
                return f"<tool_use_error>{exc}</tool_use_error>", False, None
        pattern, scope = argv[1], resolve(argv[2], cwd=root)
        if step.native == "list":
            import glob as _glob

            listed = sorted(_glob.glob(os.path.join(scope, pattern), recursive=True))[:200]
            return "\n".join(relative_to_root(h, root) for h in listed), True, 0
        try:
            rx = re.compile(pattern)
        except re.error as exc:
            return f"<tool_use_error>invalid pattern: {exc}</tool_use_error>", False, None
        hits: list[str] = []
        for dirpath, dirnames, filenames in os.walk(scope):
            dirnames[:] = [
                d for d in dirnames if d not in _PROTECTED_PARTS and not d.startswith(".")
            ]
            for f in filenames:
                p = os.path.join(dirpath, f)
                try:
                    with open(p, encoding="utf-8", errors="strict") as fh:
                        for n, line in enumerate(fh, 1):
                            if rx.search(line):
                                hits.append(
                                    f"{relative_to_root(p, root)}:{n}:{line.rstrip()[:160]}"
                                )
                                if len(hits) >= 200:
                                    break
                except (OSError, UnicodeDecodeError):
                    continue
                if len(hits) >= 200:
                    break
        return "\n".join(hits), True, 0

    def _record_run(
        self,
        macro: WorkflowMacro,
        run_id: str,
        status: str,
        done: int,
        failed_step: str,
        duration: float,
        raw_chars: int,
    ) -> None:
        now = time.time()
        turns = max(0, len(macro.steps) - 1)
        tokens = max(0, raw_chars // 4 - 150)

        def run(c: Any) -> None:
            c.execute(
                "INSERT INTO workflow_runs(run_id, macro_id, session_key, status, steps_done, steps_total, duration_ms, failed_step, ts) VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    run_id,
                    macro.macro_id,
                    self.rt.session_key,
                    status,
                    done,
                    len(macro.steps),
                    duration,
                    failed_step or None,
                    now,
                ),
            )
            if status == "success":
                c.execute(
                    "UPDATE workflow_macros SET last_used_at = ?, turns_saved = turns_saved + ?, tokens_saved = tokens_saved + ? WHERE macro_id = ?",
                    (now, turns, tokens, macro.macro_id),
                )
            else:
                c.execute(
                    "UPDATE workflow_macros SET last_used_at = ? WHERE macro_id = ?",
                    (now, macro.macro_id),
                )
            recent = [
                r[0]
                for r in c.execute(
                    "SELECT status FROM workflow_runs WHERE macro_id = ? ORDER BY ts DESC LIMIT ?",
                    (macro.macro_id, DISABLE_WINDOW),
                ).fetchall()
            ]
            total = c.execute(
                "SELECT COUNT(*), SUM(status = 'success') FROM workflow_runs WHERE macro_id = ?",
                (macro.macro_id,),
            ).fetchone()
            rate = (total[1] or 0) / total[0] if total and total[0] else 1.0
            c.execute(
                "UPDATE workflow_macros SET success_rate = ? WHERE macro_id = ?",
                (rate, macro.macro_id),
            )
            if sum(1 for s in recent if s != "success") >= DISABLE_FAILURES:
                c.execute(
                    "UPDATE workflow_macros SET disabled_reason = 'failure_policy' WHERE macro_id = ?",
                    (macro.macro_id,),
                )

        self.store.write(run)
        m = self.rt.metrics
        m.bump("workflow_invocations")
        m.bump("workflow_steps_collapsed", done)
        if status == "success":
            m.bump("workflow_turns_saved_estimate", turns)
        else:
            m.bump("workflow_failures")
        event = AgentEvent(
            event_id=stable_id(run_id, "WORKFLOW_RUN"),
            workspace_id=self.rt.workspace.workspace_id,
            session_id=self.rt.session_key,
            task_id=self.rt.task_id,
            event_type=EventType.WORKFLOW_RUN,
            success=status == "success",
            metadata={
                "macro": macro.name,
                "status": status,
                "steps": done,
                "duration_ms": int(duration),
                "origin": "headroom",
            },
        )
        self.rt.persist_events([event])
        row = self.store.query_one(
            "SELECT disabled_reason FROM workflow_macros WHERE macro_id = ?", (macro.macro_id,)
        )
        if row is not None and row["disabled_reason"] == "failure_policy":
            m.bump("workflow_auto_disabled")


def format_result(result: dict[str, Any]) -> str:
    """Compact structured text for the model (raw step outputs stay out)."""
    lines = [
        f"macro: {result.get('macro')}",
        f"status: {result.get('status')}",
        f"steps: {result.get('steps')}",
    ]
    v = result.get("verification")
    if v:
        lines.append("verification:")
        for k in ("status", "passed", "failed", "tests_run", "tiers", "failed_ids"):
            if k in v and v[k] not in (None, [], ""):
                lines.append(f"  {k}: {v[k]}")
    if "failed_step" in result:
        lines.append(f"failed_step: {result['failed_step']}")
        for d in result.get("details", []):
            if d.get("tail"):
                lines.append("failure_tail: |")
                lines.extend("  " + ln for ln in str(d["tail"]).splitlines())
                break
    for d in result.get("details", []):
        if d.get("sample"):
            lines.append(f"{d['step']} matches: {d.get('matches')}")
            lines.extend("  " + ln for ln in str(d["sample"]).splitlines()[:10])
    lines.append(f"changed_files: {result.get('changed_files', 0)}")
    lines.append(f"duration_ms: {result.get('duration_ms')}")
    if result.get("evidence_refs"):
        lines.append(f"evidence_refs: [{', '.join(result['evidence_refs'])}]")
    return "\n".join(lines)
