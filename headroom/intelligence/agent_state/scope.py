"""Feature 23: Change-Scope / Drift Firewall (plan §8).

This keeps edits aligned with the active task without becoming a path
allowlist. A :class:`ChangeContract` is compiled from task authority, in
descending order (§8.3):

1. explicit user-named paths;
2. explicit user exclusions;
3. hard constraints;
4. acceptance criteria;
5. the dirty-tree baseline;
6. graph neighbours;
7. files that build/test failures prove necessary;
8. reads.

Reading a file never authorizes writing it.

Each proposed mutation is classified ``IN_SCOPE``, ``DEPENDENCY_SCOPE``,
``GENERATED_EFFECT``, ``UNRELATED``, ``FORBIDDEN`` or ``AMBIGUOUS``. Only
deterministic ``FORBIDDEN`` cases can hard-block, and only through a real
pre-execution hook in protect mode:

* writes outside the workspace (symlinks and junctions resolved first);
* explicit user exclusions;
* ``.git`` internals;
* installed dependencies;
* credential files the user did not name;
* generated artifacts that git confirms are ignored.

Everything else at most warns, deduplicated per
``(contract revision, path, reason)``.

The git baseline taken at task start separates pre-existing user edits from
task-owned changes by hash, so the agent is never blamed for a file that was
already dirty unless that file changed after the task began.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import time
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any

from headroom.redaction import is_secret_path

from .contracts import Finding, RuleResult
from .events import AgentEvent, EventType
from .families import Family, ToolInvocation
from .ids import stable_id
from .paths import relative_to_root, resolve, top_component, within
from .proc import git
from .results import parse_diagnostics
from .store import dumps, loads, red

logger = logging.getLogger(__name__)


class ScopeClass(str, Enum):
    IN_SCOPE = "IN_SCOPE"
    DEPENDENCY_SCOPE = "DEPENDENCY_SCOPE"
    GENERATED_EFFECT = "GENERATED_EFFECT"
    UNRELATED = "UNRELATED"
    FORBIDDEN = "FORBIDDEN"
    AMBIGUOUS = "AMBIGUOUS"


class ContractMode(str, Enum):
    LEARNING = "LEARNING"
    ACTIVE = "ACTIVE"
    FROZEN = "FROZEN"


EXPANSION_REASONS = frozenset(
    {
        "COMPILER_DEPENDENCY",
        "TEST_DEPENDENCY",
        "INTERFACE_DEPENDENCY",
        "PLATFORM_REQUIREMENT",
        "CONFIG_REQUIREMENT",
        "USER_EXPANSION",
        "GRAPH_DIRECT_DEPENDENCY",
    }
)
BUDGETS = {"small": (8, 400), "normal": (20, 1500), "large": (None, None)}
_VENDOR_DIRS = ("node_modules", ".venv", "venv", "site-packages", ".tox", "bower_components")
_GENERATED_DIRS = ("build", "dist", "target", "out", "__pycache__", ".next", ".nuxt", "coverage")
_SOFT_VENDOR = ("vendor", "third_party", "thirdparty", "external")
_LOCKFILES = frozenset(
    {
        "package-lock.json",
        "yarn.lock",
        "pnpm-lock.yaml",
        "bun.lockb",
        "bun.lock",
        "Cargo.lock",
        "poetry.lock",
        "uv.lock",
        "Pipfile.lock",
        "go.sum",
        "composer.lock",
        "Gemfile.lock",
        "packages.lock.json",
    }
)
_DEP_TASK_RE = re.compile(
    r"(?i)\b(?:dependenc|upgrade|bump|install|package|lockfile|version of|pin )"
)
_EXCLUSION_RE = re.compile(
    r"(?i)(?:\b(?:do not|don't|dont|never|must not|should not|without)\s+(?:touch|touching|modify|modifying|change|changing|edit|editing|alter|altering|rewrite|rewriting)"
    r"|\bleave\b.*\b(?:alone|as is|untouched|unchanged)|\bout of scope\b|\bnot in scope\b|\bexcluded?\b)"
)
_PATH_RE = re.compile(
    r"(?<![\w@/.])(\.[A-Za-z][\w-]*(?:\.[\w-]+)*(?![\w/])|"
    r"(?:\.{0,2}/)?(?:[\w.-]+/)+[\w.*-]*|[\w-]+(?:\.[\w-]+)*\.(?:py|rs|ts|tsx|js|jsx|mjs|cjs|go|c|cc|cpp|h|hpp|java|kt|rb|cs|toml|json|ya?ml|md|cfg|ini|txt|sh|ps1|cmake|sql|proto))(?![\w])"
)
_SMALL_RE = re.compile(
    r"(?i)\b(?:fix|bug|typo|error|crash|regression|broken|patch|hotfix|one-line|small)\b"
)
_LARGE_RE = re.compile(
    r"(?i)\b(?:refactor|rewrite|migrat|restructur|overhaul|rename across|redesign|port )"
)
STATUS_TTL_S = 5.0
_TEST_NAME_RE = re.compile(
    r"(?i)(?:^|/)(?:tests?|spec|__tests__)/|(?:^|/)test_[^/]+$|_test\.\w+$|\.(?:test|spec)\.\w+$"
)


@dataclass(frozen=True)
class ChangeContract:
    contract_id: str
    task_id: str
    revision: int
    goal: str
    explicit_in_scope_paths: tuple[str, ...] = ()
    explicit_out_of_scope_paths: tuple[str, ...] = ()
    protected_paths: tuple[str, ...] = ()
    expected_subsystems: tuple[str, ...] = ()
    permitted_dependency_radius: int = 2
    allowed_operation_classes: tuple[str, ...] = (
        "READ",
        "SEARCH",
        "WRITE",
        "DELETE",
        "EXECUTE",
        "BUILD",
        "TEST",
    )
    max_change_budget: tuple[int | None, int | None] = (20, 1500)
    required_preservations: tuple[str, ...] = ()
    confidence: float = 0.5
    mode: ContractMode = ContractMode.LEARNING
    task_class: str = "normal"
    explicit_symbols: tuple[str, ...] = ()
    dependency_task: bool = False

    def to_json(self) -> dict[str, Any]:
        d = {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.__dict__.items()}
        d["mode"] = self.mode.value
        return d

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> ChangeContract:
        kw = dict(d)
        kw["mode"] = ContractMode(kw.get("mode", "LEARNING"))
        for k, v in list(kw.items()):
            if isinstance(v, list):
                kw[k] = tuple(v)
        return cls(**{k: v for k, v in kw.items() if k in cls.__dataclass_fields__})


@dataclass
class Classification:
    scope: ScopeClass
    reason: str
    deterministic: bool = False
    expansion: str = ""
    evidence_ids: tuple[str, ...] = field(default_factory=tuple)


def extract_paths(text: str) -> list[str]:
    out = []
    for m in _PATH_RE.finditer(text or ""):
        p = m.group(1).strip().rstrip(".,;:)")
        if p.startswith(("http", "www.")) or "://" in p or len(p) < 3:
            continue
        if re.fullmatch(r"[\d.]+", p):
            continue
        out.append(p.lstrip("./") if p.startswith("./") else p)
    return list(dict.fromkeys(out))


def _file_sha(path: str) -> str:
    try:
        if os.path.getsize(path) > 4 * 1024 * 1024:
            return f"size:{os.path.getsize(path)}:{int(os.path.getmtime(path))}"
        with open(path, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()[:24]
    except OSError:
        return ""


class ScopeFirewall:
    def __init__(self, runtime: Any) -> None:
        self.rt = runtime
        self.store = runtime.store
        self.contract: ChangeContract | None = None
        self._ignored_cache: dict[str, bool] = {}
        self._status_cache: tuple[float, list[tuple[str, str]]] | None = None
        self._load()

    # ---------------------------------------------------------- lifecycle
    @property
    def root(self) -> str:
        return str(self.rt.workspace.root)

    def _task(self) -> Any:
        ts = self.rt.task_state
        return ts.state if ts is not None else None

    def _load(self) -> None:
        task_id = self.rt.task_id
        if not task_id:
            return
        row = self.store.query_one(
            "SELECT contract_json FROM change_contracts WHERE task_id = ?", (task_id,)
        )
        if row is not None:
            try:
                self.contract = ChangeContract.from_json(loads(row["contract_json"], {}))
            except (TypeError, ValueError, KeyError):
                self.contract = None

    def refresh(self) -> None:
        task_id = self.rt.task_id
        if task_id and (self.contract is None or self.contract.task_id != task_id):
            self._load()
            if self.contract is None:
                self.on_task_update()

    def on_new_task(self) -> None:
        self.contract = None
        self._snapshot_baseline()
        self.on_task_update()

    def _snapshot_baseline(self) -> None:
        task_id = self.rt.task_id
        if not task_id or not self.rt.workspace.local:
            return
        if (
            self.store.query_one("SELECT 1 FROM git_baselines WHERE task_id = ?", (task_id,))
            is not None
        ):
            return
        head = branch = ""
        dirty: dict[str, str] = {}
        untracked: list[str] = []
        if git(self.root, "rev-parse", "--git-dir").ok:
            res = git(self.root, "rev-parse", "HEAD")
            head = res.stdout.strip() if res.ok else ""  # empty for a repo with no commits
            b = git(self.root, "rev-parse", "--abbrev-ref", "HEAD")
            branch = b.stdout.strip() if b.ok else ""
            for code, path in self._status_entries(force=True):
                if code == "??":
                    untracked.append(path)
                if len(dirty) < 5000:
                    dirty[path] = _file_sha(os.path.join(self.root, path))
        self.store.write(
            lambda c: c.execute(
                "INSERT OR IGNORE INTO git_baselines(task_id, head, branch, dirty_json, untracked_json, created_at) "
                "VALUES (?,?,?,?,?,?)",
                (task_id, head, branch, dumps(dirty), dumps(untracked[:2000]), time.time()),
            )
        )

    def baseline(self) -> dict[str, Any]:
        task_id = self.rt.task_id
        row = (
            self.store.query_one("SELECT * FROM git_baselines WHERE task_id = ?", (task_id,))
            if task_id
            else None
        )
        if row is None:
            return {"head": "", "branch": "", "dirty": {}, "untracked": []}
        return {
            "head": row["head"],
            "branch": row["branch"],
            "dirty": loads(row["dirty_json"], {}) or {},
            "untracked": loads(row["untracked_json"], []) or [],
        }

    def on_task_update(self) -> None:
        state = self._task()
        if state is None:
            return
        prev = self.contract
        if prev is not None and prev.task_id == state.task_id and prev.revision == state.revision:
            return
        user_atoms = [
            a
            for a in state.atoms
            if a.origin.value == "USER" and a.state.value not in ("SUPERSEDED", "REJECTED")
        ]
        in_scope: list[str] = []
        out_scope: list[str] = []
        symbols: list[str] = []
        preserve: list[str] = []
        from headroom.intelligence.task_context import extract_entities

        for atom in user_atoms:
            paths = [self._norm_rel(p) for p in extract_paths(atom.text)]
            paths = [p for p in paths if p]
            if atom.kind.value == "CONSTRAINT" and _EXCLUSION_RE.search(atom.text):
                out_scope.extend(paths)
                preserve.append(atom.text[:160])
                continue
            if atom.kind.value == "CONSTRAINT" and atom.hard:
                preserve.append(atom.text[:160])
            in_scope.extend(paths)
            ents = extract_entities(atom.text)
            symbols.extend(s for s in ents.symbols if len(s) > 3)
        goal = state.primary_goal.text if state.primary_goal else ""
        all_text = " ".join(a.text for a in user_atoms)
        task_class = (
            "large" if _LARGE_RE.search(goal) else ("small" if _SMALL_RE.search(goal) else "normal")
        )
        in_scope = [p for p in dict.fromkeys(in_scope) if p not in out_scope]
        subsystems = list(dict.fromkeys(self._subsystem(p) for p in in_scope if self._subsystem(p)))
        mode = (
            prev.mode
            if prev is not None and prev.task_id == state.task_id
            else ContractMode.LEARNING
        )
        if mode is ContractMode.LEARNING and in_scope:
            mode = ContractMode.ACTIVE
        if (
            prev is not None
            and prev.task_id == state.task_id
            and prev.mode is not ContractMode.LEARNING
        ):
            subsystems = list(dict.fromkeys([*prev.expected_subsystems, *subsystems]))
        if state.status.value == "COMPLETED":
            mode = ContractMode.FROZEN
        contract = ChangeContract(
            contract_id=stable_id(state.task_id, state.revision, n=12),
            task_id=state.task_id,
            revision=state.revision,
            goal=goal[:200],
            explicit_in_scope_paths=tuple(in_scope[:64]),
            explicit_out_of_scope_paths=tuple(dict.fromkeys(out_scope))[:64],
            protected_paths=(".git/", *[f"{d}/" for d in _VENDOR_DIRS]),
            expected_subsystems=tuple(subsystems[:16]),
            max_change_budget=BUDGETS[task_class],
            required_preservations=tuple(preserve[:8]),
            confidence=0.9 if in_scope else 0.5,
            mode=mode,
            task_class=task_class,
            explicit_symbols=tuple(dict.fromkeys(symbols))[:32],
            dependency_task=bool(_DEP_TASK_RE.search(all_text)),
        )
        self._save(contract)

    def _save(self, contract: ChangeContract) -> None:
        self.contract = contract
        payload = dumps(contract.to_json())
        self.store.write(
            lambda c: c.execute(
                "INSERT OR REPLACE INTO change_contracts(task_id, revision, contract_json, updated_at) VALUES (?,?,?,?)",
                (contract.task_id, contract.revision, payload, time.time()),
            )
        )

    def freeze(self) -> None:
        if self.contract is not None and self.contract.mode is not ContractMode.FROZEN:
            self._save(replace(self.contract, mode=ContractMode.FROZEN))

    def _activate(self, subsystem: str) -> None:
        c = self.contract
        if c is None:
            return
        subs = (
            tuple(dict.fromkeys([*c.expected_subsystems, subsystem]))
            if subsystem
            else c.expected_subsystems
        )
        self._save(
            replace(
                c,
                mode=ContractMode.ACTIVE,
                expected_subsystems=subs,
                confidence=max(c.confidence, 0.7),
            )
        )

    # ----------------------------------------------------------- helpers
    def _norm_rel(self, path: str) -> str:
        p = path.replace("\\", "/").strip()
        if not p:
            return ""
        if os.path.isabs(p) or re.match(r"^[A-Za-z]:/", p):
            rel = relative_to_root(p, self.root)
            return rel if rel != p else ""
        return p.lstrip("./") if p.startswith("./") else p

    def _subsystem(self, rel: str) -> str:
        parts = [p for p in rel.split("/") if p]
        if len(parts) <= 1:
            return ""
        depth = 2 if len(parts) > 2 else 1
        return top_component(rel, depth)

    def _under(self, rel: str, prefixes: tuple[str, ...] | list[str]) -> str:
        r = rel.casefold() if os.name == "nt" else rel
        for p in prefixes:
            q = p.rstrip("/")
            qq = q.casefold() if os.name == "nt" else q
            if not qq:
                continue
            if r == qq or r.startswith(qq + "/"):
                return p
            if "/" not in qq and os.path.basename(r) == qq:
                return p  # a bare filename the user named
        return ""

    def _is_ignored(self, rel: str) -> bool:
        if rel in self._ignored_cache:
            return self._ignored_cache[rel]
        ignored = False
        if self.rt.workspace.local:
            res = git(self.root, "check-ignore", "-q", "--", rel, timeout=2.0)
            ignored = res.returncode == 0
        self._ignored_cache[rel] = ignored
        if len(self._ignored_cache) > 4096:
            self._ignored_cache.clear()
        return ignored

    def _expansion(self, rel: str) -> tuple[str, tuple[str, ...]] | None:
        task_id = self.rt.task_id
        if not task_id:
            return None
        row = self.store.query_one(
            "SELECT reason_code, evidence_ids FROM scope_expansions WHERE task_id = ? AND path = ?",
            (task_id, rel),
        )
        if row is None:
            return None
        return row["reason_code"], tuple(loads(row["evidence_ids"], []) or ())

    def _owned(self, rel: str) -> str:
        task_id = self.rt.task_id
        if not task_id:
            return ""
        row = self.store.query_one(
            "SELECT classification FROM task_changes WHERE task_id = ? AND path = ?", (task_id, rel)
        )
        return row["classification"] if row is not None else ""

    def _graph_related(self, rel: str) -> bool:
        graph = self.rt.graph()
        c = self.contract
        if graph is None or c is None or not (c.explicit_in_scope_paths or c.explicit_symbols):
            return False
        try:
            files, _ = graph.neighborhood(
                files=list(c.explicit_in_scope_paths), symbols=list(c.explicit_symbols)
            )
        except Exception:  # noqa: BLE001
            return False
        rel_low = rel.lower()
        return any(
            rel_low.endswith(str(f).replace("\\", "/").lower().lstrip("/"))
            or str(f).replace("\\", "/").lower().endswith(rel_low)
            for f in files
        )

    def _test_of_target(self, rel: str) -> bool:
        c = self.contract
        if c is None or not _TEST_NAME_RE.search(rel):
            return False
        name = os.path.basename(rel).lower()
        for target in c.explicit_in_scope_paths:
            stem = os.path.splitext(os.path.basename(target))[0].lower()
            if stem and len(stem) > 2 and stem in name:
                return True
        return False

    # ------------------------------------------------------- classification
    def classify_path(
        self, path: str, *, op: str, inv: ToolInvocation | None = None
    ) -> Classification:
        cwd = (inv.cwd if inv is not None else "") or self.root
        full = resolve(path, cwd=cwd)
        root = resolve(self.root)
        c = self.contract
        rel = relative_to_root(full, root)
        explicitly_named = c is not None and bool(self._under(rel, c.explicit_in_scope_paths))
        if not within(full, root):
            named_abs = c is not None and any(
                full.replace("\\", "/").endswith(p) for p in c.explicit_in_scope_paths
            )
            if named_abs:
                return Classification(ScopeClass.AMBIGUOUS, "OUTSIDE_PROJECT_NAMED_BY_USER")
            return Classification(ScopeClass.FORBIDDEN, "OUTSIDE_PROJECT", deterministic=True)
        if op == "delete" and rel in (".", ""):
            return Classification(ScopeClass.FORBIDDEN, "DELETES_PROJECT_ROOT", deterministic=True)
        if c is not None and self._under(rel, c.explicit_out_of_scope_paths):
            return Classification(ScopeClass.FORBIDDEN, "USER_EXCLUDED", deterministic=True)
        parts = rel.split("/")
        if ".git" in parts[:-1] or rel == ".git":
            return Classification(ScopeClass.FORBIDDEN, "GIT_INTERNALS", deterministic=True)
        if is_secret_path(rel) and not explicitly_named:
            return Classification(ScopeClass.FORBIDDEN, "SECRET_FILE", deterministic=True)
        if any(d in parts[:-1] for d in _VENDOR_DIRS) and not explicitly_named:
            return Classification(ScopeClass.FORBIDDEN, "INSTALLED_DEPENDENCY", deterministic=True)
        if parts and parts[0] in _GENERATED_DIRS and not explicitly_named:
            ignored = self._is_ignored(rel)
            if inv is not None and inv.family is Family.BUILD:
                return Classification(ScopeClass.GENERATED_EFFECT, "BUILD_OUTPUT")
            if ignored:
                return Classification(
                    ScopeClass.FORBIDDEN, "GENERATED_ARTIFACT", deterministic=True
                )
        if explicitly_named:
            return Classification(ScopeClass.IN_SCOPE, "USER_NAMED")
        exp = self._expansion(rel)
        if exp is not None:
            return Classification(
                ScopeClass.DEPENDENCY_SCOPE, exp[0], expansion=exp[0], evidence_ids=exp[1]
            )
        owned = self._owned(rel)
        if owned in (ScopeClass.IN_SCOPE.value, ScopeClass.DEPENDENCY_SCOPE.value):
            return Classification(ScopeClass(owned), "ALREADY_IN_TASK")
        if os.path.basename(rel) in _LOCKFILES:
            if c is not None and c.dependency_task:
                return Classification(
                    ScopeClass.DEPENDENCY_SCOPE,
                    "CONFIG_REQUIREMENT",
                    expansion="CONFIG_REQUIREMENT",
                )
            return Classification(ScopeClass.AMBIGUOUS, "LOCKFILE_WITHOUT_DEPENDENCY_TASK")
        if c is None:
            return Classification(ScopeClass.AMBIGUOUS, "NO_CONTRACT")
        sub = self._subsystem(rel)
        if sub and sub in c.expected_subsystems:
            return Classification(ScopeClass.IN_SCOPE, "EXPECTED_SUBSYSTEM")
        if self._test_of_target(rel):
            return Classification(
                ScopeClass.DEPENDENCY_SCOPE, "TEST_DEPENDENCY", expansion="TEST_DEPENDENCY"
            )
        if self._graph_related(rel):
            return Classification(
                ScopeClass.DEPENDENCY_SCOPE,
                "GRAPH_DIRECT_DEPENDENCY",
                expansion="GRAPH_DIRECT_DEPENDENCY",
            )
        if parts and parts[0] in _SOFT_VENDOR:
            return Classification(ScopeClass.AMBIGUOUS, "VENDORED_SOURCE")
        if c.mode is ContractMode.LEARNING:
            return Classification(ScopeClass.IN_SCOPE, "LEARNING_CANDIDATE")
        goal_words = set(re.findall(r"[a-z0-9]{4,}", c.goal.lower()))
        path_words = set(re.findall(r"[a-z0-9]{4,}", rel.lower()))
        if goal_words & path_words:
            return Classification(ScopeClass.AMBIGUOUS, "WEAK_RELATION")
        return Classification(ScopeClass.UNRELATED, "NO_SUPPORTED_RELATION")

    def classify_proposed(self, ev: AgentEvent, *, source: str) -> tuple[list[Finding], str]:
        inv: ToolInvocation | None = ev.transient.get("invocation")
        if inv is None or self.rt.config.scope.value == "off":
            return [], ""
        findings: list[Finding] = []
        worst = ""
        rank = {
            s: i
            for i, s in enumerate(
                [
                    "IN_SCOPE",
                    "GENERATED_EFFECT",
                    "DEPENDENCY_SCOPE",
                    "AMBIGUOUS",
                    "UNRELATED",
                    "FORBIDDEN",
                ]
            )
        }
        for op, paths in (("write", inv.paths_written), ("delete", inv.paths_deleted)):
            for p in paths:
                if not p or re.search(r"[*?$%`]", p):
                    continue
                cls = self.classify_path(p, op=op, inv=inv)
                if cls.scope is ScopeClass.UNRELATED or cls.scope is ScopeClass.AMBIGUOUS:
                    cls = self._jevk5_suppress(p, cls)
                rel = relative_to_root(resolve(p, cwd=inv.cwd or self.root), resolve(self.root))
                self.rt.metrics.bump(f"scope_{cls.scope.value.lower()}")
                if not worst or rank[cls.scope.value] > rank[worst]:
                    worst = cls.scope.value
                if cls.scope is ScopeClass.FORBIDDEN:
                    findings.append(
                        Finding(
                            "scope:forbidden",
                            RuleResult.BLOCK,
                            cls.reason,
                            f"{op} of `{rel}` is out of bounds ({cls.reason})",
                            deterministic=cls.deterministic,
                        )
                    )
                    self._warn(
                        rel,
                        cls,
                        f"Out-of-bounds {op} `{rel}`: {cls.reason}.",
                        delivered=self._hook_delivers(source, cls),
                    )
                elif cls.scope is ScopeClass.UNRELATED:
                    findings.append(
                        Finding(
                            "scope:unrelated",
                            RuleResult.WARN,
                            cls.reason,
                            f"`{rel}` has no supported relation to the active task",
                            deterministic=False,
                        )
                    )
                    self._warn(
                        rel,
                        cls,
                        f"Edit `{rel}` has no supported relation to the active task. Do not continue this change unless new evidence establishes a dependency.",
                        delivered=self._hook_delivers(source, cls),
                    )
                elif cls.scope is ScopeClass.AMBIGUOUS and cls.reason in (
                    "LOCKFILE_WITHOUT_DEPENDENCY_TASK",
                    "VENDORED_SOURCE",
                    "OUTSIDE_PROJECT_NAMED_BY_USER",
                ):
                    findings.append(
                        Finding(
                            "scope:ambiguous",
                            RuleResult.WARN,
                            cls.reason,
                            f"`{rel}`: {cls.reason}",
                            deterministic=False,
                        )
                    )
                    self._warn(
                        rel,
                        cls,
                        f"`{rel}` is outside the expected change scope ({cls.reason}).",
                        delivered=self._hook_delivers(source, cls),
                    )
                elif (
                    self.contract is not None
                    and self.contract.mode is ContractMode.FROZEN
                    and not self._owned(rel)
                ):
                    findings.append(
                        Finding(
                            "scope:frozen",
                            RuleResult.WARN,
                            "FROZEN_NEW_EDIT",
                            f"`{rel}` is a new edit after the task was declared complete",
                            deterministic=False,
                        )
                    )
                    self._warn(
                        rel,
                        Classification(ScopeClass.AMBIGUOUS, "FROZEN_NEW_EDIT"),
                        f"Task was complete; `{rel}` is a new edit. Confirm it is required.",
                        delivered=self._hook_delivers(source, cls),
                    )
        return findings, worst

    def _jevk5_suppress(self, path: str, cls: Classification) -> Classification:
        from headroom.intelligence.models import DecisionFamily

        from .advice import classify

        c = self.contract
        if c is None:
            return cls
        answer = classify(
            self.rt.advisor(),
            DecisionFamily.SCOPE_NECESSITY,
            f"Active task: {c.goal}\nIn-scope: {', '.join(c.explicit_in_scope_paths[:8])}\nProposed change: {self._norm_rel(path) or path}",
            "Is this change necessary for the active task?",
            {
                "A": "clearly necessary",
                "B": "plausible dependency",
                "C": "unrelated",
                "D": "insufficient evidence",
            },
        )
        if answer is not None and answer[0] in ("A", "B") and answer[1] >= 0.75:
            # JevK5 may suppress a warning; it can never authorize a FORBIDDEN path.
            return Classification(ScopeClass.AMBIGUOUS, "JEVK5_PLAUSIBLE_DEPENDENCY")
        return cls

    def _warn(
        self, rel: str, cls: Classification, message: str, *, delivered: bool = False
    ) -> None:
        """Record a warning once per (contract revision, path, reason).

        ``delivered`` warnings already reached the model through the host hook
        (a deny reason or Claude Code ``additionalContext``) and are kept only
        for diagnostics, not re-injected.
        """
        c = self.contract
        task_id = self.rt.task_id or "none"
        revision = c.revision if c is not None else 0
        now = time.time()
        self.store.write(
            lambda conn: conn.execute(
                "INSERT OR IGNORE INTO scope_warnings(task_id, contract_revision, path, reason, classification, message, delivered, created_at) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (
                    task_id,
                    revision,
                    rel,
                    cls.reason,
                    cls.scope.value,
                    red(message, 400),
                    int(delivered),
                    now,
                ),
            )
        )

    def _hook_delivers(self, source: str, cls: Classification) -> bool:
        if source == "workflow":
            return True
        if source != "hook":
            return False
        if self.rt.agent == "claude_code":
            return True
        from .config import EnforcementMode

        return (
            cls.scope is ScopeClass.FORBIDDEN
            and cls.deterministic
            and self.rt.config.scope_mode is EnforcementMode.PROTECT
            and (self.rt.can_block() or self.rt.block_unverified())
        )

    # ------------------------------------------------------------ results
    def on_tool_result(self, ev: AgentEvent, records: list[Any]) -> None:
        inv: ToolInvocation | None = ev.transient.get("invocation")
        if inv is None:
            return
        if ev.success is not False and (inv.paths_written or inv.paths_deleted):
            self._record_changes(ev, inv)
        if inv.family in (Family.TEST, Family.BUILD) and ev.success is False:
            self._expand_from_failure(ev, inv, records)

    def _record_changes(self, ev: AgentEvent, inv: ToolInvocation) -> None:
        self._status_cache = None
        task_id = self.rt.task_id
        if not task_id:
            return
        root = resolve(self.root)
        lines = _lines_changed(ev.transient.get("input") or {}, inv)
        now = time.time()
        violations = []
        for op, paths in (("write", inv.paths_written), ("delete", inv.paths_deleted)):
            for p in paths:
                if not p or re.search(r"[*?$%`]", p):
                    continue
                full = resolve(p, cwd=inv.cwd or self.root)
                rel = relative_to_root(full, root)
                cls = self.classify_path(p, op=op, inv=inv)
                created = (
                    0 if self._owned(rel) else int(op == "write" and not self._in_baseline(rel))
                )
                if cls.scope is ScopeClass.FORBIDDEN:
                    violations.append((rel, cls))
                self.store.write(
                    lambda c, rel=rel, cls=cls, created=created: c.execute(
                        "INSERT INTO task_changes(task_id, path, classification, reason, first_event_id, last_event_id, "
                        "lines_changed, created, updated_at) VALUES (?,?,?,?,?,?,?,?,?) "
                        "ON CONFLICT(task_id, path) DO UPDATE SET last_event_id = excluded.last_event_id, "
                        "lines_changed = task_changes.lines_changed + excluded.lines_changed, updated_at = excluded.updated_at",
                        (
                            task_id,
                            rel,
                            cls.scope.value,
                            cls.reason,
                            ev.event_id,
                            ev.event_id,
                            lines,
                            created,
                            now,
                        ),
                    )
                )
                if cls.expansion and cls.expansion in EXPANSION_REASONS:
                    self.add_expansion(rel, cls.expansion, ev, cls.evidence_ids)
                c = self.contract
                if (
                    c is not None
                    and c.mode is ContractMode.LEARNING
                    and cls.reason == "LEARNING_CANDIDATE"
                ):
                    self._maybe_activate(rel)
        for rel, cls in violations:
            self._violation(ev, rel, cls)
        self._check_budget()

    def _in_baseline(self, rel: str) -> bool:
        base = self.baseline()
        return rel in base["dirty"]

    def _maybe_activate(self, rel: str) -> None:
        """LEARNING -> ACTIVE on the first edit related to the goal, or two edits in one subsystem."""
        c = self.contract
        if c is None:
            return
        sub = self._subsystem(rel) or os.path.dirname(rel)
        goal = c.goal.lower()
        stem = os.path.splitext(os.path.basename(rel))[0].lower()
        related = (len(stem) > 3 and stem in goal) or any(
            part.lower() in goal for part in rel.split("/")[:-1] if len(part) > 3
        )
        if related:
            self._activate(sub)
            return
        rows = self.store.query("SELECT path FROM task_changes WHERE task_id = ?", (c.task_id,))
        same = [
            r["path"]
            for r in rows
            if (self._subsystem(r["path"]) or os.path.dirname(r["path"])) == sub
        ]
        if len(same) >= 2:
            self._activate(sub)

    def _violation(self, ev: AgentEvent, rel: str, cls: Classification) -> None:
        self.rt.metrics.bump("scope_violations")
        event = AgentEvent(
            event_id=stable_id(ev.event_id, "SCOPE_VIOLATION", rel),
            workspace_id=self.rt.workspace.workspace_id,
            session_id=self.rt.session_key,
            task_id=self.rt.task_id,
            event_type=EventType.SCOPE_VIOLATION,
            path_refs=(rel,),
            metadata={"reason": cls.reason, "class": cls.scope.value},
        )
        self.rt.persist_events([event])
        ts = self.rt.task_state
        if ts is not None and ts.state is not None and cls.reason == "USER_EXCLUDED":
            for atom in ts.state.constraints:
                if rel in " ".join(extract_paths(atom.text)) or any(
                    rel.startswith(self._norm_rel(p).rstrip("/"))
                    for p in extract_paths(atom.text)
                    if self._norm_rel(p)
                ):
                    ts.mark_constraint_violated(ev, atom.atom_id)

    def blocking_violations(self) -> list[str]:
        task_id = self.rt.task_id
        if not task_id:
            return []
        rows = self.store.query(
            "SELECT path, reason FROM task_changes WHERE task_id = ? AND classification = 'FORBIDDEN'",
            (task_id,),
        )
        return [f"{r['path']} ({r['reason']})" for r in rows]

    def _check_budget(self) -> None:
        c = self.contract
        task_id = self.rt.task_id
        if c is None or not task_id:
            return
        max_files, max_lines = c.max_change_budget
        row = self.store.query_one(
            "SELECT COUNT(*) AS n, COALESCE(SUM(lines_changed), 0) AS l FROM task_changes WHERE task_id = ? AND classification != 'GENERATED_EFFECT'",
            (task_id,),
        )
        if row is None:
            return
        files, lines = int(row["n"]), int(row["l"])
        over = (max_files is not None and files > max_files) or (
            max_lines is not None and lines > max_lines
        )
        if over:
            self._warn(
                "*",
                Classification(ScopeClass.AMBIGUOUS, f"BUDGET_{c.task_class.upper()}"),
                f"Change budget exceeded for a {c.task_class} task: {files} files / {lines} lines "
                f"(soft threshold {max_files} files / {max_lines} lines). Re-check that every change is needed.",
            )

    def _expand_from_failure(self, ev: AgentEvent, inv: ToolInvocation, records: list[Any]) -> None:
        """Failing build/test output that points at a file proves it may need to change (§8.10)."""
        text = str(ev.transient.get("text") or "")
        diags = parse_diagnostics(text)
        reason = "COMPILER_DEPENDENCY" if inv.family is Family.BUILD else "TEST_DEPENDENCY"
        evidence_ids = tuple(
            r.evidence_id
            for r in records
            if getattr(r, "claim_type", "") in ("compiler_error", "test_aggregate", "build_status")
        )
        root = resolve(self.root)
        for d in diags[:20]:
            full = resolve(d.path, cwd=inv.cwd or self.root)
            if not within(full, root):
                continue
            rel = relative_to_root(full, root)
            if is_secret_path(rel) or any(part in _VENDOR_DIRS for part in rel.split("/")):
                continue
            cls = self.classify_path(full, op="write", inv=None)
            if cls.scope in (ScopeClass.UNRELATED, ScopeClass.AMBIGUOUS):
                self.add_expansion(rel, reason, ev, evidence_ids, confidence=0.9)

    def add_expansion(
        self,
        rel: str,
        reason: str,
        ev: AgentEvent,
        evidence_ids: tuple[str, ...] = (),
        *,
        confidence: float = 0.85,
    ) -> None:
        if reason not in EXPANSION_REASONS:
            raise ValueError(f"unsupported scope expansion reason {reason!r}")
        task_id = self.rt.task_id
        if not task_id:
            return

        def run(c: Any) -> int:
            cur = c.execute(
                "INSERT OR IGNORE INTO scope_expansions(task_id, path, reason_code, evidence_ids, event_ids, confidence, created_at) "
                "VALUES (?,?,?,?,?,?,?)",
                (
                    task_id,
                    rel,
                    reason,
                    dumps(list(evidence_ids)),
                    dumps([ev.event_id]),
                    confidence,
                    time.time(),
                ),
            )
            return cur.rowcount or 0

        if self.store.write(run, default=0):
            self.rt.metrics.bump("scope_expansions")

    # ---------------------------------------------------------- change set
    def _status_entries(self, *, force: bool = False) -> list[tuple[str, str]]:
        """``git status`` entries (files, renames resolved), cached for a few seconds.

        The cache is dropped whenever a task-owned write is recorded, so a fresh
        mutation is always seen; otherwise repeated requests reuse one call.
        """
        now = time.time()
        cached = self._status_cache
        if not force and cached is not None and now - cached[0] < STATUS_TTL_S:
            return cached[1]
        entries: list[tuple[str, str]] = []
        st = git(self.root, "status", "--porcelain=v1", "-z", "--untracked-files=all")
        if st.ok:
            parts = st.stdout.split("\0")
            i = 0
            while i < len(parts) and len(entries) < 5000:
                entry = parts[i]
                i += 1
                if len(entry) < 4:
                    continue
                code, path = entry[:2], entry[3:]
                if code[0] in "RC":
                    i += 1  # the original path of a rename/copy follows
                entries.append((code, path))
        self._status_cache = (now, entries)
        return entries

    def task_owned_changes(self, *, refresh_git: bool = False) -> list[tuple[str, str]]:
        """``(rel path, classification)`` changed during the task (tool-observed plus git delta)."""
        task_id = self.rt.task_id
        if not task_id:
            return []
        rows = self.store.query(
            "SELECT path, classification FROM task_changes WHERE task_id = ?", (task_id,)
        )
        out = {r["path"]: r["classification"] for r in rows if r["path"] != "*"}
        if refresh_git and self.rt.workspace.local:
            base = self.baseline()
            for _code, path in self._status_entries():
                if path in out:
                    continue
                before = base["dirty"].get(path)
                if before is None or before != _file_sha(os.path.join(self.root, path)):
                    out[path] = "OBSERVED_GIT_CHANGE"
        return sorted(out.items())

    # -------------------------------------------------------------- render
    def sections(self) -> list[tuple[int, str, list[str]]]:
        warnings = self.pending_warnings()
        if not warnings:
            return []
        return [(0, "scope-warning", [f"- {w}" for w in warnings[:3]])]

    def pending_warnings(self) -> list[str]:
        task_id = self.rt.task_id or "none"
        rows = self.store.query(
            "SELECT path, reason, message FROM scope_warnings WHERE task_id = ? AND delivered = 0 ORDER BY created_at LIMIT 5",
            (task_id,),
        )
        return [r["message"] for r in rows]

    def mark_delivered(self) -> None:
        task_id = self.rt.task_id or "none"
        self.store.write(
            lambda c: c.execute(
                "UPDATE scope_warnings SET delivered = 1 WHERE task_id = ?", (task_id,)
            )
        )

    def describe(self) -> dict[str, Any]:
        c = self.contract
        task_id = self.rt.task_id
        expansions = (
            [
                dict(r)
                for r in self.store.query(
                    "SELECT path, reason_code, confidence FROM scope_expansions WHERE task_id = ?",
                    (task_id,),
                )
            ]
            if task_id
            else []
        )
        return {
            "contract": c.to_json() if c is not None else None,
            "changes": self.task_owned_changes(),
            "expansions": expansions,
            "warnings": [
                dict(r)
                for r in self.store.query(
                    "SELECT path, reason, classification, message, delivered FROM scope_warnings WHERE task_id = ? ORDER BY created_at DESC LIMIT 20",
                    (task_id or "none",),
                )
            ],
            "baseline": {
                k: (v if k in ("head", "branch") else len(v)) for k, v in self.baseline().items()
            },
        }


def _lines_changed(raw: dict[str, Any], inv: ToolInvocation) -> int:
    name = inv.tool_name.lower()
    if name in ("edit",):
        return len(str(raw.get("old_string", "")).splitlines()) + len(
            str(raw.get("new_string", "")).splitlines()
        )
    if name == "multiedit":
        return sum(
            len(str(e.get("old_string", "")).splitlines())
            + len(str(e.get("new_string", "")).splitlines())
            for e in raw.get("edits") or []
            if isinstance(e, dict)
        )
    if name in ("write", "notebookedit"):
        return len(str(raw.get("content", raw.get("new_source", ""))).splitlines())
    patch = raw.get("input") if isinstance(raw.get("input"), str) else inv.command
    if patch and "*** " in patch:
        return sum(
            1
            for ln in patch.splitlines()
            if ln.startswith(("+", "-")) and not ln.startswith(("+++", "---"))
        )
    return 0
