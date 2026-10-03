"""Feature 19: Tool Contract and Argument Validator (plan §7).

Tool schemas remain the authority on syntax. A :class:`ToolContract` adds the
semantic and runtime preconditions a schema cannot express: an input path
must exist, a working directory must be a directory, an executable must be
resolvable, a git ref must be well-formed, a test selector must exist, and a
call pattern that failed three times under the same conditions is flagged.

Enforcement is truthful (plan §7.9). A call is *blocked* only when all of the
following hold:

* the mode is ``protect``;
* the finding is deterministic;
* the call arrived through a real host pre-execution hook (Claude Code or
  Codex PreToolUse) whose session has proved the hook is live.

Everywhere else the finding becomes a warning in the next live turn, and the
record says ``warned``/``observed``, never ``blocked``. JevK5 can contribute
at most a WARN.

Safe repairs (§7.8) are limited to semantics-preserving normalizations.
Today that means resolving a relative path against the established project
root for tools that require absolute paths. A repair never changes a target,
ref, selector, intent, privilege, port, or destructive flag.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import tomllib

from .config import EnforcementMode
from .events import AgentEvent
from .families import Family, OpClass, SideEffect, ToolInvocation
from .ids import stable_id
from .paths import is_absolute, join, resolve, within
from .results import failure_signature, tool_use_error
from .store import dumps, loads, red

logger = logging.getLogger(__name__)

CONTRACT_VERSION = 1
LEARN_MIN_FAILURES = 3


class RuleResult(str, Enum):
    PASS = "PASS"
    WARN = "WARN"
    REPAIRABLE = "REPAIRABLE"
    BLOCK = "BLOCK"
    UNKNOWN = "UNKNOWN"


_SEVERITY = {
    RuleResult.PASS: 0,
    RuleResult.UNKNOWN: 1,
    RuleResult.WARN: 2,
    RuleResult.REPAIRABLE: 3,
    RuleResult.BLOCK: 4,
}


@dataclass(frozen=True)
class Finding:
    rule: str
    result: RuleResult
    reason: str
    message: str
    deterministic: bool = True
    repair: dict[str, Any] | None = None


@dataclass
class ValidationOutcome:
    result: RuleResult
    findings: list[Finding]
    enforced: str  # blocked | repaired | warned | observed | allowed
    contract_id: str
    repaired_input: dict[str, Any] | None = None
    latency_ms: float = 0.0
    scope_class: str = ""

    @property
    def message(self) -> str:
        worst = [f for f in self.findings if f.result is self.result]
        return "; ".join(f.message for f in worst[:3])


@dataclass(frozen=True)
class ToolContract:
    contract_id: str
    tool_match: str
    version: int
    operation_class: OpClass
    side_effect: SideEffect
    deterministic_rules: tuple[str, ...]
    repair_rules: tuple[str, ...] = ()
    platform_rules: dict[str, Any] = field(default_factory=dict)
    source: str = "BUILTIN"
    enabled: bool = True
    allow_external_paths: bool = False


BUILTIN_CONTRACTS: dict[Family, ToolContract] = {
    Family.READ_FILE: ToolContract(
        "read_file@1",
        "read_file",
        1,
        OpClass.READ,
        SideEffect.NONE,
        ("input_exists", "range_valid"),
        ("absolute_path",),
        allow_external_paths=True,
    ),
    Family.EDIT_FILE: ToolContract(
        "edit_file@1",
        "edit_file",
        1,
        OpClass.WRITE,
        SideEffect.LOCAL_MUTATING,
        ("input_exists", "edit_effective", "patch_targets", "scope"),
        ("absolute_path",),
    ),
    Family.WRITE_FILE: ToolContract(
        "write_file@1",
        "write_file",
        1,
        OpClass.WRITE,
        SideEffect.LOCAL_MUTATING,
        ("parent_creatable", "scope"),
        ("absolute_path",),
    ),
    Family.DELETE_FILE: ToolContract(
        "delete_file@1",
        "delete_file",
        1,
        OpClass.DELETE,
        SideEffect.LOCAL_MUTATING,
        ("cwd_valid", "executables", "platform", "scope"),
    ),
    Family.SEARCH_TEXT: ToolContract(
        "search_text@1",
        "search_text",
        1,
        OpClass.SEARCH,
        SideEffect.NONE,
        ("search_scope_exists", "pattern_syntax", "cwd_valid", "executables"),
        allow_external_paths=True,
    ),
    Family.LIST_FILES: ToolContract(
        "list_files@1",
        "list_files",
        1,
        OpClass.SEARCH,
        SideEffect.NONE,
        ("search_scope_exists", "pattern_syntax"),
        allow_external_paths=True,
    ),
    Family.SHELL: ToolContract(
        "shell@1",
        "shell",
        1,
        OpClass.EXECUTE,
        SideEffect.LOCAL_MUTATING,
        ("cwd_valid", "executables", "platform", "git", "scope"),
    ),
    Family.BUILD: ToolContract(
        "build@1",
        "build",
        1,
        OpClass.BUILD,
        SideEffect.LOCAL_REVERSIBLE,
        ("cwd_valid", "executables", "platform", "build_target"),
    ),
    Family.TEST: ToolContract(
        "test@1",
        "test",
        1,
        OpClass.TEST,
        SideEffect.NONE,
        ("cwd_valid", "executables", "platform", "test_selector"),
    ),
    Family.GIT: ToolContract(
        "git@1", "git", 1, OpClass.EXECUTE, SideEffect.LOCAL_MUTATING, ("cwd_valid", "git", "scope")
    ),
    Family.NETWORK: ToolContract(
        "network@1", "web/network", 1, OpClass.NETWORK, SideEffect.EXTERNAL, ()
    ),
    Family.MCP: ToolContract("mcp@1", "mcp", 1, OpClass.OTHER, SideEffect.NONE, ("schema",)),
    Family.OTHER: ToolContract("other@1", "other", 1, OpClass.OTHER, SideEffect.NONE, ("schema",)),
}

# Builtins/cmdlets that are never resolved through PATH (plan §7.5).
SHELL_BUILTINS = frozenset(
    "cd chdir pushd popd echo printf export set unset alias unalias source . exit return true false "
    "test [ [[ type command builtin eval exec read shift trap ulimit umask wait jobs fg bg kill let "
    "local declare typeset readonly hash times getopts history dirs shopt enable help logout "
    "set-location sl get-location write-output write-host get-content gc get-childitem gci "
    "select-string sls test-path new-item ni remove-item ri copy-item cpi move-item mi "
    "set-content sc add-content ac out-file get-item resolve-path measure-object select-object "
    "where-object foreach-object sort-object format-table out-string invoke-webrequest iwr "
    "invoke-restmethod irm rename-item ren get-command get-process start-process get-filehash "
    "tee-object dir del erase copy move md rd mkdir rmdir cls ver vol mklink start call "
    "if for while do done fi then else elif case esac function time".split()
)
_CLAUDE_ABSOLUTE_TOOLS = frozenset({"read", "write", "edit", "multiedit", "notebookedit"})
_MARKERS_FOR_EXE = {
    "cmake": ("CMakeLists.txt",),
    "ctest": ("CTestTestfile.cmake", "CMakeLists.txt"),
    "make": ("Makefile", "makefile", "GNUmakefile"),
    "npm": ("package.json",),
    "pnpm": ("package.json",),
    "yarn": ("package.json",),
    "bun": ("package.json",),
    "cargo": ("Cargo.toml",),
    "go": ("go.mod",),
    "pytest": ("pyproject.toml", "pytest.ini", "setup.cfg", "tox.ini"),
    "tox": ("tox.ini", "pyproject.toml"),
    "mvn": ("pom.xml",),
    "gradle": ("build.gradle", "build.gradle.kts", "settings.gradle"),
    "dotnet": ("*.sln", "*.csproj"),
    "poetry": ("pyproject.toml",),
}


# --------------------------------------------------------------- git refs
_REF_BAD_RE = re.compile(r"(?:\.\.|@\{|[\x00-\x20\x7f~^:?*\[\\])")


def valid_ref_name(ref: str) -> bool:
    """``git check-ref-format --allow-onelevel`` rules (pure Python, no subprocess)."""
    if not ref or ref == "@" or ref.startswith("-") or ref.endswith(("/", ".", ".lock")):
        return False
    if _REF_BAD_RE.search(ref) or "//" in ref:
        return False
    return all(
        part and not part.startswith(".") and not part.endswith(".lock") for part in ref.split("/")
    )


_GIT_REF_SUBCOMMANDS = {
    "checkout",
    "switch",
    "merge",
    "rebase",
    "branch",
    "tag",
    "cherry-pick",
    "revert",
    "reset",
}
_GIT_NO_REPO = {"init", "clone", "version", "--version", "help", "config", "ls-remote", ""}


def _git_refs(argv: tuple[str, ...]) -> list[str]:
    """Ref-like positional arguments of a git command (paths after ``--`` excluded)."""
    if len(argv) < 2:
        return []
    sub_i = next((i for i, a in enumerate(argv[1:], 1) if not a.startswith("-")), None)
    if sub_i is None:
        return []
    sub = argv[sub_i]
    if sub not in _GIT_REF_SUBCOMMANDS:
        return []
    refs = []
    created: list[str] = []
    skip_next = False
    for a in argv[sub_i + 1 :]:
        if a == "--":
            break
        if skip_next:
            skip_next = False
            created.append(a)  # -b <new-branch> / -c <name>: always a ref
            continue
        if a in ("-b", "-B", "-c", "-C", "--orphan"):
            skip_next = True
            continue
        if a.startswith("-"):
            continue
        if "/" in a and os.sep in a and os.path.exists(a):
            continue
        refs.append(a)
    if sub in ("reset", "checkout", "restore"):
        # Positional args may be paths; only check things that cannot be paths.
        refs = [r for r in refs if not re.search(r"\.[A-Za-z0-9]{1,5}$", r)]
    return [*created, *refs]


# --------------------------------------------------------------- validator
class ToolContractValidator:
    def __init__(self, runtime: Any) -> None:
        self.rt = runtime
        self.store = runtime.store
        self.config = runtime.config
        self._file_index: list[str] | None = None
        self._file_index_at = 0.0

    # ------------------------------------------------------------- helpers
    def contract_for(self, inv: ToolInvocation) -> ToolContract:
        return BUILTIN_CONTRACTS.get(inv.family, BUILTIN_CONTRACTS[Family.OTHER])

    @property
    def root(self) -> str:
        return str(self.rt.workspace.root)

    @property
    def local(self) -> bool:
        return bool(self.rt.workspace.local)

    def _effective_cwd(self, inv: ToolInvocation) -> str:
        base = inv.cwd or self.root
        if inv.cd_target:
            return resolve(inv.cd_target, cwd=base)
        return base

    def _suggest(self, missing: str) -> str:
        """A same-named file inside the project, when exactly a few exist."""
        if not self.local:
            return ""
        name = os.path.basename(missing.replace("\\", "/"))
        if not name:
            return ""
        index = self._index()
        hits = [p for p in index if os.path.basename(p) == name][:3]
        if not hits:
            low = name.lower()
            hits = [p for p in index if os.path.basename(p).lower() == low][:3]
        return ", ".join(hits)

    def _index(self) -> list[str]:
        now = time.time()
        if self._file_index is not None and now - self._file_index_at < 60:
            return self._file_index
        out: list[str] = []
        skip = {
            ".git",
            "node_modules",
            ".venv",
            "venv",
            "target",
            "build",
            "dist",
            "__pycache__",
            ".tox",
            ".mypy_cache",
        }
        try:
            for dirpath, dirnames, filenames in os.walk(self.root):
                dirnames[:] = [d for d in dirnames if d not in skip and not d.startswith(".")]
                rel_dir = os.path.relpath(dirpath, self.root)
                for f in filenames:
                    out.append(f if rel_dir == "." else os.path.join(rel_dir, f).replace("\\", "/"))
                    if len(out) >= 20000:
                        raise StopIteration
        except StopIteration:
            pass
        except OSError:
            pass
        self._file_index, self._file_index_at = out, now
        return out

    # ---------------------------------------------------------- validation
    def validate(
        self,
        ev: AgentEvent,
        *,
        source: str,
        which: dict[str, str | None] | None = None,
    ) -> ValidationOutcome:
        started = time.perf_counter()
        inv: ToolInvocation | None = ev.transient.get("invocation")
        raw: dict[str, Any] = dict(ev.transient.get("input") or {})
        if inv is None:
            return ValidationOutcome(RuleResult.UNKNOWN, [], "allowed", "none")
        contract = self.contract_for(inv)
        findings: list[Finding] = []
        scope_class = ""
        rules = set(contract.deterministic_rules)
        try:
            findings.extend(self._schema_checks(inv, raw))
            if self.local:
                if "input_exists" in rules:
                    findings.extend(self._input_exists(inv))
                if "range_valid" in rules:
                    findings.extend(self._range_valid(raw))
                if "edit_effective" in rules:
                    findings.extend(self._edit_effective(inv, raw))
                if "patch_targets" in rules:
                    findings.extend(self._patch_targets(inv, raw))
                if "parent_creatable" in rules:
                    findings.extend(self._parent_creatable(inv))
                if "search_scope_exists" in rules:
                    findings.extend(self._search_scope(inv))
                if "cwd_valid" in rules:
                    findings.extend(self._cwd_valid(inv))
                if "executables" in rules:
                    findings.extend(self._executables(inv, which))
                if "platform" in rules:
                    findings.extend(self._platform(inv, which))
                if "git" in rules or any(s.executable == "git" for s in inv.segments):
                    findings.extend(self._git(inv))
                if "test_selector" in rules:
                    findings.extend(self._test_selector(inv))
                if "build_target" in rules:
                    findings.extend(self._build_target(inv))
            if "pattern_syntax" in rules:
                findings.extend(self._pattern_syntax(inv))
            if "absolute_path" in contract.repair_rules:
                findings.extend(self._absolute_path_repair(inv, raw))
            # Step 5: scope for operations that may mutate.
            scope_class = ""
            if self.rt.scope is not None and (
                inv.mutating or inv.side_effect is not SideEffect.NONE
            ):
                scope_findings, scope_class = self.rt.scope.classify_proposed(ev, source=source)
                findings.extend(scope_findings)
            # Step 6: learned narrow failure rules.
            findings.extend(self._learned(inv))
            # JevK5 (§7.11): only for otherwise-clean unknown tools, WARN at most.
            if source == "history" and not findings and inv.family in (Family.OTHER, Family.MCP):
                findings.extend(self._jevk5(inv, raw))
        except Exception:  # noqa: BLE001 - a validator bug never blocks a call
            logger.debug("tool contract validation failed", exc_info=True)
            findings = [f for f in findings if f.result is not RuleResult.BLOCK]
            scope_class = ""
        result = max(
            (f.result for f in findings), key=lambda r: _SEVERITY[r], default=RuleResult.PASS
        )
        outcome = ValidationOutcome(
            result, findings, "allowed", contract.contract_id, scope_class=scope_class
        )
        self._enforce(outcome, source=source)
        if outcome.enforced == "repaired":
            # Only a semantics-preserving rewrite survives (§7.8); otherwise warn.
            outcome.repaired_input = apply_safe_repair(
                raw, outcome.repaired_input or {}, cwd=inv.cwd, root=self.root
            )
            if outcome.repaired_input is None:
                outcome.enforced = "warned"
        outcome.latency_ms = (time.perf_counter() - started) * 1000.0
        self._record(ev, inv, outcome, source)
        return outcome

    def _enforce(self, outcome: ValidationOutcome, *, source: str) -> None:
        decide_enforcement(self.rt, outcome, source=source)

    def _record(
        self, ev: AgentEvent, inv: ToolInvocation, outcome: ValidationOutcome, source: str
    ) -> None:
        m = self.rt.metrics
        key = {
            RuleResult.PASS: "tool_validation_pass",
            RuleResult.WARN: "tool_validation_warn",
            RuleResult.REPAIRABLE: "tool_validation_repair",
            RuleResult.BLOCK: "tool_validation_block",
            RuleResult.UNKNOWN: "tool_validation_unknown",
        }[outcome.result]
        m.bump(key)
        m.bump(f"tool_validation_{outcome.enforced}")
        m.observe("validation", outcome.latency_ms)
        if any(f.rule.startswith("learned:") for f in outcome.findings):
            m.bump("learned_rule_hit")
        if outcome.result is RuleResult.PASS:
            return
        vid = stable_id(ev.event_id, source, "validation", n=24)
        rule = ",".join(sorted({f.rule for f in outcome.findings if f.result is outcome.result}))[
            :200
        ]

        def run(c: Any) -> None:
            c.execute(
                "INSERT OR IGNORE INTO validations(validation_id, session_key, tool_name, outcome, rule, "
                "reason, enforced, source, ts) VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    vid,
                    self.rt.session_key,
                    inv.tool_name,
                    outcome.result.value,
                    rule,
                    red(outcome.message, 600),
                    outcome.enforced,
                    source,
                    time.time(),
                ),
            )

        self.store.write(run)
        if self.rt.evidence is not None:
            rec = self.rt.evidence.make(
                ev,
                claim_type="tool_validation",
                subject=f"{inv.tool_name}|{ev.event_id[:12]}",
                predicate="validation",
                value={"result": outcome.result.value, "enforced": outcome.enforced, "rules": rule},
                display=f"{inv.tool_name}: {outcome.result.value} ({outcome.enforced}) {outcome.message[:120]}",
                source_kind="DERIVED",
                confidence=0.99 if all(f.deterministic for f in outcome.findings) else 0.6,
                polarity="NEGATIVE",
                source_event_id=vid,
            )
            self.rt.evidence.record(rec)
        # A hook-checked call that was allowed with a warning: the agent learns why
        # in the next live turn (§7.10). History-only calls already ran and the
        # agent saw their result, so they are recorded, not repeated back.
        contract_findings = [f for f in outcome.findings if not f.rule.startswith("scope:")]
        # Claude Code receives hook warnings immediately (additionalContext).
        if (
            outcome.enforced == "warned"
            and source == "hook"
            and contract_findings
            and self.rt.agent != "claude_code"
        ):
            self.rt.warn(
                f"contract:{rule}:{ev.event_id[:12]}",
                f"Tool-call check ({inv.tool_name}): "
                + "; ".join(f.message for f in contract_findings[:2]),
                priority=1 if outcome.result is RuleResult.BLOCK else 3,
            )

    # ------------------------------------------------------------ rules
    def _schema_checks(self, inv: ToolInvocation, raw: dict[str, Any]) -> list[Finding]:
        out: list[Finding] = []
        schema = (getattr(self.rt, "tool_schemas", {}) or {}).get(inv.tool_name)
        if isinstance(schema, dict):
            for key in ("oneOf",):
                alts = schema.get(key)
                if isinstance(alts, list) and alts:
                    satisfied = [
                        a
                        for a in alts
                        if isinstance(a, dict) and all(r in raw for r in a.get("required") or [])
                    ]
                    if len(satisfied) > 1:
                        out.append(
                            Finding(
                                "schema:exclusive",
                                RuleResult.BLOCK,
                                "MUTUALLY_EXCLUSIVE_ARGS",
                                "arguments match more than one exclusive alternative of the tool schema",
                            )
                        )
                    elif not satisfied:
                        out.append(
                            Finding(
                                "schema:alternatives",
                                RuleResult.BLOCK,
                                "MISSING_REQUIRED_ARGS",
                                "arguments satisfy none of the tool schema's alternatives",
                            )
                        )
            excl = schema.get("x-mutually-exclusive")
            if isinstance(excl, list):
                for group in excl:
                    if (
                        isinstance(group, list)
                        and sum(1 for k in group if raw.get(k) not in (None, "", False)) > 1
                    ):
                        out.append(
                            Finding(
                                "schema:exclusive",
                                RuleResult.BLOCK,
                                "MUTUALLY_EXCLUSIVE_ARGS",
                                f"arguments {', '.join(group)} are mutually exclusive",
                            )
                        )
        name = inv.tool_name.lower()
        if name == "multiedit" and not raw.get("edits"):
            out.append(
                Finding(
                    "schema:edits",
                    RuleResult.BLOCK,
                    "MISSING_REQUIRED_ARGS",
                    "MultiEdit needs at least one edit",
                )
            )
        if (
            name == "grep"
            and raw.get("output_mode") not in (None, "content")
            and any(raw.get(k) for k in ("-A", "-B", "-C", "-n"))
        ):
            out.append(
                Finding(
                    "schema:grep_context",
                    RuleResult.WARN,
                    "IGNORED_ARGS",
                    "context/line-number flags only apply with output_mode=content",
                    deterministic=False,
                )
            )
        return out

    def _input_exists(self, inv: ToolInvocation) -> list[Finding]:
        out = []
        for p in inv.paths_read:
            if not p:
                continue
            full = resolve(p, cwd=inv.cwd or self.root)
            if not os.path.exists(full):
                hint = self._suggest(p)
                msg = f"`{p}` does not exist" + (f"; did you mean {hint}?" if hint else "")
                out.append(Finding("path:input_exists", RuleResult.BLOCK, "MISSING_PATH", msg))
        return out

    def _range_valid(self, raw: dict[str, Any]) -> list[Finding]:
        out = []
        for key in ("offset", "limit"):
            v = raw.get(key)
            if isinstance(v, (int, float)) and v < 0:
                out.append(
                    Finding(
                        "args:range",
                        RuleResult.BLOCK,
                        "INVALID_RANGE",
                        f"{key} must not be negative",
                    )
                )
        return out

    def _edit_effective(self, inv: ToolInvocation, raw: dict[str, Any]) -> list[Finding]:
        old, new = raw.get("old_string"), raw.get("new_string")
        if isinstance(old, str) and isinstance(new, str) and old == new:
            return [
                Finding(
                    "args:edit_noop",
                    RuleResult.BLOCK,
                    "NO_OP_EDIT",
                    "old_string and new_string are identical; the edit would change nothing",
                )
            ]
        return []

    def _patch_targets(self, inv: ToolInvocation, raw: dict[str, Any]) -> list[Finding]:
        out: list[Finding] = []
        from .families import patch_paths

        patch = inv.command if "*** Begin Patch" in (inv.command or "") else ""
        if not patch:
            for key in ("input", "patch", "content"):
                if isinstance(raw.get(key), str) and "*** " in raw[key]:
                    patch = raw[key]
                    break
        if not patch:
            return out
        written, deleted = patch_paths(patch)
        base = self._effective_cwd(inv)
        for p in deleted:
            if not os.path.exists(resolve(p, cwd=base)):
                out.append(
                    Finding(
                        "patch:delete_missing",
                        RuleResult.BLOCK,
                        "MISSING_PATH",
                        f"patch deletes `{p}`, which does not exist",
                    )
                )
        for m in re.finditer(r"(?m)^\*\*\* Update File: (.+?)\s*$", patch):
            p = m.group(1)
            if not os.path.exists(resolve(p, cwd=base)):
                hint = self._suggest(p)
                out.append(
                    Finding(
                        "patch:update_missing",
                        RuleResult.BLOCK,
                        "MISSING_PATH",
                        f"patch updates `{p}`, which does not exist"
                        + (f"; did you mean {hint}?" if hint else ""),
                    )
                )
        return out

    def _parent_creatable(self, inv: ToolInvocation) -> list[Finding]:
        out = []
        for p in inv.paths_written:
            full = resolve(p, cwd=inv.cwd or self.root)
            parent = os.path.dirname(full)
            probe = parent
            while probe and not os.path.exists(probe):
                nxt = os.path.dirname(probe)
                if nxt == probe:
                    break
                probe = nxt
            if probe and os.path.exists(probe) and not os.path.isdir(probe):
                out.append(
                    Finding(
                        "path:parent",
                        RuleResult.BLOCK,
                        "PARENT_NOT_DIRECTORY",
                        f"`{probe}` exists and is not a directory",
                    )
                )
            if os.path.isdir(full):
                out.append(
                    Finding(
                        "path:is_dir",
                        RuleResult.BLOCK,
                        "TARGET_IS_DIRECTORY",
                        f"`{p}` is a directory",
                    )
                )
        return out

    def _search_scope(self, inv: ToolInvocation) -> list[Finding]:
        out = []
        for p in inv.paths_read:
            if p and not os.path.exists(resolve(p, cwd=inv.cwd or self.root)):
                hint = self._suggest(p)
                out.append(
                    Finding(
                        "path:search_scope",
                        RuleResult.BLOCK,
                        "MISSING_PATH",
                        f"search path `{p}` does not exist"
                        + (f"; did you mean {hint}?" if hint else ""),
                    )
                )
        return out

    def _cwd_valid(self, inv: ToolInvocation) -> list[Finding]:
        out = []
        if inv.cwd:
            full = resolve(inv.cwd)
            if not os.path.exists(full):
                out.append(
                    Finding(
                        "cwd:exists",
                        RuleResult.BLOCK,
                        "BAD_CWD",
                        f"working directory `{inv.cwd}` does not exist",
                    )
                )
            elif not os.path.isdir(full):
                out.append(
                    Finding(
                        "cwd:isdir",
                        RuleResult.BLOCK,
                        "BAD_CWD",
                        f"working directory `{inv.cwd}` is not a directory",
                    )
                )
            elif not within(full, resolve(self.root)) and inv.side_effect is not SideEffect.NONE:
                out.append(
                    Finding(
                        "cwd:outside",
                        RuleResult.WARN,
                        "CWD_OUTSIDE_PROJECT",
                        f"working directory `{inv.cwd}` is outside the project",
                        deterministic=False,
                    )
                )
        if inv.cd_target:
            target = resolve(inv.cd_target, cwd=inv.cwd or self.root)
            if not os.path.isdir(target) and not re.search(r"[$%`(]", inv.cd_target):
                out.append(
                    Finding(
                        "cwd:cd_target",
                        RuleResult.BLOCK,
                        "BAD_CWD",
                        f"`cd {inv.cd_target}` target does not exist",
                    )
                )
        return out

    def _executables(
        self, inv: ToolInvocation, which: dict[str, str | None] | None
    ) -> list[Finding]:
        out = []
        base = self._effective_cwd(inv)
        for seg in inv.segments:
            exe_tok = seg.argv[0] if seg.argv else ""
            exe = seg.executable
            if not exe or exe in SHELL_BUILTINS or re.search(r"[$%`(=]", exe_tok):
                continue
            if "/" in exe_tok or "\\" in exe_tok:
                full = resolve(exe_tok, cwd=base)
                if not os.path.exists(full):
                    out.append(
                        Finding(
                            "exe:path",
                            RuleResult.BLOCK,
                            "MISSING_EXECUTABLE",
                            f"executable `{exe_tok}` does not exist",
                        )
                    )
                continue
            if which is not None and exe_tok in which:
                found = which[exe_tok]
            else:
                found = shutil.which(exe_tok)
            if not found:
                out.append(
                    Finding(
                        "exe:resolve",
                        RuleResult.WARN,
                        "EXECUTABLE_NOT_FOUND",
                        f"`{exe_tok}` is not on PATH (it may be a shell alias or function)",
                        deterministic=False,
                    )
                )
        return out

    def _platform(self, inv: ToolInvocation, which: dict[str, str | None] | None) -> list[Finding]:
        out = []
        on_windows = sys.platform.startswith("win")
        for seg in inv.segments:
            tok = seg.argv[0] if seg.argv else ""
            low = tok.lower()
            if not on_windows:
                if re.match(r"^[A-Za-z]:[\\/]", tok):
                    out.append(
                        Finding(
                            "platform:drive",
                            RuleResult.BLOCK,
                            "WRONG_PLATFORM",
                            f"`{tok}` is a Windows drive path on a non-Windows machine",
                        )
                    )
                elif seg.executable == "cmd" and (
                    len(seg.argv) > 1 and seg.argv[1].lower() in ("/c", "/k")
                ):
                    out.append(
                        Finding(
                            "platform:cmd",
                            RuleResult.BLOCK,
                            "WRONG_PLATFORM",
                            "`cmd /c` is Windows-only; this machine is not Windows",
                        )
                    )
                elif (
                    low.endswith((".exe", ".bat", ".cmd"))
                    and not shutil.which(tok)
                    and not os.path.exists(tok)
                ):
                    out.append(
                        Finding(
                            "platform:exe",
                            RuleResult.BLOCK,
                            "WRONG_PLATFORM",
                            f"`{tok}` is a Windows executable and is not available here",
                        )
                    )
                elif re.match(r"^[A-Za-z]:[\\/]", tok):
                    out.append(
                        Finding(
                            "platform:drive",
                            RuleResult.BLOCK,
                            "WRONG_PLATFORM",
                            f"`{tok}` is a Windows drive path on a non-Windows machine",
                        )
                    )
            elif low.startswith(("/usr/", "/bin/", "/opt/")) and not os.path.exists(tok):
                out.append(
                    Finding(
                        "platform:posix",
                        RuleResult.WARN,
                        "WRONG_PLATFORM",
                        f"`{tok}` is a POSIX path; it resolves only inside a POSIX shell layer",
                        deterministic=False,
                    )
                )
        return out

    def _git(self, inv: ToolInvocation) -> list[Finding]:
        out = []
        base = self._effective_cwd(inv)
        for seg in inv.segments:
            if seg.executable != "git":
                continue
            sub = seg.subcommand
            if sub not in _GIT_NO_REPO and "-C" not in seg.argv and not _in_git_repo(base):
                out.append(
                    Finding(
                        "git:repo",
                        RuleResult.BLOCK,
                        "NOT_A_GIT_REPO",
                        f"`git {sub}` needs a repository; `{base}` is not inside one",
                    )
                )
            for ref in _git_refs(seg.argv):
                if (
                    not valid_ref_name(ref)
                    and not re.match(r"^(?:HEAD|FETCH_HEAD|ORIG_HEAD)(?:[~^]\d*)*$", ref)
                    and not re.match(r"^[0-9a-f]{4,40}(?:[~^]\d*)*$", ref)
                    and not re.search(r"[~^]\d*$", ref)
                ):
                    out.append(
                        Finding(
                            "git:ref",
                            RuleResult.BLOCK,
                            "INVALID_REF",
                            f"`{ref}` is not a valid git ref name",
                        )
                    )
        return out

    def _test_selector(self, inv: ToolInvocation) -> list[Finding]:
        out = []
        base = self._effective_cwd(inv)
        for seg in inv.segments:
            argv = seg.argv
            is_pytest = seg.executable == "pytest" or ("-m" in argv and "pytest" in argv)
            if is_pytest:
                for a in argv[1:]:
                    if a.startswith("-") or a in ("pytest", "-m"):
                        continue
                    path = a.split("::", 1)[0]
                    if not re.search(r"\.py$|[\\/]", path):
                        continue
                    if not os.path.exists(resolve(path, cwd=base)):
                        hint = self._suggest(path)
                        out.append(
                            Finding(
                                "test:selector",
                                RuleResult.BLOCK,
                                "MISSING_TEST_SELECTOR",
                                f"test path `{path}` does not exist"
                                + (f"; did you mean {hint}?" if hint else ""),
                            )
                        )
            if seg.executable == "cargo" and seg.subcommand == "test":
                pkgs = _cargo_packages(base)
                for i, a in enumerate(argv):
                    if (
                        a in ("-p", "--package")
                        and i + 1 < len(argv)
                        and pkgs
                        and argv[i + 1].split("@")[0] not in pkgs
                    ):
                        out.append(
                            Finding(
                                "test:cargo_package",
                                RuleResult.BLOCK,
                                "UNKNOWN_PACKAGE",
                                f"cargo package `{argv[i + 1]}` is not in this workspace",
                            )
                        )
            if seg.executable in ("npm", "pnpm", "yarn", "bun") and seg.subcommand == "run":
                out.extend(self._npm_script(seg.argv, base))
        return out

    def _build_target(self, inv: ToolInvocation) -> list[Finding]:
        out = []
        base = self._effective_cwd(inv)
        for seg in inv.segments:
            if seg.executable in ("npm", "pnpm", "yarn", "bun") and seg.subcommand == "run":
                out.extend(self._npm_script(seg.argv, base))
            if seg.executable == "cargo":
                pkgs = _cargo_packages(base)
                for i, a in enumerate(seg.argv):
                    if (
                        a in ("-p", "--package")
                        and i + 1 < len(seg.argv)
                        and pkgs
                        and seg.argv[i + 1].split("@")[0] not in pkgs
                    ):
                        out.append(
                            Finding(
                                "build:cargo_package",
                                RuleResult.BLOCK,
                                "UNKNOWN_PACKAGE",
                                f"cargo package `{seg.argv[i + 1]}` is not in this workspace",
                            )
                        )
        return out

    def _npm_script(self, argv: tuple[str, ...], base: str) -> list[Finding]:
        rest = [a for a in argv[2:] if not a.startswith("-")]
        if not rest:
            return []
        script = rest[0]
        pkg = os.path.join(base, "package.json")
        try:
            import json

            with open(pkg, encoding="utf-8") as fh:
                scripts = json.load(fh).get("scripts") or {}
        except (OSError, ValueError):
            return []
        if isinstance(scripts, dict) and script not in scripts:
            return [
                Finding(
                    "build:npm_script",
                    RuleResult.BLOCK,
                    "MISSING_SCRIPT",
                    f"package.json has no `{script}` script",
                )
            ]
        return []

    def _pattern_syntax(self, inv: ToolInvocation) -> list[Finding]:
        out = []
        for pattern in inv.patterns:
            try:
                re.compile(pattern)
            except re.error as exc:
                if (
                    "unbalanced parenthesis" in str(exc)
                    or "unterminated character set" in str(exc)
                    or "missing )" in str(exc)
                ):
                    out.append(
                        Finding(
                            "pattern:regex",
                            RuleResult.WARN,
                            "INVALID_REGEX",
                            f"pattern `{pattern[:60]}` looks invalid ({exc.msg})",
                            deterministic=False,
                        )
                    )
        glob = str(inv.extra.get("glob") or "")
        if glob and (glob.count("[") != glob.count("]") or glob.count("{") != glob.count("}")):
            out.append(
                Finding(
                    "pattern:glob",
                    RuleResult.WARN,
                    "INVALID_GLOB",
                    f"glob `{glob[:60]}` has unbalanced brackets",
                    deterministic=False,
                )
            )
        return out

    def _absolute_path_repair(self, inv: ToolInvocation, raw: dict[str, Any]) -> list[Finding]:
        """Claude Code's file tools take absolute paths; a relative one is resolved against the root."""
        if inv.tool_name.lower() not in _CLAUDE_ABSOLUTE_TOOLS:
            return []
        for key in ("file_path", "notebook_path"):
            value = raw.get(key)
            if (
                isinstance(value, str)
                and value.strip()
                and not is_absolute(value.strip())
                and self.root
            ):
                fixed = join(self.root, value.strip())
                return [
                    Finding(
                        "repair:absolute_path",
                        RuleResult.REPAIRABLE,
                        "RELATIVE_PATH",
                        f"`{value}` is relative; this tool needs an absolute path (`{fixed}`)",
                        repair={key: fixed},
                    )
                ]
        return []

    def _jevk5(self, inv: ToolInvocation, raw: dict[str, Any]) -> list[Finding]:
        from headroom.intelligence.models import DecisionFamily

        from .advice import classify

        schema = (getattr(self.rt, "tool_schemas", {}) or {}).get(inv.tool_name) or {}
        description = (
            str((schema or {}).get("description", ""))[:400] if isinstance(schema, dict) else ""
        )
        if not description:
            return []
        answer = classify(
            self.rt.advisor(),
            DecisionFamily.TOOL_CONTRACT,
            f"Tool: {inv.tool_name}\nPurpose: {description}\nArguments: {red(dumps(raw), 600)}",
            "Is this proposed invocation consistent with the tool's documented purpose?",
            {"A": "yes", "B": "no", "C": "ambiguous"},
        )
        if answer is not None and answer[0] == "B" and answer[1] >= 0.8:
            return [
                Finding(
                    "jevk5:purpose",
                    RuleResult.WARN,
                    "PURPOSE_MISMATCH",
                    f"{inv.tool_name} call looks inconsistent with its documented purpose",
                    deterministic=False,
                )
            ]
        return []

    # -------------------------------------------------------- learned rules
    def features(self, inv: ToolInvocation) -> dict[str, Any]:
        exe = inv.executable
        feats: dict[str, Any] = {}
        if not exe or not self.local:
            return feats
        base = self._effective_cwd(inv)
        for marker in _MARKERS_FOR_EXE.get(exe, ()):
            if "*" in marker:
                try:
                    suffix = marker.lstrip("*")
                    present = any(n.endswith(suffix) for n in os.listdir(base))
                except OSError:
                    present = False
            else:
                present = os.path.exists(os.path.join(base, marker))
            feats[f"has:{marker}"] = present
        if exe not in SHELL_BUILTINS:
            feats["exe_found"] = bool(shutil.which(exe))
        return feats

    def shape(self, inv: ToolInvocation) -> str:
        if not inv.segments:
            return inv.family.value
        parts = []
        for seg in inv.segments[:4]:
            toks = []
            for a in seg.argv[:8]:
                if re.search(r"[\\/.]", a) and not a.startswith("-"):
                    toks.append("<path>")
                elif re.fullmatch(r"\d+", a):
                    toks.append("<n>")
                else:
                    toks.append(a.lower())
            parts.append(" ".join(toks))
        return red(" | ".join(parts), 300)

    def on_tool_result(self, ev: AgentEvent, records: list[Any]) -> None:
        inv: ToolInvocation | None = ev.transient.get("invocation")
        if inv is None or not inv.command or ev.success is None:
            return
        text = str(ev.transient.get("text") or "")
        if "headroom" in text.lower() and "blocked" in text.lower():
            return  # our own block, not an environment failure
        exe = inv.executable
        sub = inv.segments[0].subcommand if inv.segments else ""
        shape = self.shape(inv)
        feats = self.features(inv)
        sig = failure_signature(tool_use_error(text) or text[-2000:]) if ev.success is False else ""

        def run(c: Any) -> None:
            c.execute(
                "INSERT OR IGNORE INTO tool_outcomes(event_id, family, executable, subcommand, shape, "
                "signature, features, success, ts) VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    ev.event_id,
                    inv.family.value,
                    exe,
                    sub,
                    shape,
                    sig,
                    dumps(feats),
                    int(bool(ev.success)),
                    time.time(),
                ),
            )

        self.store.write(run)
        if ev.success:
            self._invalidate_by_success(inv.family.value, exe, sub, shape, feats)
        else:
            self._maybe_learn(inv.family.value, exe, sub, shape, sig, text)

    def _invalidate_by_success(
        self, family: str, exe: str, sub: str, shape: str, feats: dict[str, Any]
    ) -> None:
        for row in self.store.query(
            "SELECT rule_id, shape, predicate FROM learned_rules WHERE family = ? AND executable = ? "
            "AND subcommand = ? AND disabled_reason IS NULL",
            (family, exe, sub),
        ):
            pred = loads(row["predicate"], {}) or {}
            if (pred and all(feats.get(k) == v for k, v in pred.items())) or (
                not pred and row["shape"] == shape
            ):
                self.store.write(
                    lambda c, rid=row["rule_id"]: c.execute(
                        "UPDATE learned_rules SET disabled_reason = 'counterexample' WHERE rule_id = ?",
                        (rid,),
                    )
                )
                self.rt.metrics.bump("learned_rules_invalidated")

    def _maybe_learn(
        self, family: str, exe: str, sub: str, shape: str, sig: str, text: str
    ) -> None:
        if not sig:
            return
        rows = self.store.query(
            "SELECT shape, features FROM tool_outcomes WHERE family = ? AND executable = ? AND subcommand = ? "
            "AND signature = ? AND success = 0 ORDER BY ts DESC LIMIT 50",
            (family, exe, sub, sig),
        )
        if len(rows) < LEARN_MIN_FAILURES:
            return
        feats = [loads(r["features"], {}) or {} for r in rows]
        common = {k: v for k, v in feats[0].items() if all(f.get(k) == v for f in feats[1:])}
        # Only features that *distinguish* failure: a feature value also seen on a
        # success with the same command would make the rule too broad.
        successes = self.store.query(
            "SELECT shape, features FROM tool_outcomes WHERE family = ? AND executable = ? AND subcommand = ? AND success = 1",
            (family, exe, sub),
        )
        succ_feats = [loads(r["features"], {}) or {} for r in successes]
        predicate = dict(common)
        if predicate and any(
            all(sf.get(k) == v for k, v in predicate.items()) for sf in succ_feats
        ):
            predicate = {}
        rule_shape = "*" if predicate else shape
        if not predicate:
            same_shape = [r for r in rows if r["shape"] == shape]
            if len(same_shape) < LEARN_MIN_FAILURES:
                return
            if any(r["shape"] == shape for r in successes):
                return
        reason = _reason_for(exe, predicate, text)
        rule_id = stable_id(family, exe, sub, rule_shape, sig, dumps(predicate), n=20)
        now = time.time()

        def run(c: Any) -> None:
            c.execute(
                "INSERT INTO learned_rules(rule_id, family, executable, subcommand, shape, signature, predicate, "
                "reason, failures, created_at, last_reinforced) VALUES (?,?,?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(rule_id) DO UPDATE SET failures = excluded.failures, "
                "last_reinforced = excluded.last_reinforced, disabled_reason = NULL",
                (
                    rule_id,
                    family,
                    exe,
                    sub,
                    rule_shape,
                    sig,
                    dumps(predicate),
                    reason,
                    len(rows),
                    now,
                    now,
                ),
            )

        self.store.write(run)
        self.rt.metrics.bump("learned_rules_created")

    def _learned(self, inv: ToolInvocation) -> list[Finding]:
        exe = inv.executable
        if not exe:
            return []
        sub = inv.segments[0].subcommand if inv.segments else ""
        rules = self.store.query(
            "SELECT * FROM learned_rules WHERE family = ? AND executable = ? AND subcommand = ? "
            "AND disabled_reason IS NULL",
            (inv.family.value, exe, sub),
        )
        if not rules:
            return []
        feats = self.features(inv)
        shape = self.shape(inv)
        out = []
        for r in rules:
            pred = loads(r["predicate"], {}) or {}
            hit = (pred and all(feats.get(k) == v for k, v in pred.items())) or (
                not pred and r["shape"] == shape
            )
            if hit:
                out.append(
                    Finding(
                        f"learned:{r['rule_id'][:8]}",
                        RuleResult.WARN,
                        r["reason"],
                        f"`{exe} {sub}`".strip()
                        + f" failed {r['failures']}x here under the same conditions ({r['reason']})",
                        deterministic=False,
                    )
                )
                self.store.write(
                    lambda c, rid=r["rule_id"]: c.execute(
                        "UPDATE learned_rules SET hits = hits + 1, last_hit = ? WHERE rule_id = ?",
                        (time.time(), rid),
                    )
                )
        return out

    def learned_rules(self) -> list[dict[str, Any]]:
        return [
            dict(r)
            for r in self.store.query(
                "SELECT * FROM learned_rules ORDER BY last_reinforced DESC LIMIT 50"
            )
        ]


# Keys a safe repair may touch (plan §7.8). A target, ref, selector, command,
# port, flag or privilege is never in this set.
REPAIRABLE_KEYS = frozenset({"file_path", "notebook_path", "path", "cwd", "workdir"})


def apply_safe_repair(
    raw: dict[str, Any], repair: dict[str, Any], *, cwd: str, root: str
) -> dict[str, Any] | None:
    """Apply ``repair`` to ``raw`` only when it is semantics-preserving; else None.

    Every repaired value must resolve to exactly the path the original value
    resolves to (against the explicit ``cwd``, or the established project root).
    The repair may change spelling (relative to absolute, separators), never
    the target.
    """
    if not repair:
        return None
    out = dict(raw)
    for key, new in repair.items():
        if key not in REPAIRABLE_KEYS or not isinstance(new, str):
            return None
        old = raw.get(key)
        if not isinstance(old, str) or not old.strip():
            # Filling a missing optional cwd with the established project root.
            if key in ("cwd", "workdir") and resolve(new) == resolve(root):
                out[key] = new
                continue
            return None
        base = raw.get("cwd") or raw.get("workdir") or cwd or root
        if resolve(old, cwd=str(base)) != resolve(new, cwd=str(base)):
            return None  # would change the target
        out[key] = new
    return out


def decide_enforcement(rt: Any, outcome: ValidationOutcome, *, source: str) -> None:
    """Set ``outcome.enforced`` truthfully (plan §7.9, §8.8).

    ``blocked`` requires protect mode, a deterministic BLOCK, a hook-sourced
    call and a session whose hook has been observed live. A call seen only in
    history already ran, so it is ``observed``: never ``blocked``.
    """
    cfg = rt.config
    if outcome.result is RuleResult.PASS:
        outcome.enforced = "allowed"
        return
    blocking = [f for f in outcome.findings if f.result is outcome.result]
    scope_only = bool(blocking) and all(f.rule.startswith("scope:") for f in blocking)
    mode: EnforcementMode = cfg.scope_mode if scope_only else cfg.contract_mode
    # Pre-execution control exists for a live host hook and for Headroom's own
    # macro executor; a call seen only in history has already run.
    controlled = source == "workflow" or (source == "hook")
    if mode is EnforcementMode.OBSERVE or not controlled:
        outcome.enforced = "observed"
        return
    if outcome.result is RuleResult.BLOCK:
        deterministic = all(f.deterministic for f in blocking)
        if source == "workflow":
            # Headroom executes macro steps itself: any BLOCK stops the step.
            outcome.enforced = "blocked"
        elif mode is EnforcementMode.PROTECT and deterministic and rt.can_block():
            outcome.enforced = "blocked"
        elif mode is EnforcementMode.PROTECT and deterministic and rt.block_unverified():
            # Host hook seen but its blocking semantics are not yet proven
            # (Codex): the hook is asked to block, and the record says so.
            outcome.enforced = "block_requested"
        else:
            outcome.enforced = "warned"
        return
    if outcome.result is RuleResult.REPAIRABLE:
        repairs = [f.repair for f in outcome.findings if f.repair]
        if (
            cfg.contract_auto_repair
            and repairs
            and (source == "workflow" or rt.capabilities.can_rewrite_safe_args)
        ):
            merged: dict[str, Any] = {}
            for r in repairs:
                merged.update(r or {})
            outcome.repaired_input = merged
            outcome.enforced = "repaired"
            return
    outcome.enforced = "warned"


def _reason_for(exe: str, predicate: dict[str, Any], text: str) -> str:
    missing = [k[4:] for k, v in predicate.items() if k.startswith("has:") and v is False]
    if missing:
        return f"MISSING_PROJECT_ROOT (cwd lacks {missing[0]})"
    if predicate.get("exe_found") is False or re.search(
        r"(?i)command not found|not recognized", text
    ):
        return "EXECUTABLE_NOT_FOUND"
    return "REPEATED_FAILURE"


def _in_git_repo(path: str) -> bool:
    p = os.path.abspath(path or ".")
    while True:
        if os.path.exists(os.path.join(p, ".git")):
            return True
        parent = os.path.dirname(p)
        if parent == p:
            return False
        p = parent


_CARGO_CACHE: dict[str, tuple[float, frozenset[str]]] = {}


def _cargo_packages(base: str) -> frozenset[str]:
    """Package names of the Cargo workspace containing ``base`` (no subprocess)."""
    p = os.path.abspath(base or ".")
    root_manifest = ""
    while True:
        cand = os.path.join(p, "Cargo.toml")
        if os.path.exists(cand):
            root_manifest = cand
        parent = os.path.dirname(p)
        if parent == p:
            break
        p = parent
    if not root_manifest:
        return frozenset()
    try:
        mtime = os.path.getmtime(root_manifest)
    except OSError:
        return frozenset()
    hit = _CARGO_CACHE.get(root_manifest)
    if hit and hit[0] == mtime:
        return hit[1]
    names: set[str] = set()
    try:
        with open(root_manifest, "rb") as fh:
            data = tomllib.load(fh)
        if isinstance(data.get("package"), dict) and data["package"].get("name"):
            names.add(str(data["package"]["name"]))
        root_dir = os.path.dirname(root_manifest)
        import glob as _glob

        for member in (data.get("workspace") or {}).get("members") or []:
            for d in _glob.glob(os.path.join(root_dir, member)):
                try:
                    with open(os.path.join(d, "Cargo.toml"), "rb") as fh:
                        sub = tomllib.load(fh)
                    if isinstance(sub.get("package"), dict) and sub["package"].get("name"):
                        names.add(str(sub["package"]["name"]))
                except (OSError, tomllib.TOMLDecodeError):
                    continue
    except (OSError, tomllib.TOMLDecodeError):
        return frozenset()
    result = frozenset(names)
    _CARGO_CACHE[root_manifest] = (mtime, result)
    return result
