"""Feature 24: Test Impact and Verification Planner (plan §9).

For the task-owned change set (from the Scope Firewall, not raw ``git
status``), the planner selects the smallest verification set that gives
useful confidence, then escalates by deterministic risk:

* **Adapters** (§9.4): pytest, Cargo, CTest, Jest/Vitest, plus a generic
  adapter for project-defined commands. An adapter without enough confidence
  invents no selector.
* **Impact edges** (§9.5): explicit project rule 1.00, coverage 0.95, failure
  0.90, direct static import 0.80, history up to 0.80, naming convention 0.45,
  transitive static 0.40–0.70. Edges combine as ``1 - Π(1 - w)``. Historical
  edges decay linearly over 90 days unless reinforced.
* **Risk** (§9.7)::

      R = .25·scope + .20·interface + .15·config + .15·history
        + .10·platform + .10·uncertainty + .05·size

* **Tiers** (§9.8–9.9): ``R < tier2`` runs Tier 1 only, ``tier2 ≤ R < tier3``
  adds Tier 2 on pass, ``R ≥ tier3`` adds Tier 3 on pass. Mandatory Tier 3
  categories (wire/serialization, migrations, auth, dependency resolution,
  root build config, shared CLI parsing, core proxy model, or an explicit user
  request) force Tier 3 regardless of score.
* **Execution** stops at the first failing tier. A known-flaky test is rerun
  at most once, and differing outcomes mark it ``FLAKY`` rather than passing.
  Results flow into the Evidence Ledger and Task State.

Commands are argv lists built per platform (``.cmd`` shims resolved on
Windows), never shell strings. The planner works with JevK5 disabled.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass, field
from typing import Any

import tomllib

from .events import AgentEvent, EventType
from .ids import stable_id
from .results import TestResult, parse_test_output
from .store import dumps, loads

logger = logging.getLogger(__name__)

# A source module whose name matches ``test_*.py``: never collect it as tests.
__test__ = False

EDGE_WEIGHTS = {
    "PROJECT_RULE": 1.00,
    "COVERAGE": 0.95,
    "FAILURE": 0.90,
    "STATIC": 0.80,
    "HISTORICAL": 0.80,  # cap
    "NAME": 0.45,
}
STATIC_TRANSITIVE = {2: 0.70, 3: 0.55, 4: 0.40}
DECAY_DAYS = 90.0
TIER1_SCORE = 0.75
_SELECTOR_RE = re.compile(
    r"(?:::|\.py\b|\.rs\b|\.[cm]?[jt]sx?\b|\s-k\s|\s-p\s|--test\s|\s-R\s|--package)"
)
TIER2_SCORE = 0.45
SKIP_DIRS = frozenset(
    {
        ".git",
        "node_modules",
        ".venv",
        "venv",
        "target",
        "dist",
        "__pycache__",
        ".tox",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "site-packages",
        ".next",
        "coverage",
    }
)
MAX_SCAN_FILES = 20000


def combine(weights: list[float]) -> float:
    """``1 - Π(1 - w)``, capped at 1.0 (independent evidence never sums past 1)."""
    p = 1.0
    for w in weights:
        p *= 1.0 - max(0.0, min(1.0, w))
    return min(1.0, 1.0 - p)


def decayed(weight: float, last_observed: float, *, now: float | None = None) -> float:
    now = time.time() if now is None else now
    age_days = max(0.0, (now - last_observed) / 86400.0)
    return max(0.0, weight * (1.0 - age_days / DECAY_DAYS))


@dataclass(frozen=True)
class TestCase:
    test_id: str
    framework: str
    canonical_name: str
    file_path: str = ""
    tags: tuple[str, ...] = ()
    package: str = ""


@dataclass(frozen=True)
class TestSelection:
    test_id: str
    framework: str
    target: str  # file path / package / name passed to the runner
    score: float
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class Check:
    name: str
    argv: tuple[str, ...]


@dataclass(frozen=True)
class CommandSpec:
    framework: str
    argv: tuple[str, ...]
    cwd: str
    selections: tuple[str, ...] = ()

    def display(self) -> str:
        return display_argv(self.argv)

    def brief(self, root: str) -> str:
        """The model-facing form: an executable inside ``root`` is shown relative to it.

        The command runs from ``root``, so ``.venv/bin/python3`` (or
        ``.venv\\Scripts\\python.exe``) is equivalent and much shorter.
        """
        argv = list(self.argv)
        same_cwd = os.path.normcase(os.path.abspath(self.cwd)) == os.path.normcase(
            os.path.abspath(root)
        )
        if argv and same_cwd and os.path.isabs(argv[0]):
            try:
                rel = os.path.relpath(argv[0], root)
            except ValueError:  # a different drive on Windows
                rel = ""
            if rel and not rel.startswith(".."):
                argv[0] = rel
        return display_argv(argv)


@dataclass
class VerificationPlan:
    plan_id: str
    task_id: str | None
    change_set_hash: str
    risk_score: float
    components: dict[str, float]
    tier1: list[TestSelection] = field(default_factory=list)
    tier2: list[TestSelection] = field(default_factory=list)
    tier3: list[TestSelection] = field(default_factory=list)
    commands: dict[int, list[CommandSpec]] = field(default_factory=dict)
    required_non_test_checks: list[Check] = field(default_factory=list)
    escalation_rules: list[str] = field(default_factory=list)
    rationale_codes: list[str] = field(default_factory=list)
    mandatory_tier3: list[str] = field(default_factory=list)
    changed: list[str] = field(default_factory=list)
    tests_considered: int = 0

    @property
    def max_tier(self) -> int:
        if self.mandatory_tier3:
            return 3
        return 3 if self.risk_score >= self._t3 else (2 if self.risk_score >= self._t2 else 1)

    _t2: float = 0.35
    _t3: float = 0.70

    def to_json(self) -> dict[str, Any]:
        def sel(xs: list[TestSelection]) -> list[dict[str, Any]]:
            return [
                {
                    "test_id": s.test_id,
                    "target": s.target,
                    "score": round(s.score, 3),
                    "reasons": list(s.reasons),
                }
                for s in xs
            ]

        return {
            "plan_id": self.plan_id,
            "task_id": self.task_id,
            "change_set_hash": self.change_set_hash,
            "risk_score": round(self.risk_score, 4),
            "components": {k: round(v, 4) for k, v in self.components.items()},
            "tier1": sel(self.tier1),
            "tier2": sel(self.tier2),
            "tier3": sel(self.tier3),
            "commands": {
                str(k): [{"framework": c.framework, "argv": list(c.argv), "cwd": c.cwd} for c in v]
                for k, v in self.commands.items()
            },
            "required_non_test_checks": [
                {"name": c.name, "argv": list(c.argv)} for c in self.required_non_test_checks
            ],
            "escalation_rules": self.escalation_rules,
            "rationale_codes": self.rationale_codes,
            "mandatory_tier3": self.mandatory_tier3,
            "changed": self.changed,
            "max_tier": self.max_tier,
        }


def display_argv(argv: tuple[str, ...] | list[str]) -> str:
    if sys.platform.startswith("win"):
        import subprocess

        return subprocess.list2cmdline(list(argv))
    import shlex

    return " ".join(shlex.quote(a) for a in argv)


def _walk(root: str, *, limit: int = MAX_SCAN_FILES) -> list[str]:
    out: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and not d.startswith("."))
        rel_dir = os.path.relpath(dirpath, root).replace("\\", "/")
        for f in sorted(filenames):
            out.append(f if rel_dir == "." else f"{rel_dir}/{f}")
            if len(out) >= limit:
                return out
    return out


def resolve_executable(name: str) -> str:
    """Absolute path of ``name`` (``.cmd``/``.exe`` shims on Windows), or the name."""
    found = shutil.which(name)
    return found or name


# ------------------------------------------------------------------ adapters
class TestAdapter:
    framework = "generic"

    def __init__(self, root: str, files: list[str]) -> None:
        self.root = root
        self.files = files

    def detect(self) -> float:
        return 0.0

    def discover(self) -> list[TestCase]:
        return []

    def is_test_file(self, rel: str) -> bool:
        return False

    def static_edges(
        self, changed: list[str], tests: list[TestCase]
    ) -> dict[str, list[tuple[str, str, float]]]:
        """test_id -> [(source rel, edge type, weight)]."""
        return {}

    def select(self, selections: list[TestSelection]) -> list[CommandSpec]:
        return []

    def full_suite(self) -> CommandSpec | None:
        return None

    def compile_checks(self, changed: list[str]) -> list[Check]:
        return []

    def parse_result(self, text: str) -> TestResult | None:
        return parse_test_output(text, self.framework)


_PYTHON_CACHE: dict[str, list[str]] = {}


def _python_cmd(root: str) -> list[str]:
    """The project's interpreter: its venv first, else the first on PATH that has pytest.

    The probe runs once per project root (cached for the process).
    """
    cached = _PYTHON_CACHE.get(root)
    if cached is not None:
        return cached
    candidates: list[str] = []
    for cand in (
        ".venv/bin/python",
        "venv/bin/python",
        ".venv/Scripts/python.exe",
        "venv/Scripts/python.exe",
    ):
        p = os.path.join(root, cand)
        if os.path.exists(p):
            candidates.append(p)
    for name in ("python3", "python", "py"):
        found = shutil.which(name)
        if found and found not in candidates:
            candidates.append(found)
    if sys.executable and sys.executable not in candidates:
        candidates.append(sys.executable)
    from .proc import run_argv

    chosen = candidates[0] if candidates else sys.executable
    for cand in candidates:
        if run_argv([cand, "-c", "import pytest"], cwd=root, timeout=15).ok:
            chosen = cand
            break
    _PYTHON_CACHE[root] = [chosen]
    return [chosen]


_PY_IMPORT_RE = re.compile(
    r"(?m)^\s*(?:from\s+([\w.]+)\s+import|import\s+([\w.]+(?:\s*,\s*[\w.]+)*))"
)


class PytestAdapter(TestAdapter):
    framework = "pytest"

    def detect(self) -> float:
        names = set(self.files[:5000])
        score = 0.0
        if "pytest.ini" in names or "conftest.py" in names:
            score = 0.95
        for cfg in ("pyproject.toml", "setup.cfg", "tox.ini"):
            if cfg in names:
                try:
                    with open(
                        os.path.join(self.root, cfg), encoding="utf-8", errors="replace"
                    ) as fh:
                        body = fh.read(200_000)
                    if "pytest" in body:
                        score = max(score, 0.95)
                except OSError:
                    pass
        if score == 0.0 and any(self.is_test_file(f) for f in self.files):
            score = 0.6
        return score

    def is_test_file(self, rel: str) -> bool:
        base = rel.rsplit("/", 1)[-1]
        return base.endswith(".py") and (base.startswith("test_") or base.endswith("_test.py"))

    def discover(self) -> list[TestCase]:
        return [TestCase(f"pytest:{f}", "pytest", f, f) for f in self.files if self.is_test_file(f)]

    def _module_files(self) -> dict[str, str]:
        mods: dict[str, str] = {}
        for f in self.files:
            if not f.endswith(".py"):
                continue
            stem = f[:-3]
            for prefix in ("", "src/", "lib/", "python/"):
                if prefix and not stem.startswith(prefix):
                    continue
                mod = stem[len(prefix) :].replace("/", ".")
                if mod.endswith(".__init__"):
                    mod = mod[: -len(".__init__")]
                mods.setdefault(mod, f)
        return mods

    def _imports(self, rel: str) -> list[str]:
        try:
            with open(os.path.join(self.root, rel), encoding="utf-8", errors="replace") as fh:
                body = fh.read(400_000)
        except OSError:
            return []
        out = []
        for m in _PY_IMPORT_RE.finditer(body):
            if m.group(1):
                out.append(m.group(1))
            else:
                out.extend(x.strip() for x in m.group(2).split(","))
        return out

    def static_edges(
        self, changed: list[str], tests: list[TestCase]
    ) -> dict[str, list[tuple[str, str, float]]]:
        mods = self._module_files()
        changed_set = set(changed)
        file_imports: dict[str, list[str]] = {}

        def resolve_mod(name: str) -> str:
            parts = name.split(".")
            while parts:
                f = mods.get(".".join(parts))
                if f:
                    return f
                parts.pop()
            return ""

        def deps(rel: str) -> list[str]:
            if rel not in file_imports:
                file_imports[rel] = [d for d in (resolve_mod(m) for m in self._imports(rel)) if d]
            return file_imports[rel]

        edges: dict[str, list[tuple[str, str, float]]] = {}
        for t in tests:
            frontier = {t.file_path}
            seen = {t.file_path}
            for depth in range(1, 5):
                nxt: set[str] = set()
                for f in frontier:
                    for d in deps(f):
                        if d in seen:
                            continue
                        seen.add(d)
                        nxt.add(d)
                        if d in changed_set:
                            w = (
                                EDGE_WEIGHTS["STATIC"]
                                if depth == 1
                                else STATIC_TRANSITIVE.get(depth, 0.4)
                            )
                            edges.setdefault(t.test_id, []).append((d, "STATIC", w))
                frontier = nxt
                if not frontier or len(seen) > 400:
                    break
        return edges

    def collect(self, *, cache: dict[str, Any] | None = None, config_hash: str = "") -> list[str]:
        """Node ids via ``pytest --collect-only -q`` (cached per configuration hash)."""
        if cache is not None and config_hash and cache.get("hash") == config_hash:
            return list(cache.get("ids") or [])
        from .proc import run_argv

        res = run_argv(
            [*_python_cmd(self.root), "-m", "pytest", "--collect-only", "-q"],
            cwd=self.root,
            timeout=120,
        )
        ids = parse_collect_only(res.stdout) if res.returncode in (0, 5) else []
        if cache is not None:
            cache.update({"hash": config_hash, "ids": ids})
        return ids

    def select(self, selections: list[TestSelection]) -> list[CommandSpec]:
        targets = sorted({s.target for s in selections if s.target})
        if not targets:
            return []
        return [
            CommandSpec(
                "pytest",
                (*_python_cmd(self.root), "-m", "pytest", "-q", *targets),
                self.root,
                tuple(targets),
            )
        ]

    def full_suite(self) -> CommandSpec | None:
        return CommandSpec("pytest", (*_python_cmd(self.root), "-m", "pytest", "-q"), self.root)

    def compile_checks(self, changed: list[str]) -> list[Check]:
        py = [
            c for c in changed if c.endswith(".py") and os.path.exists(os.path.join(self.root, c))
        ]
        if not py:
            return []
        return [Check("py_compile", (*_python_cmd(self.root), "-m", "py_compile", *py[:50]))]


class CargoAdapter(TestAdapter):
    framework = "cargo"

    def detect(self) -> float:
        return 0.95 if "Cargo.toml" in self.files else 0.0

    def packages(self) -> dict[str, str]:
        """package name -> package dir (rel)."""
        out: dict[str, str] = {}
        for f in self.files:
            if not f.endswith("Cargo.toml"):
                continue
            try:
                with open(os.path.join(self.root, f), "rb") as fh:
                    data = tomllib.load(fh)
            except (OSError, tomllib.TOMLDecodeError):
                continue
            pkg = data.get("package")
            if isinstance(pkg, dict) and pkg.get("name"):
                d = f.rsplit("/", 1)[0] if "/" in f else ""
                out[str(pkg["name"])] = d
        return out

    def is_test_file(self, rel: str) -> bool:
        return rel.endswith(".rs") and "/tests/" in "/" + rel

    def discover(self) -> list[TestCase]:
        out = []
        for name, d in self.packages().items():
            out.append(TestCase(f"cargo:{name}:lib", "cargo", f"{name} (unit)", d, package=name))
            prefix = f"{d}/tests/" if d else "tests/"
            for f in self.files:
                if f.startswith(prefix) and f.endswith(".rs") and f.count("/") == prefix.count("/"):
                    stem = f.rsplit("/", 1)[-1][:-3]
                    out.append(
                        TestCase(
                            f"cargo:{name}:test:{stem}",
                            "cargo",
                            f"{name} --test {stem}",
                            f,
                            package=name,
                        )
                    )
        return out

    def static_edges(
        self, changed: list[str], tests: list[TestCase]
    ) -> dict[str, list[tuple[str, str, float]]]:
        pkgs = self.packages()
        edges: dict[str, list[tuple[str, str, float]]] = {}
        for c in changed:
            if not c.endswith(".rs"):
                continue
            owner = max(
                (n for n, d in pkgs.items() if not d or c.startswith(d + "/")),
                key=lambda n: len(pkgs[n]),
                default="",
            )
            if not owner:
                continue
            stem = c.rsplit("/", 1)[-1][:-3]
            for t in tests:
                if t.package != owner:
                    continue
                if t.test_id.endswith(":lib"):
                    edges.setdefault(t.test_id, []).append((c, "STATIC", EDGE_WEIGHTS["STATIC"]))
                elif t.file_path == c:
                    continue
                else:
                    w = (
                        EDGE_WEIGHTS["NAME"]
                        if stem in t.test_id.rsplit(":", 1)[-1]
                        else STATIC_TRANSITIVE[3]
                    )
                    edges.setdefault(t.test_id, []).append(
                        (c, "NAME" if w == EDGE_WEIGHTS["NAME"] else "STATIC", w)
                    )
        return edges

    def select(self, selections: list[TestSelection]) -> list[CommandSpec]:
        cargo = resolve_executable("cargo")
        by_pkg: dict[str, list[str]] = {}
        for s in selections:
            parts = s.test_id.split(":")
            if len(parts) < 3:
                continue
            by_pkg.setdefault(parts[1], [])
            if parts[2] == "test":
                by_pkg[parts[1]].append(parts[3])
            else:
                by_pkg[parts[1]].append("--lib")
        out = []
        for pkg, targets in sorted(by_pkg.items()):
            args: list[str] = [cargo, "test", "-p", pkg]
            if "--lib" not in targets:
                for t in sorted(set(targets)):
                    args += ["--test", t]
            out.append(CommandSpec("cargo", tuple(args), self.root, (pkg,)))
        return out

    def full_suite(self) -> CommandSpec | None:
        return CommandSpec("cargo", (resolve_executable("cargo"), "test", "--workspace"), self.root)

    def compile_checks(self, changed: list[str]) -> list[Check]:
        if not any(c.endswith(".rs") for c in changed):
            return []
        return [
            Check("cargo_check", (resolve_executable("cargo"), "check", "--workspace", "--quiet"))
        ]


class CTestAdapter(TestAdapter):
    framework = "ctest"

    def build_dir(self) -> str:
        for f in self.files:
            if f.endswith("CTestTestfile.cmake") and f.count("/") <= 3:
                return os.path.join(self.root, f.rsplit("/", 1)[0])
        return ""

    def detect(self) -> float:
        if "CMakeLists.txt" not in self.files:
            return 0.0
        try:
            with open(
                os.path.join(self.root, "CMakeLists.txt"), encoding="utf-8", errors="replace"
            ) as fh:
                body = fh.read(400_000)
        except OSError:
            return 0.0
        if not re.search(r"(?i)\b(?:enable_testing|add_test|ctest)\b", body):
            return 0.0
        return 0.9 if self.build_dir() else 0.5  # no configured build tree: never run configure

    def discover(self) -> list[TestCase]:
        bdir = self.build_dir()
        ctest = shutil.which("ctest")
        if not bdir or not ctest:
            return []
        from .proc import run_argv

        res = run_argv(
            [ctest, "--show-only=json-v1", "--test-dir", bdir], cwd=self.root, timeout=20
        )
        names = parse_ctest_json(res.stdout) if res.ok else []
        if not names:
            res = run_argv([ctest, "-N", "--test-dir", bdir], cwd=self.root, timeout=20)
            names = parse_ctest_n(res.stdout) if res.ok else []
        return [TestCase(f"ctest:{n}", "ctest", n) for n in names]

    def static_edges(
        self, changed: list[str], tests: list[TestCase]
    ) -> dict[str, list[tuple[str, str, float]]]:
        edges: dict[str, list[tuple[str, str, float]]] = {}
        for c in changed:
            stem = os.path.splitext(c.rsplit("/", 1)[-1])[0].lower()
            if len(stem) < 3:
                continue
            for t in tests:
                if stem in t.canonical_name.lower():
                    edges.setdefault(t.test_id, []).append((c, "NAME", EDGE_WEIGHTS["NAME"]))
        return edges

    def select(self, selections: list[TestSelection]) -> list[CommandSpec]:
        names = sorted({s.target for s in selections})
        bdir = self.build_dir()
        if not names or not bdir:
            return []
        regex = "^(" + "|".join(re.escape(n) for n in names) + ")$"
        return [
            CommandSpec(
                "ctest",
                (
                    resolve_executable("ctest"),
                    "--test-dir",
                    bdir,
                    "--output-on-failure",
                    "-R",
                    regex,
                ),
                self.root,
                tuple(names),
            )
        ]

    def full_suite(self) -> CommandSpec | None:
        bdir = self.build_dir()
        if not bdir:
            return None
        return CommandSpec(
            "ctest",
            (resolve_executable("ctest"), "--test-dir", bdir, "--output-on-failure"),
            self.root,
        )


def parse_ctest_json(text: str) -> list[str]:
    """Test names from ``ctest --show-only=json-v1``."""
    try:
        data = json.loads(text or "")
    except ValueError:
        return []
    return [
        str(t.get("name")) for t in data.get("tests", []) if isinstance(t, dict) and t.get("name")
    ]


def parse_collect_only(text: str) -> list[str]:
    """Node ids from ``pytest --collect-only -q`` (summary/warning lines dropped)."""
    out = []
    for line in (text or "").replace("\r\n", "\n").splitlines():
        line = line.strip()
        if "::" in line and not line.startswith(("=", "<", "ERROR", "WARNING")) and " " not in line:
            out.append(line)
    return out


def parse_ctest_n(text: str) -> list[str]:
    return re.findall(r"(?m)^\s*Test\s+#\d+:\s+(\S+)", text or "")


_JS_TEST_RE = re.compile(r"(?:^|/)(?:__tests__/.+|[^/]+\.(?:test|spec)\.(?:[cm]?[jt]sx?))$")
_JS_IMPORT_RE = re.compile(r"""(?:from\s+|require\(\s*|import\(\s*)['"](\.{1,2}/[^'"]+)['"]""")


def package_manager(root: str, files: list[str]) -> str:
    """Manager the project uses (lockfile / ``packageManager``), never a substitute."""
    names = set(files)
    try:
        with open(os.path.join(root, "package.json"), encoding="utf-8") as fh:
            declared = str(json.load(fh).get("packageManager") or "")
        if declared:
            return declared.split("@", 1)[0]
    except (OSError, ValueError):
        pass
    for lock, pm in (
        ("pnpm-lock.yaml", "pnpm"),
        ("yarn.lock", "yarn"),
        ("bun.lockb", "bun"),
        ("bun.lock", "bun"),
        ("package-lock.json", "npm"),
    ):
        if lock in names:
            return pm
    return "npm"


def _pm_exec(pm: str, tool: str) -> list[str]:
    if pm == "pnpm":
        return [resolve_executable("pnpm"), "exec", tool]
    if pm == "yarn":
        return [resolve_executable("yarn"), tool]
    if pm == "bun":
        return [resolve_executable("bunx"), tool]
    return [resolve_executable("npx"), "--no-install", tool]


class JsAdapter(TestAdapter):
    def __init__(self, root: str, files: list[str]) -> None:
        super().__init__(root, files)
        self.framework = ""
        self.pm = "npm"
        self.scripts: dict[str, str] = {}

    def detect(self) -> float:
        if "package.json" not in self.files:
            return 0.0
        try:
            with open(os.path.join(self.root, "package.json"), encoding="utf-8") as fh:
                pkg = json.load(fh)
        except (OSError, ValueError):
            return 0.0
        deps = {**(pkg.get("dependencies") or {}), **(pkg.get("devDependencies") or {})}
        self.scripts = pkg.get("scripts") or {}
        test_script = str(self.scripts.get("test", ""))
        if (
            "vitest" in deps
            or "vitest" in test_script
            or any(f.startswith("vitest.config.") for f in self.files)
        ):
            self.framework = "vitest"
        elif (
            "jest" in deps
            or "jest" in test_script
            or any(f.startswith("jest.config.") for f in self.files)
        ):
            self.framework = "jest"
        else:
            return 0.0
        self.pm = package_manager(self.root, self.files)
        return 0.9

    def is_test_file(self, rel: str) -> bool:
        return bool(_JS_TEST_RE.search(rel))

    def discover(self) -> list[TestCase]:
        return [
            TestCase(f"{self.framework}:{f}", self.framework, f, f)
            for f in self.files
            if self.is_test_file(f)
        ]

    def static_edges(
        self, changed: list[str], tests: list[TestCase]
    ) -> dict[str, list[tuple[str, str, float]]]:
        changed_set = set(changed)
        edges: dict[str, list[tuple[str, str, float]]] = {}
        exts = ("", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", "/index.ts", "/index.js")
        files = set(self.files)
        for t in tests:
            try:
                with open(
                    os.path.join(self.root, t.file_path), encoding="utf-8", errors="replace"
                ) as fh:
                    body = fh.read(300_000)
            except OSError:
                continue
            base = os.path.dirname(t.file_path)
            for spec in _JS_IMPORT_RE.findall(body):
                target = os.path.normpath(os.path.join(base, spec)).replace("\\", "/")
                for ext in exts:
                    if target + ext in files and target + ext in changed_set:
                        edges.setdefault(t.test_id, []).append(
                            (target + ext, "STATIC", EDGE_WEIGHTS["STATIC"])
                        )
                        break
        return edges

    def select(self, selections: list[TestSelection]) -> list[CommandSpec]:
        targets = sorted({s.target for s in selections if s.target})
        if not targets:
            return []
        if self.framework == "vitest":
            argv = (*_pm_exec(self.pm, "vitest"), "run", *targets)
        else:
            argv = (*_pm_exec(self.pm, "jest"), *targets)
        return [CommandSpec(self.framework, tuple(argv), self.root, tuple(targets))]

    def related(self, changed: list[str]) -> CommandSpec | None:
        src = [c for c in changed if re.search(r"\.[cm]?[jt]sx?$", c) and not self.is_test_file(c)]
        if not src:
            return None
        if self.framework == "vitest":
            argv = (*_pm_exec(self.pm, "vitest"), "related", "--run", *src)
        else:
            argv = (*_pm_exec(self.pm, "jest"), "--findRelatedTests", *src)
        return CommandSpec(self.framework, tuple(argv), self.root, tuple(src))

    def full_suite(self) -> CommandSpec | None:
        if "test" in self.scripts:
            pm = resolve_executable(self.pm)
            return CommandSpec(
                self.framework,
                (pm, "run", "test") if self.pm != "yarn" else (pm, "test"),
                self.root,
            )
        return CommandSpec(
            self.framework,
            (*_pm_exec(self.pm, self.framework), *(["run"] if self.framework == "vitest" else [])),
            self.root,
        )


class GenericAdapter(TestAdapter):
    """Project-defined test commands only; never invents a selector."""

    framework = "generic"

    def detect(self) -> float:
        return 0.5 if self.full_suite() is not None else 0.0

    def full_suite(self) -> CommandSpec | None:
        names = set(self.files)
        for mk in ("Makefile", "makefile", "GNUmakefile"):
            if mk in names:
                try:
                    with open(
                        os.path.join(self.root, mk), encoding="utf-8", errors="replace"
                    ) as fh:
                        if re.search(r"(?m)^test\s*:", fh.read(400_000)):
                            return CommandSpec(
                                "generic", (resolve_executable("make"), "test"), self.root
                            )
                except OSError:
                    pass
        if "tox.ini" in names and shutil.which("tox"):
            return CommandSpec("generic", (resolve_executable("tox"),), self.root)
        return None


ADAPTERS: tuple[type[TestAdapter], ...] = (
    PytestAdapter,
    CargoAdapter,
    CTestAdapter,
    JsAdapter,
    GenericAdapter,
)


# ------------------------------------------------------------------- risk
_INTERFACE_RE = re.compile(
    r"(?i)(?:^|/)(?:api|apis|cli|public|schema|schemas|proto|protocol|interfaces?)(?:/|\.)|\.proto$|serializ|openapi|_pb2|"
    r"(?:^|/)__init__\.py$|\.d\.ts$|(?:^|/)include/.+\.h(?:pp)?$|(?:^|/)lib\.rs$|(?:^|/)mod\.rs$"
)
_SHARED_RE = re.compile(
    r"(?i)(?:^|/)(?:base|common|shared|utils?|helpers?|types|core|models?)(?:/|\.|_)"
)
_CONFIG_RE = re.compile(
    r"(?i)(?:^|/)(?:pyproject\.toml|setup\.py|setup\.cfg|requirements[^/]*\.txt|package\.json|Cargo\.toml|CMakeLists\.txt|"
    r"[^/]+\.cmake|Makefile|makefile|GNUmakefile|tox\.ini|noxfile\.py|Dockerfile|go\.mod|build\.gradle(?:\.kts)?|pom\.xml|"
    r"tsconfig[^/]*\.json|vite\.config\.[^/]+|jest\.config\.[^/]+|vitest\.config\.[^/]+|pytest\.ini|conftest\.py|"
    r"\.github/workflows/.+|azure-pipelines\.yml|\.gitlab-ci\.yml)$"
)
_PLATFORM_RE = re.compile(
    r"(?i)(?:windows|win32|win64|(?:^|[/_.-])win(?=[_.-])|winapi|msvc|posix|darwin|macos|linux|_nt\b|msvcrt|fcntl)"
)
_CROSS_PLATFORM_RE = re.compile(
    r"(?i)(?:^|/)(?:compat|platform|paths?|subprocess|_subprocess|os_utils|shell)\.\w+$"
)
MANDATORY_TIER3: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "WIRE_PROTOCOL_OR_SERIALIZATION",
        re.compile(r"(?i)(?:\.proto$|serializ|(?:^|/)wire|codec|(?:^|/)protocol)"),
    ),
    (
        "DATABASE_MIGRATION_OR_SCHEMA",
        re.compile(r"(?i)(?:(?:^|/)migrations?/|alembic|schema\.sql$|(?:^|/)migrate|_migration)"),
    ),
    (
        "AUTH_CORE",
        re.compile(
            r"(?i)(?:^|/)(?:auth|authn|authz|authentication|authorization|permissions?)(?:/|\.|_)"
        ),
    ),
    (
        "DEPENDENCY_RESOLUTION",
        re.compile(
            r"(?i)(?:^|/)(?:requirements[^/]*\.txt|poetry\.lock|uv\.lock|Pipfile(?:\.lock)?|package-lock\.json|yarn\.lock|pnpm-lock\.yaml|Cargo\.lock|go\.sum|bun\.lockb?)$"
        ),
    ),
    (
        "ROOT_BUILD_CONFIG",
        re.compile(
            r"^(?:pyproject\.toml|setup\.py|setup\.cfg|Cargo\.toml|CMakeLists\.txt|package\.json|Makefile|go\.mod|build\.gradle(?:\.kts)?|pom\.xml)$"
        ),
    ),
    (
        "SHARED_CLI_PARSING",
        re.compile(r"(?i)(?:^|/)cli/(?:__init__|main|app|root)\.\w+$|(?:^|/)(?:__main__|cli)\.py$"),
    ),
    (
        "CORE_PROXY_MODEL",
        re.compile(r"(?i)(?:^|/)proxy/(?:server|models|request|response|handlers/base)\.\w+$"),
    ),
)
_FULL_SUITE_REQUEST_RE = re.compile(
    r"(?i)\b(?:full (?:test )?suite|entire (?:test )?suite|all (?:the )?tests|whole (?:test )?suite|run everything)\b"
)


def risk_components(
    *,
    changed: list[tuple[str, str]],
    expected_subsystems: tuple[str, ...],
    unresolved_scope_warnings: int,
    historical_failure: float,
    evidence_confidences: list[float],
    change_files: int,
    change_lines: int,
    budget: tuple[int | None, int | None],
    importers: dict[str, int] | None = None,
    platforms_with_evidence: int = 1,
    path_text: dict[str, str] | None = None,
) -> dict[str, float]:
    """Deterministic risk components, each 0..1 (plan §9.7 definitions)."""
    paths = [p for p, _ in changed]
    classes = [c for _, c in changed]
    subs = {"/".join(p.split("/")[:2]) for p in paths if "/" in p}
    new_subs = [s for s in subs if s not in expected_subsystems]
    if unresolved_scope_warnings or len(new_subs) > 1 or "UNRELATED" in classes:
        scope = 1.0
    elif "DEPENDENCY_SCOPE" in classes or new_subs:
        scope = 0.5
    else:
        scope = 0.0
    importers = importers or {}
    if any(_INTERFACE_RE.search(p) for p in paths):
        interface = 1.0
    elif any(_SHARED_RE.search(p) or importers.get(p, 0) >= 3 for p in paths):
        interface = 0.5
    elif paths:
        interface = 0.1
    else:
        interface = 0.0
    config = 1.0 if any(_CONFIG_RE.search(p) for p in paths) else 0.0
    texts = path_text or {}
    if any(_PLATFORM_RE.search(p) or _PLATFORM_RE.search(texts.get(p, "")) for p in paths):
        platform = 1.0 if platforms_with_evidence <= 1 else 0.5
    elif any(_CROSS_PLATFORM_RE.search(p) for p in paths):
        platform = 0.5
    else:
        platform = 0.0
    uncertainty = (
        max(0.0, 1.0 - (sum(evidence_confidences) / len(evidence_confidences)))
        if evidence_confidences
        else 1.0
    )
    max_files, max_lines = budget
    ratios = []
    if max_files:
        ratios.append(change_files / max_files)
    if max_lines:
        ratios.append(change_lines / max_lines)
    r = max(ratios) if ratios else 0.0
    size = 0.0 if r <= 1.0 else min(1.0, (r - 1.0) / 2.0)
    return {
        "scope_risk": scope,
        "interface_risk": interface,
        "config_build_risk": config,
        "historical_failure_risk": min(1.0, max(0.0, historical_failure)),
        "platform_risk": platform,
        "evidence_uncertainty": uncertainty,
        "change_size_risk": size,
    }


RISK_WEIGHTS = {
    "scope_risk": 0.25,
    "interface_risk": 0.20,
    "config_build_risk": 0.15,
    "historical_failure_risk": 0.15,
    "platform_risk": 0.10,
    "evidence_uncertainty": 0.10,
    "change_size_risk": 0.05,
}


def risk_score(components: dict[str, float]) -> float:
    return round(sum(RISK_WEIGHTS[k] * components.get(k, 0.0) for k in RISK_WEIGHTS), 6)


def mandatory_tier3(paths: list[str], *, user_text: str = "") -> list[str]:
    out = []
    for code, rx in MANDATORY_TIER3:
        if any(rx.search(p) for p in paths):
            out.append(code)
    if _FULL_SUITE_REQUEST_RE.search(user_text or ""):
        out.append("USER_REQUESTED_FULL_SUITE")
    return out


def tiers_to_run(
    risk: float, *, t2: float, t3: float, mandatory: bool, criteria_demand_tier2: bool
) -> int:
    """Highest tier the escalation rules allow (each only after the previous passes)."""
    if mandatory or risk >= t3:
        return 3
    if risk >= t2:
        return 2
    return 2 if criteria_demand_tier2 else 1


# ------------------------------------------------------------------ planner
class TestImpactPlanner:
    def __init__(self, runtime: Any) -> None:
        self.rt = runtime
        self.store = runtime.store
        self.config = runtime.config
        self._plan_cache: tuple[str, VerificationPlan] | None = None

    @property
    def root(self) -> str:
        return str(self.rt.workspace.root)

    # ------------------------------------------------------------ project
    def project(self) -> tuple[list[str], list[tuple[TestAdapter, float]]]:
        facts = self.rt.workspace.facts
        cached = facts.get("test_project")
        now = time.time()
        if cached and now - cached[0] < 60:
            return cached[1], cached[2]
        files = _walk(self.root) if self.rt.workspace.local else []
        adapters = []
        for cls in ADAPTERS:
            a = cls(self.root, files)
            try:
                conf = a.detect()
            except Exception:  # noqa: BLE001
                conf = 0.0
            if conf >= 0.5:
                adapters.append((a, conf))
        facts["test_project"] = (now, files, adapters)
        return files, adapters

    def config_hash(self) -> str:
        """Identity of the project's build/test configuration (macro invalidation)."""
        files, adapters = self.project()
        h = hashlib.sha256()
        for f in files:
            if _CONFIG_RE.search(f) or f.endswith(("pytest.ini", "conftest.py")):
                try:
                    with open(os.path.join(self.root, f), "rb") as fh:
                        h.update(f.encode() + b"\0" + fh.read(1_000_000))
                except OSError:
                    continue
        h.update(",".join(sorted(a.framework for a, _ in adapters)).encode())
        return h.hexdigest()[:20]

    # ----------------------------------------------------------- change set
    def change_set(self) -> list[tuple[str, str]]:
        if self.rt.scope is not None:
            return [
                c
                for c in self.rt.scope.task_owned_changes(refresh_git=self.rt.workspace.local)
                if c[1] != "GENERATED_EFFECT"
            ]
        task_id = self.rt.task_id
        rows = (
            self.store.query(
                "SELECT DISTINCT path_refs FROM events WHERE task_id = ? AND event_type IN ('FILE_WRITE','FILE_DELETE')",
                (task_id,),
            )
            if task_id
            else []
        )
        out = set()
        for r in rows:
            for p in loads(r["path_refs"], []) or []:
                out.add(self.rt.workspace.relpath(p))
        return sorted((p, "IN_SCOPE") for p in out)

    # ------------------------------------------------------------- plan
    def plan(self, *, force: bool = False) -> VerificationPlan | None:
        changed = self.change_set()
        if not changed:
            return None
        chash = hashlib.sha256(json.dumps(changed).encode()).hexdigest()[:16]
        if not force and self._plan_cache is not None and self._plan_cache[0] == chash:
            return self._plan_cache[1]
        started = time.perf_counter()
        plan = self._build(changed, chash)
        self._plan_cache = (chash, plan)
        self.rt.metrics.observe("test_plan", (time.perf_counter() - started) * 1000.0)
        self.rt.metrics.bump("verification_plans")
        self.store.write(
            lambda c: c.execute(
                "INSERT OR REPLACE INTO verification_plans(plan_id, task_id, change_set_hash, risk, plan_json, created_at) VALUES (?,?,?,?,?,?)",
                (
                    plan.plan_id,
                    plan.task_id,
                    chash,
                    plan.risk_score,
                    dumps(plan.to_json()),
                    time.time(),
                ),
            )
        )
        return plan

    def _build(self, changed: list[tuple[str, str]], chash: str) -> VerificationPlan:
        files, adapters = self.project()
        paths = [p for p, _ in changed]
        tests_all: list[tuple[TestAdapter, TestCase]] = []
        weights: dict[str, list[tuple[str, str, float]]] = {}
        adapter_of: dict[str, TestAdapter] = {}
        for a, _conf in adapters:
            tests = a.discover()
            for t in tests:
                tests_all.append((a, t))
                adapter_of[t.test_id] = a
            for tid, edges in a.static_edges(paths, tests).items():
                weights.setdefault(tid, []).extend(edges)
            for t in tests:
                if t.file_path and t.file_path in paths:
                    weights.setdefault(t.test_id, []).append((t.file_path, "PROJECT_RULE", 1.0))
                for c in paths:
                    stem = os.path.splitext(c.rsplit("/", 1)[-1])[0].lower()
                    base = (
                        t.file_path.rsplit("/", 1)[-1].lower()
                        if t.file_path
                        else t.canonical_name.lower()
                    )
                    if (
                        len(stem) > 2
                        and c != t.file_path
                        and re.search(rf"(?:^|[_./-]){re.escape(stem)}(?:[_./-]|$)", base)
                    ):
                        weights.setdefault(t.test_id, []).append((c, "NAME", EDGE_WEIGHTS["NAME"]))
        # Persisted (historical / failure / coverage / project-rule) edges.
        now = time.time()
        for c in paths:
            for r in self.store.query(
                "SELECT * FROM impact_edges WHERE source_resource_id = ?", (f"file:{c}",)
            ):
                w = float(r["weight"])
                if r["edge_type"] == "HISTORICAL":
                    w = decayed(
                        min(w, EDGE_WEIGHTS["HISTORICAL"]), float(r["last_observed_at"]), now=now
                    )
                if w > 0:
                    weights.setdefault(r["test_id"], []).append((c, r["edge_type"], w))
        # Tests that failed for these resources before (Tier 1 by rule).
        failing_before = {
            r["test_id"]
            for r in self.store.query(
                "SELECT test_id FROM test_cases WHERE last_status IN ('fail','error','flaky')"
            )
        }
        scores = {tid: combine([w for _, _, w in edges]) for tid, edges in weights.items()}
        selections: dict[str, TestSelection] = {}
        for a, t in tests_all:
            score = scores.get(t.test_id, 0.0)
            reasons = tuple(sorted({e for _, e, _ in weights.get(t.test_id, [])}))
            target = (
                t.file_path
                if a.framework in ("pytest", "jest", "vitest")
                else (t.canonical_name if a.framework == "ctest" else t.test_id)
            )
            if t.test_id in failing_before and score > 0:
                reasons = (*reasons, "PREVIOUSLY_FAILED")
                score = max(score, TIER1_SCORE)
            selections[t.test_id] = TestSelection(t.test_id, a.framework, target, score, reasons)
        explicit = self._user_explicit_tests(tests_all)
        tier1 = sorted(
            [s for s in selections.values() if s.score >= TIER1_SCORE or s.test_id in explicit],
            key=lambda s: (-s.score, s.test_id),
        )
        tier2 = sorted(
            [
                s
                for s in selections.values()
                if TIER2_SCORE <= s.score < TIER1_SCORE and s.test_id not in explicit
            ],
            key=lambda s: (-s.score, s.test_id),
        )
        # Risk.
        scope = self.rt.scope
        contract = scope.contract if scope is not None else None
        warnings = len(scope.pending_warnings()) if scope is not None else 0
        hist = self._historical_failure([s.test_id for s in (*tier1, *tier2)])
        confidences = [s.score for s in (*tier1, *tier2)]
        lines = self._lines(changed)
        budget = contract.max_change_budget if contract is not None else (20, 1500)
        comps = risk_components(
            changed=changed,
            expected_subsystems=contract.expected_subsystems if contract is not None else (),
            unresolved_scope_warnings=warnings,
            historical_failure=hist,
            evidence_confidences=confidences,
            change_files=len(changed),
            change_lines=lines,
            budget=budget,
            path_text=self._path_text(paths),
        )
        risk = risk_score(comps)
        user_text = self._user_text()
        mandatory = mandatory_tier3(paths, user_text=user_text)
        plan = VerificationPlan(
            plan_id="P" + stable_id(self.rt.session_key, chash, n=12),
            task_id=self.rt.task_id,
            change_set_hash=chash,
            risk_score=risk,
            components=comps,
            tier1=tier1,
            tier2=tier2,
            mandatory_tier3=mandatory,
            changed=paths,
            tests_considered=len(tests_all),
        )
        plan._t2, plan._t3 = self.config.test_risk_tier2, self.config.test_risk_tier3
        # Commands per tier.
        by_adapter1: dict[int, list[TestSelection]] = {}
        for s in tier1:
            by_adapter1.setdefault(id(adapter_of[s.test_id]), []).append(s)
        plan.commands[1] = [
            cmd for a, _ in adapters for cmd in a.select(by_adapter1.get(id(a), []))
        ]
        by_adapter2: dict[int, list[TestSelection]] = {}
        for s in tier2:
            by_adapter2.setdefault(id(adapter_of[s.test_id]), []).append(s)
        plan.commands[2] = [
            cmd for a, _ in adapters for cmd in a.select(by_adapter2.get(id(a), []))
        ]
        for a, _ in adapters:
            if isinstance(a, JsAdapter):
                rel = a.related(paths)
                if rel is not None:
                    plan.commands[2].append(rel)
            plan.required_non_test_checks.extend(a.compile_checks(paths))
        plan.commands[3] = [c for c in (a.full_suite() for a, _ in adapters) if c is not None]
        plan.tier3 = [
            TestSelection(f"{c.framework}:full", c.framework, "", 1.0, ("FULL_SUITE",))
            for c in plan.commands[3]
        ]
        demand2 = self._criteria_demand_tier2()
        top = tiers_to_run(
            risk, t2=plan._t2, t3=plan._t3, mandatory=bool(mandatory), criteria_demand_tier2=demand2
        )
        plan.escalation_rules = [
            f"R={risk:.2f}: run tier 1"
            + (
                ""
                if top == 1
                else f", then tier {'2' if top == 2 else '2 then 3'} if the previous tier passes"
            ),
            "stop at the first failing tier; fix it before running broader tests",
            "a known-flaky test is re-run once; differing outcomes are FLAKY, not passing",
        ]
        codes = []
        if not tier1 and not tier2:
            codes.append("NO_IMPACTED_TESTS_FOUND")
        codes += [k.upper() for k, v in comps.items() if v >= 0.5]
        codes += mandatory
        if demand2:
            codes.append("FEATURE_TASK_REQUIRES_TIER2")
        plan.rationale_codes = codes
        return plan

    def _lines(self, changed: list[tuple[str, str]]) -> int:
        task_id = self.rt.task_id
        if not task_id:
            return 0
        row = self.store.query_one(
            "SELECT COALESCE(SUM(lines_changed),0) AS l FROM task_changes WHERE task_id = ?",
            (task_id,),
        )
        return int(row["l"]) if row is not None else 0

    def _path_text(self, paths: list[str]) -> dict[str, str]:
        out = {}
        for p in paths[:20]:
            try:
                with open(os.path.join(self.root, p), encoding="utf-8", errors="replace") as fh:
                    out[p] = fh.read(20_000)
            except OSError:
                continue
        return out

    def _historical_failure(self, test_ids: list[str]) -> float:
        if not test_ids:
            return 0.0
        rates = []
        for tid in test_ids[:100]:
            row = self.store.query_one("SELECT history FROM test_cases WHERE test_id = ?", (tid,))
            hist = loads(row["history"], []) if row is not None else []
            recent = hist[-10:]
            if recent:
                rates.append(
                    sum(1 for h in recent if h.get("status") in ("fail", "error")) / len(recent)
                )
        return max(rates) if rates else 0.0

    def _user_text(self) -> str:
        ts = self.rt.task_state
        state = ts.state if ts is not None else None
        if state is None:
            return ""
        return " ".join(
            a.text
            for a in state.atoms
            if a.origin.value == "USER" and a.state.value not in ("SUPERSEDED", "REJECTED")
        )

    def _user_explicit_tests(self, tests: list[tuple[TestAdapter, TestCase]]) -> set[str]:
        ts = self.rt.task_state
        state = ts.state if ts is not None else None
        if state is None:
            return set()
        targets = {t.split("::")[0] for c in state.acceptance_criteria for t in c.targets}
        return {t.test_id for _, t in tests if t.file_path and t.file_path in targets}

    def _criteria_demand_tier2(self) -> bool:
        ts = self.rt.task_state
        state = ts.state if ts is not None else None
        if state is None or state.primary_goal is None:
            return False
        goal = state.primary_goal.text.lower()
        return bool(
            re.search(r"\b(?:implement|add|feature|refactor|support|introduce|build)\b", goal)
        )

    # ----------------------------------------------------- history learning
    def on_tool_result(self, ev: AgentEvent, records: list[Any]) -> None:
        inv = ev.transient.get("invocation")
        if inv is not None and (inv.paths_written or inv.paths_deleted):
            self._plan_cache = None  # §11.5: invalidate; rebuild lazily
        for rec in records:
            if getattr(rec, "claim_type", "") == "test_aggregate" and isinstance(rec.value, dict):
                self.learn_from_run(rec.value, source_event_id=ev.event_id, tier=None)

    def learn_from_run(
        self, value: dict[str, Any], *, source_event_id: str, tier: int | None
    ) -> None:
        fw = str(value.get("framework") or "generic")
        failed = [str(x) for x in value.get("failed_ids") or []]
        passed = [str(x) for x in value.get("passed_ids") or []]
        command = str(value.get("command") or "")
        changed = self.change_set() if self.rt.task_id else []
        changed_paths = [p for p, _ in changed]
        snapshot = {p: _quick_hash(os.path.join(self.root, p)) for p in changed_paths[:50]}
        chash = hashlib.sha256(json.dumps(sorted(snapshot.items())).encode()).hexdigest()[:16]
        now = time.time()
        run_id = "R" + stable_id(source_event_id, command, n=14)
        inserted = self.store.write(
            lambda c: (
                c.execute(
                    "INSERT OR IGNORE INTO test_runs(run_id, task_id, framework, command_fingerprint, change_set_hash, changed_resources, passed, failed, skipped, errors, failed_ids, passed_ids, tier, ts) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        run_id,
                        self.rt.task_id,
                        fw,
                        command[:200],
                        chash,
                        dumps(changed_paths[:100]),
                        int(value.get("passed") or 0),
                        int(value.get("failed") or 0),
                        int(value.get("skipped") or 0),
                        int(value.get("errors") or 0),
                        dumps(failed[:200]),
                        dumps(passed[:200]),
                        tier,
                        now,
                    ),
                ).rowcount
            ),
            default=0,
        )
        if not inserted:
            return  # replayed event: no double learning
        self.rt.metrics.bump("verification_runs")
        if value.get("full_suite"):
            self.rt.metrics.bump("full_suite_runs")
        mapped_ids = {self._case_id(fw, t, command): t for t in [*failed, *passed]}
        for case_id, raw_id in mapped_ids.items():
            status = "fail" if raw_id in failed else "pass"
            self._update_case(case_id, fw, raw_id, status, chash, snapshot, now)

    def _case_id(self, fw: str, raw_id: str, command: str) -> str:
        if fw in ("pytest", "jest", "vitest"):
            return f"{fw}:{raw_id.split('::')[0].split(' > ')[0]}"
        if fw == "ctest":
            return f"ctest:{raw_id}"
        return f"{fw}:{raw_id}"

    def _update_case(
        self,
        case_id: str,
        fw: str,
        raw_id: str,
        status: str,
        chash: str,
        snapshot: dict[str, str],
        now: float,
    ) -> None:
        row = self.store.query_one("SELECT * FROM test_cases WHERE test_id = ?", (case_id,))
        history = loads(row["history"], []) if row is not None else []
        flake = float(row["flake_score"]) if row is not None else 0.0
        prev = history[-1] if history else None
        if prev is not None:
            if prev.get("status") != status and prev.get("change") == chash:
                flake = min(1.0, flake + 0.34)  # outcome flipped with no relevant code change
            else:
                flake = max(0.0, flake * 0.9)
        history.append({"ts": now, "status": status, "change": chash, "files": snapshot})
        history = history[-10:]
        file_path = raw_id.split("::")[0] if fw in ("pytest", "jest", "vitest") else ""

        def run(c: Any) -> None:
            c.execute(
                "INSERT INTO test_cases(test_id, framework, canonical_name, file_path, flake_score, last_status, last_run_at, last_change_hash, history) "
                "VALUES (?,?,?,?,?,?,?,?,?) ON CONFLICT(test_id) DO UPDATE SET flake_score = excluded.flake_score, "
                "last_status = excluded.last_status, last_run_at = excluded.last_run_at, last_change_hash = excluded.last_change_hash, history = excluded.history",
                (case_id, fw, raw_id[:300], file_path, flake, status, now, chash, dumps(history)),
            )
            # Historical association: failures count more than passes (§9.6).
            for p in snapshot:
                src = f"file:{p}"
                inc = 0.10 if status == "fail" else 0.02
                c.execute(
                    "INSERT INTO impact_edges(source_resource_id, test_id, edge_type, weight, evidence_count, last_observed_at) VALUES (?,?,?,?,?,?) "
                    "ON CONFLICT(source_resource_id, test_id, edge_type) DO UPDATE SET evidence_count = impact_edges.evidence_count + 1, "
                    "weight = MIN(?, impact_edges.weight + ?), last_observed_at = excluded.last_observed_at",
                    (src, case_id, "HISTORICAL", inc, 1, now, EDGE_WEIGHTS["HISTORICAL"], inc),
                )
            # Fail -> fix -> pass: the files that changed in between get a FAILURE edge.
            if status == "pass" and prev is not None and prev.get("status") in ("fail", "error"):
                before = prev.get("files") or {}
                fixed = [p for p in set(before) | set(snapshot) if before.get(p) != snapshot.get(p)]
                if 0 < len(fixed) <= 10:
                    for p in fixed:
                        c.execute(
                            "INSERT INTO impact_edges(source_resource_id, test_id, edge_type, weight, evidence_count, last_observed_at) VALUES (?,?,?,?,?,?) "
                            "ON CONFLICT(source_resource_id, test_id, edge_type) DO UPDATE SET evidence_count = impact_edges.evidence_count + 1, last_observed_at = excluded.last_observed_at",
                            (f"file:{p}", case_id, "FAILURE", EDGE_WEIGHTS["FAILURE"], 1, now),
                        )

        self.store.write(run)

    def preexisting_failures(self, failed_ids: list[str], framework: str) -> list[str]:
        """Failures already present before any task-owned change (§9.10)."""
        task_id = self.rt.task_id
        if not task_id or not failed_ids:
            return []
        if not self.change_set():
            return list(failed_ids)  # nothing task-owned has changed yet
        base = self.store.query_one(
            "SELECT created_at FROM git_baselines WHERE task_id = ?", (task_id,)
        )
        task_row = self.store.query_one(
            "SELECT created_at FROM tasks WHERE task_id = ?", (task_id,)
        )
        started = (
            float(base["created_at"])
            if base is not None
            else (float(task_row["created_at"]) if task_row is not None else 0.0)
        )
        out = []
        for raw in failed_ids:
            cid = self._case_id(framework, raw, "")
            row = self.store.query_one("SELECT history FROM test_cases WHERE test_id = ?", (cid,))
            hist = loads(row["history"], []) if row is not None else []
            if any(
                h.get("status") in ("fail", "error")
                and (h.get("ts", 0) < started or not h.get("files"))
                for h in hist
            ):
                out.append(raw)
        return out

    # ------------------------------------------------------------- execute
    def run_plan(
        self, plan: VerificationPlan, *, max_tier: int | None = None, timeout: float = 900.0
    ) -> dict[str, Any]:
        """Execute tiers in order, stopping at the first failure (§9.9-§9.11)."""
        from .proc import run_argv

        top = max_tier or tiers_to_run(
            plan.risk_score,
            t2=self.config.test_risk_tier2,
            t3=self.config.test_risk_tier3,
            mandatory=bool(plan.mandatory_tier3),
            criteria_demand_tier2=self._criteria_demand_tier2(),
        )
        summary: dict[str, Any] = {
            "plan_id": plan.plan_id,
            "risk": plan.risk_score,
            "tiers": [],
            "status": "passed",
            "evidence_refs": [],
        }
        for check in plan.required_non_test_checks:
            res = run_argv(check.argv, cwd=self.root, timeout=min(timeout, 300))
            if not res.ok:
                summary["status"] = "failed"
                summary["failed_check"] = {
                    "name": check.name,
                    "error": res.output[-600:] or res.error,
                }
                return summary
        tests_run = 0
        for tier in (1, 2, 3):
            if tier > top:
                break
            cmds = plan.commands.get(tier) or []
            if not cmds:
                continue
            tier_result: dict[str, Any] = {
                "tier": tier,
                "commands": [],
                "passed": 0,
                "failed": 0,
                "flaky": [],
            }
            for spec in cmds:
                res = run_argv(spec.argv, cwd=spec.cwd, timeout=timeout)
                parsed = parse_test_output(res.output, spec.framework)
                ok = res.ok and (parsed is None or parsed.ok)
                refs = self._record_run(spec, res, parsed, tier)
                summary["evidence_refs"].extend(refs)
                if parsed is not None:
                    tests_run += parsed.total
                    tier_result["passed"] += parsed.passed
                    tier_result["failed"] += parsed.failed + parsed.errors
                if not ok and parsed is not None and parsed.failed_ids:
                    flaky = self._flaky_ids(spec.framework, parsed.failed_ids)
                    if flaky and set(flaky) == set(parsed.failed_ids):
                        rerun = run_argv(
                            spec.argv, cwd=spec.cwd, timeout=timeout
                        )  # one rerun maximum
                        reparsed = parse_test_output(rerun.output, spec.framework)
                        summary["evidence_refs"].extend(
                            self._record_run(spec, rerun, reparsed, tier)
                        )
                        if reparsed is not None and reparsed.ok:
                            tier_result["flaky"] = flaky
                            self._mark_flaky(spec.framework, flaky)
                            ok = False  # differing outcomes: FLAKY/AMBIGUOUS, not passing
                tier_result["commands"].append(
                    {
                        "command": spec.display(),
                        "ok": ok,
                        "exit": res.returncode,
                        "timed_out": res.timed_out,
                    }
                )
                if not ok:
                    summary["status"] = "flaky" if tier_result["flaky"] else "failed"
                    if parsed is not None:
                        summary["failed_ids"] = parsed.failed_ids[:10]
                    else:
                        summary["error"] = res.output[-600:] or res.error
            summary["tiers"].append(tier_result)
            self.rt.metrics.bump("verification_tier_run")
            if summary["status"] != "passed":
                break  # do not run broader tests after a direct failure
        considered = max(plan.tests_considered, tests_run)
        summary["tests_run"] = tests_run
        self.rt.metrics.bump("tests_selected", tests_run)
        if top < 3 and plan.commands.get(3):
            self.rt.metrics.bump("full_suite_avoided")
            self.rt.metrics.bump("tests_avoided_estimate", max(0, considered - tests_run))
        return summary

    def _record_run(
        self, spec: CommandSpec, res: Any, parsed: TestResult | None, tier: int
    ) -> list[str]:
        sei = stable_id(self.rt.session_key, "verify", spec.display(), time.time(), n=24)
        ev = AgentEvent(
            event_id=sei,
            workspace_id=self.rt.workspace.workspace_id,
            session_id=self.rt.session_key,
            task_id=self.rt.task_id,
            event_type=EventType.VERIFICATION_RESULT,
            exit_code=res.returncode,
            success=bool(res.ok and (parsed is None or parsed.ok)),
            metadata={"tier": tier, "framework": spec.framework, "origin": "headroom"},
        )
        self.rt.persist_events([ev])
        refs: list[str] = []
        if parsed is not None and self.rt.evidence is not None:
            recs = self.rt.evidence.record_test_result(
                None, parsed, command=spec.display(), framework=spec.framework, source_event_id=sei
            )
            refs = [r.evidence_id for r in recs if r.claim_type == "test_aggregate"]
            agg = next((r for r in recs if r.claim_type == "test_aggregate"), None)
            if agg is not None:
                self.learn_from_run(agg.value, source_event_id=sei, tier=tier)
                if self.rt.task_state is not None:
                    self.rt.task_state.on_tool_result(ev, recs)
        return refs

    def _flaky_ids(self, fw: str, failed_ids: list[str]) -> list[str]:
        out = []
        for raw in failed_ids:
            row = self.store.query_one(
                "SELECT flake_score FROM test_cases WHERE test_id = ?",
                (self._case_id(fw, raw, ""),),
            )
            if row is not None and float(row["flake_score"]) >= 0.3:
                out.append(raw)
        return out

    def _mark_flaky(self, fw: str, ids: list[str]) -> None:
        for raw in ids:
            cid = self._case_id(fw, raw, "")
            self.store.write(
                lambda c, cid=cid: c.execute(
                    "UPDATE test_cases SET last_status = 'flaky' WHERE test_id = ?", (cid,)
                )
            )

    # ------------------------------------------------------------- render
    def sections(self) -> list[tuple[int, str, list[str]]]:
        """Inject a compact plan only when it is actionable.

        That is: task-owned changes lack current passing evidence *and* the agent
        is verifying (its last tool call ran tests or a build) or is declaring
        the task complete. A plan is not re-injected after every edit.
        """
        ts = self.rt.task_state
        declared = bool(ts is not None and ts.state is not None and ts.state.declared_complete)
        verifying = self.rt.last_tool_family in ("test", "build")
        if not declared and not verifying:
            return []
        plan = self.plan()
        if plan is None:
            return []
        tier1_ok, full_ok = self._coverage(plan)
        top = tiers_to_run(
            plan.risk_score,
            t2=self.config.test_risk_tier2,
            t3=self.config.test_risk_tier3,
            mandatory=bool(plan.mandatory_tier3),
            criteria_demand_tier2=self._criteria_demand_tier2(),
        )
        if full_ok or (tier1_ok and top == 1):
            return []  # current changes already have sufficient passing evidence
        if not declared and not tier1_ok:
            # Mid-iteration with failing or partial runs: the agent is already
            # verifying; re-sending the plan on every run adds nothing.
            return []
        lines = [f"risk={plan.risk_score:.2f}"]
        nxt = plan.commands.get(2) or []
        full = plan.commands.get(3) or []
        if tier1_ok:
            lines.append("tier1: passed")
            if top >= 2 and nxt:
                lines.append("now: " + " && ".join(c.brief(self.root) for c in nxt[:2])[:300])
            elif top >= 3 and full:
                lines.append("now: " + full[0].brief(self.root)[:200])
        else:
            now = plan.commands.get(1) or []
            if now:
                lines.append("now: " + " && ".join(c.brief(self.root) for c in now[:2])[:300])
            elif plan.required_non_test_checks:
                lines.append("now: " + display_argv(plan.required_non_test_checks[0].argv)[:200])
            else:
                lines.append("now: no directly impacted tests found")
            if top >= 2 and nxt:
                lines.append(
                    "next-if-pass: " + " && ".join(c.brief(self.root) for c in nxt[:2])[:300]
                )
        if top >= 3 and full:
            why = ",".join(plan.mandatory_tier3) or "risk"
            cmd = full[0].brief(self.root)[:160]
            if any(line.endswith(": " + cmd) for line in lines):
                cmd = "the command above"
            lines.append(f"full-suite: required after tiers pass ({why}): {cmd}")
        else:
            lines.append("full-suite: not yet required")
        codes = [c for c in plan.rationale_codes if c][:4]
        if codes:
            lines.append("reason: " + ", ".join(codes).lower())
        return [(35, "verify", lines)]

    def _coverage(self, plan: VerificationPlan) -> tuple[bool, bool]:
        """(tier-1 selections passed, full suite passed) since the last change."""
        snapshot = self._current_snapshot_hash()
        rows = self.store.query(
            "SELECT command_fingerprint, failed, errors FROM test_runs WHERE task_id = ? AND change_set_hash = ? ORDER BY ts DESC LIMIT 20",
            (self.rt.task_id, snapshot),
        )
        if not rows or rows[0]["failed"] or rows[0]["errors"]:
            return False, False
        good = [r["command_fingerprint"] for r in rows if not r["failed"] and not r["errors"]]
        full = any(not _SELECTOR_RE.search(" " + c) for c in good)
        commands = " ".join(good)
        tier1 = full or all(s.target and s.target in commands for s in plan.tier1)
        return tier1, full

    def _verified(self, plan: VerificationPlan) -> bool:
        """True when a passing run since the last change covered every tier-1 selection."""
        snapshot = self._current_snapshot_hash()
        rows = self.store.query(
            "SELECT command_fingerprint, failed, errors FROM test_runs WHERE task_id = ? AND change_set_hash = ? ORDER BY ts DESC LIMIT 20",
            (self.rt.task_id, snapshot),
        )
        if not rows or rows[0]["failed"] or rows[0]["errors"]:
            return False
        good = [r["command_fingerprint"] for r in rows if not r["failed"] and not r["errors"]]
        if any(not _SELECTOR_RE.search(" " + c) for c in good):
            return True  # a full-suite run passed after the last change
        commands = " ".join(good)
        return all(s.target and s.target in commands for s in plan.tier1)

    def _current_snapshot_hash(self) -> str:
        changed = self.change_set()
        snapshot = {p: _quick_hash(os.path.join(self.root, p)) for p, _ in changed[:50]}
        return hashlib.sha256(json.dumps(sorted(snapshot.items())).encode()).hexdigest()[:16]


def _quick_hash(path: str) -> str:
    try:
        st = os.stat(path)
        if st.st_size > 2 * 1024 * 1024:
            return f"{st.st_size}:{int(st.st_mtime)}"
        with open(path, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()[:16]
    except OSError:
        return "missing"
