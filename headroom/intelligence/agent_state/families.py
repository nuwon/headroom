"""Canonical tool families and effect classification (plan §7.4, §10.2).

Every Phase 2 system reasons about a tool call through one
:class:`ToolInvocation` instead of parsing provider- or agent-specific tool
shapes on its own. The normalizer covers:

* **Claude Code**: ``Read``, ``Write``, ``Edit``, ``MultiEdit``,
  ``NotebookEdit``, ``Glob``, ``Grep``, ``LS``, ``Bash``, ``PowerShell``,
  ``WebFetch``, ``WebSearch`` and ``mcp__*``.
* **Codex**: ``shell`` / ``exec_command`` / ``local_shell`` (argv or string,
  ``workdir``), ``apply_patch`` (as a tool or as a shell command), and
  ``update_plan``.
* **Shells**: POSIX shells, PowerShell and ``cmd`` (``-Command`` / ``/c``
  unwrapping, ``.exe``/``.cmd``/``.bat`` suffixes, ``Set-Location``/``cd``
  prefixes, redirects, chains and pipes).

An unknown tool stays ``OTHER``/``UNKNOWN``; it is never forced into a
category that would let a stricter or looser rule apply.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Family(str, Enum):
    READ_FILE = "read_file"
    WRITE_FILE = "write_file"
    EDIT_FILE = "edit_file"
    DELETE_FILE = "delete_file"
    SEARCH_TEXT = "search_text"
    LIST_FILES = "list_files"
    SHELL = "shell"
    BUILD = "build"
    TEST = "test"
    GIT = "git"
    NETWORK = "network"
    MCP = "mcp"
    OTHER = "other"


class OpClass(str, Enum):
    READ = "READ"
    SEARCH = "SEARCH"
    WRITE = "WRITE"
    DELETE = "DELETE"
    EXECUTE = "EXECUTE"
    BUILD = "BUILD"
    TEST = "TEST"
    NETWORK = "NETWORK"
    OTHER = "OTHER"


class SideEffect(str, Enum):
    NONE = "NONE"
    LOCAL_REVERSIBLE = "LOCAL_REVERSIBLE"
    LOCAL_MUTATING = "LOCAL_MUTATING"
    EXTERNAL = "EXTERNAL"


class SafetyClass(str, Enum):
    READ_ONLY = "READ_ONLY"
    VERIFICATION = "VERIFICATION"
    LOCAL_REVERSIBLE = "LOCAL_REVERSIBLE"
    MUTATING = "MUTATING"
    EXTERNAL_SIDE_EFFECT = "EXTERNAL_SIDE_EFFECT"
    UNKNOWN = "UNKNOWN"


# Severity order: the effective class of a sequence is its most severe step.
_SAFETY_RANK = {
    SafetyClass.READ_ONLY: 0,
    SafetyClass.VERIFICATION: 1,
    SafetyClass.LOCAL_REVERSIBLE: 2,
    SafetyClass.UNKNOWN: 3,
    SafetyClass.MUTATING: 4,
    SafetyClass.EXTERNAL_SIDE_EFFECT: 5,
}


def max_safety(classes: list[SafetyClass] | tuple[SafetyClass, ...]) -> SafetyClass:
    if not classes:
        return SafetyClass.UNKNOWN
    return max(classes, key=lambda c: _SAFETY_RANK[c])


@dataclass(frozen=True)
class CommandSegment:
    argv: tuple[str, ...]
    executable: str  # basename, lower-cased, without .exe/.cmd/.bat/.ps1
    subcommand: str
    safety: SafetyClass
    kind: str  # read | search | list | test | build | lint | git | write | delete | network | other
    writes: tuple[str, ...] = ()
    deletes: tuple[str, ...] = ()
    reads: tuple[str, ...] = ()
    privileged: bool = False


@dataclass(frozen=True)
class ToolInvocation:
    tool_name: str
    family: Family
    op_class: OpClass
    side_effect: SideEffect
    safety: SafetyClass
    paths_read: tuple[str, ...] = ()
    paths_written: tuple[str, ...] = ()
    paths_deleted: tuple[str, ...] = ()
    cwd: str = ""
    cd_target: str = ""
    command: str = ""
    shell_kind: str = ""  # posix | powershell | cmd | ""
    segments: tuple[CommandSegment, ...] = ()
    test_framework: str = ""
    network: bool = False
    privileged: bool = False
    argv_safe: bool = False  # one plain command: executable directly as argv, no shell syntax
    patterns: tuple[str, ...] = ()  # regex/glob patterns the call carries
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def mutating(self) -> bool:
        return self.side_effect in (SideEffect.LOCAL_MUTATING, SideEffect.EXTERNAL) or bool(
            self.paths_written or self.paths_deleted
        )

    @property
    def executable(self) -> str:
        return self.segments[0].executable if self.segments else ""

    @property
    def argv(self) -> tuple[str, ...]:
        return self.segments[0].argv if len(self.segments) == 1 else ()


# --------------------------------------------------------------------- tools
_CLAUDE_READ = {"read", "notebookread", "view", "read_file", "readfile", "open_file"}
_CLAUDE_WRITE = {"write", "write_file", "create_file"}
_CLAUDE_EDIT = {"edit", "multiedit", "notebookedit", "str_replace_editor", "edit_file"}
_CLAUDE_DELETE = {"delete_file", "remove_file"}
_SEARCH = {"grep", "search", "search_files", "codebase_search", "ripgrep", "find_in_files"}
_LIST = {"glob", "ls", "list", "list_dir", "list_directory", "find_files"}
_SHELL = {
    "bash",
    "shell",
    "local_shell",
    "shell_command",
    "powershell",
    "pwsh",
    "exec_command",
    "run_terminal_cmd",
    "container.exec",
}
_NETWORK_TOOLS = {"webfetch", "websearch", "web_search", "fetch", "web_fetch", "browser"}
_PLANNING = {"todowrite", "update_plan", "task", "exitplanmode", "todoread"}

_PATH_KEYS = ("file_path", "path", "notebook_path", "filename", "file", "target_file")


def _first_str(d: dict[str, Any], *keys: str) -> str:
    for key in keys:
        v = d.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _abs(path: str, cwd: str) -> str:
    p = (path or "").strip().strip("'\"")
    if not p:
        return ""
    if _is_absolute(p) or not cwd:
        return p
    sep = "\\" if ("\\" in cwd and "/" not in cwd) else "/"
    return cwd.rstrip("\\/") + sep + p


_WIN_ABS_RE = re.compile(r"^(?:[A-Za-z]:[\\/]|\\\\)")


def _is_absolute(path: str) -> bool:
    return bool(path) and (path.startswith(("/", "~")) or bool(_WIN_ABS_RE.match(path)))


# ------------------------------------------------------------------- shells
_TOKEN_RE = re.compile(r'"(?:[^"\\]|\\.)*"|\'[^\']*\'|[^\s"\']+')
_SEPARATORS = {"&&", "||", ";", "|", "&"}
# ``&`` splits only as a separator: never inside ``2>&1`` / ``&>file``.
_SPLIT_RE = re.compile(r'("(?:[^"\\]|\\.)*"|\'[^\']*\'|&&|\|\||(?<![>&])&(?![>&])|[;|\n])')
_SHELL_META_RE = re.compile(r"[|;&<>`$(){}*?\n]")
_WRAPPER_EXES = {"sudo", "doas", "time", "nohup", "env", "command", "exec", "nice", "stdbuf"}
_PRIV_EXES = {"sudo", "doas", "runas", "su", "gsudo"}
_SHELL_EXES = {"bash", "sh", "zsh", "dash", "fish", "powershell", "pwsh", "cmd"}

_READ_EXES = {
    "cat",
    "head",
    "tail",
    "less",
    "more",
    "nl",
    "bat",
    "type",
    "get-content",
    "gc",
    "od",
    "xxd",
    "hexdump",
}
_SEARCH_EXES = {"grep", "rg", "ag", "ack", "egrep", "fgrep", "select-string", "sls", "findstr"}
_LIST_EXES = {"ls", "dir", "tree", "find", "fd", "get-childitem", "gci", "du", "locate"}
_INFO_EXES = {
    "pwd",
    "echo",
    "printf",
    "which",
    "where",
    "whereis",
    "wc",
    "file",
    "stat",
    "df",
    "sort",
    "uniq",
    "cut",
    "diff",
    "cmp",
    "jq",
    "yq",
    "awk",
    "realpath",
    "readlink",
    "basename",
    "dirname",
    "date",
    "uname",
    "whoami",
    "hostname",
    "printenv",
    "test-path",
    "get-item",
    "resolve-path",
    "get-location",
    "measure-object",
    "column",
    "tr",
    "true",
    "false",
    "sleep",
    "get-command",
    "get-process",
    "ps",
    "id",
    "nproc",
    "free",
    "lscpu",
    "sha256sum",
    "md5sum",
    "get-filehash",
    "certutil",
    "cd",
    "chdir",
    "pushd",
    "popd",
    "set-location",
    "sl",
    "exit",
    "write-output",
    "write-host",
    "out-string",
    "format-table",
    "select-object",
    "sort-object",
    "where-object",
    "foreach-object",
}
_DELETE_EXES = {"rm", "del", "erase", "rmdir", "rd", "remove-item", "ri", "unlink", "shred"}
_WRITE_EXES = {
    "mv",
    "move",
    "move-item",
    "mi",
    "cp",
    "copy",
    "copy-item",
    "cpi",
    "xcopy",
    "robocopy",
    "mkdir",
    "md",
    "new-item",
    "ni",
    "touch",
    "tee",
    "tee-object",
    "chmod",
    "chown",
    "chgrp",
    "ln",
    "set-content",
    "sc",
    "add-content",
    "ac",
    "out-file",
    "rename-item",
    "ren",
    "truncate",
    "patch",
    "apply_patch",
    "applypatch",
    "dd",
    "install",
}
_NETWORK_EXES = {
    "curl",
    "wget",
    "invoke-webrequest",
    "iwr",
    "invoke-restmethod",
    "irm",
    "ssh",
    "scp",
    "sftp",
    "rsync",
    "ftp",
    "nc",
    "netcat",
    "telnet",
    "gh",
    "aws",
    "gcloud",
    "az",
    "kubectl",
    "helm",
    "terraform",
    "pulumi",
    "heroku",
    "vercel",
    "netlify",
    "flyctl",
    "fly",
}
_TEST_RE = re.compile(
    r"(?i)^(?:pytest|py\.test|tox|nox|ctest|jest|vitest|mocha|phpunit|rspec|ava|tap|karma|"
    r"nextest|unittest)$"
)
_LINT_EXES = {
    "ruff",
    "mypy",
    "pyright",
    "flake8",
    "pylint",
    "eslint",
    "tsc",
    "shellcheck",
    "black",
    "isort",
    "prettier",
    "clang-tidy",
    "cppcheck",
    "golangci-lint",
    "stylelint",
    "biome",
}
_BUILD_EXES = {"make", "ninja", "cmake", "msbuild", "bazel", "buck", "meson", "gradle", "mvn"}
_PKG_MANAGERS = {"npm", "pnpm", "yarn", "bun", "npx", "bunx", "pnpx"}
_PY_EXES = re.compile(r"^(?:python[\d.]*|py|pypy[\d.]*|uv|poetry|pipenv|hatch|pdm|rye)$")

_GIT_READ = {
    "status",
    "diff",
    "log",
    "show",
    "rev-parse",
    "ls-files",
    "blame",
    "grep",
    "describe",
    "shortlog",
    "reflog",
    "cat-file",
    "ls-tree",
    "merge-base",
    "rev-list",
    "check-ignore",
    "check-ref-format",
    "for-each-ref",
    "name-rev",
    "whatchanged",
    "count-objects",
    "version",
    "help",
}
_GIT_NETWORK = {"push", "fetch", "pull", "clone", "ls-remote", "submodule", "send-email"}


def _base_exe(token: str) -> str:
    t = token.strip().strip("'\"").replace("\\", "/").rsplit("/", 1)[-1].lower()
    for suffix in (".exe", ".cmd", ".bat", ".ps1", ".com"):
        if t.endswith(suffix):
            t = t[: -len(suffix)]
            break
    return t


def _strip_quotes(token: str) -> str:
    if len(token) >= 2 and token[0] == token[-1] and token[0] in "\"'":
        return token[1:-1]
    return token


def tokenize(command: str) -> list[str]:
    """Quote-aware split that keeps Windows backslashes intact."""
    return [_strip_quotes(t) for t in _TOKEN_RE.findall(command or "")]


def split_segments(command: str) -> tuple[list[str], list[str]]:
    """Split a shell line on ``&&``, ``||``, ``;``, ``|``, ``&`` and newlines.

    Returns ``(segments, separators)``. Quoted text is never split.
    """
    parts = _SPLIT_RE.split(command or "")
    segments: list[str] = []
    seps: list[str] = []
    buf = ""
    for part in parts:
        if not part:
            continue
        if part in _SEPARATORS or part == "\n":
            if buf.strip():
                segments.append(buf.strip())
            seps.append(part)
            buf = ""
        else:
            buf += part
    if buf.strip():
        segments.append(buf.strip())
    return segments, seps


_REDIRECT_OP_RE = re.compile(r"^(?:\d?>>?|&>>?|\*>>?)(.*)$")
_NULL_SINKS = {"/dev/null", "nul", "$null", "/dev/stderr", "/dev/stdout"}
_HEREDOC_RE = re.compile(r"<<-?\s*['\"]?(\w+)['\"]?")


def _redirect_targets(segment: str) -> list[str]:
    """Files a segment redirects output into (quoted text is never parsed)."""
    raw = _TOKEN_RE.findall(segment or "")
    out: list[str] = []
    i = 0
    while i < len(raw):
        tok = raw[i]
        m = _REDIRECT_OP_RE.match(tok) if tok[:1] not in "\"'" else None
        if m:
            target = m.group(1)
            if not target and i + 1 < len(raw):
                i += 1
                target = raw[i]
            target = _strip_quotes(target)
            if target and not target.startswith("&") and target.lower() not in _NULL_SINKS:
                out.append(target)
        i += 1
    return out


def _unwrap_shell(argv: list[str]) -> tuple[str, str] | None:
    """``powershell -Command "x"`` / ``bash -lc "x"`` / ``cmd /c x`` -> (kind, inner)."""
    if not argv:
        return None
    exe = _base_exe(argv[0])
    if exe not in _SHELL_EXES:
        return None
    kind = "powershell" if exe in ("powershell", "pwsh") else ("cmd" if exe == "cmd" else "posix")
    for i, tok in enumerate(argv[1:], start=1):
        low = tok.lower()
        if kind == "cmd" and low in ("/c", "/k"):
            return kind, " ".join(argv[i + 1 :])
        if kind == "powershell" and low in ("-command", "-c", "-commandwithargs"):
            return kind, " ".join(argv[i + 1 :])
        if kind == "posix" and low.startswith("-") and "c" in low.lstrip("-"):
            return kind, " ".join(argv[i + 1 :])
    return None


def _non_flag_args(argv: list[str], start: int = 1) -> list[str]:
    return [a for a in argv[start:] if a and not a.startswith("-") and not a.startswith("/")]


def _path_args(argv: list[str], start: int = 1) -> list[str]:
    out = []
    for a in argv[start:]:
        if not a or a.startswith("-"):
            continue
        if a.startswith("/") and len(a) <= 3 and os.name == "nt":
            continue  # cmd switches like /s /q
        out.append(a)
    return out


def _strip_redirects(segment: str) -> list[str]:
    """Tokens of a segment without redirect operators, their targets or heredocs."""
    raw = _TOKEN_RE.findall(segment or "")
    out: list[str] = []
    i = 0
    while i < len(raw):
        tok = raw[i]
        if tok[:1] not in "\"'":
            m = _REDIRECT_OP_RE.match(tok)
            if m:
                if not m.group(1):
                    i += 1  # the target is the next token
                i += 1
                continue
            if tok.startswith("<<") or tok == "<":
                i += 2 if tok in ("<<", "<<-", "<") else 1
                continue
        out.append(_strip_quotes(tok))
        i += 1
    return out


def classify_segment(segment: str) -> CommandSegment:
    tokens = _strip_redirects(segment)
    privileged = False
    # Peel env assignments and wrappers (sudo is recorded, never added).
    while tokens:
        head = tokens[0]
        low = _base_exe(head)
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", head):
            tokens = tokens[1:]
            continue
        if low in _WRAPPER_EXES:
            privileged = privileged or low in _PRIV_EXES
            tokens = tokens[1:]
            while tokens and tokens[0].startswith("-"):
                tokens = tokens[1:]
            continue
        if low in _PRIV_EXES:
            privileged = True
            tokens = tokens[1:]
            continue
        if head == "&":  # PowerShell call operator
            tokens = tokens[1:]
            continue
        break
    if not tokens:
        return CommandSegment((), "", "", SafetyClass.READ_ONLY, "other", privileged=privileged)
    exe = _base_exe(tokens[0])
    args = tokens[1:]
    sub = ""
    for a in args:
        if not a.startswith("-"):
            sub = a.lower()
            break
    redirect_writes = tuple(_redirect_targets(segment))
    argv = tuple(tokens)

    def seg(
        safety: SafetyClass,
        kind: str,
        *,
        writes: tuple[str, ...] = (),
        deletes: tuple[str, ...] = (),
        reads: tuple[str, ...] = (),
    ) -> CommandSegment:
        all_writes = tuple(dict.fromkeys((*writes, *redirect_writes)))
        if all_writes and safety in (SafetyClass.READ_ONLY, SafetyClass.VERIFICATION):
            safety = SafetyClass.MUTATING
        if privileged and safety is not SafetyClass.EXTERNAL_SIDE_EFFECT:
            safety = max_safety([safety, SafetyClass.MUTATING])
        return CommandSegment(
            argv,
            exe,
            sub,
            safety,
            kind,
            writes=all_writes,
            deletes=deletes,
            reads=reads,
            privileged=privileged,
        )

    if exe in _READ_EXES:
        looks_like_path = [a for a in _path_args(list(argv)) if re.search(r"[./\\]", a)]
        return seg(SafetyClass.READ_ONLY, "read", reads=tuple(looks_like_path))
    if exe == "sed":
        if any(a == "-i" or a.startswith("-i") or a == "--in-place" for a in args):
            targets = [a for a in _non_flag_args(list(argv)) if not a.startswith(("s/", "/"))]
            return seg(SafetyClass.MUTATING, "write", writes=tuple(targets[1:] or targets))
        return seg(SafetyClass.READ_ONLY, "read")
    if exe in _SEARCH_EXES:
        return seg(SafetyClass.READ_ONLY, "search")
    if exe in _LIST_EXES:
        if exe == "find" and any(a in ("-delete", "-exec", "-execdir", "-ok") for a in args):
            return seg(SafetyClass.MUTATING, "delete")
        return seg(SafetyClass.READ_ONLY, "list")
    if exe in _INFO_EXES:
        return seg(SafetyClass.READ_ONLY, "info")
    if exe in _DELETE_EXES:
        return seg(SafetyClass.MUTATING, "delete", deletes=tuple(_path_args(list(argv))))
    if exe in _WRITE_EXES:
        paths = tuple(_path_args(list(argv)))
        if exe in ("mv", "move", "move-item", "mi", "rename-item", "ren"):
            return seg(SafetyClass.MUTATING, "write", writes=paths[-1:], deletes=paths[:-1])
        if exe in ("cp", "copy", "copy-item", "cpi", "xcopy", "robocopy"):
            return seg(SafetyClass.MUTATING, "write", writes=paths[-1:])
        return seg(SafetyClass.MUTATING, "write", writes=paths)
    if exe in _NETWORK_EXES:
        return seg(SafetyClass.EXTERNAL_SIDE_EFFECT, "network")
    if exe == "git":
        gsub = sub
        if gsub in _GIT_READ:
            return seg(SafetyClass.READ_ONLY, "git")
        if (
            gsub == "branch"
            and not _non_flag_args(list(argv), 2)
            and not any(
                a in ("-d", "-D", "-m", "-M", "--delete", "--move", "-c", "-C") for a in args
            )
        ):
            return seg(SafetyClass.READ_ONLY, "git")
        if gsub in ("remote", "config", "tag", "stash") and (
            len(args) == 1 or any(a in ("-v", "-l", "--list", "--get", "show") for a in args)
        ):
            return seg(SafetyClass.READ_ONLY, "git")
        if gsub in _GIT_NETWORK:
            return seg(SafetyClass.EXTERNAL_SIDE_EFFECT, "git")
        return seg(SafetyClass.MUTATING, "git")
    if exe == "cargo":
        if sub in ("test", "nextest"):
            return seg(SafetyClass.VERIFICATION, "test")
        if sub in ("build", "check", "clippy", "doc", "bench", "metadata", "tree"):
            return seg(SafetyClass.VERIFICATION, "build" if sub != "clippy" else "lint")
        if sub == "fmt":
            if "--check" in args:
                return seg(SafetyClass.VERIFICATION, "lint")
            return seg(SafetyClass.MUTATING, "write")
        if sub in ("publish", "login", "yank", "owner"):
            return seg(SafetyClass.EXTERNAL_SIDE_EFFECT, "network")
        return seg(SafetyClass.MUTATING, "other")
    if exe == "go":
        if sub == "test":
            return seg(SafetyClass.VERIFICATION, "test")
        if sub in ("build", "vet", "list", "version", "env"):
            return seg(SafetyClass.VERIFICATION, "build")
        if sub in ("get", "install", "mod"):
            return seg(SafetyClass.EXTERNAL_SIDE_EFFECT, "network")
        return seg(SafetyClass.UNKNOWN, "other")
    if exe in ("dotnet", "mvn", "gradle", "gradlew", "mix"):
        if sub == "test" or "test" in [a.lower() for a in args]:
            return seg(SafetyClass.VERIFICATION, "test")
        if sub in ("build", "compile", "package", "assemble", "check", "restore"):
            return seg(SafetyClass.VERIFICATION, "build")
        return seg(SafetyClass.UNKNOWN, "other")
    if _TEST_RE.match(exe):
        return seg(SafetyClass.VERIFICATION, "test")
    if _PY_EXES.match(exe):
        mod = ""
        if "-m" in args:
            i = args.index("-m")
            mod = args[i + 1].lower() if i + 1 < len(args) else ""
        elif exe in ("uv", "poetry", "pipenv", "hatch", "pdm", "rye") and sub == "run":
            rest = _non_flag_args(args, 1)
            mod = _base_exe(rest[0]) if rest else ""
        if mod in ("pytest", "unittest", "nose2", "tox", "nox"):
            return seg(SafetyClass.VERIFICATION, "test")
        if mod in ("mypy", "pyright", "flake8", "pylint", "compileall", "py_compile"):
            return seg(SafetyClass.VERIFICATION, "lint")
        if mod == "ruff":
            if "--fix" in args or "format" in [a.lower() for a in args]:
                return seg(SafetyClass.MUTATING, "write")
            return seg(SafetyClass.VERIFICATION, "lint")
        if mod in ("build",):
            return seg(SafetyClass.VERIFICATION, "build")
        if mod in ("pip",) or sub in ("add", "install", "remove", "sync", "lock"):
            return seg(SafetyClass.EXTERNAL_SIDE_EFFECT, "network")
        if mod in ("json.tool", "site", "platform", "this"):
            return seg(SafetyClass.READ_ONLY, "info")
        if "--version" in args or "-V" in args:
            return seg(SafetyClass.READ_ONLY, "info")
        return seg(SafetyClass.UNKNOWN, "other")
    if exe in ("pip", "pip3", "pipx", "conda", "mamba", "apt", "apt-get", "brew", "choco"):
        if sub in ("list", "show", "freeze", "--version", "search", "info"):
            return seg(SafetyClass.READ_ONLY, "info")
        return seg(SafetyClass.EXTERNAL_SIDE_EFFECT, "network")
    if exe in ("winget", "scoop", "yum", "dnf", "pacman", "snap"):
        return seg(SafetyClass.EXTERNAL_SIDE_EFFECT, "network")
    if exe in _PKG_MANAGERS:
        script = sub
        if script == "run":
            rest = _non_flag_args(args, 1)
            script = rest[0].lower() if rest else ""
        if exe in ("npx", "bunx", "pnpx") or script in ("exec", "dlx"):
            target = _non_flag_args(args, 1 if script in ("exec", "dlx") else 0)
            tool = _base_exe(target[0]) if target else ""
            if tool in ("jest", "vitest", "mocha", "playwright", "cypress", "ava"):
                return seg(SafetyClass.VERIFICATION, "test")
            if tool in ("tsc", "eslint", "prettier", "biome") and not any(
                a in ("--fix", "--write", "-w") for a in args
            ):
                return seg(SafetyClass.VERIFICATION, "lint")
            return seg(SafetyClass.UNKNOWN, "other")
        if script.startswith("test") or script in ("jest", "vitest"):
            return seg(SafetyClass.VERIFICATION, "test")
        if script in ("lint", "typecheck", "type-check", "check", "tsc"):
            return seg(SafetyClass.VERIFICATION, "lint")
        if script == "build":
            return seg(SafetyClass.VERIFICATION, "build")
        if script in ("ls", "list", "outdated", "view", "info", "why", "--version", "-v"):
            return seg(SafetyClass.READ_ONLY, "info")
        if script in ("install", "i", "ci", "add", "remove", "update", "upgrade", "uninstall"):
            return seg(SafetyClass.EXTERNAL_SIDE_EFFECT, "network")
        if script in ("publish", "deploy", "login", "release"):
            return seg(SafetyClass.EXTERNAL_SIDE_EFFECT, "network")
        return seg(SafetyClass.UNKNOWN, "other")
    if exe in _LINT_EXES:
        if any(a in ("--fix", "--write", "-w", "format", "--in-place") for a in args) or (
            exe in ("black", "isort") and "--check" not in args and "--diff" not in args
        ):
            return seg(SafetyClass.MUTATING, "write")
        return seg(SafetyClass.VERIFICATION, "lint")
    if exe in _BUILD_EXES:
        if (
            exe == "cmake"
            and "--build" not in args
            and not any(a in ("--version", "-E", "--help") for a in args)
        ):
            # Configure step: writes a build tree (generated, reversible).
            return seg(SafetyClass.LOCAL_REVERSIBLE, "build")
        if exe == "make" and any(a in ("install", "clean", "distclean") for a in args):
            return seg(SafetyClass.MUTATING, "build")
        return seg(SafetyClass.VERIFICATION, "build")
    if exe in ("docker", "podman"):
        if sub in ("ps", "images", "inspect", "logs", "version", "info"):
            return seg(SafetyClass.READ_ONLY, "info")
        return seg(SafetyClass.EXTERNAL_SIDE_EFFECT, "network")
    return seg(SafetyClass.UNKNOWN, "other")


def _shell_kind_for(tool_name: str, argv: list[str] | None, command: str) -> str:
    low = tool_name.lower()
    if low in ("powershell", "pwsh"):
        return "powershell"
    if argv:
        unwrapped = _unwrap_shell(argv)
        if unwrapped:
            return unwrapped[0]
    if os.name == "nt" and re.search(r"(?i)\b(?:Get-|Set-|New-|Remove-)\w+", command):
        return "powershell"
    return "posix"


_PATCH_FILE_RE = re.compile(r"^\*\*\* (Add|Update|Delete) File: (.+?)\s*$", re.M)
_PATCH_MOVE_RE = re.compile(r"^\*\*\* Move to: (.+?)\s*$", re.M)


def patch_paths(patch: str) -> tuple[list[str], list[str]]:
    """``(written, deleted)`` paths named by a Codex ``apply_patch`` envelope."""
    written: list[str] = []
    deleted: list[str] = []
    for m in _PATCH_FILE_RE.finditer(patch or ""):
        (deleted if m.group(1) == "Delete" else written).append(m.group(2).strip())
    written.extend(m.group(1).strip() for m in _PATCH_MOVE_RE.finditer(patch or ""))
    return written, deleted


_CD_PREFIX_RE = re.compile(
    r"^\s*(?:cd|chdir|pushd|set-location|sl)\s+(\"[^\"]+\"|'[^']+'|[^&;|\s]+)\s*(?:&&|;)\s*",
    re.I,
)


def _cd_prefix(command: str) -> tuple[str, str]:
    """Peel ``cd X && `` prefixes; return (last cd target, remaining command)."""
    target = ""
    rest = command
    while True:
        m = _CD_PREFIX_RE.match(rest)
        if not m:
            break
        target = _strip_quotes(m.group(1))
        rest = rest[m.end() :]
    return target, rest


def normalize_tool_call(
    tool_name: str, tool_input: Any, *, default_cwd: str = ""
) -> ToolInvocation:
    """Normalize one tool call (any agent) into a :class:`ToolInvocation`."""
    name = str(tool_name or "")
    inp: dict[str, Any] = tool_input if isinstance(tool_input, dict) else {}
    if isinstance(tool_input, str) and name.lower() in ("apply_patch", "applypatch"):
        inp = {"input": tool_input}
    low = name.lower()
    base = low.split("__")[-1] if low.startswith("mcp__") else low
    cwd = _first_str(inp, "workdir", "cwd", "working_directory", "directory_path") or default_cwd

    if low.startswith("mcp__"):
        if base in ("headroom_workflow",):
            return ToolInvocation(
                name, Family.MCP, OpClass.TEST, SideEffect.NONE, SafetyClass.VERIFICATION, cwd=cwd
            )
        if base.startswith("headroom_"):
            return ToolInvocation(
                name, Family.MCP, OpClass.READ, SideEffect.NONE, SafetyClass.READ_ONLY, cwd=cwd
            )
        return ToolInvocation(
            name, Family.MCP, OpClass.OTHER, SideEffect.NONE, SafetyClass.UNKNOWN, cwd=cwd
        )
    if base in _CLAUDE_READ:
        path = _abs(_first_str(inp, *_PATH_KEYS), cwd)
        return ToolInvocation(
            name,
            Family.READ_FILE,
            OpClass.READ,
            SideEffect.NONE,
            SafetyClass.READ_ONLY,
            paths_read=(path,) if path else (),
            cwd=cwd,
        )
    if base in _CLAUDE_WRITE:
        path = _abs(_first_str(inp, *_PATH_KEYS), cwd)
        return ToolInvocation(
            name,
            Family.WRITE_FILE,
            OpClass.WRITE,
            SideEffect.LOCAL_MUTATING,
            SafetyClass.MUTATING,
            paths_written=(path,) if path else (),
            cwd=cwd,
        )
    if base in _CLAUDE_EDIT:
        path = _abs(_first_str(inp, *_PATH_KEYS), cwd)
        return ToolInvocation(
            name,
            Family.EDIT_FILE,
            OpClass.WRITE,
            SideEffect.LOCAL_MUTATING,
            SafetyClass.MUTATING,
            paths_read=(path,) if path else (),
            paths_written=(path,) if path else (),
            cwd=cwd,
        )
    if base in _CLAUDE_DELETE:
        path = _abs(_first_str(inp, *_PATH_KEYS), cwd)
        return ToolInvocation(
            name,
            Family.DELETE_FILE,
            OpClass.DELETE,
            SideEffect.LOCAL_MUTATING,
            SafetyClass.MUTATING,
            paths_deleted=(path,) if path else (),
            cwd=cwd,
        )
    if base in ("apply_patch", "applypatch"):
        patch = _first_str(inp, "input", "patch", "content")
        written, deleted = patch_paths(patch)
        return ToolInvocation(
            name,
            Family.EDIT_FILE,
            OpClass.WRITE,
            SideEffect.LOCAL_MUTATING,
            SafetyClass.MUTATING,
            paths_written=tuple(_abs(p, cwd) for p in written),
            paths_deleted=tuple(_abs(p, cwd) for p in deleted),
            cwd=cwd,
        )
    if base in _SEARCH:
        path = _first_str(inp, "path", "directory", "include")
        pattern = _first_str(inp, "pattern", "query", "regex", "search")
        return ToolInvocation(
            name,
            Family.SEARCH_TEXT,
            OpClass.SEARCH,
            SideEffect.NONE,
            SafetyClass.READ_ONLY,
            paths_read=(_abs(path, cwd),) if path else (),
            cwd=cwd,
            patterns=(pattern,) if pattern else (),
            extra={"glob": _first_str(inp, "glob")},
        )
    if base in _LIST:
        path = _first_str(inp, "path", "directory", "dir", "target_directory")
        pattern = _first_str(inp, "pattern", "glob", "glob_pattern")
        return ToolInvocation(
            name,
            Family.LIST_FILES,
            OpClass.SEARCH,
            SideEffect.NONE,
            SafetyClass.READ_ONLY,
            paths_read=(_abs(path, cwd),) if path else (),
            cwd=cwd,
            extra={"glob": pattern},
        )
    if base in _NETWORK_TOOLS:
        return ToolInvocation(
            name,
            Family.NETWORK,
            OpClass.NETWORK,
            SideEffect.EXTERNAL,
            SafetyClass.EXTERNAL_SIDE_EFFECT,
            cwd=cwd,
            network=True,
        )
    if base in _PLANNING:
        return ToolInvocation(
            name, Family.OTHER, OpClass.OTHER, SideEffect.NONE, SafetyClass.READ_ONLY, cwd=cwd
        )
    if base in _SHELL:
        return _normalize_shell(name, inp, cwd)
    return ToolInvocation(
        name, Family.OTHER, OpClass.OTHER, SideEffect.NONE, SafetyClass.UNKNOWN, cwd=cwd
    )


def _normalize_shell(name: str, inp: dict[str, Any], cwd: str) -> ToolInvocation:
    raw = inp.get("command", inp.get("cmd", inp.get("args")))
    action = inp.get("action")
    if raw is None and isinstance(action, dict):
        raw = action.get("command")
        cwd = str(action.get("working_directory") or cwd)
    argv_in: list[str] | None = None
    if isinstance(raw, list):
        argv_in = [str(x) for x in raw]
        unwrapped = _unwrap_shell(argv_in)
        if unwrapped:
            shell_kind, command = unwrapped
        else:
            shell_kind = "posix"
            if argv_in and _base_exe(argv_in[0]) in ("apply_patch", "applypatch"):
                patch = argv_in[1] if len(argv_in) > 1 else ""
                written, deleted = patch_paths(patch)
                return ToolInvocation(
                    name,
                    Family.EDIT_FILE,
                    OpClass.WRITE,
                    SideEffect.LOCAL_MUTATING,
                    SafetyClass.MUTATING,
                    paths_written=tuple(_abs(p, cwd) for p in written),
                    paths_deleted=tuple(_abs(p, cwd) for p in deleted),
                    cwd=cwd,
                    command="apply_patch",
                )
            command = " ".join(
                (f'"{a}"' if (" " in a and not a.startswith(('"', "'"))) else a) for a in argv_in
            )
    else:
        command = str(raw or "")
        shell_kind = _shell_kind_for(name, None, command)
    command = command.strip()
    heredoc_patch = ""
    if re.match(r"^\s*apply_patch\b", command):
        heredoc_patch = command
    cd_target, rest = _cd_prefix(command)
    if heredoc_patch:
        written, deleted = patch_paths(heredoc_patch)
        effective_cwd = _abs(cd_target, cwd) if cd_target else cwd
        return ToolInvocation(
            name,
            Family.EDIT_FILE,
            OpClass.WRITE,
            SideEffect.LOCAL_MUTATING,
            SafetyClass.MUTATING,
            paths_written=tuple(_abs(p, effective_cwd) for p in written),
            paths_deleted=tuple(_abs(p, effective_cwd) for p in deleted),
            cwd=cwd,
            cd_target=cd_target,
            command=command[:2000],
            shell_kind=shell_kind,
        )
    # Heredoc bodies are data, not commands.
    body_free = rest
    hd = _HEREDOC_RE.search(rest)
    if hd:
        line_end = rest.find("\n", hd.end())
        body_free = rest if line_end < 0 else rest[:line_end]
    seg_texts, seps = split_segments(body_free)
    segments = tuple(classify_segment(s) for s in seg_texts if s)
    effective_cwd = _abs(cd_target, cwd) if cd_target else cwd
    writes = tuple(dict.fromkeys(_abs(p, effective_cwd) for s in segments for p in s.writes if p))
    deletes = tuple(dict.fromkeys(_abs(p, effective_cwd) for s in segments for p in s.deletes if p))
    reads = tuple(dict.fromkeys(_abs(p, effective_cwd) for s in segments for p in s.reads if p))
    safety = max_safety([s.safety for s in segments]) if segments else SafetyClass.READ_ONLY
    kinds = {s.kind for s in segments}
    if (writes or deletes) and not kinds & {"test", "build", "lint"}:
        kinds = kinds | ({"delete"} if deletes and not writes else {"write"})
        kinds -= {"read", "info", "search", "list"}
    if "test" in kinds:
        family, op = Family.TEST, OpClass.TEST
    elif "build" in kinds or "lint" in kinds:
        family, op = Family.BUILD, OpClass.BUILD
    elif "git" in kinds and kinds <= {"git", "info"}:
        family, op = Family.GIT, OpClass.EXECUTE
    elif kinds & {"delete"}:
        family, op = Family.DELETE_FILE, OpClass.DELETE
    elif kinds & {"write"}:
        family, op = Family.SHELL, OpClass.WRITE
    elif kinds and kinds <= {"read", "info"}:
        family, op = Family.READ_FILE if "read" in kinds else Family.SHELL, OpClass.READ
    elif kinds and kinds <= {"search", "list", "info", "read"}:
        family, op = Family.SEARCH_TEXT, OpClass.SEARCH
    elif "network" in kinds:
        family, op = Family.NETWORK, OpClass.NETWORK
    else:
        family, op = Family.SHELL, OpClass.EXECUTE
    if safety is SafetyClass.EXTERNAL_SIDE_EFFECT:
        side = SideEffect.EXTERNAL
    elif safety is SafetyClass.MUTATING or writes or deletes:
        side = SideEffect.LOCAL_MUTATING
    elif safety is SafetyClass.LOCAL_REVERSIBLE:
        side = SideEffect.LOCAL_REVERSIBLE
    elif safety is SafetyClass.UNKNOWN:
        side = SideEffect.LOCAL_MUTATING
    else:
        side = SideEffect.NONE
    framework = ""
    if family is Family.TEST:
        framework = detect_test_framework(command)
    argv_safe = (
        len(segments) == 1
        and not seps
        and not hd
        and not cd_target
        and not _SHELL_META_RE.search(rest.replace("\\", "/"))
        and bool(segments[0].argv)
    )
    return ToolInvocation(
        name,
        family,
        op,
        side,
        safety,
        paths_read=reads,
        paths_written=writes,
        paths_deleted=deletes,
        cwd=cwd,
        cd_target=cd_target,
        command=command[:2000],
        shell_kind=shell_kind,
        segments=segments,
        test_framework=framework,
        network=safety is SafetyClass.EXTERNAL_SIDE_EFFECT or "network" in kinds,
        privileged=any(s.privileged for s in segments),
        argv_safe=argv_safe,
    )


_FRAMEWORK_RES = (
    ("pytest", re.compile(r"(?i)\b(?:pytest|py\.test|-m\s+pytest)\b")),
    ("unittest", re.compile(r"(?i)-m\s+unittest\b")),
    ("cargo", re.compile(r"(?i)\bcargo\s+(?:test|nextest)\b")),
    ("ctest", re.compile(r"(?i)\bctest\b")),
    ("vitest", re.compile(r"(?i)\bvitest\b")),
    ("jest", re.compile(r"(?i)\bjest\b")),
    ("go", re.compile(r"(?i)\bgo\s+test\b")),
    ("dotnet", re.compile(r"(?i)\bdotnet\s+test\b")),
    ("npm", re.compile(r"(?i)\b(?:npm|pnpm|yarn|bun)\s+(?:run\s+)?test")),
)


def detect_test_framework(command: str) -> str:
    for name, rx in _FRAMEWORK_RES:
        if rx.search(command or ""):
            return name
    return "generic"
