"""PreToolUse hook client for Claude Code and Codex (standard library only).

Installed by ``headroom wrap`` and run by the host before each matched tool
call. The script never imports the ``headroom`` package, so startup stays at
interpreter cost. It forwards the host's hook JSON (plus executable
resolutions from the agent's own environment) to the local Headroom proxy and
maps the answer to the host protocol:

* ``deny``: the reason goes to stderr and the script exits 2 (the host blocks
  the call).
* ``allow`` with context (Claude Code): ``hookSpecificOutput.additionalContext``
  goes to stdout and the script exits 0.
* Anything else, including any error, timeout or unreachable proxy: exit 0
  with no output. The host proceeds exactly as without Headroom (fail-open).

Usage::

    python hook_client.py --agent claude --port 8787
"""

from __future__ import annotations

import json
import re
import shutil
import sys
import urllib.error
import urllib.request

TIMEOUT_S = 2.0
MARKER = "headroom-agent-state-hook"
_SEG_RE = re.compile(r"&&|\|\||;|\||\n")
_SHELL_TOOLS = {"bash", "powershell", "shell", "exec_command", "local_shell", "shell_command"}
_SKIP = {
    "cd",
    "export",
    "set",
    "echo",
    "source",
    ".",
    "true",
    "false",
    "test",
    "[",
    "if",
    "for",
    "while",
}


def _arg(name: str, default: str = "") -> str:
    argv = sys.argv[1:]
    for i, tok in enumerate(argv):
        if tok == name and i + 1 < len(argv):
            return argv[i + 1]
        if tok.startswith(name + "="):
            return tok.split("=", 1)[1]
    return default


def _which(hook: dict) -> dict:
    """Resolve each command segment's executable in the *agent's* environment."""
    if str(hook.get("tool_name", "")).lower() not in _SHELL_TOOLS:
        return {}
    raw = (hook.get("tool_input") or {}).get("command")
    if isinstance(raw, list):
        raw = " ".join(str(x) for x in raw)
    if not isinstance(raw, str):
        return {}
    out = {}
    for seg in _SEG_RE.split(raw)[:12]:
        toks = seg.strip().split()
        while toks and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", toks[0]):
            toks = toks[1:]
        if not toks:
            continue
        exe = toks[0].strip("'\"")
        if exe in _SKIP or "/" in exe or "\\" in exe or not re.match(r"^[\w.+-]+$", exe):
            continue
        out[exe] = shutil.which(exe)
    return out


def main() -> int:
    agent = _arg("--agent", "claude")
    port = _arg("--port", "8787")
    try:
        hook = json.loads(sys.stdin.read() or "{}")
    except (ValueError, OSError):
        return 0
    if not isinstance(hook, dict):
        return 0
    body = json.dumps({"agent": agent, "hook": hook, "which": _which(hook)}).encode("utf-8")
    req = urllib.request.Request(
        f"http://127.0.0.1:{int(port)}/v1/agent-state/hook",
        data=body,
        headers={"content-type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:  # noqa: S310 - loopback only
            answer = json.loads(resp.read().decode("utf-8") or "{}")
    except (urllib.error.URLError, OSError, ValueError, TimeoutError):
        return 0
    if not isinstance(answer, dict):
        return 0
    if answer.get("decision") == "deny":
        sys.stderr.write(str(answer.get("reason") or "Headroom blocked this tool call") + "\n")
        return 2
    context = answer.get("context")
    if context and agent.startswith("claude"):
        sys.stdout.write(
            json.dumps(
                {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "additionalContext": str(context),
                    }
                }
            )
        )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001 - a hook must never break the host
        sys.exit(0)
