"""Installing the agent-state PreToolUse hook for ``headroom wrap``.

* **Claude Code**: an entry in the project's ``.claude/settings.local.json``,
  the file ``wrap`` already manages for its self-heal hook. The entry is
  deduplicated by marker and rewritten in place when the proxy port or the
  interpreter changes.
* **Codex**: an entry in ``$CODEX_HOME/hooks.json``. Codex is launched with
  ``-c features.hooks=true`` for the wrapped session only.

The command runs the standard-library hook client with the interpreter that
runs Headroom, so it works the same in PowerShell, cmd and POSIX shells.
When the proxy is not running, the client exits 0 immediately and the agent
proceeds unaffected. ``headroom unwrap`` removes the entries.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

MARKER = "headroom-agent-state-hook"
CLAUDE_MATCHER = "Edit|Write|MultiEdit|NotebookEdit|Bash|PowerShell"
CODEX_MATCHER = "Bash|shell|exec_command|local_shell|apply_patch"
HOOK_TIMEOUT_S = 5


def hook_client_path() -> str:
    return str(Path(__file__).with_name("hook_client.py"))


def hook_command(agent: str, port: int) -> str:
    from headroom.cli.init import _command_string

    return _command_string(
        [
            sys.executable,
            hook_client_path(),
            "--agent",
            agent,
            "--port",
            str(int(port)),
            "--marker",
            MARKER,
        ]
    )


def hooks_wanted(environ: Any = None) -> bool:
    """True when the validator or the firewall is on (they are what the hook serves)."""
    from headroom.rollout import resolve_rollout

    from .config import AgentStateConfig, AgentStateConfigError

    env = os.environ if environ is None else environ
    try:
        cfg = AgentStateConfig.from_env(env, rollout=resolve_rollout(env))
    except AgentStateConfigError:
        return False  # the proxy reports the configuration error itself
    return cfg.contracts.enabled or cfg.scope.enabled


def _read(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _write(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _upsert(payload: dict[str, Any], matcher: str, command: str) -> bool:
    hooks = dict(payload.get("hooks") or {}) if isinstance(payload.get("hooks"), dict) else {}
    entries = (
        list(hooks.get("PreToolUse") or []) if isinstance(hooks.get("PreToolUse"), list) else []
    )
    retained = []
    current = None
    for entry in entries:
        items = entry.get("hooks") if isinstance(entry, dict) else None
        if isinstance(items, list) and any(
            isinstance(i, dict) and MARKER in str(i.get("command", "")) for i in items
        ):
            current = entry
            continue
        retained.append(entry)
    wanted = {
        "matcher": matcher,
        "hooks": [{"type": "command", "command": command, "timeout": HOOK_TIMEOUT_S}],
    }
    if current == wanted and len(retained) == len(entries) - 1:
        return False
    retained.append(wanted)
    hooks["PreToolUse"] = retained
    payload["hooks"] = hooks
    return True


def _remove(payload: dict[str, Any]) -> bool:
    hooks = payload.get("hooks")
    if not isinstance(hooks, dict) or not isinstance(hooks.get("PreToolUse"), list):
        return False
    kept = []
    changed = False
    for entry in hooks["PreToolUse"]:
        items = entry.get("hooks") if isinstance(entry, dict) else None
        if isinstance(items, list):
            left = [
                i
                for i in items
                if not (isinstance(i, dict) and MARKER in str(i.get("command", "")))
            ]
            if len(left) != len(items):
                changed = True
                if left:
                    kept.append({**entry, "hooks": left})
                continue
        kept.append(entry)
    if not changed:
        return False
    if kept:
        hooks["PreToolUse"] = kept
    else:
        hooks.pop("PreToolUse", None)
    if hooks:
        payload["hooks"] = hooks
    else:
        payload.pop("hooks", None)
    return True


def ensure_claude_hook(settings_path: Path, port: int) -> bool:
    """Install or update the hook in ``.claude/settings.local.json``. Returns True if written."""
    payload = _read(settings_path)
    if not _upsert(payload, CLAUDE_MATCHER, hook_command("claude", port)):
        return False
    _write(settings_path, payload)
    return True


def remove_claude_hook(settings_path: Path) -> bool:
    if not settings_path.exists():
        return False
    payload = _read(settings_path)
    if not _remove(payload):
        return False
    _write(settings_path, payload)
    return True


def ensure_codex_hook(hooks_path: Path, port: int) -> bool:
    payload = _read(hooks_path)
    if not _upsert(payload, CODEX_MATCHER, hook_command("codex", port)):
        return False
    _write(hooks_path, payload)
    return True


def remove_codex_hook(hooks_path: Path) -> bool:
    if not hooks_path.exists():
        return False
    payload = _read(hooks_path)
    if not _remove(payload):
        return False
    _write(hooks_path, payload)
    return True
