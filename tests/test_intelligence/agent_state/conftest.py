"""Shared fixtures for the Phase 2 agent-state tests."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

from headroom.intelligence.agent_state.config import AgentStateConfig
from headroom.intelligence.agent_state.runtime import AgentStateService

SESSION = "11111111-2222-3333-4444-555555555555"
CLAUDE_HEADERS = {"user-agent": "claude-cli/2.0.0 (external, cli)"}


def git(root: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        env={
            **os.environ,
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@e",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@e",
        },
    )


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "proj"
    (root / "src" / "pkg").mkdir(parents=True)
    (root / "src" / "other").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "src" / "pkg" / "__init__.py").write_text("")
    (root / "src" / "pkg" / "core.py").write_text("def f():\n    return 1\n")
    (root / "src" / "pkg" / "iface.py").write_text("def g():\n    return 2\n")
    (root / "src" / "other" / "cost.py").write_text("RATE = 3\n")
    (root / "tests" / "test_core.py").write_text(
        "from pkg.core import f\n\n\ndef test_f():\n    assert f() == 1\n"
    )
    (root / "tests" / "test_iface.py").write_text(
        "from pkg.iface import g\n\n\ndef test_g():\n    assert g() == 2\n"
    )
    (root / "pyproject.toml").write_text("[tool.pytest.ini_options]\npythonpath = ['src']\n")
    (root / ".gitignore").write_text("build/\n")
    git(root, "init", "-q")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "init")
    return root


@pytest.fixture
def make_service():
    def _make(**overrides: Any) -> AgentStateService:
        env = {k: str(v) for k, v in overrides.pop("env", {}).items()}
        cfg = AgentStateConfig.from_env(env)
        return AgentStateService(cfg, **overrides)

    return _make


@pytest.fixture
def service(make_service) -> AgentStateService:
    return make_service()


def claude_body(session: str = SESSION, tools: list[str] | None = None) -> dict[str, Any]:
    return {
        "model": "claude-opus",
        "metadata": {"user_id": f"user_abc_account_x_session_{session}"},
        "tools": [
            {"name": n, "input_schema": {"type": "object"}}
            for n in (tools or ["Bash", "Read", "Edit", "Write"])
        ],
    }


class Conversation:
    """Builds Anthropic-wire histories turn by turn."""

    def __init__(self, root: Path, first: str) -> None:
        self.root = root
        self.messages: list[dict[str, Any]] = [{"role": "user", "content": first}]
        self.n = 0

    def call(
        self, name: str, inp: dict[str, Any], result: str, *, error: bool = False, text: str = ""
    ) -> str:
        self.n += 1
        cid = f"toolu_{self.n:04d}"
        content: list[dict[str, Any]] = []
        if text:
            content.append({"type": "text", "text": text})
        content.append({"type": "tool_use", "id": cid, "name": name, "input": inp})
        self.messages.append({"role": "assistant", "content": content})
        block: dict[str, Any] = {"type": "tool_result", "tool_use_id": cid, "content": result}
        if error:
            block["is_error"] = True
        self.messages.append({"role": "user", "content": [block]})
        return cid

    def bash(self, command: str, output: str, *, code: int = 0) -> str:
        text = output if code == 0 else f"Exit code {code}\n{output}"
        return self.call("Bash", {"command": command}, text, error=code != 0)

    def edit(self, rel: str, old: str = "a", new: str = "b") -> str:
        return self.call(
            "Edit",
            {"file_path": str(self.root / rel), "old_string": old, "new_string": new},
            "The file has been updated.",
        )

    def read(self, rel: str) -> str:
        return self.call("Read", {"file_path": str(self.root / rel)}, "     1\tcontent")

    def say(self, text: str) -> None:
        self.messages.append({"role": "assistant", "content": [{"type": "text", "text": text}]})

    def user(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})


def send(
    service: AgentStateService, convo: Conversation, *, session: str = SESSION, commit: bool = True
):
    rs = service.begin_anthropic(
        claude_body(session), CLAUDE_HEADERS, convo.messages, cwd=str(convo.root)
    )
    if rs is not None:
        out = service.apply_anthropic(rs, convo.messages, convo.messages)
        if commit:
            service.commit(rs)
        rs.outgoing = out  # type: ignore[attr-defined]
    return rs


def hook(
    service: AgentStateService,
    root: Path,
    tool: str,
    inp: dict[str, Any],
    *,
    tool_use_id: str = "hk1",
    agent: str = "claude",
    session: str = SESSION,
):
    return service.on_pretool_hook(
        {
            "agent": agent,
            "hook": {
                "session_id": session,
                "cwd": str(root),
                "hook_event_name": "PreToolUse",
                "tool_name": tool,
                "tool_input": inp,
                "tool_use_id": tool_use_id,
            },
        }
    )


PYTEST_FAIL = "F.\nFAILED tests/test_core.py::test_f - assert 2 == 1\n1 failed, 1 passed in 0.10s"
PYTEST_PASS = "..\n2 passed in 0.05s"
