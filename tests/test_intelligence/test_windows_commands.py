"""Windows (PowerShell / cmd.exe) command classification for Codex and Claude Code.

Read protection keeps file reads byte-exact so the agent can patch them; on
Windows those reads arrive as PowerShell/cmd commands, often wrapped in
``powershell.exe -Command`` by Codex.
"""

from __future__ import annotations

import pytest

from headroom.transforms.content_router import (
    ContentRouterConfig,
    _bash_command_is_search,
    _is_read_command,
)

READS = [
    'powershell.exe -NoProfile -Command "Get-Content src\\app.py"',
    'pwsh -c "gc .\\main.rs -TotalCount 80"',
    "cmd /c type C:\\repo\\a.py",
    "cmd.exe /c type a.py",
    '"C:\\Program Files\\Git\\usr\\bin\\cat.exe" a.py',
    "Set-Location C:\\repo; Get-Content a.py",
    "cd C:\\repo && type a.py",
    "& 'C:\\tools\\cat.exe' a.py",
    "Get-Content -Path .\\src\\lib.rs",
    "bash -lc \"sed -n '1,40p' a.py\"",
]
NOT_READS = [
    "Get-Content a.py | Set-Content b.py",
    "Get-Content a.py | Out-File b.py",
    "gc a.py | Tee-Object -FilePath copy.py",
    "powershell -File script.ps1",
    "powershell -EncodedCommand ZQBjAGgAbwA=",
    'powershell -Command "Select-String -Path *.py -Pattern foo"',
    "type package-lock.json",
    "Get-Content C:\\repo\\Cargo.lock",
    'echo "cat a.py"',
]


@pytest.mark.parametrize("command", READS)
def test_windows_reads_are_protected(command):
    assert _is_read_command(command)


@pytest.mark.parametrize("command", NOT_READS)
def test_windows_non_reads(command):
    assert not _is_read_command(command)


@pytest.mark.parametrize(
    "command",
    [
        'powershell -Command "Select-String -Path *.py -Pattern foo"',
        "sls foo *.cs",
        "findstr /n /s TODO *.py",
        "cmd /c findstr /s foo *.cs",
        "& 'C:\\tools\\rg.exe' -n foo",
    ],
)
def test_windows_searches_fold_losslessly(command):
    assert _bash_command_is_search(command, ContentRouterConfig().bash_search_commands)


def test_powershell_tool_names_are_shells():
    names = ContentRouterConfig().bash_tool_names
    assert {"powershell", "pwsh", "shell_command"} <= names


def test_relative_shell_read_resolves_against_workdir():
    from headroom.intelligence.resources import resource_for

    a = resource_for("shell_command", {"command": "type src\\app.py", "workdir": "C:\\dev\\repoA"})
    b = resource_for("shell_command", {"command": "type src\\app.py", "workdir": "C:\\dev\\repoB"})
    assert a is not None and b is not None
    # Same relative name in two checkouts must be two resources (never a
    # delta base for each other).
    assert a.identity != b.identity
    assert a.identity == "file:c:/dev/repoa/src/app.py#cmd"
    # An absolute target ignores the workdir.
    c = resource_for(
        "shell",
        {
            "command": [
                "powershell.exe",
                "-Command",
                "Get-Content -Path C:\\dev\\repoA\\src\\app.py",
            ],
            "workdir": "D:\\elsewhere",
        },
    )
    assert c is not None and c.identity == a.identity
    posix = resource_for("exec_command", {"cmd": "cat src/app.py", "workdir": "/home/u/repo"})
    assert posix is not None and posix.identity == "file:/home/u/repo/src/app.py#cmd"
