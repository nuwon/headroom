"""Phase 1: shared runtime, config/flags, identities, store, events, injection, families."""

from __future__ import annotations

import os
import sqlite3
import threading
from pathlib import Path

import pytest

from headroom.intelligence.agent_state import build_service
from headroom.intelligence.agent_state.config import (
    AgentStateConfig,
    AgentStateConfigError,
    EnforcementMode,
    FeatureMode,
    disabled_agent_state_config,
)
from headroom.intelligence.agent_state.events import AgentEvent, EventType, bound_metadata
from headroom.intelligence.agent_state.families import Family, SafetyClass, normalize_tool_call
from headroom.intelligence.agent_state.ids import (
    claude_session_id,
    codex_session_id,
    detect_agent,
    find_project_root,
    workspace_id,
)
from headroom.intelligence.agent_state.injection import StickyInjector, anchor_hash
from headroom.intelligence.agent_state.normalizer import clean_user_text, history_items
from headroom.intelligence.agent_state.store import MIGRATIONS, SCHEMA_VERSION, AgentStateStore
from headroom.redaction import is_secret_path, redact_text
from headroom.rollout import FEATURES, resolve_rollout

from .conftest import CLAUDE_HEADERS, SESSION, Conversation, claude_body, send


# ------------------------------------------------------------------ config
def test_six_rollout_flags_are_stable_and_default_on() -> None:
    snap = resolve_rollout({})
    for name in (
        "task_state_compiler",
        "evidence_ledger",
        "tool_contract_validator",
        "scope_firewall",
        "test_impact_planner",
        "workflow_macro_compiler",
    ):
        assert name in FEATURES
        assert snap.is_enabled(name), name


@pytest.mark.parametrize(
    ("var", "attr"),
    [
        ("HEADROOM_TASK_STATE", "task_state"),
        ("HEADROOM_EVIDENCE_LEDGER", "evidence"),
        ("HEADROOM_TOOL_CONTRACTS", "contracts"),
        ("HEADROOM_SCOPE_FIREWALL", "scope"),
        ("HEADROOM_TEST_IMPACT", "test_impact"),
        ("HEADROOM_WORKFLOW_MACROS", "workflows"),
    ],
)
def test_overrides_auto_on_off(var: str, attr: str) -> None:
    for raw, mode in (("auto", FeatureMode.AUTO), ("on", FeatureMode.ON), ("off", FeatureMode.OFF)):
        env = {var: raw}
        cfg = AgentStateConfig.from_env(env, rollout=resolve_rollout(env))
        assert getattr(cfg, attr) is mode


def test_disable_features_kill_switch() -> None:
    env = {"HEADROOM_DISABLE_FEATURES": "scope_firewall"}
    cfg = AgentStateConfig.from_env(env, rollout=resolve_rollout(env))
    assert cfg.scope is FeatureMode.OFF
    assert cfg.task_state is FeatureMode.AUTO


@pytest.mark.parametrize(
    "env",
    [
        {"HEADROOM_TASK_STATE": "maybe"},
        {"HEADROOM_SCOPE_MODE": "block"},
        {"HEADROOM_TOOL_CONTRACT_MODE": "loud"},
        {"HEADROOM_AGENT_STATE_MAX_TOKENS": "50"},
        {"HEADROOM_AGENT_STATE_MAX_TOKENS": "lots"},
        {"HEADROOM_WORKFLOW_MIN_OBSERVATIONS": "1"},
        {"HEADROOM_WORKFLOW_AUTO_CLASSES": "read_only,mutating"},
        {"HEADROOM_WORKFLOW_AUTO_CLASSES": "everything"},
        {"HEADROOM_TEST_RISK_TIER2": "0.8", "HEADROOM_TEST_RISK_TIER3": "0.7"},
        {"HEADROOM_TEST_RISK_TIER3": "1.5"},
    ],
)
def test_bad_values_fail_loudly(env: dict[str, str]) -> None:
    with pytest.raises(AgentStateConfigError):
        AgentStateConfig.from_env(env)


def test_knobs_and_roundtrip() -> None:
    cfg = AgentStateConfig.from_env(
        {
            "HEADROOM_AGENT_STATE_MAX_TOKENS": "900",
            "HEADROOM_SCOPE_MODE": "warn",
            "HEADROOM_TOOL_CONTRACT_MODE": "observe",
            "HEADROOM_WORKFLOW_AUTO_CLASSES": "read_only",
            "HEADROOM_TEST_RISK_TIER2": "0.3",
        }
    )
    assert cfg.max_tokens == 900
    assert cfg.scope_mode is EnforcementMode.WARN
    assert cfg.contract_mode is EnforcementMode.OBSERVE
    assert cfg.workflow_auto_classes == frozenset({"read_only"})
    assert AgentStateConfig.from_dict(cfg.to_dict()).to_dict() == cfg.to_dict()


def test_all_off_builds_no_service() -> None:
    assert build_service(disabled_agent_state_config()) is None


# ---------------------------------------------------------------- identity
def test_workspace_id_is_stable_hash_not_path(tmp_path: Path) -> None:
    wid = workspace_id(str(tmp_path))
    assert len(wid) == 24 and str(tmp_path.name) not in wid
    assert workspace_id(str(tmp_path) + "/") == wid
    assert workspace_id(str(tmp_path / ".." / tmp_path.name)) == wid


def test_project_root_walks_to_vcs_marker(repo: Path) -> None:
    assert find_project_root(str(repo / "src" / "pkg")) == str(repo.resolve())


def test_session_extraction() -> None:
    assert claude_session_id(claude_body()) == SESSION
    assert (
        claude_session_id({"metadata": {"user_id": '{"session_id": "abc-123-def"}'}})
        == "abc-123-def"
    )
    assert claude_session_id({}, {"x-claude-code-session-id": "sess-9876"}) == "sess-9876"
    assert codex_session_id({"prompt_cache_key": "0199-codex-thread"}) == "0199-codex-thread"
    assert detect_agent(CLAUDE_HEADERS, {}, "anthropic") == "claude_code"
    assert detect_agent({"originator": "codex_cli_rs"}, {}, "openai") == "codex"
    assert detect_agent({"user-agent": "python-httpx"}, {}, "openai") == ""


# ---------------------------------------------------------------- redaction
def test_shared_redactor_covers_secret_shapes() -> None:
    text = (
        "API_KEY=abc123secret Authorization: Bearer abcdefghijklmnop "
        "sk-ant-0123456789abcdef ghp_abcdefghijklmnopqrstuvwxyz0123 "
        "https://user:hunter2@example.com -----BEGIN RSA PRIVATE KEY-----\nMIIE\n-----END RSA PRIVATE KEY-----"
    )
    out = redact_text(text)
    for secret in (
        "abc123secret",
        "abcdefghijklmnop",
        "0123456789abcdef",
        "ghp_abcdef",
        "hunter2",
        "MIIE",
    ):
        assert secret not in out
    assert (
        is_secret_path(".env")
        and is_secret_path("keys/id_rsa")
        and not is_secret_path(".env.example")
    )


def test_metadata_is_bounded_and_redacted() -> None:
    meta = bound_metadata({"token": "x", "note": "PASSWORD=hunter2", "big": "y" * 5000})
    assert "token" not in meta
    assert "hunter2" not in meta["note"]
    assert len(str(meta)) < 2100


# -------------------------------------------------------------------- store
def test_store_pragmas_and_schema(tmp_path: Path) -> None:
    store = AgentStateStore(tmp_path / "db.sqlite3", workspace_id="w")
    assert store.available
    assert store.schema_version() == SCHEMA_VERSION
    mode = store.query_one("PRAGMA journal_mode")
    assert mode is not None and str(mode[0]).lower() == "wal"
    assert int(store.query_one("PRAGMA foreign_keys")[0]) == 1
    assert int(store.query_one("PRAGMA busy_timeout")[0]) >= 5000


@pytest.mark.parametrize("start", list(range(0, SCHEMA_VERSION)))
def test_migrations_from_every_version(tmp_path: Path, start: int) -> None:
    path = tmp_path / "db.sqlite3"
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE schema_version (version INTEGER PRIMARY KEY, description TEXT NOT NULL, applied_at REAL NOT NULL)"
    )
    for version, desc, ddl in MIGRATIONS:
        if version > start:
            break
        conn.executescript(ddl)
        conn.execute("INSERT INTO schema_version VALUES (?,?,0)", (version, desc))
    conn.commit()
    conn.close()
    store = AgentStateStore(path, workspace_id="w")
    assert store.available and store.schema_version() == SCHEMA_VERSION
    assert store.query_one("SELECT COUNT(*) FROM workflow_macros") is not None


def test_corrupt_db_is_quarantined_not_deleted(tmp_path: Path) -> None:
    path = tmp_path / "db.sqlite3"
    path.write_bytes(b"this is not a database" * 100)
    store = AgentStateStore(path, workspace_id="w")
    assert store.available
    assert store.quarantined and Path(store.quarantined[0]).exists()
    assert Path(store.quarantined[0]).read_bytes().startswith(b"this is not a database")


def test_newer_schema_disables_persistence_without_touching_file(tmp_path: Path) -> None:
    path = tmp_path / "db.sqlite3"
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE schema_version (version INTEGER PRIMARY KEY, description TEXT, applied_at REAL)"
    )
    conn.execute("INSERT INTO schema_version VALUES (999, 'future', 0)")
    conn.commit()
    conn.close()
    store = AgentStateStore(path, workspace_id="w")
    assert not store.available and "newer" in store.disabled_reason
    assert not store.quarantined
    assert store.write(lambda c: 1, default="fail-open") == "fail-open"


def test_concurrent_writers_do_not_lose_rows(tmp_path: Path) -> None:
    path = tmp_path / "db.sqlite3"
    stores = [AgentStateStore(path, workspace_id="w") for _ in range(4)]

    def work(i: int) -> None:
        for j in range(25):
            stores[i].write(
                lambda c, i=i, j=j: c.execute(
                    "INSERT INTO processed(consumer, event_id, ts) VALUES (?,?,0)",
                    (f"c{i}", f"e{j}"),
                )
            )

    threads = [threading.Thread(target=work, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert stores[0].query_one("SELECT COUNT(*) AS n FROM processed")["n"] == 100


def test_retention_runs_at_most_hourly(tmp_path: Path) -> None:
    store = AgentStateStore(tmp_path / "db.sqlite3", workspace_id="w")
    assert store.maybe_maintain(now=1_000_000.0, force=True)
    assert not store.maybe_maintain(now=1_000_100.0)
    assert store.maybe_maintain(now=1_000_000.0 + 3700)


# ------------------------------------------------------------- normalizer
def test_history_items_strip_harness_noise() -> None:
    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "<system-reminder>ctx</system-reminder>"},
                {"type": "text", "text": "Fix the parser"},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Looking"},
                {"type": "tool_use", "id": "t1", "name": "Read", "input": {"file_path": "/x.py"}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "ok"},
                {"type": "text", "text": "<system-reminder>r</system-reminder>"},
            ],
        },
        {"role": "user", "content": "[Request interrupted by user]"},
    ]
    items = history_items(msgs)
    kinds = [i.kind for i in items]
    assert kinds == ["user", "assistant", "call", "result", "user"]
    assert items[0].text == "Fix the parser"
    assert items[3].name == "Read"
    assert items[4].interrupted
    assert clean_user_text("<environment_context><cwd>/r</cwd></environment_context>") == ""


def test_codex_items_normalize_with_call_ids() -> None:
    from headroom.intelligence.messages import responses_items_to_messages

    items = [
        {
            "type": "message",
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "<environment_context><cwd>/r</cwd></environment_context>",
                }
            ],
        },
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "run the tests"}],
        },
        {
            "type": "function_call",
            "call_id": "c1",
            "name": "shell",
            "arguments": '{"command":["bash","-lc","pytest -q"],"workdir":"/r"}',
        },
        {
            "type": "function_call_output",
            "call_id": "c1",
            "output": "Exit code: 1\nWall time: 0.2 seconds\nOutput:\n1 failed in 0.1s",
        },
    ]
    hist = history_items(responses_items_to_messages(items))
    assert [h.kind for h in hist] == ["user", "call", "result"]
    assert hist[2].name == "shell" and hist[2].input["workdir"] == "/r"


# -------------------------------------------------------------- families
@pytest.mark.parametrize(
    ("name", "inp", "family", "safety"),
    [
        ("Read", {"file_path": "/r/a.py"}, Family.READ_FILE, SafetyClass.READ_ONLY),
        ("Edit", {"file_path": "/r/a.py"}, Family.EDIT_FILE, SafetyClass.MUTATING),
        ("Bash", {"command": "pytest -q 2>&1 | tail -5"}, Family.TEST, SafetyClass.VERIFICATION),
        ("Bash", {"command": "git status && git diff --stat"}, Family.GIT, SafetyClass.READ_ONLY),
        ("Bash", {"command": "git push origin main"}, Family.GIT, SafetyClass.EXTERNAL_SIDE_EFFECT),
        ("Bash", {"command": "rm -rf build"}, Family.DELETE_FILE, SafetyClass.MUTATING),
        (
            "Bash",
            {"command": "curl https://example.com"},
            Family.NETWORK,
            SafetyClass.EXTERNAL_SIDE_EFFECT,
        ),
        ("Bash", {"command": "echo hi > out.txt"}, Family.SHELL, SafetyClass.MUTATING),
        ("Bash", {"command": "sudo make install"}, Family.BUILD, SafetyClass.MUTATING),
        (
            "exec_command",
            {"cmd": ["powershell.exe", "-Command", "Get-Content src\\a.py"], "workdir": "C:\\repo"},
            Family.READ_FILE,
            SafetyClass.READ_ONLY,
        ),
        ("shell", {"command": ["cmd", "/c", "dir"]}, Family.SEARCH_TEXT, SafetyClass.READ_ONLY),
        ("mcp__srv__thing", {}, Family.MCP, SafetyClass.UNKNOWN),
        ("SomethingNew", {}, Family.OTHER, SafetyClass.UNKNOWN),
    ],
)
def test_tool_families(name, inp, family, safety) -> None:
    inv = normalize_tool_call(name, inp, default_cwd="/r")
    assert inv.family is family
    assert inv.safety is safety


def test_privilege_is_recorded_never_added() -> None:
    inv = normalize_tool_call("Bash", {"command": "sudo rm -rf /tmp/x"})
    assert inv.privileged and inv.paths_deleted == ("/tmp/x",)


def test_apply_patch_paths() -> None:
    patch = "*** Begin Patch\n*** Update File: src/a.py\n@@\n-x\n+y\n*** Delete File: src/b.py\n*** End Patch"
    inv = normalize_tool_call("apply_patch", {"input": patch}, default_cwd="/r")
    assert inv.paths_written == ("/r/src/a.py",) and inv.paths_deleted == ("/r/src/b.py",)
    shell = normalize_tool_call("shell", {"command": ["apply_patch", patch], "workdir": "/r"})
    assert shell.paths_written == ("/r/src/a.py",)


# -------------------------------------------------------------- injection
def test_sticky_injection_replays_identical_bytes() -> None:
    inj = StickyInjector()
    turn1 = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "go", "cache_control": {"type": "ephemeral"}}],
        }
    ]
    out1, pending = inj.apply_anthropic("s", turn1, turn1, "<state1>", "h1")
    inj.commit(pending)
    # Next turn: the client moved its breakpoint and appended history.
    turn2 = [
        {"role": "user", "content": [{"type": "text", "text": "go"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
        {"role": "user", "content": "next"},
    ]
    out2, pending2 = inj.apply_anthropic("s", turn2, turn2, None, "h1")
    assert out2[0]["content"][-1] == {"type": "text", "text": "<state1>"}
    assert pending2.replayed == 1 and pending2.entry is None
    assert anchor_hash(turn1[0]) == anchor_hash(turn2[0])
    # Untouched messages are the same objects (byte-identical by construction).
    assert out2[1] is turn2[1]


def test_uncommitted_injection_is_not_replayed() -> None:
    inj = StickyInjector()
    msgs = [{"role": "user", "content": "x"}]
    inj.apply_anthropic("s", msgs, msgs, "<b>", "h")
    assert inj.apply_anthropic("s", msgs, msgs, None, "h") is None


def test_responses_injection_inserts_after_anchor() -> None:
    inj = StickyInjector()
    items = [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "a"}]}]
    out, pending = inj.apply_responses("s", items, items, "<b>", "h", incremental=False)
    inj.commit(pending)
    assert out[1]["role"] == "user" and out[1]["content"][0]["text"] == "<b>"
    later = [*items, {"type": "function_call_output", "call_id": "c", "output": "x"}]
    out2, _ = inj.apply_responses("s", later, later, None, "h", incremental=False)
    assert [i.get("type") for i in out2] == ["message", "message", "function_call_output"]


# ------------------------------------------------------- service behavior
def test_non_agent_traffic_is_untouched(service, repo: Path) -> None:
    convo = Conversation(repo, "hello")
    assert (
        service.begin_anthropic(
            {"tools": []}, {"user-agent": "curl"}, convo.messages, cwd=str(repo)
        )
        is None
    )


def test_sessions_and_workspaces_are_isolated(service, repo: Path, tmp_path: Path) -> None:
    a = Conversation(repo, "Fix the parser. Do not modify docs/.")
    send(service, a, session="aaaaaaaa-1111")
    b = Conversation(repo, "Write a changelog entry.")
    send(service, b, session="bbbbbbbb-2222")
    rts = list(service._runtimes.values())
    goals = {rt.agent_session: rt.task_state.state.primary_goal.text for rt in rts}
    assert goals["aaaaaaaa-1111"].startswith("Fix the parser")
    assert goals["bbbbbbbb-2222"].startswith("Write a changelog")
    other = tmp_path / "other"
    other.mkdir()
    (other / ".git").mkdir()
    c = Conversation(other, "Unrelated project task")
    rs = send(service, c, session="aaaaaaaa-1111")
    assert rs.runtime.workspace.workspace_id != rts[0].workspace.workspace_id
    assert rs.runtime.task_state.state.primary_goal.text.startswith("Unrelated")
    db_a = rts[0].store.path
    db_c = rs.runtime.store.path
    assert db_a != db_c and str(repo) not in str(db_a)


def test_replayed_request_is_idempotent(service, repo: Path) -> None:
    convo = Conversation(repo, "Fix the parser. Tests must pass.")
    convo.bash("pytest -q", "1 failed in 0.1s", code=1)
    send(service, convo)
    rt = next(iter(service._runtimes.values()))
    before = rt.store.query_one("SELECT COUNT(*) AS n FROM evidence")["n"]
    revision = rt.task_state.state.revision
    send(service, convo)
    send(service, convo)
    assert rt.store.query_one("SELECT COUNT(*) AS n FROM evidence")["n"] == before
    assert rt.task_state.state.revision == revision


def test_disabled_features_are_noops(make_service, repo: Path) -> None:
    env = dict.fromkeys(
        (
            "HEADROOM_TASK_STATE",
            "HEADROOM_EVIDENCE_LEDGER",
            "HEADROOM_TOOL_CONTRACTS",
            "HEADROOM_SCOPE_FIREWALL",
            "HEADROOM_TEST_IMPACT",
            "HEADROOM_WORKFLOW_MACROS",
        ),
        "off",
    )
    svc = make_service(env=env)
    convo = Conversation(repo, "Fix the parser. Do not touch docs/.")
    assert svc.begin_anthropic(claude_body(), CLAUDE_HEADERS, convo.messages, cwd=str(repo)) is None
    assert not list(Path(os.environ["HEADROOM_AGENT_STATE_DIR"]).glob("*/agent_state.sqlite3"))


def test_single_feature_off_leaves_others(make_service, repo: Path) -> None:
    svc = make_service(env={"HEADROOM_SCOPE_FIREWALL": "off"})
    convo = Conversation(repo, "Fix the parser. Do not modify tests/test_core.py.")
    rs = send(svc, convo)
    assert rs.runtime.scope is None and rs.runtime.task_state is not None
    assert (
        hook(
            svc,
            repo,
            "Edit",
            {"file_path": str(repo / "tests/test_core.py"), "old_string": "a", "new_string": "b"},
        )["decision"]
        == "allow"
    )


def test_store_unavailable_fails_open(make_service, repo: Path, monkeypatch) -> None:
    from headroom.intelligence.agent_state import runtime as rt_mod

    class Dead:
        available = False
        disabled_reason = "unavailable: test"

    monkeypatch.setattr(rt_mod, "get_store", lambda *a, **k: Dead())
    svc = make_service()
    convo = Conversation(repo, "Fix it")
    assert svc.begin_anthropic(claude_body(), CLAUDE_HEADERS, convo.messages, cwd=str(repo)) is None
    assert svc.metrics.counters["store_unavailable"] >= 1


def test_events_are_persisted_with_monotonic_seq(service, repo: Path) -> None:
    convo = Conversation(repo, "Fix the parser")
    convo.read("src/pkg/core.py")
    send(service, convo)
    rt = next(iter(service._runtimes.values()))
    rows = rt.store.query(
        "SELECT event_type, seq FROM events WHERE session_key = ? ORDER BY seq", (rt.session_key,)
    )
    seqs = [r["seq"] for r in rows]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)
    types = {r["event_type"] for r in rows}
    assert {
        "SESSION_START",
        "USER_MESSAGE",
        "TOOL_CALL_PROPOSED",
        "TOOL_CALL_FINISHED",
        "FILE_READ",
    } <= types


def hook(svc, root, tool, inp):
    from .conftest import hook as _hook

    return _hook(svc, root, tool, inp)


def test_event_model_fields() -> None:
    ev = AgentEvent(
        event_id="e",
        workspace_id="w",
        session_id="s",
        event_type=EventType.TEST_RUN,
        metadata={"api_key": "x", "n": 1},
    )
    row = ev.to_row()
    assert row["metadata"] == {"n": 1}
    assert ev.timestamp_utc.tzinfo is not None
