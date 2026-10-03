"""Token economy of the sticky agent-state block: deltas, events, goal, back-off.

Every block stays in history byte-for-byte (prompt-cache safety) and is re-sent
on each later request, so these tests pin down what keeps that cost small
without giving up correctness: a delta only when its base is provably present,
a cleared section reported as ``none``, announce-once sections that never force
or fake a change, no repeated sentences on the goal line, and minor churn that
backs off as the state already in history grows.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from headroom.intelligence.agent_state import runtime as runtime_mod
from headroom.intelligence.agent_state.injection import MemoEntry
from headroom.intelligence.agent_state.serialization import compose_state, delta_block
from headroom.intelligence.agent_state.task_state import AtomKind, _goal_line, normalize_key
from headroom.intelligence.agent_state.test_impact import CommandSpec

from .conftest import CLAUDE_HEADERS, PYTEST_FAIL, PYTEST_PASS, Conversation, claude_body, send

GOAL = (
    "Add retry support to src/pkg/core.py. Do not change the public API. "
    "Never log credentials. All tests must pass."
)


def _blocks(messages: list[dict]) -> list[str]:
    out = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            out.extend(
                b["text"]
                for b in content
                if isinstance(b, dict)
                and str(b.get("text", "")).startswith("<headroom_agent_state")
            )
    return out


def _new_block(rs) -> str | None:  # noqa: ANN001
    entry = rs.pending.entry if rs is not None and rs.pending is not None else None
    return entry.text if entry is not None else None


def test_goal_line_drops_sentences_listed_as_atoms(service, repo: Path) -> None:
    rs = send(service, Conversation(repo, GOAL))
    block = _new_block(rs)
    assert block is not None
    assert "goal: Add retry support to src/pkg/core.py.\n" in block
    assert block.count("Do not change the public API.") == 1  # constraints only
    assert block.count("Never log credentials.") == 1


def test_goal_line_is_unchanged_when_every_sentence_is_an_atom() -> None:
    atoms = [SimpleNamespace(kind=AtomKind.CONSTRAINT, normalized_key=normalize_key("Keep it."))]
    assert _goal_line("Keep it.", atoms) == "Keep it."
    assert _goal_line("Fix the bug. Keep it.", atoms) == "Fix the bug."


def test_second_block_is_a_delta_and_the_prefix_is_unchanged(service, repo: Path) -> None:
    convo = Conversation(repo, GOAL)
    first = send(service, convo)
    full = _new_block(first)
    assert full is not None and "changes-since" not in full
    convo.edit("src/pkg/core.py")
    convo.bash("python -m pytest -q", PYTEST_FAIL, code=1)
    rs = send(service, convo)
    delta = _new_block(rs)
    assert delta is not None
    assert 'changes-since="' in delta
    assert "blocked:" in delta
    assert "constraints:" not in delta  # unchanged sections are not repeated
    assert rs.block is not None and len(delta) < len(rs.block)  # vs a full block now
    # The earlier full block is replayed byte-identically ahead of the delta.
    assert _blocks(rs.outgoing)[0] == full
    assert rs.runtime.service.metrics.counters["injections_delta"] == 1


def test_a_cleared_blocker_is_never_shown_as_current(service, repo: Path) -> None:
    convo = Conversation(repo, GOAL)
    send(service, convo)
    convo.edit("src/pkg/core.py")
    convo.bash("python -m pytest -q", PYTEST_FAIL, code=1)
    assert "blocked:" in (_new_block(send(service, convo)) or "")
    convo.bash("python -m pytest -q", PYTEST_PASS)
    block = _new_block(send(service, convo))
    assert block is not None
    # Either a delta that clears it, or a full block that no longer lists it.
    if "changes-since" in block:
        assert "blocked: none" in block
    else:
        assert "blocked:" not in block and "B1" not in block


def test_delta_reports_a_removed_section_as_none(service) -> None:
    cfg = service.config
    attrs = {"task": "T1", "revision": "6"}
    goal = (30, "goal", ["Ship it."])
    before = compose_state(
        [(10, "blocked", ["- B1 pytest: 1 failing"]), goal], cfg=cfg, attrs=attrs
    )
    after = compose_state([goal], cfg=cfg, attrs={**attrs, "revision": "7"})
    assert before is not None and after is not None
    delta = delta_block(after, before.digests(), "6")
    assert delta is not None
    assert "blocked: none" in delta and "goal:" not in delta
    assert 'changes-since="6"' in delta and 'revision="7"' in delta


def test_full_block_when_the_delta_base_left_the_history(service, repo: Path) -> None:
    convo = Conversation(repo, GOAL)
    send(service, convo)
    convo.edit("src/pkg/core.py")
    convo.bash("python -m pytest -q", PYTEST_FAIL, code=1)
    send(service, convo)
    # The client compacted: none of the earlier anchors survive.
    compacted = Conversation(repo, GOAL)
    compacted.messages[0] = {"role": "user", "content": [{"type": "text", "text": GOAL}]}
    compacted.bash("python -m pytest -q", PYTEST_FAIL, code=1)
    rs = send(service, compacted)
    block = _new_block(rs)
    assert block is not None
    assert "changes-since" not in block and "constraints:" in block


def test_codex_incremental_requests_always_get_full_blocks(service, repo: Path) -> None:
    convo = Conversation(repo, GOAL)
    send(service, convo)
    convo.edit("src/pkg/core.py")
    convo.bash("python -m pytest -q", PYTEST_FAIL, code=1)
    rs = service.begin_anthropic(claude_body(), CLAUDE_HEADERS, convo.messages, cwd=str(repo))
    assert rs is not None and rs.block is not None
    text, full = service._encode(rs, convo.messages, incremental=True)  # noqa: SLF001
    assert full and text == rs.block


def test_event_sections_stay_out_of_state_and_are_never_none(service) -> None:
    cfg = service.config
    attrs = {"task": "T1", "revision": "4"}
    base = [(20, "constraints", ["- C1 Do not change the public API."])]
    announce = [(60, "workflows", ["- headroom_workflow macro=m: a -> b"])]
    quiet = compose_state(base, cfg=cfg, attrs=attrs)
    loud = compose_state([*base, *announce], cfg=cfg, attrs=attrs)
    assert quiet is not None and loud is not None
    assert loud.has_events and not quiet.has_events
    assert loud.state_hash == quiet.state_hash  # an announcement is not a state change
    assert loud.digests() == quiet.digests()
    # Dropping the announcement afterwards is not a change either.
    assert delta_block(quiet, loud.digests(), "4") is None
    # A fresh announcement is always carried by the next block.
    delta = delta_block(loud, quiet.digests(), "4")
    assert delta is not None and "workflows:" in delta and "constraints" not in delta


def test_status_change_alone_still_produces_a_delta(service) -> None:
    cfg = service.config
    sections = [(20, "constraints", ["- C1 x"])]
    active = compose_state(sections, cfg=cfg, attrs={"task": "T1", "revision": "4"})
    done = compose_state(
        sections, cfg=cfg, attrs={"task": "T1", "revision": "5", "status": "complete"}
    )
    assert active is not None and done is not None
    delta = delta_block(done, active.digests(), "4")
    assert delta is not None and 'status="complete"' in delta


def test_memo_entries_written_before_deltas_load_as_full_blocks() -> None:
    legacy = MemoEntry.from_json([3, "anchor", "<headroom_agent_state>", "a:b"])
    assert legacy is not None and legacy.full and legacy.digests == ""
    entry = MemoEntry(1, "x", "t", "h", "@task=T;goal=abc", False, "7")
    assert MemoEntry.from_json(entry.to_json()) == entry


def test_minor_churn_backs_off_as_state_history_grows(
    make_service, repo: Path, monkeypatch
) -> None:
    def injections(soft_tokens: int) -> int:
        monkeypatch.setattr(runtime_mod, "STATE_HISTORY_SOFT_TOKENS", soft_tokens)
        service = make_service()
        convo = Conversation(repo, GOAL)
        send(service, convo)
        count = 0
        for i in range(14):
            convo.edit("src/pkg/core.py", f"o{i}", f"n{i}")
            send(service, convo)
            convo.bash("python -m pytest -q tests/test_core.py", PYTEST_PASS)
            rs = send(service, convo)
            count += _new_block(rs) is not None
        assert service.live_state_tokens(rs.runtime, convo.messages) > 0
        return count

    default = injections(runtime_mod.STATE_HISTORY_SOFT_TOKENS)
    backed_off = injections(1)  # any state in history multiplies the interval
    assert backed_off < default


def test_verify_commands_show_project_interpreters_relative(tmp_path: Path) -> None:
    root = str(tmp_path)
    venv_python = str(tmp_path / ".venv" / "bin" / "python3")
    spec = CommandSpec("pytest", (venv_python, "-m", "pytest", "-q"), root)
    assert spec.brief(root).startswith(".venv")
    assert spec.display().startswith(venv_python)  # evidence matching is unchanged
    elsewhere = CommandSpec("pytest", ("/usr/bin/python3", "-m", "pytest"), root)
    assert elsewhere.brief(root) == elsewhere.display()
