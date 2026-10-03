"""Normalized agent events (plan §4.5–§4.8).

Six Phase 2 systems consume one event stream rather than parsing provider
wire formats on their own. Events are derived from the conversation history
each request carries (Claude Code ``tool_use``/``tool_result`` blocks, Codex
Responses ``function_call``/``*_output`` items), from host pre-tool hooks and
from Headroom's own workflow executor.

Persisted fields are bounded and redacted. ``transient`` carries in-process
payloads (message text, tool output, raw tool input) that consumers may read
during the request that produced the event, but that are never written to the
store. Raw outputs stay with CCR.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

METADATA_MAX_CHARS = 2048


class EventType(str, Enum):
    SESSION_START = "SESSION_START"
    SESSION_END = "SESSION_END"
    USER_MESSAGE = "USER_MESSAGE"
    ASSISTANT_MESSAGE = "ASSISTANT_MESSAGE"
    TOOL_CALL_PROPOSED = "TOOL_CALL_PROPOSED"
    TOOL_CALL_STARTED = "TOOL_CALL_STARTED"
    TOOL_CALL_FINISHED = "TOOL_CALL_FINISHED"
    TOOL_CALL_FAILED = "TOOL_CALL_FAILED"
    FILE_READ = "FILE_READ"
    FILE_WRITE = "FILE_WRITE"
    FILE_DELETE = "FILE_DELETE"
    COMMAND_RUN = "COMMAND_RUN"
    BUILD_RUN = "BUILD_RUN"
    TEST_RUN = "TEST_RUN"
    SEARCH_RUN = "SEARCH_RUN"
    GIT_STATE = "GIT_STATE"
    GIT_DIFF = "GIT_DIFF"
    TASK_STATE_CHANGED = "TASK_STATE_CHANGED"
    SCOPE_VIOLATION = "SCOPE_VIOLATION"
    VERIFICATION_RESULT = "VERIFICATION_RESULT"
    WORKFLOW_RUN = "WORKFLOW_RUN"


Scalar = str | int | float | bool | None


def bound_metadata(meta: dict[str, Any] | None) -> dict[str, Any]:
    """Scalar/list-only metadata, redacted, under :data:`METADATA_MAX_CHARS`."""
    if not meta:
        return {}
    from headroom.redaction import redact_text, should_redact_key

    out: dict[str, Any] = {}
    budget = METADATA_MAX_CHARS
    for key in sorted(meta):
        value = meta[key]
        if should_redact_key(key):
            continue
        if isinstance(value, (list, tuple)):
            items: list[Any] = []
            for item in value[:32]:
                if isinstance(item, (int, float, bool)) or item is None:
                    items.append(item)
                else:
                    items.append(redact_text(str(item))[:256])
            value = items
        elif isinstance(value, str):
            value = redact_text(value)[:512]
        elif not (isinstance(value, (int, float, bool)) or value is None):
            value = redact_text(str(value))[:256]
        cost = len(key) + len(str(value)) + 8
        if cost > budget:
            break
        budget -= cost
        out[key] = value
    return out


@dataclass(frozen=True)
class AgentEvent:
    event_id: str
    workspace_id: str
    session_id: str
    event_type: EventType
    task_id: str | None = None
    monotonic_seq: int = 0
    timestamp: float = field(default_factory=time.time)
    tool_name: str | None = None
    operation: str | None = None
    resource_ids: tuple[str, ...] = ()
    path_refs: tuple[str, ...] = ()
    command_fingerprint: str | None = None
    exit_code: int | None = None
    success: bool | None = None
    content_hash: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    source_ref: str | None = None
    # Never persisted: in-process payload for consumers of this request only.
    transient: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)

    @property
    def timestamp_utc(self) -> datetime:
        return datetime.fromtimestamp(self.timestamp, tz=timezone.utc)

    def to_row(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "workspace_id": self.workspace_id,
            "session_key": self.session_id,
            "task_id": self.task_id,
            "event_type": self.event_type.value,
            "ts": self.timestamp,
            "tool_name": self.tool_name,
            "operation": self.operation,
            "resource_ids": list(self.resource_ids)[:32],
            "path_refs": list(self.path_refs)[:32],
            "command_fingerprint": self.command_fingerprint,
            "exit_code": self.exit_code,
            "success": self.success,
            "content_hash": self.content_hash,
            "metadata": bound_metadata(self.metadata),
            "source_ref": self.source_ref,
        }


def command_fingerprint(command: str) -> str:
    """Stable identity for a normalized command (whitespace-insensitive)."""
    from .ids import stable_id

    return "cmd:" + stable_id(" ".join((command or "").split()), n=20)
