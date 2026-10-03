"""Secret redaction policy for opt-in proxy wire debug capture."""

from __future__ import annotations

from typing import Any

from headroom.redaction import REDACTED as WIRE_DEBUG_REDACTED
from headroom.redaction import SECRET_KEYS as WIRE_DEBUG_SECRET_KEYS
from headroom.redaction import redact_value, should_redact_key

__all__ = [
    "WIRE_DEBUG_REDACTED",
    "WIRE_DEBUG_SECRET_KEYS",
    "redact_for_wire_debug",
    "should_redact_key",
]


def redact_for_wire_debug(value: Any) -> Any:
    """Redact credential fields while preserving request/response shape.

    Key-based only: wire debug captures bodies verbatim apart from fields
    whose name marks them as credentials (shared policy in
    :mod:`headroom.redaction`).
    """
    return redact_value(value, text=False)
