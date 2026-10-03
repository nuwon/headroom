"""Cross-turn semantic delta encoding (Optimization 5, plan §12).

When the agent re-reads / re-searches / re-runs the same resource, send what
changed instead of another near-identical copy. Content-aware strategies:

* JSON arrays (or an object's main array): element delta keyed by a stable
  key (``id``/``uuid``/``key``/``name``/``path``) when one exists, else by the
  canonical element hash;
* search results / listings: order-preserving line-set delta;
* files / command output / logs: line diff with one line of context.

The representation states what was omitted, what was added/removed/changed,
the base and current content hashes, and (added by the caller after a
verified CCR store) the hash of the complete current version. A delta is
never computed against a partial base, and is rejected when the change is so
large that a delta is no clearer than the full output.
"""

from __future__ import annotations

import difflib
import json
from dataclasses import dataclass
from typing import Any

from .resources import content_hash

MAX_DIFF_LINES = 20_000
STABLE_KEYS = ("id", "uuid", "key", "name", "path", "file", "url", "sha", "ref")


@dataclass(frozen=True)
class DeltaBody:
    strategy: str  # json_keyed | json_hashed | line_set | line_diff | identical
    body: str
    added: int
    removed: int
    changed: int
    unchanged: int
    changed_text: str  # added/changed material only (for invariant + volatility checks)


def _as_array(text: str) -> list[Any] | None:
    stripped = text.strip()
    if not stripped or stripped[0] not in "[{":
        return None
    try:
        value = json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        return None
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        arrays = [v for v in value.values() if isinstance(v, list)]
        if arrays:
            return max(arrays, key=len)
    return None


def _stable_key(items: list[Any]) -> str | None:
    if not items or not all(isinstance(i, dict) for i in items):
        return None
    for key in STABLE_KEYS:
        values = [i.get(key) for i in items]
        if all(isinstance(v, (str, int)) for v in values) and len(set(map(str, values))) == len(
            values
        ):
            return key
    return None


def _canon(item: Any) -> str:
    return json.dumps(item, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _json_delta(base: list[Any], cur: list[Any]) -> DeltaBody | None:
    key = _stable_key(base) if _stable_key(cur) == _stable_key(base) else None
    lines: list[str] = []
    changed_parts: list[str] = []
    if key is not None:
        b = {str(i[key]): i for i in base}
        c = {str(i[key]): i for i in cur}
        added = [k for k in c if k not in b]
        removed = [k for k in b if k not in c]
        changed = [k for k in c if k in b and _canon(b[k]) != _canon(c[k])]
        unchanged = len(c) - len(added) - len(changed)
        for k in added:
            s = _canon(c[k])
            lines.append(f"+ {s}")
            changed_parts.append(s)
        for k in changed:
            s = _canon(c[k])
            lines.append(f"~ {s}")
            changed_parts.append(s)
        for k in removed:
            lines.append(f"- {key}={k}")
        return DeltaBody(
            "json_keyed",
            "\n".join(lines),
            len(added),
            len(removed),
            len(changed),
            unchanged,
            "\n".join(changed_parts),
        )
    bh = [_canon(i) for i in base]
    ch = [_canon(i) for i in cur]
    bset, cset = set(bh), set(ch)
    added_l = [s for s in ch if s not in bset]
    removed_l = [s for s in bh if s not in cset]
    for s in added_l:
        lines.append(f"+ {s}")
    for s in removed_l:
        lines.append(f"- {s}")
    unchanged = sum(1 for s in ch if s in bset)
    return DeltaBody(
        "json_hashed",
        "\n".join(lines),
        len(added_l),
        len(removed_l),
        0,
        unchanged,
        "\n".join(added_l),
    )


def _lines(text: str) -> list[str]:
    return text.replace("\r\n", "\n").split("\n")


def _line_set_delta(base: str, cur: str) -> DeltaBody:
    b, c = _lines(base), _lines(cur)
    bset, cset = set(b), set(c)
    added = [ln for ln in c if ln not in bset]
    removed = [ln for ln in b if ln not in cset]
    body = "\n".join([*(f"+ {ln}" for ln in added), *(f"- {ln}" for ln in removed)])
    unchanged = sum(1 for ln in c if ln in bset)
    return DeltaBody("line_set", body, len(added), len(removed), 0, unchanged, "\n".join(added))


def _line_diff(base: str, cur: str) -> DeltaBody | None:
    b, c = _lines(base), _lines(cur)
    if len(b) > MAX_DIFF_LINES or len(c) > MAX_DIFF_LINES:
        return None
    matcher = difflib.SequenceMatcher(a=b, b=c, autojunk=False)
    out: list[str] = []
    added = removed = changed = 0
    changed_parts: list[str] = []
    for group in matcher.get_grouped_opcodes(1):
        first, last = group[0], group[-1]
        out.append(
            f"@@ -{first[1] + 1},{last[2] - first[1]} +{first[3] + 1},{last[4] - first[3]} @@"
        )
        for tag, i1, i2, j1, j2 in group:
            if tag == "equal":
                out.extend(f"  {ln}" for ln in c[j1:j2])
                continue
            if tag in ("replace", "delete"):
                out.extend(f"- {ln}" for ln in b[i1:i2])
                removed += i2 - i1
            if tag in ("replace", "insert"):
                out.extend(f"+ {ln}" for ln in c[j1:j2])
                added += j2 - j1
                changed_parts.extend(c[j1:j2])
            if tag == "replace":
                changed += min(i2 - i1, j2 - j1)
    unchanged = sum(t[2] - t[1] for t in matcher.get_opcodes() if t[0] == "equal")
    return DeltaBody(
        "line_diff", "\n".join(out), added, removed, changed, unchanged, "\n".join(changed_parts)
    )


def compute_delta(base: str, current: str, resource_kind: str) -> DeltaBody | None:
    """Delta body from ``base`` to ``current`` (``None`` when not applicable)."""
    if content_hash(base) == content_hash(current):
        return DeltaBody("identical", "", 0, 0, 0, len(_lines(current)), "")
    b_arr, c_arr = _as_array(base), _as_array(current)
    if b_arr is not None and c_arr is not None:
        return _json_delta(b_arr, c_arr)
    if resource_kind in ("search", "list"):
        return _line_set_delta(base, current)
    return _line_diff(base, current)


def render_delta(
    body: DeltaBody,
    *,
    label: str,
    base_message_index: int,
    base_hash: str,
    current_hash: str,
    stability: str,
) -> str:
    """Human/model-facing delta header + body (marker appended by the caller)."""
    unit = "items" if body.strategy.startswith("json") else "lines"
    if body.strategy == "identical":
        head = (
            f"headroom delta: {label} — identical to the output in message {base_message_index} "
            f"(content {current_hash[:12]}); repeated copy omitted."
        )
        return head
    head = (
        f"headroom delta: {label} — same resource as message {base_message_index} "
        f"(base {base_hash[:12]} -> current {current_hash[:12]}, {stability} change). "
        f"{body.unchanged} unchanged {unit} omitted; {body.added} added, {body.removed} removed"
        + (f", {body.changed} changed" if body.changed else "")
        + "."
    )
    return f"{head}\n{body.body}" if body.body else head
