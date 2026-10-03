"""One deterministic ``<headroom_agent_state>`` block per live turn (plan §5.8, §12.1).

All six systems contribute *sections*. A section is a priority, a heading and
some lines. Sections merge into one block under hard caps:

* task state: 900 tokens, target 600;
* evidence: 350;
* scope warnings: 180;
* verification plan: 250;
* workflow catalog: 250;
* combined: ``HEADROOM_AGENT_STATE_MAX_TOKENS`` (default 1200).

Under pressure, completed history goes first and hard warnings and blockers
go last, which is the plan's priority order. Empty headings are never emitted
and the user's text is never rewritten. Output depends only on state
(sections are sorted, with no timestamps and no dict iteration order), so
identical state renders identical bytes.

The *material hash* covers everything except completed-history and revision
churn. A new block is injected only when it changes (see :mod:`.injection`).
"""

from __future__ import annotations

import hashlib
from typing import Any

Section = tuple[int, str, list[str]]

_CAP_KEYS = {
    "scope-warning": "scope_tokens",
    "evidence-conflicts": "evidence_tokens",
    "failing": "evidence_tokens",
    "compiler-errors": "evidence_tokens",
    "verify": "verification_tokens",
    "workflows": "workflow_tokens",
}
_TASK_HEADINGS = {
    "blocked",
    "unresolved",
    "constraints",
    "goal",
    "acceptance",
    "pending",
    "decisions",
    "done",
}
_NON_MATERIAL = {"done"}
NOTE = "automated task state from Headroom (not a user message)"


def tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4) if text else 0


def _clip_section(lines: list[str], budget: int) -> list[str]:
    out: list[str] = []
    used = 0
    for line in lines:
        cost = tokens(line) + 1
        if used + cost > budget:
            if not out and budget > 8:
                out.append(line[: max(8, budget * 4 - 4)].rstrip() + "…")
            break
        out.append(line)
        used += cost
    return out


def _render_section(heading: str, lines: list[str]) -> list[str]:
    if not lines:
        return []
    if heading == "goal" and len(lines) == 1:
        return [f"goal: {lines[0]}"]
    return [f"{heading}:", *lines]


def compose(
    sections: list[Section],
    *,
    cfg: Any,
    attrs: dict[str, str],
) -> tuple[str, str] | None:
    """Merge sections under the budgets; return ``(block, material_hash)`` or None."""
    if not sections:
        return None
    per_feature_used: dict[str, int] = {}
    task_used = 0
    total_budget = int(cfg.max_tokens)
    header = (
        "<headroom_agent_state "
        + " ".join(f'{k}="{v}"' for k, v in sorted(attrs.items()))
        + f' note="{NOTE}">'
    )
    footer = "</headroom_agent_state>"
    used = tokens(header) + tokens(footer)
    kept: list[tuple[int, str, list[str]]] = []
    for prio, heading, lines in sorted(sections, key=lambda s: (s[0], s[1])):
        cap_key = _CAP_KEYS.get(heading)
        if cap_key is not None:
            cap = int(getattr(cfg, cap_key)) - per_feature_used.get(cap_key, 0)
        elif heading in _TASK_HEADINGS:
            target = (
                int(cfg.task_state_target_tokens)
                if heading == "done"
                else int(cfg.task_state_tokens)
            )
            cap = target - task_used
        else:
            cap = int(cfg.scope_tokens)
        cap = min(cap, total_budget - used)
        if cap <= 4:
            continue
        clipped = _clip_section(lines, cap - tokens(heading) - 1)
        if not clipped:
            continue
        cost = sum(tokens(x) + 1 for x in clipped) + tokens(heading) + 1
        used += cost
        if cap_key is not None:
            per_feature_used[cap_key] = per_feature_used.get(cap_key, 0) + cost
        elif heading in _TASK_HEADINGS:
            task_used += cost
        kept.append((prio, heading, clipped))
    if not kept:
        return None
    body: list[str] = []
    material: list[str] = [attrs.get("task", ""), attrs.get("status", "")]
    for _, heading, lines in kept:
        rendered = _render_section(heading, lines)
        body.extend(rendered)
        if heading not in _NON_MATERIAL:
            material.extend(x for x in rendered if not x.startswith("risk="))
    block = "\n".join([header, *body, footer])
    digest = hashlib.sha256("\n".join(material).encode("utf-8", "surrogatepass")).hexdigest()[:20]
    return block, digest


def render_agent_state(rt: Any) -> tuple[str, str] | None:
    """Collect sections from every enabled system of ``rt`` and compose the block."""
    sections: list[Section] = []
    warnings = rt.take_warnings()
    scope_lines: list[str] = []
    if rt.scope is not None:
        scope_lines = rt.scope.pending_warnings()
        if scope_lines:
            rt.scope.mark_delivered()
    seen = set()
    merged: list[str] = []
    for w in [*scope_lines, *warnings]:
        if w not in seen:
            seen.add(w)
            merged.append(f"- {w}")
    if merged:
        sections.append((0, "scope-warning" if scope_lines else "warnings", merged))
    extra: list[Section] = []
    if rt.evidence is not None:
        extra.extend(rt.evidence.sections())
    if rt.test_impact is not None:
        try:
            extra.extend((42, h, ls) for _, h, ls in rt.test_impact.sections())
        except Exception:  # noqa: BLE001
            pass
    if rt.workflows is not None:
        extra.extend(rt.workflows.sections())
    sections.extend(extra)
    ts = rt.task_state
    attrs: dict[str, str] = {}
    if ts is not None and ts.state is not None:
        if ts.wants_injection() or sections:
            sections.extend(ts.sections())
            attrs = ts.header_attrs()
    if not sections:
        return None
    if attrs.get("revision") is None:
        attrs["revision"] = "0"
    return compose(sections, cfg=rt.config, attrs=attrs)
