"""JevK5 prompt + option-letter readout, without heavy dependencies.

When the upstream ``jevk5`` package (pinned ``v0.3.0``, installed with
``pip install --no-deps``) is importable, its ``jevk5.prompt`` module is used
directly so Headroom builds exactly the prompt the model was trained on. When
it is not installed (offline machine, locked-down Python, setup not yet run),
this module falls back to a faithful mirror of ``jevk5/prompt.py`` v0.3.0.
``tests/test_intelligence/test_jevk5_protocol.py`` asserts the mirror and the
upstream module agree byte-for-byte on prompts and numerically on readouts.

Mirror source: https://github.com/allebee/jevk5 (Apache-2.0), itself
following SemIf (TheoLeeCJ/SemIf, MIT). See NOTICE.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Callable, Sequence
from typing import Any

logger = logging.getLogger(__name__)

LETTERS = "ABCDEFGHIJKLMNOP"
METHODS = ("knockout", "tree")
TEMPERATURES = {"knockout": 0.77, "tree": 1.0}
SYSTEM = (
    "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. "
    "Respond with only its uppercase letter, with no explanation or reasoning."
)
CHAT_TEMPLATE = (
    "<|im_start|>system\n{system}<|im_end|>\n"
    "<|im_start|>user\n{user}<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n\n</think>\n\n"
)
#: A letter outside the returned top-k sits at least this far below the last.
MISSING_MARGIN = 2.0

Reader = Callable[[list[str]], Sequence[float]]


def _mirror_prompt_text(state: Any, criterion: str, options: list[str]) -> str:
    payload = {
        "evidence": state,
        "criterion": criterion,
        "options": [{"letter": LETTERS[i], "description": d} for i, d in enumerate(options)],
    }
    return CHAT_TEMPLATE.format(system=SYSTEM, user=json.dumps(payload, ensure_ascii=False))


def _mirror_decision_options(question: dict[str, Any]) -> list[tuple[str, str]]:
    crit: Any = question.get("criteria")
    pairs: list[tuple[str, str]]
    if question["type"] == "noul":
        pairs = [(k, (crit or {}).get(k) or f"The proposition is {k}.") for k in ("true", "false")]
    elif question["type"] == "choice":
        if isinstance(crit, list):
            crit = dict.fromkeys(crit)
        pairs = [(k, v or k) for k, v in (crit or {}).items()]
    else:
        pairs = [(str(i), level) for i, level in enumerate(crit or [])]
    return [(k, f"{k}: {d}") for k, d in pairs]


def _groups(n: int, count: int) -> list[range]:
    base, extra = divmod(n, count)
    runs, start = [], 0
    for g in range(count):
        stop = start + base + (g < extra)
        runs.append(range(start, stop))
        start = stop
    return runs


def _mirror_spread(
    read: Reader, texts: list[str], method: str = "knockout", temperature: float | None = None
) -> list[float]:
    if len(texts) <= len(LETTERS):
        return list(read(texts))
    probs = _combine(read, texts, method)
    temperature = TEMPERATURES[method] if temperature is None else temperature
    if temperature != 1.0:
        probs = [q ** (1 / temperature) for q in probs]
        total = sum(probs)
        probs = [q / total for q in probs]
    return probs


def _combine(read: Reader, texts: list[str], method: str) -> list[float]:
    if len(texts) <= len(LETTERS):
        return list(read(texts))
    if method == "knockout":
        weights = _knockout(read, texts)
    elif method == "tree":
        weights = _tree(read, texts)
    else:
        raise ValueError(f"unknown method {method!r}; use one of {METHODS}")
    total = sum(weights)
    return [w / total for w in weights]


def _knockout(read: Reader, texts: list[str]) -> list[float]:
    runs = _groups(len(texts), -(-len(texts) // len(LETTERS)))
    inner = [list(read([texts[i] for i in run])) for run in runs]
    inner = [[q / sum(p) for q in p] for p in inner]
    keep = max(1, len(LETTERS) // len(runs))
    ranked = [sorted(range(len(p)), key=lambda j: -p[j]) for p in inner]
    chosen = {(g, j) for g, order in enumerate(ranked) for j in order[:keep]}
    rest = sorted(
        ((g, j) for g, order in enumerate(ranked) for j in order[keep:]),
        key=lambda gj: -inner[gj[0]][gj[1]],
    )
    chosen.update(rest[: max(0, len(LETTERS) - len(chosen))])
    tops = [sorted(j for h, j in chosen if h == g) for g in range(len(runs))]
    final = _combine(read, [texts[run[j]] for run, top in zip(runs, tops) for j in top], "knockout")
    shares, at = [], 0
    for top in tops:
        shares.append(dict(zip(top, final[at : at + len(top)])))
        at += len(top)
    in_final = sum(sum(f.values()) * sum(p[j] for j in f) for p, f in zip(inner, shares))
    weights: list[float] = []
    for p, f in zip(inner, shares):
        mass = sum(f.values())
        weights += [f[j] * in_final if j in f else mass * q for j, q in enumerate(p)]
    return weights


def _tree(read: Reader, texts: list[str]) -> list[float]:
    runs = _groups(len(texts), min(len(LETTERS), -(-len(texts) // len(LETTERS))))
    outer = read(["One of: " + "; ".join(texts[i] for i in run) for run in runs])
    weights: list[float] = []
    for run, share in zip(runs, outer):
        weights += [share * q for q in _combine(read, [texts[i] for i in run], "tree")]
    return weights


def _mirror_answer(
    question: dict[str, Any], probs: dict[str, float], tokens: int
) -> dict[str, Any]:
    kind = question["type"]
    out: dict[str, Any] = {"type": kind, "confidence": max(probs.values()), "input_tokens": tokens}
    if kind == "noul":
        out["noul"] = probs["true"]
    elif kind == "choice":
        out.update(choice=max(probs, key=probs.get), probabilities=probs)  # type: ignore[arg-type]
    else:
        out.update(score=sum(int(k) * v for k, v in probs.items()), probabilities=probs)
    return out


# ---------------------------------------------------------------------------
# Upstream-first binding
# ---------------------------------------------------------------------------

UPSTREAM_VERSION: str | None = None
try:  # pragma: no cover - exercised only when the pinned package is installed
    import jevk5 as _upstream_pkg  # type: ignore[import-not-found]
    from jevk5 import prompt as _upstream  # type: ignore[import-not-found]

    UPSTREAM_VERSION = getattr(_upstream_pkg, "__version__", None)
    prompt_text = _upstream.prompt_text
    decision_options = _upstream.decision_options
    spread = _upstream.spread
    answer = _upstream.answer
except Exception:  # noqa: BLE001 - any import problem means "use the mirror"
    prompt_text = _mirror_prompt_text
    decision_options = _mirror_decision_options
    spread = _mirror_spread
    answer = _mirror_answer


def protocol_source() -> str:
    """``upstream:<version>`` or ``mirror:v0.3.0`` — reported by ``doctor``."""
    return f"upstream:{UPSTREAM_VERSION}" if UPSTREAM_VERSION else "mirror:v0.3.0"


def letter_distribution(
    seen: dict[str, float], n_options: int, temperature: float
) -> tuple[list[float], bool]:
    """Calibrated distribution over the first ``n_options`` letters.

    ``seen`` maps tokens to log-probabilities from llama-server's top-k. A
    letter missing from the top-k is assigned ``min(seen) - MISSING_MARGIN``
    (identical to ``JevK5GGUF``). Returns ``(probs, any_missing)``.
    """
    floor = min(seen.values(), default=0.0) - MISSING_MARGIN
    logprobs = [seen.get(LETTERS[i], floor) for i in range(n_options)]
    missing = any(LETTERS[i] not in seen for i in range(n_options))
    top = max(logprobs)
    weights = [math.exp((z - top) / temperature) for z in logprobs]
    total = sum(weights)
    return [w / total for w in weights], missing


def validate_question(question: dict[str, Any]) -> dict[str, Any]:
    """Normalize/validate a typed question exactly like ``jevk5.server``."""
    if question.get("type") not in ("noul", "choice", "score"):
        raise ValueError(f"unknown question type {question.get('type')!r}")
    if "instructions" not in question:
        raise ValueError("question is missing instructions")
    criteria = question.get("criteria")
    if question["type"] == "choice":
        if isinstance(criteria, list):
            criteria = dict.fromkeys(criteria)
        if not isinstance(criteria, dict) or len(criteria) < 2:
            raise ValueError("choice criteria must name at least two options")
    if question["type"] == "score" and (not isinstance(criteria, list) or len(criteria) < 2):
        raise ValueError("score criteria must list at least two levels")
    return {**question, "criteria": criteria}
