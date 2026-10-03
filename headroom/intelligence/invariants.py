"""Semantic invariant / quality guard (Optimization 7, plan §9).

Invariants are extracted from the ORIGINAL content before any candidate is
chosen, then every candidate representation is validated against them.

Hard vetoes (a candidate failing any of these is never selected):

* an explicit user-named entity disappears without exact CCR recovery;
* an exit/status code changes value (or disappears unrecoverably);
* a numeric value is changed (a number-with-unit appears that the original
  never contained — fabrication, not omission);
* error lines or test-summary counts disappear unrecoverably;
* a partial input is represented as complete (truncation notice lost);
* a retrieval marker points at content the store does not hold;
* JSON / diff structure required by the representation becomes invalid;
* the candidate is not smaller than the original (after marker overhead).

Everything else contributes to ``soft_invariant_recall`` in ``[0, 1]``.
Headroom's own annotation text (``[N items compressed …]``, ``<<ccr:…>>``,
``… lines elided``) is stripped before the "fabricated value" checks so a
compressor's bookkeeping numbers are never mistaken for changed data.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from .task_context import (
    _EXIT_CODE_RE,
    _NUMBER_UNIT_RE,
    Provenance,
    TaskContext,
    extract_entities,
)

# Markers Headroom (or a lossless fold) emits. Any of these makes the
# candidate's omissions recoverable *if* the referenced hash resolves.
_CCR_HASH_RES = (
    re.compile(r"<<ccr:([0-9a-fA-F]{8,64})"),
    re.compile(r"hash=([0-9a-fA-F]{8,64})"),
    re.compile(r"headroom_retrieve\(\s*hash\s*=\s*[\"']?([0-9a-fA-F]{8,64})"),
)
_ANNOTATION_LINE_RE = re.compile(
    r"(?im)^.*(?:<<ccr:|hash=[0-9a-f]{6,}|\bcompressed\b|\belided\b|\bomitted\b|"
    r"\bheadroom\b|\bretrieve\b|\(repeated \d+|\bunchanged\b|\bdelta\b|\bsummar(?:y|ized)\b|"
    r"\bmore (?:items|rows|lines|matches|results)\b|\btruncated\b).*$"
)
_BRACKET_ANNOTATION_RE = re.compile(
    r"\[[^\]\n]*(?:compress|omit|elid|retriev|hash=|headroom|repeated|more)[^\]\n]*\]", re.I
)
_TEST_SUMMARY_RE = re.compile(
    r"\b\d+\s+(?:passed|failed|errors?|skipped|xfailed|xpassed|warnings?|deselected|tests? ran)\b"
    r"|\bTests?:\s+\d+[^\n]{0,80}"
    r"|\btest result: (?:ok|FAILED)\.[^\n]{0,120}",
    re.I,
)
_DIFF_FILE_RE = re.compile(r"^(?:\+\+\+|---) (?:[ab]/)?(\S+)", re.M)
_HUNK_RE = re.compile(r"^@@ [^@]+ @@", re.M)
_FILE_LOC_RE = re.compile(r"[\w./\\-]+\.[A-Za-z0-9]{1,6}:\d+(?::\d+)?")
_SIGNATURE_RE = re.compile(
    r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?(?:def|class|fn|func|function|interface|struct|enum|trait|impl|type)\s+[A-Za-z_][\w]*",
    re.M,
)


def ccr_hashes_in(text: str) -> tuple[str, ...]:
    if not text:
        return ()
    found: list[str] = []
    for pattern in _CCR_HASH_RES:
        for m in pattern.finditer(text):
            h = m.group(1).lower()
            if h not in found:
                found.append(h)
    return tuple(found)


def strip_annotations(text: str) -> str:
    """Remove Headroom/compressor bookkeeping text before value checks."""
    text = _BRACKET_ANNOTATION_RE.sub(" ", text)
    return _ANNOTATION_LINE_RE.sub(" ", text)


def _content_kind(text: str) -> str:
    stripped = text.lstrip()
    if stripped[:1] in ("{", "["):
        try:
            json.loads(text)
            return "json"
        except (json.JSONDecodeError, ValueError):
            pass
    if "\n@@ " in text or text.startswith("diff --git") or "\ndiff --git " in text:
        return "diff"
    if len(_SIGNATURE_RE.findall(text[:20_000])) >= 3:
        return "code"
    return "text"


@dataclass(frozen=True)
class InvariantSet:
    """Invariants extracted from one original block."""

    kind: str = "text"
    user_entities: tuple[str, ...] = ()
    error_lines: tuple[str, ...] = ()
    exit_codes: tuple[str, ...] = ()
    test_summaries: tuple[str, ...] = ()
    file_locations: tuple[str, ...] = ()
    numbers: tuple[str, ...] = ()
    diff_files: tuple[str, ...] = ()
    hunk_headers: tuple[str, ...] = ()
    signatures: tuple[str, ...] = ()
    ccr_hashes: tuple[str, ...] = ()
    provenance: Provenance = Provenance()
    # True when extraction itself was uncertain (e.g. the original was too
    # large to scan exhaustively): callers should prefer conservative choices.
    uncertain: bool = False

    @property
    def soft_items(self) -> tuple[str, ...]:
        return (
            *self.error_lines,
            *self.file_locations,
            *self.numbers,
            *self.signatures,
            *self.test_summaries,
        )


_MAX_SCAN = 400_000


def extract_invariants(
    original: str,
    task: TaskContext | None = None,
    provenance: Provenance | None = None,
) -> InvariantSet:
    """Extract the invariant set for ``original`` (deterministic, bounded)."""
    if not original:
        return InvariantSet(provenance=provenance or Provenance())
    uncertain = len(original) > _MAX_SCAN
    scan = original if not uncertain else original[: _MAX_SCAN // 2] + original[-_MAX_SCAN // 2 :]
    ents = extract_entities(scan, max_items=256)
    lowered = scan.lower()
    user_entities: list[str] = []
    if task is not None:
        for entity in task.explicit_entities:
            if len(entity) >= 2 and entity.lower() in lowered:
                user_entities.append(entity)
    kind = _content_kind(scan)
    diff_files = tuple(dict.fromkeys(_DIFF_FILE_RE.findall(scan))) if kind == "diff" else ()
    hunks = tuple(dict.fromkeys(_HUNK_RE.findall(scan))) if kind == "diff" else ()
    signatures = (
        tuple(dict.fromkeys(s.strip() for s in _SIGNATURE_RE.findall(scan)))[:200]
        if kind == "code"
        else ()
    )
    return InvariantSet(
        kind=kind,
        user_entities=tuple(user_entities),
        error_lines=tuple(e[:160] for e in ents.error_signals[:64]),
        exit_codes=ents.exit_codes,
        test_summaries=tuple(
            dict.fromkeys(m.group(0).strip() for m in _TEST_SUMMARY_RE.finditer(scan))
        )[:32],
        file_locations=tuple(dict.fromkeys(_FILE_LOC_RE.findall(scan)))[:128],
        numbers=ents.numbers_with_units[:128],
        diff_files=diff_files,
        hunk_headers=hunks,
        signatures=signatures,
        ccr_hashes=ccr_hashes_in(scan),
        provenance=provenance or Provenance(),
        uncertain=uncertain,
    )


@dataclass(frozen=True)
class InvariantReport:
    hard_invariants_preserved: bool
    soft_invariant_recall: float
    numeric_integrity: bool
    structure_integrity: bool
    truncation_provenance_preserved: bool
    recoverable: bool
    violations: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.hard_invariants_preserved


def _present(item: str, candidate: str, candidate_lower: str) -> bool:
    if item in candidate:
        return True
    probe = item.strip()
    if len(probe) > 80:
        probe = probe[:80]
    return probe.lower() in candidate_lower


def _json_structure_ok(candidate: str) -> bool:
    text = candidate.lstrip()
    if not text or text[0] not in "[{":
        # Representation changed away from JSON entirely (e.g. a tabular
        # compaction or a preview + marker); not a JSON syntax violation.
        return True
    try:
        json.JSONDecoder().raw_decode(text)
        return True
    except (json.JSONDecodeError, ValueError):
        return False


def validate_candidate(
    original: str,
    candidate: str,
    invariants: InvariantSet,
    *,
    original_tokens: int,
    candidate_tokens: int,
    lossless: bool = False,
    store_has: Callable[[str], bool] | None = None,
    is_original: bool = False,
) -> InvariantReport:
    """Validate ``candidate`` against the invariants of ``original``."""
    if is_original or candidate == original:
        return InvariantReport(True, 1.0, True, True, True, True)

    violations: list[str] = []
    cand_lower = candidate.lower()

    # --- recoverability -------------------------------------------------
    new_hashes = [h for h in ccr_hashes_in(candidate) if h not in invariants.ccr_hashes]
    marker_valid = True
    if new_hashes and store_has is not None:
        for h in new_hashes:
            try:
                if not store_has(h):
                    marker_valid = False
                    break
            except Exception:  # noqa: BLE001 - an unreachable store is "not recoverable"
                marker_valid = False
                break
    if not marker_valid:
        violations.append("retrieval_marker_unresolvable")
    recoverable = lossless or (bool(new_hashes) and marker_valid)

    # --- token savings --------------------------------------------------
    if candidate_tokens >= original_tokens:
        violations.append("non_positive_savings")

    # --- explicit user entities ----------------------------------------
    missing_entities = [
        e for e in invariants.user_entities if not _present(e, candidate, cand_lower)
    ]
    if missing_entities and not recoverable:
        violations.append("user_entity_dropped")

    # --- values that must not change ------------------------------------
    numeric_ok = True
    cand_values = strip_annotations(candidate)
    orig_exit = set(invariants.exit_codes)
    cand_exit = set(_EXIT_CODE_RE.findall(cand_values))
    if cand_exit - orig_exit and orig_exit:
        violations.append("exit_code_changed")
        numeric_ok = False
    if orig_exit and not recoverable:
        # An exit/status code the original reported must still be reported
        # (anywhere in the candidate, annotations included) unless the full
        # original can be brought back.
        if orig_exit - set(_EXIT_CODE_RE.findall(candidate)):
            violations.append("exit_code_dropped")
    orig_numbers_lower = {n.lower().replace(" ", "") for n in invariants.numbers}
    if orig_numbers_lower:
        fabricated = [
            m.group(0)
            for m in _NUMBER_UNIT_RE.finditer(cand_values)
            if m.group(0).strip().lower().replace(" ", "") not in orig_numbers_lower
            and m.group(0).strip() not in original
        ]
        if fabricated:
            violations.append("numeric_value_changed")
            numeric_ok = False

    # --- errors and test summaries --------------------------------------
    missing_errors = [e for e in invariants.error_lines if not _present(e, candidate, cand_lower)]
    if missing_errors and not recoverable:
        violations.append("error_signal_dropped")
    missing_tests = [t for t in invariants.test_summaries if not _present(t, candidate, cand_lower)]
    if missing_tests and not recoverable:
        violations.append("test_summary_dropped")

    # --- truncation provenance -------------------------------------------
    trunc_ok = True
    prov = invariants.provenance
    if prov.is_partial:
        boundary = prov.truncation_boundary
        labelled = (
            "[partial" in cand_lower or "partial input" in cand_lower or "truncated" in cand_lower
        )
        if boundary:
            trunc_ok = boundary in candidate or labelled
        else:
            trunc_ok = labelled or "complete" not in cand_lower
        if re.search(
            r"\b(?:complete|entire|all \d+|full) (?:file|output|listing|results?)\b", cand_lower
        ):
            trunc_ok = False
        if not trunc_ok:
            violations.append("partial_represented_as_complete")

    # --- structure ------------------------------------------------------
    structure_ok = True
    if invariants.kind == "json" and not _json_structure_ok(candidate):
        structure_ok = False
        violations.append("json_structure_invalid")
    if invariants.kind == "diff":
        cand_hunks = set(_HUNK_RE.findall(candidate))
        if cand_hunks - set(invariants.hunk_headers):
            structure_ok = False
            violations.append("diff_hunk_fabricated")
        named_files = [
            f
            for f in invariants.diff_files
            if any(e in f or f in e for e in invariants.user_entities)
        ]
        if named_files and not recoverable and any(f not in candidate for f in named_files):
            structure_ok = False
            violations.append("diff_named_file_dropped")

    # --- soft recall ----------------------------------------------------
    soft = invariants.soft_items
    if soft:
        hit = sum(1 for s in soft if _present(s, candidate, cand_lower))
        recall = hit / len(soft)
    else:
        recall = 1.0
    if invariants.user_entities:
        ent_recall = 1.0 - len(missing_entities) / len(invariants.user_entities)
        recall = 0.5 * recall + 0.5 * ent_recall

    return InvariantReport(
        hard_invariants_preserved=not violations,
        soft_invariant_recall=round(recall, 6),
        numeric_integrity=numeric_ok,
        structure_integrity=structure_ok,
        truncation_provenance_preserved=trunc_ok,
        recoverable=recoverable,
        violations=tuple(violations),
    )


def summarize_violations(reports: Iterable[InvariantReport]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for report in reports:
        for v in report.violations:
            counts[v] = counts.get(v, 0) + 1
    return counts
