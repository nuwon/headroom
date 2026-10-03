"""Python half of the Python/Rust intelligence parity contract.

``tests/fixtures/intelligence_policy_parity.json`` is asserted here against
``headroom.intelligence.policy`` and in Rust by
``compression_policy::tests::policy_admission_matches_python_parity_fixture``.
A semantic change on either side breaks one of the two.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from headroom.intelligence.policy import admit_candidate, classify_change

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "intelligence_policy_parity.json"

POLICIES = {
    # Mirrors CompressionPolicy::for_mode (crates/headroom-core/src/compression_policy.rs).
    "payg": SimpleNamespace(max_lossy_ratio=0.45, volatile_token_threshold=128),
    "subscription": SimpleNamespace(max_lossy_ratio=0.25, volatile_token_threshold=32),
    "none": None,
}


def _fixture() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", _fixture()["admission"], ids=lambda c: str(c)[:80])
def test_admission_matches_fixture(case):
    d = admit_candidate(
        POLICIES[case["policy"]],
        case["original_tokens"],
        case["candidate_tokens"],
        recoverable=case["recoverable"],
        lossless=case["lossless"],
        marker_valid=case["marker_valid"],
    )
    assert d.allowed is case["allowed"]
    assert d.reason == case["reason"]
    assert d.ratios.wire_drop_ratio == pytest.approx(case["wire_drop_ratio"], abs=1e-9)
    assert d.ratios.irreversible_drop_ratio == pytest.approx(
        case["irreversible_drop_ratio"], abs=1e-9
    )


@pytest.mark.parametrize("case", _fixture()["classify_change"], ids=lambda c: str(c))
def test_classify_change_matches_fixture(case):
    assert classify_change(POLICIES[case["policy"]], case["changed_tokens"]) == case["class"]


def test_rust_policy_table_matches_python_policies():
    """The Rust per-mode constants the fixture relies on."""
    src = (
        Path(__file__).resolve().parents[2] / "crates/headroom-core/src/compression_policy.rs"
    ).read_text(encoding="utf-8")
    for needle in (
        "MAX_LOSSY_RATIO_PAYG: f32 = 0.45",
        "MAX_LOSSY_RATIO_SUBSCRIPTION: f32 = 0.25",
        "VOLATILE_TOKEN_THRESHOLD_PAYG: u32 = 128",
        "VOLATILE_TOKEN_THRESHOLD_SUBSCRIPTION: u32 = 32",
    ):
        assert needle in src, needle


def test_email_pattern_tld_rejects_pipe_like_rust():
    from headroom.relevance.hybrid import HybridScorer

    pattern = HybridScorer._EMAIL_PATTERN
    assert pattern.search("contact alice@example.com today")
    assert pattern.search("bob@SUB.EXAMPLE.IO")
    assert not pattern.search("x@host.c|om")
    assert not pattern.search("x@host.||")
