"""Foundation tests: config, TaskContext, invariants, policy admission, arbiter."""

from __future__ import annotations

import json

import pytest

from headroom.intelligence.arbiter import (
    TIER_AGGRESSIVE,
    TIER_CONSERVATIVE,
    TIER_INDEXED,
    TIER_STRUCTURAL,
    ArbiterSession,
    Candidate,
    arbitrate,
)
from headroom.intelligence.config import (
    MAX_ADVISORY_WEIGHT,
    ArbiterWeights,
    IntelligenceConfig,
    level_features,
    manifest_for,
)
from headroom.intelligence.invariants import extract_invariants, validate_candidate
from headroom.intelligence.models import AdvisoryScores, DecisionFamily
from headroom.intelligence.policy import admit_candidate, classify_change, drop_ratios
from headroom.intelligence.task_context import (
    build_task_context,
    detect_provenance,
    extract_entities,
)
from headroom.proxy.auth_mode import AuthMode
from headroom.transforms.compression_policy import policy_for_mode


def tok(text: str) -> int:
    return max(1, len(text) // 4) if text else 0


PAYG = policy_for_mode(AuthMode.PAYG)
SUB = policy_for_mode(AuthMode.SUBSCRIPTION)


# --------------------------------------------------------------------- config
class TestConfig:
    def test_default_is_off(self):
        cfg = IntelligenceConfig.from_env({})
        assert cfg.level == "off"
        assert not cfg.any_enabled
        assert cfg.jevk5.mode == "auto"

    def test_safe_level_enables_safe_features_only(self):
        cfg = IntelligenceConfig.from_env({"HEADROOM_INTELLIGENCE": "safe"})
        assert cfg.task_query and cfg.invariant_guard and cfg.policy_budget and cfg.arbiter
        assert cfg.ccr_search
        assert not cfg.delta and not cfg.admission and not cfg.tool_catalog
        assert not cfg.effort_routing

    def test_full_level_and_per_feature_override(self):
        cfg = IntelligenceConfig.from_env(
            {"HEADROOM_INTELLIGENCE": "full", "HEADROOM_DELTA": "0", "HEADROOM_EFFORT_ROUTING": "1"}
        )
        assert cfg.admission and cfg.tool_catalog and cfg.budget_allocator
        assert not cfg.delta
        assert cfg.effort_routing  # explicit opt-in only

    def test_feature_on_without_level(self):
        cfg = IntelligenceConfig.from_env({"HEADROOM_CCR_SEARCH": "1"})
        assert cfg.ccr_search and not cfg.arbiter

    def test_effort_routing_never_in_a_posture(self):
        assert "effort_routing" not in level_features("full")
        assert "delta_reads" not in level_features("full")

    def test_weights_sum_to_one(self):
        ArbiterWeights().validate()
        assert abs(ArbiterWeights().total() - 1.0) < 1e-12
        with pytest.raises(ValueError):
            ArbiterWeights(latency=0.5).validate()

    def test_advisory_weight_clamped(self):
        cfg = IntelligenceConfig.from_env({"HEADROOM_JEVK5_ADVISORY_WEIGHT": "0.9"})
        assert cfg.jevk5.advisory_weight == MAX_ADVISORY_WEIGHT
        cfg = IntelligenceConfig.from_env({"HEADROOM_JEVK5_ADVISORY_WEIGHT": "-1"})
        assert cfg.jevk5.advisory_weight == 0.0

    def test_jevk5_manifest_calibration(self):
        m = manifest_for("jevk5-4b-v0.3-Q8_0.gguf")
        assert m is not None
        assert (m.temperature, m.knockout_temperature) == (1.22, 0.93)
        cfg = IntelligenceConfig.from_env({})
        assert cfg.jevk5.temperature == 1.22 and cfg.jevk5.knockout_temperature == 0.93
        assert cfg.jevk5.model_repo == "alibiserikbay/JevK5-GGUF"

    def test_jevk5_modes(self):
        assert IntelligenceConfig.from_env({"HEADROOM_JEVK5": "off"}).jevk5.mode == "off"
        on = IntelligenceConfig.from_env({"HEADROOM_JEVK5": "on"}).jevk5
        assert on.mode == "on" and on.allow_download and on.allow_build
        auto = IntelligenceConfig.from_env({}).jevk5
        assert not auto.allow_download and not auto.allow_build

    def test_bad_values_degrade(self):
        cfg = IntelligenceConfig.from_env(
            {"HEADROOM_INTELLIGENCE": "bogus", "HEADROOM_JEVK5_TIMEOUT_MS": "abc"}
        )
        assert cfg.level == "off" and cfg.jevk5.timeout_ms == 1200


# ---------------------------------------------------------------- task context
def _anthropic_convo(user_text: str, tool_name: str, tool_input: dict, result: str) -> list:
    return [
        {"role": "user", "content": user_text},
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "t1", "name": tool_name, "input": tool_input}],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t1", "content": result}],
        },
    ]


class TestTaskContext:
    def test_entities(self):
        e = extract_entities(
            "Fix `parse_config` in src/app/config.py:42 and C:\\proj\\main.rs; "
            "see https://x.io/a and 3f2a9c1d7e. user 550e8400-e29b-41d4-a716-446655440000 took 120ms"
        )
        assert "parse_config" in e.quoted
        assert any(p.endswith("config.py:42") or p.endswith("config.py") for p in e.paths)
        assert any("main.rs" in p for p in e.paths)
        assert "https://x.io/a" in e.urls
        assert "3f2a9c1d7e" in e.hashes
        assert "550e8400-e29b-41d4-a716-446655440000" in e.identifiers
        assert "120ms" in e.numbers_with_units

    def test_build_skips_tool_result_turns(self):
        msgs = _anthropic_convo(
            "Why does `UserService.login` fail?", "Grep", {"pattern": "login"}, "a.py:1:def login"
        )
        ctx = build_task_context(msgs)
        assert ctx.current_user_text.startswith("Why does")
        assert ctx.tool_name == "Grep"
        assert "pattern=login" in ctx.tool_input_summary
        assert ctx.turn_kind == "tool_continuation"
        assert "UserService.login" in ctx.explicit_entities
        q = ctx.relevance_query()
        assert "UserService.login" in q and "Grep" in q

    def test_query_never_drops_exact_before_prose(self):
        long_prose = "please investigate " * 400
        msgs = [{"role": "user", "content": f"{long_prose} `ORDER-99812` deadbeef42"}]
        ctx = build_task_context(msgs)
        q = ctx.relevance_query(max_tokens=40)
        assert "ORDER-99812" in q and "deadbeef42" in q

    def test_openai_shape(self):
        msgs = [
            {"role": "user", "content": "list failing tests"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {
                            "name": "shell",
                            "arguments": json.dumps({"command": "pytest -q"}),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "c1",
                "content": "FAILED tests/test_a.py::test_x - AssertionError\n1 failed, 3 passed",
            },
        ]
        ctx = build_task_context(msgs)
        assert ctx.tool_name == "shell"
        assert ctx.error_signals
        assert ctx.turn_kind == "unknown" or ctx.turn_kind in ("tool_continuation", "new_user_ask")

    def test_provenance(self):
        assert detect_provenance("x", "Read", {"offset": 10}).is_partial
        p = detect_provenance("line\n[... 400 lines truncated ...]\nend")
        assert p.is_partial and "truncated" in p.truncation_boundary
        assert not detect_provenance("all good").is_partial
        assert detect_provenance('{"items": [], "has_more": true}').is_partial


# ----------------------------------------------------------------- invariants
class TestInvariants:
    def test_user_entity_veto_unless_recoverable(self):
        ctx = build_task_context([{"role": "user", "content": "find order `ORD-7781`"}])
        original = "\n".join(f"row {i} ORD-{1000 + i} ok" for i in range(6000, 6900))
        inv = extract_invariants(original, ctx)
        assert "ORD-7781" in inv.user_entities
        cand = "row 0 ORD-8000 ok\n[899 rows omitted]"
        rep = validate_candidate(
            original, cand, inv, original_tokens=tok(original), candidate_tokens=tok(cand)
        )
        assert not rep.hard_invariants_preserved and "user_entity_dropped" in rep.violations
        cand2 = cand + "\n<<ccr:abcdef012345>>"
        rep2 = validate_candidate(
            original,
            cand2,
            inv,
            original_tokens=tok(original),
            candidate_tokens=tok(cand2),
            store_has=lambda h: h == "abcdef012345",
        )
        assert rep2.hard_invariants_preserved and rep2.recoverable

    def test_unresolvable_marker_vetoed(self):
        original = "x " * 2000
        inv = extract_invariants(original)
        cand = "x x [compressed] <<ccr:abcdef012345>>"
        rep = validate_candidate(
            original,
            cand,
            inv,
            original_tokens=tok(original),
            candidate_tokens=tok(cand),
            store_has=lambda h: False,
        )
        assert "retrieval_marker_unresolvable" in rep.violations

    def test_exit_code_and_numeric_change(self):
        original = "build step took 120ms\n" * 50 + "Process exited with code 2\n"
        inv = extract_invariants(original)
        bad = "build step took 999ms\nProcess exited with code 2\n"
        rep = validate_candidate(
            original, bad, inv, original_tokens=tok(original), candidate_tokens=tok(bad)
        )
        assert "numeric_value_changed" in rep.violations
        bad2 = "build step took 120ms\nProcess exited with code 0\n"
        rep2 = validate_candidate(
            original, bad2, inv, original_tokens=tok(original), candidate_tokens=tok(bad2)
        )
        assert "exit_code_changed" in rep2.violations
        good = "build step took 120ms (repeated 50 times)\nProcess exited with code 2\n"
        rep3 = validate_candidate(
            original, good, inv, original_tokens=tok(original), candidate_tokens=tok(good)
        )
        assert rep3.hard_invariants_preserved, rep3.violations

    def test_errors_and_tests(self):
        original = (
            "PASSED a\n" * 300
            + "FAILED tests/test_x.py::test_y - AssertionError: 1 != 2\n5 failed, 300 passed\n"
        )
        inv = extract_invariants(original)
        cand = "PASSED a (repeated 300 times)\n"
        rep = validate_candidate(
            original, cand, inv, original_tokens=tok(original), candidate_tokens=tok(cand)
        )
        assert "error_signal_dropped" in rep.violations and "test_summary_dropped" in rep.violations

    def test_partial_input_must_stay_partial(self):
        original = "def a():\n    pass\n" * 200 + "[... 900 lines truncated ...]\n"
        inv = extract_invariants(original, provenance=detect_provenance(original))
        assert inv.provenance.is_partial
        cand = "Complete file outline: def a()"
        rep = validate_candidate(
            original, cand, inv, original_tokens=tok(original), candidate_tokens=tok(cand)
        )
        assert "partial_represented_as_complete" in rep.violations
        cand_ok = "def a(): ...\n[... 900 lines truncated ...]\n"
        rep_ok = validate_candidate(
            original, cand_ok, inv, original_tokens=tok(original), candidate_tokens=tok(cand_ok)
        )
        assert rep_ok.truncation_provenance_preserved

    def test_json_structure(self):
        original = json.dumps([{"id": i, "v": "x" * 20} for i in range(200)])
        inv = extract_invariants(original)
        assert inv.kind == "json"
        broken = original[: len(original) // 3]
        rep = validate_candidate(
            original, broken, inv, original_tokens=tok(original), candidate_tokens=tok(broken)
        )
        assert "json_structure_invalid" in rep.violations

    def test_non_positive_savings(self):
        original = "abc " * 100
        inv = extract_invariants(original)
        rep = validate_candidate(
            original, original + " x", inv, original_tokens=10, candidate_tokens=11
        )
        assert "non_positive_savings" in rep.violations


# --------------------------------------------------------------------- policy
class TestPolicy:
    def test_payg_cap_rejects_irreversible_over_45(self):
        assert not admit_candidate(PAYG, 1000, 500, recoverable=False, lossless=False).allowed
        assert admit_candidate(PAYG, 1000, 560, recoverable=False, lossless=False).allowed

    def test_subscription_cap(self):
        d = admit_candidate(SUB, 1000, 700, recoverable=False, lossless=False)
        assert not d.allowed and d.reason == "max_lossy_ratio"
        assert admit_candidate(SUB, 1000, 760, recoverable=False, lossless=False).allowed

    def test_ccr_backed_can_exceed_wire_cap(self):
        d = admit_candidate(SUB, 100_000, 300, recoverable=True, lossless=False)
        assert (
            d.allowed and d.ratios.wire_drop_ratio > 0.99 and d.ratios.irreversible_drop_ratio == 0
        )

    def test_invalid_marker_rejected(self):
        d = admit_candidate(PAYG, 1000, 100, recoverable=True, lossless=False, marker_valid=False)
        assert not d.allowed and d.reason == "invalid_marker"

    def test_zero_and_one_caps(self):
        class P:
            def __init__(self, r):
                self.max_lossy_ratio = r
                self.volatile_token_threshold = 32

        assert not admit_candidate(P(0.0), 100, 99, recoverable=False, lossless=False).allowed
        assert admit_candidate(P(0.0), 100, 10, recoverable=False, lossless=True).allowed
        assert admit_candidate(P(1.0), 100, 1, recoverable=False, lossless=False).allowed

    def test_non_positive(self):
        assert (
            admit_candidate(PAYG, 100, 100, recoverable=True, lossless=True).reason
            == "non_positive_savings"
        )

    def test_drop_ratios_and_volatility(self):
        r = drop_ratios(200, 50, recoverable=False, lossless=False)
        assert r.wire_drop_ratio == 0.75 and r.irreversible_drop_ratio == 0.75
        assert classify_change(SUB, 32) == "stable" and classify_change(SUB, 33) == "volatile"
        assert classify_change(PAYG, 128) == "stable" and classify_change(PAYG, 129) == "volatile"


# -------------------------------------------------------------------- arbiter
def _log_original() -> str:
    lines = [f"INFO step {i} ok elapsed 5ms" for i in range(400)]
    lines.insert(250, "ERROR db.connect failed: timeout after 30s (host=db-7.internal)")
    return "\n".join(lines)


class TestArbiter:
    def test_unsafe_aggressive_rejected_conservative_chosen(self):
        original = _log_original()
        ctx = build_task_context([{"role": "user", "content": "why did db.connect fail?"}])
        aggressive = "INFO step 0 ok elapsed 5ms\n[399 lines omitted]"
        conservative = (
            "INFO step 0 ok elapsed 5ms\n[398 similar INFO lines omitted; retrieve: <<ccr:0123456789abcdef01234567>>]\n"
            "ERROR db.connect failed: timeout after 30s (host=db-7.internal)"
        )
        decision = arbitrate(
            original,
            [
                Candidate("a", "log_aggressive", aggressive, tier=TIER_AGGRESSIVE),
                Candidate("c", "log_conservative", conservative, tier=TIER_CONSERVATIVE),
            ],
            count_tokens=tok,
            task=ctx,
            policy=PAYG,
            store_has=lambda h: True,
        )
        assert decision.selected.candidate_id == "c"
        agg = next(c for c in decision.candidates if c.candidate_id == "a")
        assert agg.rejected_reason is not None
        assert agg.report is not None and "error_signal_dropped" in agg.report.violations

    def test_pareto_returns_original_when_nothing_safe(self):
        original = _log_original()
        decision = arbitrate(
            original,
            [Candidate("a", "bad", "nothing useful", tier=TIER_AGGRESSIVE)],
            count_tokens=tok,
            policy=PAYG,
        )
        assert decision.kept_original

    def test_externalization_beats_lossy_under_tight_policy(self):
        original = "\n".join(
            f'{{"id": {i}, "name": "item{i}", "price": "{i}.00 USD"}}' for i in range(500)
        )
        lossy = "\n".join(original.splitlines()[:50])  # 90% irreversible drop
        indexed = (
            "\n".join(original.splitlines()[:20])
            + "\n[480 more rows; retrieve: <<ccr:abcdef0123456789abcdef01>>]"
        )
        decision = arbitrate(
            original,
            [
                Candidate("lossy", "smart_crusher", lossy, tier=TIER_AGGRESSIVE),
                Candidate("idx", "indexed", indexed, tier=TIER_INDEXED),
            ],
            count_tokens=tok,
            policy=SUB,
            store_has=lambda h: True,
        )
        assert decision.selected.candidate_id == "idx"
        assert decision.rejections.get("policy:max_lossy_ratio") == 1

    def test_lossless_candidate_is_safe(self):
        original = "a/b/c.py:1:x\n" * 300
        decision = arbitrate(
            original,
            [
                Candidate(
                    "l",
                    "lossless_search",
                    "a/b/c.py\n" + "1:x\n" * 300,
                    tier=TIER_STRUCTURAL,
                    lossless=True,
                )
            ],
            count_tokens=tok,
            policy=SUB,
        )
        assert decision.selected.candidate_id == "l"

    def test_cache_penalty_keeps_original(self):
        original = "word " * 1000
        decision = arbitrate(
            original,
            [Candidate("x", "lossless", "word " * 900, lossless=True, tier=TIER_STRUCTURAL)],
            count_tokens=tok,
            cache_penalty_tokens=10_000,
        )
        assert decision.kept_original and "cache_net_cost" in decision.rejections

    def test_advisory_fusion_bounded_and_deterministic_without_advice(self):
        original = "alpha beta gamma delta " * 400
        cands = lambda: [  # noqa: E731
            Candidate("x", "s1", "alpha beta " * 300, lossless=True, tier=TIER_STRUCTURAL),
            Candidate("y", "s2", "gamma delta " * 290, lossless=True, tier=TIER_STRUCTURAL),
        ]
        s = ArbiterSession(original, count_tokens=tok, content_type="text")
        s.prepare(cands())
        base = s.select(None).selected.candidate_id
        s2 = ArbiterSession(original, count_tokens=tok, content_type="text")
        s2.prepare(cands())
        assert (
            s2.select(
                AdvisoryScores(DecisionFamily.TRANSFORM_CANDIDATE, {}, 0.0, weight=0.0)
            ).selected.candidate_id
            == base
        )
        req = s2.advisory_request()
        assert req is not None and set(req.options) == {"x", "y"}
        other = "y" if base == "x" else "x"
        adv = AdvisoryScores(
            DecisionFamily.TRANSFORM_CANDIDATE, {other: 1.0, base: 0.0}, 1.0, weight=5.0
        )
        decision = s2.select(adv)
        assert decision.advised
        # weight is clamped to MAX_ADVISORY_WEIGHT whatever the caller passes
        for c in decision.candidates:
            if c.advisory_p is not None:
                expected = (
                    1 - MAX_ADVISORY_WEIGHT
                ) * c.deterministic_utility + MAX_ADVISORY_WEIGHT * c.advisory_p
                assert abs(c.final_utility - expected) < 1e-9

    def test_advice_never_overrides_hard_veto(self):
        original = _log_original()
        ctx = build_task_context([{"role": "user", "content": "why did db.connect fail?"}])
        s = ArbiterSession(original, count_tokens=tok, task=ctx, policy=PAYG)
        s.prepare([Candidate("bad", "x", "INFO\n[omitted]", tier=TIER_AGGRESSIVE)])
        decision = s.select(
            AdvisoryScores(DecisionFamily.TRANSFORM_CANDIDATE, {"bad": 1.0}, 1.0, weight=0.4)
        )
        assert decision.kept_original

    def test_order_independent(self):
        original = "alpha beta gamma delta " * 400
        a = Candidate("x", "s1", "alpha beta " * 300, lossless=True)
        b = Candidate("y", "s2", "gamma delta " * 300, lossless=True)
        d1 = arbitrate(original, [a, b], count_tokens=tok)
        a2 = Candidate("x", "s1", "alpha beta " * 300, lossless=True)
        b2 = Candidate("y", "s2", "gamma delta " * 300, lossless=True)
        d2 = arbitrate(original, [b2, a2], count_tokens=tok)
        assert d1.selected.candidate_id == d2.selected.candidate_id
