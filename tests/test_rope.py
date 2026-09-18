import pytest

from bayes_ab_kit.decisions import Decision
from bayes_ab_kit.posteriors import BetaBinomialPosterior
from bayes_ab_kit.rope import (
    ExpectedLossStopResult,
    RopeResult,
    expected_loss_stop,
    rope_decision,
)


def b_variant(conversions, trials):
    return BetaBinomialPosterior.from_counts(conversions, trials)


def test_ship_b_when_interval_is_above_rope():
    result = rope_decision(b_variant(40, 1000), b_variant(95, 1000), rope=0.01)
    assert result.decision is Decision.SHIP_B
    assert result.diff_ci_lower > result.rope_upper
    assert result.prob_above_rope > 0.95


def test_ship_a_when_interval_is_below_rope():
    result = rope_decision(b_variant(95, 1000), b_variant(40, 1000), rope=0.01)
    assert result.decision is Decision.SHIP_A
    assert result.diff_ci_upper < result.rope_lower
    assert result.prob_below_rope > 0.95


def test_practical_equivalence_when_interval_sits_inside_rope():
    result = rope_decision(
        b_variant(10_000, 100_000),
        b_variant(10_020, 100_000),
        rope=0.01,
        n_samples=40_000,
    )
    assert result.decision is Decision.PRACTICAL_EQUIVALENCE
    assert result.rope_lower <= result.diff_ci_lower
    assert result.diff_ci_upper <= result.rope_upper
    assert result.prob_in_rope > 0.95


def test_keep_running_when_interval_straddles_rope_boundary():
    result = rope_decision(b_variant(52, 1000), b_variant(56, 1000), rope=0.01)
    assert result.decision is Decision.KEEP_RUNNING
    assert result.diff_ci_lower < result.rope_upper
    assert result.diff_ci_upper > result.rope_lower


def test_wide_rope_turns_a_close_race_into_equivalence():
    tight = rope_decision(b_variant(52, 1000), b_variant(56, 1000), rope=0.01)
    wide = rope_decision(b_variant(52, 1000), b_variant(56, 1000), rope=0.05)
    assert tight.decision is Decision.KEEP_RUNNING
    assert wide.decision is Decision.PRACTICAL_EQUIVALENCE


def test_asymmetric_rope_interval_is_honoured():
    result = rope_decision(
        b_variant(40, 1000),
        b_variant(95, 1000),
        rope=(-0.002, 0.002),
        n_samples=20_000,
    )
    assert result.rope_lower == pytest.approx(-0.002)
    assert result.rope_upper == pytest.approx(0.002)
    assert result.decision is Decision.SHIP_B


def test_mass_inside_and_outside_rope_sums_to_one():
    result = rope_decision(b_variant(80, 900), b_variant(92, 900), n_samples=30_000)
    total = result.prob_in_rope + result.prob_below_rope + result.prob_above_rope
    assert total == pytest.approx(1.0, abs=1e-9)


def test_result_is_deterministic_under_seed():
    a, b = b_variant(70, 800), b_variant(82, 800)
    first = rope_decision(a, b, rope=0.01, n_samples=15_000, seed=11)
    second = rope_decision(a, b, rope=0.01, n_samples=15_000, seed=11)
    assert first == second
    assert isinstance(first, RopeResult)


def test_invalid_rope_and_ci_rejected():
    a, b = b_variant(5, 50), b_variant(6, 50)
    with pytest.raises(ValueError, match="half-width"):
        rope_decision(a, b, rope=0.0)
    with pytest.raises(ValueError, match="two endpoints"):
        rope_decision(a, b, rope=(0.01,))
    with pytest.raises(ValueError, match="lower bound"):
        rope_decision(a, b, rope=(0.02, -0.01))
    with pytest.raises(ValueError, match="credible level"):
        rope_decision(a, b, ci=1.0)
    with pytest.raises(ValueError, match="n_samples"):
        rope_decision(a, b, n_samples=0)


def test_expected_loss_stops_for_a_clear_leader():
    result = expected_loss_stop(b_variant(35, 1000), b_variant(120, 1000))
    assert isinstance(result, ExpectedLossStopResult)
    assert result.should_stop is True
    assert result.leader == "b"
    assert result.decision is Decision.SHIP_B
    assert result.expected_loss < result.threshold


def test_expected_loss_keeps_running_when_threshold_is_tight():
    result = expected_loss_stop(
        b_variant(48, 1000), b_variant(52, 1000), threshold=0.001
    )
    assert result.should_stop is False
    assert result.decision is Decision.KEEP_RUNNING


def test_looser_loss_threshold_stops_sooner():
    tight = expected_loss_stop(b_variant(52, 1000), b_variant(64, 1000), threshold=0.0005)
    loose = expected_loss_stop(b_variant(52, 1000), b_variant(64, 1000), threshold=0.02)
    assert tight.should_stop is False
    assert loose.should_stop is True
    assert loose.decision is Decision.SHIP_B


def test_expected_loss_stop_recognizes_leading_a():
    result = expected_loss_stop(b_variant(130, 1000), b_variant(60, 1000), threshold=0.005)
    assert result.leader == "a"
    assert result.should_stop is True
    assert result.decision is Decision.SHIP_A


def test_expected_loss_stop_rejects_invalid_threshold():
    with pytest.raises(ValueError):
        expected_loss_stop(b_variant(5, 100), b_variant(6, 100), threshold=0.0)
