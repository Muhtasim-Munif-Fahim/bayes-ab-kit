import numpy as np
import pytest

from bayes_ab_kit.decisions import (
    ComparisonResult,
    Decision,
    difference_credible_interval,
    prob_b_beats_a_mc,
    prob_b_beats_a_quad,
    superiority_decision,
)
from bayes_ab_kit.posteriors import BetaBinomialPosterior


def b_variant(conversions, trials):
    return BetaBinomialPosterior.from_counts(conversions, trials)


def test_symmetric_posteriors_give_half_probability():
    a = b_variant(10, 100)
    assert prob_b_beats_a_quad(a, a) == pytest.approx(0.5)


def test_quad_matches_closed_case_extreme_counts():
    a = b_variant(20, 100)
    b = b_variant(60, 100)
    quad = prob_b_beats_a_quad(a, b)
    mc = prob_b_beats_a_mc(a, b, n_samples=40_000)
    assert abs(quad - mc) < 0.01


def test_prob_increases_with_true_gap():
    base = b_variant(50, 1000)
    slight = prob_b_beats_a_quad(base, b_variant(55, 1000))
    large = prob_b_beats_a_quad(base, b_variant(80, 1000))
    assert 0.5 < slight < large < 1.0


def test_mc_is_deterministic_for_fixed_seed():
    a = b_variant(30, 200)
    b = b_variant(36, 200)
    first = prob_b_beats_a_mc(a, b, n_samples=5000)
    second = prob_b_beats_a_mc(a, b, n_samples=5000)
    assert first == second


def test_difference_interval_contains_zero_when_close():
    result = difference_credible_interval(b_variant(100, 1000), b_variant(104, 1000))
    lower, upper = result
    assert lower < 0 < upper


def test_difference_interval_positive_for_dominant_b():
    lower, upper = difference_credible_interval(b_variant(50, 1000), b_variant(120, 1000))
    assert 0 <= lower < upper


def test_higher_level_widens_interval():
    a = b_variant(70, 800)
    b = b_variant(90, 800)
    lo80, hi80 = difference_credible_interval(a, b, ci=0.80, n_samples=30_000)
    lo99, hi99 = difference_credible_interval(a, b, ci=0.99, n_samples=30_000)
    assert hi99 - lo99 > hi80 - lo80


def test_ship_b_when_b_clearly_wins():
    result = superiority_decision(b_variant(40, 1000), b_variant(95, 1000), ci=0.95)
    assert result.decision is Decision.SHIP_B
    assert result.diff_ci_lower > 0
    assert result.prob_b_beats_a > 0.95


def test_ship_a_when_a_clearly_wins():
    result = superiority_decision(b_variant(95, 1000), b_variant(40, 1000), ci=0.95)
    assert result.decision is Decision.SHIP_A
    assert result.diff_ci_upper < 0


def test_keep_running_on_ambiguous_data():
    result = superiority_decision(b_variant(52, 1000), b_variant(56, 1000), ci=0.95)
    assert result.decision is Decision.KEEP_RUNNING
    assert result.diff_ci_lower < 0 < result.diff_ci_upper


def test_result_carries_requested_level():
    result = superiority_decision(
        b_variant(40, 1000), b_variant(95, 1000), ci=0.90, n_samples=20_000
    )
    assert isinstance(result, ComparisonResult)
    assert result.ci == pytest.approx(0.90)


def test_invalid_credible_level_rejected():
    with pytest.raises(ValueError):
        superiority_decision(b_variant(1, 10), b_variant(2, 10), ci=0.0)
    with pytest.raises(ValueError):
        difference_credible_interval(b_variant(1, 10), b_variant(2, 10), ci=1.5)


def test_nonpositive_sample_count_rejected():
    with pytest.raises(ValueError):
        prob_b_beats_a_mc(b_variant(1, 10), b_variant(2, 10), n_samples=0)