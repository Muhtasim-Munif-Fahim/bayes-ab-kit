import math

import numpy as np
import pytest
from scipy.special import betaln

from bayes_ab_kit.decisions import difference_credible_interval
from bayes_ab_kit.posteriors import BetaBinomialPosterior
from bayes_ab_kit.rope import rope_decision
from bayes_ab_kit.uplift import (
    ControlUpliftResult,
    UpliftResult,
    VariantVsControl,
    expected_absolute_uplift,
    expected_relative_uplift,
    variant_vs_control,
    variants_vs_control,
)


def b_variant(conversions, trials, alpha0=1.0, beta0=1.0):
    return BetaBinomialPosterior.from_counts(
        conversions, trials, alpha0=alpha0, beta0=beta0
    )


def cook_p_x_gt_y(alpha_x, beta_x, alpha_y, beta_y) -> float:
    """P(X > Y) via the Cook (2005) series when ``alpha_x`` is a positive integer.

    X ~ Beta(alpha_x, beta_x), Y ~ Beta(alpha_y, beta_y), independent.
    """
    n = int(alpha_x)
    if n != alpha_x or n < 1:
        raise ValueError("Cook series requires integer alpha_x >= 1")
    log_by = betaln(alpha_y, beta_y)
    total = 0.0
    for i in range(n):
        log_term = (
            betaln(alpha_y + i, beta_y + beta_x)
            - math.log(beta_x + i)
            - betaln(1 + i, beta_x)
            - log_by
        )
        total += math.exp(log_term)
    return total


def conjugate_relative(control: BetaBinomialPosterior, variant: BetaBinomialPosterior) -> float:
    mean_inv = (control.alpha + control.beta - 1.0) / (control.alpha - 1.0)
    return variant.mean() * mean_inv - 1.0


def test_identical_posteriors_half_probability_and_zero_absolute_uplift():
    arm = b_variant(40, 400)
    result = variant_vs_control(arm, arm, n_samples=8_000)
    assert result.prob_beats_control == pytest.approx(0.5)
    assert result.expected_uplift == pytest.approx(0.0)
    assert result.expected_uplift == pytest.approx(arm.mean() - arm.mean())
    assert result.uplift_ci_lower < 0 < result.uplift_ci_upper


def test_absolute_uplift_matches_difference_of_beta_means():
    control = b_variant(95, 1000)
    variant = b_variant(130, 1000)
    expected = variant.mean() - control.mean()
    assert expected_absolute_uplift(control, variant) == pytest.approx(expected)
    result = variant_vs_control(control, variant, n_samples=5_000)
    assert result.expected_uplift == pytest.approx(expected)
    assert expected == pytest.approx((131 / 1002) - (96 / 1002))


def test_relative_uplift_matches_conjugate_e_inv_control():
    control = b_variant(20, 200)
    variant = b_variant(30, 200)
    closed = conjugate_relative(control, variant)
    assert control.alpha == pytest.approx(21.0)
    assert expected_relative_uplift(control, variant) == pytest.approx(closed)
    result = variant_vs_control(control, variant, n_samples=5_000)
    assert result.expected_relative_uplift == pytest.approx(closed)
    assert closed != pytest.approx((variant.mean() - control.mean()) / control.mean())


def test_iid_relative_uplift_is_positive_by_jensen():
    arm = b_variant(50, 200)
    rel = expected_relative_uplift(arm, arm)
    plug_in = (arm.mean() - arm.mean()) / arm.mean()
    assert plug_in == pytest.approx(0.0)
    assert rel > 0.0
    assert rel == pytest.approx(conjugate_relative(arm, arm))


def test_uniform_control_prob_equals_variant_mean():
    control = BetaBinomialPosterior(alpha0=1.0, beta0=1.0)
    variant = b_variant(8, 10)
    result = variant_vs_control(control, variant, n_samples=4_000)
    assert result.prob_beats_control == pytest.approx(variant.mean(), abs=1e-8)
    assert math.isinf(result.expected_relative_uplift)
    assert result.expected_uplift == pytest.approx(variant.mean() - 0.5)


def test_zero_control_conversions_relative_expectation_diverges():
    control = b_variant(0, 80)
    variant = b_variant(12, 80)
    assert control.alpha == pytest.approx(1.0)
    assert math.isinf(expected_relative_uplift(control, variant))
    result = variant_vs_control(control, variant, n_samples=3_000)
    assert math.isinf(result.expected_relative_uplift)
    assert np.isfinite(result.relative_uplift_ci_lower)
    assert np.isfinite(result.relative_uplift_ci_upper)


def test_quad_prob_matches_cook_series_for_integer_shapes():
    control = b_variant(10, 50)
    variant = b_variant(18, 50)
    result = variant_vs_control(control, variant, n_samples=4_000)
    exact = cook_p_x_gt_y(variant.alpha, variant.beta, control.alpha, control.beta)
    assert result.prob_beats_control == pytest.approx(exact, rel=1e-8, abs=1e-8)


def test_cook_series_is_half_for_identical_integer_betas():
    arm = b_variant(7, 20)
    exact = cook_p_x_gt_y(arm.alpha, arm.beta, arm.alpha, arm.beta)
    assert exact == pytest.approx(0.5, abs=1e-10)


def test_prob_increases_with_true_gap():
    control = b_variant(50, 1000)
    slight = variant_vs_control(control, b_variant(55, 1000), n_samples=2_000)
    large = variant_vs_control(control, b_variant(90, 1000), n_samples=2_000)
    assert 0.5 < slight.prob_beats_control < large.prob_beats_control < 1.0
    assert slight.expected_uplift < large.expected_uplift


def test_clear_winner_has_positive_interval_and_high_probability():
    result = variant_vs_control(
        b_variant(40, 1000), b_variant(95, 1000), n_samples=20_000
    )
    assert result.prob_beats_control > 0.95
    assert result.expected_uplift > 0
    assert result.expected_relative_uplift > 0
    assert result.uplift_ci_lower > 0
    assert result.relative_uplift_ci_lower > 0


def test_absolute_ci_matches_difference_interval_and_rope_at_same_seed():
    control, variant = b_variant(70, 800), b_variant(88, 800)
    kwargs = dict(ci=0.90, n_samples=15_000, seed=11)
    result = variant_vs_control(control, variant, **kwargs)
    lower, upper = difference_credible_interval(control, variant, **kwargs)
    rope = rope_decision(control, variant, rope=0.01, **kwargs)
    assert result.uplift_ci_lower == pytest.approx(lower)
    assert result.uplift_ci_upper == pytest.approx(upper)
    assert result.uplift_ci_lower == pytest.approx(rope.diff_ci_lower)
    assert result.uplift_ci_upper == pytest.approx(rope.diff_ci_upper)


def test_concentrated_relative_ci_covers_conjugate_expectation():
    control = b_variant(2_000, 10_000)
    variant = b_variant(2_400, 10_000)
    result = variant_vs_control(control, variant, n_samples=25_000)
    expected = conjugate_relative(control, variant)
    assert result.relative_uplift_ci_lower < expected < result.relative_uplift_ci_upper
    assert result.uplift_ci_lower < result.expected_uplift < result.uplift_ci_upper


def test_higher_level_widens_both_intervals():
    control, variant = b_variant(70, 800), b_variant(90, 800)
    tight = variant_vs_control(control, variant, ci=0.80, n_samples=20_000)
    wide = variant_vs_control(control, variant, ci=0.99, n_samples=20_000)
    assert wide.uplift_ci_upper - wide.uplift_ci_lower > tight.uplift_ci_upper - tight.uplift_ci_lower
    assert (
        wide.relative_uplift_ci_upper - wide.relative_uplift_ci_lower
        > tight.relative_uplift_ci_upper - tight.relative_uplift_ci_lower
    )


def test_result_is_deterministic_under_seed():
    control, variant = b_variant(60, 700), b_variant(75, 700)
    first = variant_vs_control(control, variant, n_samples=12_000, seed=19)
    second = variant_vs_control(control, variant, n_samples=12_000, seed=19)
    assert first == second
    assert isinstance(first, UpliftResult)
    assert first.ci == pytest.approx(0.95)


def test_variants_vs_control_matches_pairwise_and_preserves_order():
    arms = {
        "gamma": b_variant(10, 100),
        "alpha": b_variant(12, 100),
        "beta": b_variant(11, 100),
    }
    multi = variants_vs_control(arms, control="gamma", n_samples=8_000, seed=5)
    assert isinstance(multi, ControlUpliftResult)
    assert [arm.name for arm in multi.variants] == ["alpha", "beta"]
    assert multi.control_name == "gamma"
    assert multi.control_conversions == 10
    assert multi.control_trials == 100
    pairwise_alpha = variant_vs_control(arms["gamma"], arms["alpha"], n_samples=8_000, seed=5)
    pairwise_beta = variant_vs_control(arms["gamma"], arms["beta"], n_samples=8_000, seed=5)
    assert multi.variants[0].uplift == pairwise_alpha
    assert multi.variants[1].uplift == pairwise_beta
    assert all(isinstance(arm, VariantVsControl) for arm in multi.variants)


def test_variants_vs_control_does_not_rank_like_p_best():
    result = variants_vs_control(
        {
            "control": b_variant(40, 1000),
            "weak": b_variant(45, 1000),
            "winner": b_variant(150, 1000),
        },
        control="control",
        n_samples=10_000,
    )
    probs = [arm.uplift.prob_beats_control for arm in result.variants]
    assert sum(probs) != pytest.approx(1.0)
    by_name = {arm.name: arm.uplift for arm in result.variants}
    assert by_name["winner"].prob_beats_control > 0.95
    assert by_name["winner"].expected_uplift > by_name["weak"].expected_uplift


def test_carries_requested_level_and_sample_count():
    result = variants_vs_control(
        {"A": b_variant(20, 200), "B": b_variant(30, 200)},
        control="A",
        ci=0.90,
        n_samples=3_000,
    )
    assert result.ci == pytest.approx(0.90)
    assert result.n_samples == 3_000
    assert result.variants[0].uplift.ci == pytest.approx(0.90)


def test_rejects_invalid_inputs():
    control, variant = b_variant(5, 50), b_variant(6, 50)
    with pytest.raises(ValueError, match="credible level"):
        variant_vs_control(control, variant, ci=1.0)
    with pytest.raises(ValueError, match="n_samples"):
        variant_vs_control(control, variant, n_samples=0)
    with pytest.raises(ValueError, match="BetaBinomialPosterior"):
        variant_vs_control(control, "nope")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="at least two"):
        variants_vs_control({"A": control}, control="A")
    with pytest.raises(ValueError, match="control arm not found"):
        variants_vs_control({"A": control, "B": variant}, control="Z")
    with pytest.raises(ValueError, match="arm name"):
        variants_vs_control({"A": control, "  ": variant}, control="A")
    with pytest.raises(ValueError, match="control name"):
        variants_vs_control({"A": control, "B": variant}, control="  ")
    with pytest.raises(ValueError, match="BetaBinomialPosterior"):
        variants_vs_control({"A": control, "B": "nope"}, control="A")


def test_rope_import_available_for_joint_use():
    # Keep the uplift summary orthogonal to ROPE decisions: same posteriors
    # can be passed to both APIs.
    control, variant = b_variant(40, 1000), b_variant(95, 1000)
    uplift = variant_vs_control(control, variant, n_samples=8_000)
    rope = rope_decision(control, variant, rope=0.01, n_samples=8_000)
    assert uplift.expected_uplift > 0
    assert rope.diff_ci_lower > rope.rope_upper
