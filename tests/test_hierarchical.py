import numpy as np
import pytest
from scipy import stats

from bayes_ab_kit.hierarchical import (
    HierarchicalBetaResult,
    ShrunkArm,
    _beta_binomial_loglik,
    _log_marginal_and_grad,
    hierarchical_beta_shrinkage,
)
from bayes_ab_kit.multiarms import probability_of_being_best
from bayes_ab_kit.posteriors import BetaBinomialPosterior
from bayes_ab_kit.rope import rope_decision
from bayes_ab_kit.uplift import variant_vs_control


def _baseline_catalog():
    arms = {f"base-{i}": (500, 5000) for i in range(12)}
    arms["noisy"] = (8, 20)
    arms["precise"] = (2000, 5000)
    return arms


def test_loglik_matches_scipy_beta_binomial():
    conversions = np.array([8, 500, 2000], dtype=float)
    trials = np.array([20, 5000, 5000], dtype=float)
    alpha, beta = 12.0, 80.0
    got = _beta_binomial_loglik(alpha, beta, conversions, trials)
    expected = stats.betabinom.logpmf(conversions, trials, alpha, beta).sum()
    assert got == pytest.approx(expected)


def test_marginal_gradient_matches_finite_differences():
    conversions = np.array([10.0, 50.0, 8.0, 90.0])
    trials = np.array([100.0, 400.0, 20.0, 1000.0])
    mu, kappa = 0.14, 35.0
    _, d_mu, d_kappa = _log_marginal_and_grad(mu, kappa, conversions, trials)
    eps = 1e-5
    ll_hi, _, _ = _log_marginal_and_grad(mu + eps, kappa, conversions, trials)
    ll_lo, _, _ = _log_marginal_and_grad(mu - eps, kappa, conversions, trials)
    assert d_mu == pytest.approx((ll_hi - ll_lo) / (2 * eps), rel=1e-4, abs=1e-4)
    ll_hi, _, _ = _log_marginal_and_grad(mu, kappa + eps, conversions, trials)
    ll_lo, _, _ = _log_marginal_and_grad(mu, kappa - eps, conversions, trials)
    assert d_kappa == pytest.approx((ll_hi - ll_lo) / (2 * eps), rel=1e-4, abs=1e-4)


def test_posterior_mean_is_shrinkage_toward_prior_mean():
    result = hierarchical_beta_shrinkage(_baseline_catalog())
    for arm in result.arms:
        blended = (1.0 - arm.prior_weight) * arm.raw_rate + arm.prior_weight * result.prior_mean
        assert arm.posterior_mean == pytest.approx(blended, abs=1e-12)
        assert 0.0 < arm.prior_weight < 1.0


def test_noisy_arm_shrinks_more_than_precise_arm():
    result = hierarchical_beta_shrinkage(_baseline_catalog())
    by_name = {arm.name: arm for arm in result.arms}
    noisy = by_name["noisy"]
    precise = by_name["precise"]
    assert noisy.raw_rate == pytest.approx(0.40)
    assert precise.raw_rate == pytest.approx(0.40)
    assert noisy.prior_weight > precise.prior_weight
    assert abs(noisy.posterior_mean - result.prior_mean) < abs(noisy.raw_rate - result.prior_mean)
    assert abs(precise.posterior_mean - precise.raw_rate) < abs(
        noisy.posterior_mean - noisy.raw_rate
    )
    assert result.prior_mean < noisy.posterior_mean < noisy.raw_rate
    assert result.prior_mean < precise.posterior_mean < precise.raw_rate


def test_low_noisy_arm_is_pulled_up_toward_grand_mean():
    arms = {f"base-{i}": (1000, 5000) for i in range(10)}
    arms["noisy"] = (0, 25)
    arms["precise"] = (100, 5000)
    result = hierarchical_beta_shrinkage(arms, ci=None)
    noisy = next(arm for arm in result.arms if arm.name == "noisy")
    precise = next(arm for arm in result.arms if arm.name == "precise")
    assert noisy.raw_rate == 0.0
    assert precise.raw_rate < result.prior_mean
    assert noisy.raw_rate < noisy.posterior_mean < result.prior_mean
    assert precise.raw_rate < precise.posterior_mean < result.prior_mean
    assert noisy.prior_weight > precise.prior_weight
    assert abs(noisy.posterior_mean - noisy.raw_rate) > abs(
        precise.posterior_mean - precise.raw_rate
    )


def test_homogeneous_arms_pool_near_the_common_rate():
    arms = {f"arm-{i}": (100, 1000) for i in range(15)}
    result = hierarchical_beta_shrinkage(arms)
    assert result.prior_mean == pytest.approx(0.10, abs=1e-3)
    assert result.prior_strength > 1_000
    for arm in result.arms:
        assert arm.posterior_mean == pytest.approx(0.10, abs=1e-3)


def test_fitted_prior_beats_nearby_alternatives():
    arms = _baseline_catalog()
    result = hierarchical_beta_shrinkage(arms, ci=None)
    conversions = np.array([conv for conv, _ in arms.values()], dtype=float)
    trials = np.array([n for _, n in arms.values()], dtype=float)
    best = _beta_binomial_loglik(result.prior_alpha, result.prior_beta, conversions, trials)
    for mu in np.linspace(0.05, 0.45, 9):
        for kappa in (1.0, 5.0, 20.0, 100.0, 1_000.0, 1e5):
            candidate = _beta_binomial_loglik(mu * kappa, (1.0 - mu) * kappa, conversions, trials)
            assert candidate <= best + 1e-5


def test_credible_interval_matches_beta_posterior_and_covers_mean():
    result = hierarchical_beta_shrinkage(_baseline_catalog(), ci=0.9)
    assert result.ci == pytest.approx(0.9)
    posts = result.posteriors()
    widths = {}
    for arm in result.arms:
        lower, upper = posts[arm.name].credible_interval(0.9)
        assert arm.ci_lower == pytest.approx(lower)
        assert arm.ci_upper == pytest.approx(upper)
        assert arm.posterior_mean == pytest.approx(posts[arm.name].mean())
        assert 0.0 <= lower < arm.posterior_mean < upper <= 1.0
        widths[arm.name] = upper - lower
    assert widths["noisy"] > widths["precise"]
    assert widths["noisy"] > widths["base-0"]


def test_ci_none_omits_intervals():
    result = hierarchical_beta_shrinkage(
        {"A": (20, 200), "B": (30, 200), "C": (25, 200)},
        ci=None,
    )
    assert result.ci is None
    for arm in result.arms:
        assert arm.ci_lower is None
        assert arm.ci_upper is None
        assert isinstance(arm, ShrunkArm)


def test_input_order_and_determinism_are_preserved():
    arms = {"C": (30, 400), "A": (10, 50), "B": (80, 400)}
    first = hierarchical_beta_shrinkage(arms)
    second = hierarchical_beta_shrinkage(arms)
    assert first == second
    assert isinstance(first, HierarchicalBetaResult)
    assert [arm.name for arm in first.arms] == ["C", "A", "B"]


def test_posterior_objects_ignore_their_original_prior():
    counts = {"A": (8, 40), "B": (90, 800), "C": (110, 800)}
    posts = {
        "A": BetaBinomialPosterior.from_counts(8, 40, alpha0=1, beta0=1),
        "B": BetaBinomialPosterior.from_counts(90, 800, alpha0=40, beta0=1),
        "C": BetaBinomialPosterior.from_counts(110, 800, alpha0=2, beta0=50),
    }
    from_counts = hierarchical_beta_shrinkage(counts, ci=None)
    from_posts = hierarchical_beta_shrinkage(posts, ci=None)
    assert from_posts.prior_alpha == pytest.approx(from_counts.prior_alpha)
    assert from_posts.prior_beta == pytest.approx(from_counts.prior_beta)
    assert posts["B"].alpha0 == pytest.approx(40.0)
    assert [arm.posterior_mean for arm in from_posts.arms] == pytest.approx(
        [arm.posterior_mean for arm in from_counts.arms]
    )


def test_shrunk_posteriors_plug_into_rope_pbest_and_uplift():
    result = hierarchical_beta_shrinkage(
        {"A": (40, 800), "B": (70, 800), "C": (55, 800)},
        ci=0.95,
    )
    posts = result.posteriors()
    rope = rope_decision(posts["A"], posts["B"], n_samples=8_000, seed=1)
    uplift = variant_vs_control(posts["A"], posts["B"], n_samples=8_000, seed=1)
    best = probability_of_being_best(posts, n_samples=8_000, seed=1)
    assert rope.decision.value in {"ship_a", "ship_b", "practical_equivalence", "keep_running"}
    assert uplift.expected_uplift == pytest.approx(
        posts["B"].mean() - posts["A"].mean()
    )
    assert set(best.probabilities) == {"A", "B", "C"}
    assert sum(best.probabilities.values()) == pytest.approx(1.0, abs=1e-12)


def test_names_are_stripped_before_duplicate_check():
    result = hierarchical_beta_shrinkage({" A ": (10, 100), "B": (20, 100)})
    assert result.arms[0].name == "A"
    with pytest.raises(ValueError, match="duplicate arm name"):
        hierarchical_beta_shrinkage({"A": (10, 100), " A ": (12, 100)})


@pytest.mark.parametrize(
    "arms",
    [
        {},
        {"only": (10, 100)},
        [("A", (10, 100)), ("B", (12, 100))],
    ],
)
def test_too_few_or_non_mapping_arms_rejected(arms):
    with pytest.raises(ValueError):
        hierarchical_beta_shrinkage(arms)


def test_invalid_counts_and_intervals_rejected():
    with pytest.raises(ValueError, match="conversion count"):
        hierarchical_beta_shrinkage({"A": (11, 10), "B": (3, 10)})
    with pytest.raises(ValueError, match="positive"):
        hierarchical_beta_shrinkage({"A": (0, 0), "B": (3, 10)})
    with pytest.raises(ValueError, match="integer"):
        hierarchical_beta_shrinkage({"A": (1.5, 10), "B": (3, 10)})
    with pytest.raises(ValueError, match="non-empty"):
        hierarchical_beta_shrinkage({"": (1, 10), "B": (3, 10)})
    with pytest.raises(ValueError, match="credible level"):
        hierarchical_beta_shrinkage({"A": (1, 10), "B": (3, 10)}, ci=1.0)
    with pytest.raises(ValueError, match="BetaBinomialPosterior"):
        hierarchical_beta_shrinkage({"A": (1, 10, 0), "B": (3, 10)})


def test_extreme_all_success_and_all_failure_stay_finite():
    zeros = hierarchical_beta_shrinkage({f"z{i}": (0, 40) for i in range(6)})
    ones = hierarchical_beta_shrinkage({f"o{i}": (40, 40) for i in range(6)})
    assert all(0.0 <= arm.posterior_mean < 0.05 for arm in zeros.arms)
    assert all(0.95 < arm.posterior_mean <= 1.0 for arm in ones.arms)
    assert np.isfinite(zeros.prior_strength) and zeros.prior_strength > 0
    assert np.isfinite(ones.prior_strength) and ones.prior_strength > 0
    for result in (zeros, ones):
        posts = result.posteriors()
        for arm in result.arms:
            lower, upper = posts[arm.name].credible_interval(result.ci)
            assert arm.ci_lower == lower
            assert arm.ci_upper == upper
            assert 0.0 <= lower <= upper <= 1.0
            assert np.isfinite(arm.posterior_mean)
