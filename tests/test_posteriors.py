import numpy as np
import pytest

from bayes_ab_kit.posteriors import BetaBinomialPosterior


def test_uniformative_prior_mean_is_half():
    post = BetaBinomialPosterior()
    assert post.mean() == pytest.approx(0.5)


def test_conjugate_update_matches_counts():
    post = BetaBinomialPosterior.from_counts(conversions=8, trials=10)
    assert post.alpha == pytest.approx(9.0)
    assert post.beta == pytest.approx(3.0)
    assert post.trials == 10


def test_incremental_update_equals_batch_update():
    batch = BetaBinomialPosterior.from_counts(30, 100)
    incremental = BetaBinomialPosterior()
    for _ in range(10):
        incremental.update(successes=3, failures=7)
    assert incremental.alpha == pytest.approx(batch.alpha)
    assert incremental.beta == pytest.approx(batch.beta)


def test_posterior_shrinks_toward_prior_for_small_samples():
    weak = BetaBinomialPosterior.from_counts(1, 2, alpha0=50, beta0=50)
    strong_prior = BetaBinomialPosterior(alpha0=50, beta0=50).mean()
    assert abs(weak.mean() - strong_prior) < 0.05


def test_mode_formula_when_defined():
    post = BetaBinomialPosterior.from_counts(8, 10)
    assert post.mode() == pytest.approx((9.0 - 1) / (9.0 + 3.0 - 2))


def test_mode_none_near_uniform():
    post = BetaBinomialPosterior(alpha0=1, beta0=1)
    assert post.mode() is None


def test_variance_decreases_with_more_data():
    small = BetaBinomialPosterior.from_counts(8, 10)
    large = BetaBinomialPosterior.from_counts(80, 100)
    assert large.variance() < small.variance()


def test_variance_matches_closed_form():
    post = BetaBinomialPosterior.from_counts(8, 10)
    ab = post.alpha * post.beta
    denom = (post.alpha + post.beta) ** 2 * (post.alpha + post.beta + 1)
    assert post.variance() == pytest.approx(ab / denom)


def test_credible_interval_contains_mean():
    post = BetaBinomialPosterior.from_counts(20, 100)
    lower, upper = post.credible_interval(0.95)
    assert lower < post.mean() < upper


def test_credible_interval_bounds_within_unit_interval():
    post = BetaBinomialPosterior.from_counts(3, 5)
    lower, upper = post.credible_interval()
    assert 0.0 <= lower < upper <= 1.0


def test_wider_coverage_gives_wider_interval():
    post = BetaBinomialPosterior.from_counts(20, 100)
    lo80, hi80 = post.credible_interval(0.80)
    lo99, hi99 = post.credible_interval(0.99)
    assert hi99 - lo99 > hi80 - lo80


def test_quantile_monotonic_and_consistent_with_cdf():
    post = BetaBinomialPosterior.from_counts(12, 40)
    q25, q75 = post.quantile(0.25), post.quantile(0.75)
    assert q25 < q75
    assert post.cdf(q25) == pytest.approx(0.25, abs=1e-6)


def test_prob_above_boundaries_and_interior():
    post = BetaBinomialPosterior.from_counts(8, 10)
    assert post.prob_above(0.0) == 1.0
    assert post.prob_above(1.0) == 0.0
    interior = post.prob_above(post.quantile(0.4))
    assert interior == pytest.approx(0.6, abs=1e-6)


def test_sample_is_reproducible_with_seeded_rng():
    post = BetaBinomialPosterior.from_counts(15, 60)
    rng_a = np.random.default_rng(42)
    rng_b = np.random.default_rng(42)
    a = post.sample(rng_a, size=(200,))
    b = post.sample(rng_b, size=(200,))
    assert a.shape == b.shape == (200,)
    assert np.array_equal(a, b)


def test_sample_values_inside_support():
    post = BetaBinomialPosterior.from_counts(15, 60)
    draws = post.sample(np.random.default_rng(7), size=1000)
    assert np.all(draws >= 0.0)
    assert np.all(draws <= 1.0)
    assert draws.mean() == pytest.approx(post.mean(), abs=0.02)


def test_invalid_prior_rejected():
    with pytest.raises(ValueError):
        BetaBinomialPosterior(alpha0=0)
    with pytest.raises(ValueError):
        BetaBinomialPosterior(beta0=-1)


def test_invalid_counts_rejected():
    with pytest.raises(ValueError):
        BetaBinomialPosterior.from_counts(conversions=11, trials=10)
    with pytest.raises(ValueError):
        BetaBinomialPosterior().update(successes=-3, failures=0)

def test_zero_conversions_keeps_prior_influenced_posterior():
    post = BetaBinomialPosterior.from_counts(0, 20)
    assert post.alpha == pytest.approx(1.0)
    assert post.beta == pytest.approx(21.0)
    lo, hi = post.credible_interval()
    assert 0.0 <= lo < hi <= 1.0
    assert post.mode() is None


def test_all_conversions_extreme_posterior_stays_finite():
    post = BetaBinomialPosterior.from_counts(50, 50)
    assert np.isfinite(post.mean())
    assert post.prob_above(0.9) > 0.99


def test_empty_history_matches_prior():
    prior = BetaBinomialPosterior(alpha0=2, beta0=5)
    assert prior.mean() == pytest.approx(2.0 / 7.0)