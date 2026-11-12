import numpy as np
import pytest
from scipy import integrate, special, stats

from bayes_ab_kit import (
    compare_with_historical_control,
    normalized_power_prior,
    power_prior_posterior,
)


def test_fixed_a0_endpoints_are_no_borrowing_and_pooling():
    none = power_prior_posterior(30, 600, 100, 2000, a0=0.0)
    assert (none.alpha, none.beta) == (31.0, 571.0)
    full = power_prior_posterior(30, 600, 100, 2000, a0=1.0)
    assert (full.alpha, full.beta) == (131.0, 2471.0)
    half = power_prior_posterior(30, 600, 100, 2000, a0=0.5, alpha0=2.0, beta0=3.0)
    assert half.alpha == pytest.approx(2.0 + 50.0 + 30.0)
    assert half.beta == pytest.approx(3.0 + 950.0 + 570.0)


def test_a0_posterior_matches_direct_quadrature():
    x, n, x0, n0 = 18, 300, 70, 1000
    res = normalized_power_prior(x, n, x0, n0, grid_size=2001)

    def dens(a):
        return np.exp(
            special.betaln(1 + a * x0 + x, 1 + a * (n0 - x0) + n - x)
            - special.betaln(1 + a * x0, 1 + a * (n0 - x0))
        )

    z = integrate.quad(dens, 0, 1)[0]
    mean = integrate.quad(lambda a: a * dens(a), 0, 1)[0] / z
    assert res.a0_mean == pytest.approx(mean, rel=1e-4)
    assert res.a0_weights.sum() == pytest.approx(1.0)
    assert res.borrowed_sample_size == pytest.approx(mean * n0, rel=1e-4)


def test_congruent_history_is_borrowed_conflicting_is_discounted():
    agree = normalized_power_prior(50, 1000, 250, 5000)
    conflict = normalized_power_prior(50, 1000, 600, 5000)
    assert agree.a0_mean > 0.5
    assert agree.a0_eb == pytest.approx(1.0)
    assert conflict.a0_mean < 0.05
    assert conflict.a0_eb < 0.02
    # borrowing narrows the interval relative to current data alone
    lo, hi = agree.credible_interval()
    alone = stats.beta(51, 951).interval(0.95)
    assert hi - lo < alone[1] - alone[0]
    # conflicting history barely moves the current-data mean (pooled would be ~0.108)
    assert conflict.mean() == pytest.approx(51 / 1002, rel=0.1)
    assert abs(conflict.mean() - 51 / 1002) < 0.1 * abs(651 / 6002 - 51 / 1002)


def test_mixture_mean_quantile_and_sampling_agree():
    res = normalized_power_prior(40, 800, 120, 3000)
    draws = res.sample(200_000, np.random.default_rng(0))
    assert draws.mean() == pytest.approx(res.mean(), rel=0.01)
    med = res.quantile(0.5)
    assert res.cdf(med) == pytest.approx(0.5, abs=1e-8)
    assert np.mean(draws <= med) == pytest.approx(0.5, abs=0.01)


def test_compare_with_historical_control():
    out = compare_with_historical_control(
        control_conversions=48, control_trials=1000,
        treatment_conversions=62, treatment_trials=1000,
        hist_conversions=240, hist_trials=5000, n_samples=100_000, seed=1,
    )
    assert 0.5 < out.prob_treatment_better < 1.0
    assert out.lift_interval[0] < out.expected_lift < out.lift_interval[1]
    assert out.control.a0_mean > 0.4
    # borrowing a congruent, tighter control sharpens the comparison
    assert out.prob_treatment_better > out.prob_treatment_better_no_borrowing - 0.02


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(conversions=5, trials=3, hist_conversions=1, hist_trials=10),
        dict(conversions=1, trials=3, hist_conversions=11, hist_trials=10),
        dict(conversions=1, trials=3, hist_conversions=1, hist_trials=10, a0_prior=(0, 1)),
        dict(conversions=1, trials=3, hist_conversions=1, hist_trials=10, grid_size=2),
        dict(conversions=1, trials=3, hist_conversions=1, hist_trials=10, alpha0=-1),
    ],
)
def test_normalized_power_prior_validation(kwargs):
    with pytest.raises(ValueError):
        normalized_power_prior(**kwargs)


def test_fixed_a0_validation():
    with pytest.raises(ValueError):
        power_prior_posterior(1, 10, 1, 10, a0=1.5)
