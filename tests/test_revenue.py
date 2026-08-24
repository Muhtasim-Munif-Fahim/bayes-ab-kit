import numpy as np
import pytest

from bayes_ab_kit.revenue import LognormalRevenueModel, NormalRevenueModel
from bayes_ab_kit.sampling import make_rng


def test_conjugate_update_math_exact():
    model = NormalRevenueModel.from_samples(
        [5.0] * 9, known_sigma=2.0, mu0=0.0, tau0=1000.0
    )
    assert model.n == 9
    assert model.posterior_var == pytest.approx(1.0 / (1.0 / 1000.0**2 + 9.0 / 4.0))
    assert model.mean() == pytest.approx(5.0, rel=1e-6)


def test_weak_prior_defers_to_sample_mean():
    model = NormalRevenueModel.from_samples(
        np.array([30.0, 34.0, 26.0]), known_sigma=4.0, mu0=0.0, tau0=500.0
    )
    assert model.mean() == pytest.approx(np.mean([30.0, 34.0, 26.0]), abs=0.05)


def test_strong_prior_dominates_single_observation():
    model = NormalRevenueModel.from_samples(
        [100.0], known_sigma=10.0, mu0=10.0, tau0=0.1
    )
    assert model.mean() < 12.0


def test_credible_interval_is_symmetric():
    model = NormalRevenueModel.from_samples(
        np.arange(1.0, 21.0), known_sigma=3.0, mu0=0.0, tau0=50.0
    )
    lower, upper = model.credible_interval(0.95)
    assert lower < model.mean() < upper
    assert model.mean() - lower == pytest.approx(upper - model.mean())


def test_normal_sample_reproducible_and_centered():
    model = NormalRevenueModel.from_samples(
        [40.0] * 25, known_sigma=6.0, mu0=0.0, tau0=100.0
    )
    a = model.sample(make_rng(7), size=8000)
    b = model.sample(make_rng(7), size=8000)
    assert np.array_equal(a, b)
    assert a.mean() == pytest.approx(model.mean(), abs=0.15)


def test_invalid_noise_parameters_rejected():
    with pytest.raises(ValueError):
        NormalRevenueModel(known_sigma=0)
    with pytest.raises(ValueError):
        NormalRevenueModel(tau0=-1)


def test_empty_samples_rejected():
    with pytest.raises(ValueError):
        NormalRevenueModel.from_samples([], known_sigma=1.0)


def test_lognormal_recovers_true_expected_value():
    rng = np.random.default_rng(2026)
    true_mu, true_sigma = np.log(20.0), 0.8
    values = rng.lognormal(mean=true_mu, sigma=true_sigma, size=2500)
    model = LognormalRevenueModel.from_revenues(values, tau0=10.0)
    truth = np.exp(true_mu + true_sigma**2 / 2.0)
    assert model.expected_value() == pytest.approx(truth, rel=0.03)


def test_lognormal_draws_match_analytic_expectation():
    values = np.random.default_rng(11).lognormal(mean=2.0, sigma=0.5, size=1500)
    model = LognormalRevenueModel.from_revenues(values)
    draws = model.expected_value_draws(make_rng(13), size=4000)
    assert np.all(draws > 0)
    assert np.array_equal(draws, model.expected_value_draws(make_rng(13), size=4000))
    assert draws.mean() == pytest.approx(model.expected_value(), rel=0.01)


def test_lognormal_rejects_nonpositive_values():
    with pytest.raises(ValueError):
        LognormalRevenueModel.from_revenues([10.0, 0.0, 5.0])
    with pytest.raises(ValueError):
        LognormalRevenueModel.from_revenues([])


def test_lognormal_mu_interval_contains_posterior_mean_of_logs():
    values = np.random.default_rng(3).lognormal(mean=3.0, sigma=0.7, size=900)
    model = LognormalRevenueModel.from_revenues(values)
    lower, upper = model.credible_interval_for_mu()
    assert lower < model.mu_posterior.mean() < upper
def test_single_value_falls_back_to_default_dispersion():
    model = LognormalRevenueModel.from_revenues([25.0])
    assert model.n == 1
    assert model.sigma_log == pytest.approx(1.0)


def test_zero_spread_orders_require_explicit_sigma():
    with pytest.raises(ValueError, match="no dispersion"):
        LognormalRevenueModel.from_revenues([7.0, 7.0, 7.0])
    ok = LognormalRevenueModel.from_revenues([7.0, 7.0, 7.0], sigma_log=0.5)
    assert ok.sigma_log == pytest.approx(0.5)