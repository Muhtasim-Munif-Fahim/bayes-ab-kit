"""Tests for the Weibull time-to-event comparison."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import integrate, special

from bayes_ab_kit import (
    GammaPoissonPosterior,
    WeibullPosterior,
    compare_survival,
    simulate_weibull_survival,
    weibull_median,
    weibull_rmst,
)
from bayes_ab_kit.cli import main


def test_marginal_likelihood_matches_numerical_integration():
    t, e = simulate_weibull_survival(30, 1.3, 2.0, censor_time=2.5, seed=4)
    a0, b0 = 2.0, 1.5
    grid = np.array([0.7, 1.3, 2.1])
    post = WeibullPosterior(t, e, shape_grid=grid, a0=a0, b0=b0)
    for k, logm in zip(grid, post.log_marginal):
        def integrand(lam):
            loglik = np.sum(e * (np.log(lam * k) + (k - 1) * np.log(t))) - lam * np.sum(t**k)
            logprior = a0 * np.log(b0) - special.gammaln(a0) + (a0 - 1) * np.log(lam) - b0 * lam
            return np.exp(loglik + logprior - logm)
        val, _ = integrate.quad(integrand, 0, np.inf, limit=200)
        assert val == pytest.approx(1.0, rel=1e-6)


def test_exponential_special_case_matches_gamma_poisson():
    t, e = simulate_weibull_survival(200, 1.0, 5.0, censor_time=6.0, seed=1)
    post = WeibullPosterior(t, e, shape_grid=[1.0], a0=0.5, b0=1e-6)
    gp = GammaPoissonPosterior.from_counts(int(e.sum()), float(t.sum()), shape0=0.5, rate0=1e-6)
    assert post.post_shape == pytest.approx(gp.shape)
    assert post.post_rate[0] == pytest.approx(gp.rate)
    k, lam = post.sample(np.random.default_rng(0), 200_000)
    assert np.all(k == 1.0)
    assert lam.mean() == pytest.approx(gp.mean(), rel=0.01)


def test_rmst_and_median_closed_forms():
    for k, lam, tau in [(1.5, 0.03, 7.0), (0.6, 0.4, 3.0), (1.0, 0.2, 10.0)]:
        quad, _ = integrate.quad(lambda s: np.exp(-lam * s**k), 0, tau)
        assert weibull_rmst(k, lam, tau) == pytest.approx(quad, rel=1e-9)
        med = weibull_median(k, lam)
        assert np.exp(-lam * med**k) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        weibull_rmst(1.0, 1.0, 0.0)


def test_recovers_shape_with_heavy_censoring():
    t, e = simulate_weibull_survival(1500, 2.0, 10.0, censor_time=8.0, seed=11)
    assert 0.3 < e.mean() < 0.6
    post = WeibullPosterior(t, e)
    lo, hi = post.shape_interval(0.95)
    assert lo < 2.0 < hi
    mean, lower, upper = post.survival([5.0, 8.0])
    true = np.exp(-((np.array([5.0, 8.0]) / 10.0) ** 2))
    assert np.all(lower < true) and np.all(true < upper)


def test_ignoring_censoring_is_biased_but_model_is_not():
    t, e = simulate_weibull_survival(2000, 1.0, 10.0, censor_time=6.0, seed=3)
    true_median = 10.0 * np.log(2.0)
    proper = WeibullPosterior(t, e, shape_grid=[1.0])
    naive = WeibullPosterior(t, None, shape_grid=[1.0])  # treats censored as events
    k, lam = proper.sample(np.random.default_rng(1), 50_000)
    kn, lamn = naive.sample(np.random.default_rng(1), 50_000)
    assert np.median(weibull_median(k, lam)) == pytest.approx(true_median, rel=0.08)
    assert np.median(weibull_median(kn, lamn)) < 0.6 * true_median


def test_compare_detects_longer_survival_and_is_symmetric():
    ta, ea = simulate_weibull_survival(800, 1.5, 10, censor_time=15, seed=1)
    tb, eb = simulate_weibull_survival(800, 1.5, 12, censor_time=15, seed=2)
    res = compare_survival(ta, ea, tb, eb)
    assert res.prob_b_better > 0.99
    assert res.rmst_difference_interval[0] > 0
    assert res.median_ratio_interval[0] > 1.0
    assert res.expected_loss_choose_b < res.expected_loss_choose_a
    assert res.tau == pytest.approx(15.0)
    flipped = compare_survival(ta, ea, tb, eb, longer_is_better=False)
    assert flipped.prob_b_better == pytest.approx(1.0 - res.prob_b_better, abs=1e-9)
    assert flipped.expected_loss_choose_a == pytest.approx(res.expected_loss_choose_b)


def test_aa_comparison_is_uncertain():
    ta, ea = simulate_weibull_survival(400, 0.8, 5, censor_time=10, seed=21)
    tb, eb = simulate_weibull_survival(400, 0.8, 5, censor_time=10, seed=22)
    res = compare_survival(ta, ea, tb, eb, tau=8.0)
    assert 0.05 < res.prob_b_better < 0.95
    lo, hi = res.rmst_difference_interval
    assert lo < 0 < hi


@pytest.mark.parametrize(
    "times, events",
    [([], None), ([1.0, -2.0], None), ([1.0, 2.0], [1, 2]), ([1.0, 2.0], [0, 0]), ([1.0], [1, 0])],
)
def test_invalid_inputs(times, events):
    with pytest.raises(ValueError):
        WeibullPosterior(times, events)


def test_cli_survival(tmp_path, capsys):
    ta, ea = simulate_weibull_survival(300, 1.2, 10, censor_time=12, seed=5)
    tb, eb = simulate_weibull_survival(300, 1.2, 14, censor_time=12, seed=6)
    path = tmp_path / "d.csv"
    lines = ["arm,time,event"]
    lines += [f"A,{t:.5f},{e}" for t, e in zip(ta, ea)]
    lines += [f"B,{t:.5f},{e}" for t, e in zip(tb, eb)]
    path.write_text("\n".join(lines) + "\n")
    assert main(["survival", "--data", str(path), "--samples", "5000"]) == 0
    out = capsys.readouterr().out
    assert "Weibull time-to-event comparison" in out
    assert "RMST difference B - A" in out
    assert "P(B better)" in out


def test_cli_survival_missing_columns(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("arm,value\nA,1\nB,2\n")
    with pytest.raises(SystemExit):
        main(["survival", "--data", str(path)])
