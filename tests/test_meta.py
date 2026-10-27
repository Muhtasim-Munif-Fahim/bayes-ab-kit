import numpy as np
import pytest
from scipy.stats import norm

from bayes_ab_kit import meta_analysis
from bayes_ab_kit.cli import main

SCHOOLS_Y = [28, 8, -3, 7, -1, 1, 18, 12]
SCHOOLS_S = [15, 10, 16, 11, 9, 11, 10, 18]


def test_eight_schools_matches_bda():
    # Gelman et al. BDA3 §5.5 (uniform prior on tau): E[mu|y] ~ 8, sd ~ 5, theta_A ~ 11
    r = meta_analysis(SCHOOLS_Y, SCHOOLS_S, tau_prior="uniform", tau_max=40)
    assert r.mu_mean == pytest.approx(7.9, abs=0.3)
    assert r.mu_sd == pytest.approx(5.1, abs=0.4)
    assert 10.5 < r.theta_means[0] < 12.5
    assert 0 < r.tau_interval[0] < r.tau_median < r.tau_interval[1] < 25
    # partial pooling: every estimate is pulled towards mu
    assert np.all(np.abs(r.theta_means - r.mu_mean) < np.abs(np.array(SCHOOLS_Y) - r.mu_mean))
    # noisier experiments are pooled more
    assert r.pooling[7] > r.pooling[6]
    assert r.predictive_sd > r.mu_sd
    assert r.predictive_interval[0] < r.mu_interval[0] and r.predictive_interval[1] > r.mu_interval[1]


def test_monte_carlo_agrees_with_grid():
    r = meta_analysis(SCHOOLS_Y, SCHOOLS_S, tau_prior="half-cauchy", tau_scale=5)
    rng = np.random.default_rng(0)
    n = 200_000
    tau = rng.choice(r.tau_grid, size=n, p=r.tau_weights)
    y = np.array(SCHOOLS_Y, float)
    s = np.array(SCHOOLS_S, float)
    var = s[None, :] ** 2 + tau[:, None] ** 2
    v_mu = 1 / (1 / var).sum(1)
    mu = rng.normal((y / var).sum(1) * v_mu, np.sqrt(v_mu))
    v0 = 1 / (1 / s[0] ** 2 + 1 / np.maximum(tau, 1e-12) ** 2)
    th0 = rng.normal(v0 * (y[0] / s[0] ** 2 + mu / np.maximum(tau, 1e-12) ** 2), np.sqrt(v0))
    assert mu.mean() == pytest.approx(r.mu_mean, abs=0.06)
    assert mu.std() == pytest.approx(r.mu_sd, abs=0.06)
    assert (mu > 0).mean() == pytest.approx(r.prob_mu_above, abs=0.005)
    assert th0.mean() == pytest.approx(r.theta_means[0], abs=0.08)
    assert np.quantile(th0, 0.975) == pytest.approx(r.theta_intervals[0][1], abs=0.3)


def test_homogeneous_data_pools_to_weighted_mean():
    y = [0.010, 0.012, 0.011, 0.009, 0.0105]
    s = [0.004, 0.005, 0.003, 0.006, 0.004]
    r = meta_analysis(y, s, tau_prior="half-normal", tau_scale=0.0005)
    w = 1 / np.square(s)
    fixed = float(np.dot(w, y) / w.sum())
    assert r.mu_mean == pytest.approx(fixed, rel=0.02)
    assert r.mu_sd == pytest.approx(w.sum() ** -0.5, rel=0.05)
    assert np.all(r.pooling > 0.9)
    np.testing.assert_allclose(r.theta_means, r.mu_mean, atol=5e-4)
    assert r.prob_mu_above == pytest.approx(norm.cdf(fixed * np.sqrt(w.sum())), abs=0.01)


def test_heterogeneous_precise_data_barely_pooled():
    r = meta_analysis([-5.0, 0.0, 5.0, 10.0], [0.1, 0.1, 0.1, 0.1], tau_prior="half-cauchy", tau_scale=5)
    np.testing.assert_allclose(r.theta_means, [-5, 0, 5, 10], atol=0.01)
    assert np.all(r.pooling < 0.01)
    assert 3 < r.tau_median < 15


def test_threshold_and_validation():
    r0 = meta_analysis(SCHOOLS_Y, SCHOOLS_S)
    r5 = meta_analysis(SCHOOLS_Y, SCHOOLS_S, threshold=5)
    assert r5.prob_mu_above < r0.prob_mu_above
    assert np.all(r5.prob_theta_above < r0.prob_theta_above)
    assert r0.names[0] == "exp1"
    with pytest.raises(ValueError):
        meta_analysis([1.0], [1.0])
    with pytest.raises(ValueError):
        meta_analysis([1, 2], [1, 0])
    with pytest.raises(ValueError):
        meta_analysis([1, 2], [1, 1], tau_prior="uniform")
    with pytest.raises(ValueError):
        meta_analysis([1, 2], [1, 1], tau_prior="gamma")
    with pytest.raises(ValueError):
        meta_analysis([1, 2], [1, 1], names=["a"])
    with pytest.raises(ValueError):
        meta_analysis([1, 2], [1, 1], ci=1.0)


def test_cli_meta(capsys):
    argv = ["meta"]
    for name, y, s in zip("ABCDEFGH", SCHOOLS_Y, SCHOOLS_S):
        argv += ["--experiment", f"{name}:{y}:{s}"]
    assert main(argv + ["--tau-prior", "uniform"]) == 0
    out = capsys.readouterr().out
    assert "8 experiments" in out and "overall effect mu" in out and "new experiment effect" in out
    with pytest.raises(SystemExit):
        main(["meta", "--experiment", "A:1", "--experiment", "B:2:1"])
    assert "NAME:ESTIMATE:SE" in capsys.readouterr().err
