"""Tests for the mixture SPRT (always-valid inference)."""

import math

import numpy as np
import pytest

from bayes_ab_kit import (
    msprt_log_likelihood_ratio,
    msprt_monitor,
    msprt_proportions,
    msprt_radius,
    simulate_msprt_false_positives,
)
from bayes_ab_kit.cli import main
from bayes_ab_kit.msprt import _normal_quantile

LOOKS = [(48, 500, 62, 500), (98, 1000, 131, 1000), (150, 1500, 205, 1500), (199, 2000, 281, 2000)]


def test_closed_form_matches_numerical_mixture():
    delta, v, tau, theta0 = 0.031, 0.012**2, 0.02, 0.005
    theta = np.linspace(theta0 - 12 * tau, theta0 + 12 * tau, 200_001)
    prior = np.exp(-0.5 * ((theta - theta0) / tau) ** 2) / (tau * math.sqrt(2 * math.pi))
    ratio = np.exp((-(delta - theta) ** 2 + (delta - theta0) ** 2) / (2 * v))
    numeric = np.sum(prior * ratio) * (theta[1] - theta[0])
    assert msprt_log_likelihood_ratio(delta, v, tau, theta0) == pytest.approx(math.log(numeric), rel=1e-6)


def test_radius_is_boundary_of_rejection_region():
    v, tau, alpha, delta = 0.01**2, 0.03, 0.05, 0.02
    r = msprt_radius(v, tau, alpha)
    for theta0 in (delta - r, delta + r):
        assert msprt_log_likelihood_ratio(delta, v, tau, theta0) == pytest.approx(math.log(1 / alpha))
    assert msprt_log_likelihood_ratio(delta, v, tau, delta + 0.9 * r) < math.log(1 / alpha)


def test_monitor_p_values_monotone_and_consistent_with_sequence():
    res = msprt_proportions(LOOKS, alpha=0.05, tau=0.02)
    pvals = [look.p_value for look in res.looks]
    assert all(b <= a for a, b in zip(pvals, pvals[1:]))
    lows = [look.ci_low for look in res.looks]
    highs = [look.ci_high for look in res.looks]
    assert all(b >= a for a, b in zip(lows, lows[1:]))
    assert all(b <= a for a, b in zip(highs, highs[1:]))
    assert res.stopped_at == 3 and res.rejected
    assert res.p_value <= 0.05
    # rejection of theta0 = 0 happens exactly when 0 leaves the sequence
    for look in res.looks:
        assert (look.p_value <= 0.05) == (look.ci_low > 0 or look.ci_high < 0)
    assert res.confidence_interval[0] > 0


def test_proportions_estimate_and_variance():
    res = msprt_proportions([(10, 100, 20, 100)], tau=0.05)
    look = res.looks[0]
    assert look.delta == pytest.approx(0.10)
    sa, sb = 10.5 / 101, 20.5 / 101
    assert look.std_error == pytest.approx(math.sqrt(sa * (1 - sa) / 100 + sb * (1 - sb) / 100))
    # zero conversions still give a finite, positive variance
    assert msprt_proportions([(0, 50, 0, 50)]).looks[0].std_error > 0


def test_false_positive_control_under_peeking():
    sim = simulate_msprt_false_positives(rate=0.1, per_look=400, looks=25, simulations=3000, seed=7)
    assert sim.naive_false_positive_rate > 0.15
    assert sim.msprt_false_positive_rate <= 0.05 + 3 * math.sqrt(0.05 * 0.95 / 3000)


def test_confidence_sequence_covers_true_effect_uniformly():
    rng = np.random.default_rng(11)
    misses = 0
    sims, looks, per = 600, 30, 50
    for _ in range(sims):
        x = rng.normal(0.3, 1.0, size=looks * per)
        n = per * np.arange(1, looks + 1)
        means = np.cumsum(x)[n - 1] / n
        res = msprt_monitor(means, 1.0 / n, tau=0.5, alpha=0.1)
        if any(not (look.ci_low <= 0.3 <= look.ci_high) for look in res.looks):
            misses += 1
    assert misses / sims <= 0.1 + 3 * math.sqrt(0.09 / sims)


def test_power_under_real_lift():
    rng = np.random.default_rng(3)
    stops = 0
    for _ in range(200):
        a = np.cumsum(rng.binomial(1000, 0.10, size=20))
        b = np.cumsum(rng.binomial(1000, 0.13, size=20))
        n = 1000 * np.arange(1, 21)
        stops += msprt_proportions(list(zip(a, n, b, n))).rejected
    assert stops / 200 > 0.9


def test_validation():
    with pytest.raises(ValueError, match="alpha"):
        msprt_monitor([0.1], [0.01], tau=0.1, alpha=1.5)
    with pytest.raises(ValueError, match="tau"):
        msprt_monitor([0.1], [0.01], tau=0.0)
    with pytest.raises(ValueError, match="variance"):
        msprt_monitor([0.1], [0.0], tau=0.1)
    with pytest.raises(ValueError, match="same length"):
        msprt_monitor([0.1, 0.2], [0.01], tau=0.1)
    with pytest.raises(ValueError, match="non-decreasing"):
        msprt_proportions([(10, 100, 10, 100), (9, 200, 20, 200)])
    with pytest.raises(ValueError, match="exceed"):
        msprt_proportions([(101, 100, 10, 100)])
    with pytest.raises(ValueError, match="at least one look"):
        msprt_proportions([])


def test_normal_quantile():
    assert _normal_quantile(0.975) == pytest.approx(1.959964, abs=1e-5)


def test_cli_msprt(capsys):
    argv = ["msprt"]
    for row in LOOKS:
        argv += ["--look", ":".join(str(x) for x in row)]
    assert main(argv) == 0
    out = capsys.readouterr().out
    assert "always-valid p" in out and "reject p_B - p_A = 0 at look 4" in out
    assert main(["msprt", "--simulate", "--sims", "300", "--looks", "10"]) == 0
    assert "mSPRT always-valid FPR" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main(["msprt", "--look", "1:2:3"])
