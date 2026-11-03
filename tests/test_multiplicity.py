import numpy as np
import pytest
from scipy.stats import norm

from bayes_ab_kit import (
    BayesianFdrResult,
    MetricScreenResult,
    bayesian_fdr_select,
    lfsr_from_draws,
    lfsr_normal,
    screen_metrics,
)


def test_lfsr_from_draws_matches_normal():
    rng = np.random.default_rng(0)
    means = np.array([0.0, 1.0, -2.0, 0.3])
    draws = means + rng.normal(size=(200_000, 4))
    np.testing.assert_allclose(lfsr_from_draws(draws), lfsr_normal(means, np.ones(4)), atol=4e-3)
    assert lfsr_from_draws(np.array([1.0, 2.0, 3.0]))[0] == 0.0
    # zeros count against both signs
    assert lfsr_from_draws(np.zeros(10))[0] == 1.0


def test_lfsr_normal_formula():
    np.testing.assert_allclose(lfsr_normal([1.96], [1.0]), [norm.cdf(-1.96)])
    np.testing.assert_allclose(lfsr_normal([-3.0, 3.0], [1.5, 1.5]), [norm.cdf(-2)] * 2)


def test_bayesian_fdr_selection_rule():
    e = [0.01, 0.2, 0.02, 0.5, 0.06, 0.15]
    r = bayesian_fdr_select(e, q=0.05, names=list("abcdef"))
    assert isinstance(r, BayesianFdrResult)
    # sorted: .01 .02 .06 .15 .2 .5 -> running means .01 .015 .03 .06 .088 .1567
    assert r.selected_names == ("a", "c", "e")
    assert r.n_selected == 3
    assert r.expected_false_discoveries == pytest.approx(0.09)
    assert r.expected_fdr == pytest.approx(0.03)
    assert r.threshold == pytest.approx(0.06)
    np.testing.assert_allclose(r.q_values, [0.01, 0.088, 0.015, 0.1566666667, 0.03, 0.06])
    assert "3/6 selected" in r.summary()


def test_selection_q_value_consistency():
    rng = np.random.default_rng(1)
    e = rng.uniform(0, 0.5, size=50)
    for q in (0.02, 0.05, 0.1, 0.2):
        r = bayesian_fdr_select(e, q=q)
        np.testing.assert_array_equal(r.selected, r.q_values <= q + 1e-15)
        assert r.expected_fdr <= q + 1e-12


def test_nothing_selected():
    r = bayesian_fdr_select([0.4, 0.3], q=0.05)
    assert r.n_selected == 0 and r.expected_fdr == 0.0 and r.selected_names == ()


def test_screen_metrics_shrinks_and_selects():
    y = [0.05, 0.002, -0.004, 0.001, -0.06, 0.003, 0.0, -0.002]
    s = [0.01] * 8
    r = screen_metrics(y, s, names=[f"k{i}" for i in range(8)], q=0.05)
    assert isinstance(r, MetricScreenResult)
    assert r.tau > 0
    assert set(r.selected_names) == {"k0", "k4"}
    assert np.all(np.abs(r.posterior_means) <= np.abs(r.estimates) + 1e-15)
    rows = r.table()
    assert rows[0]["lfsr"] <= rows[-1]["lfsr"]
    assert {"metric", "lfsr", "q_value", "selected"} <= set(rows[0])
    assert "tau" in r.summary()


def test_pure_noise_shrinks_to_zero_and_selects_almost_nothing():
    rng = np.random.default_rng(3)
    taus, n_sel = [], []
    for _ in range(50):
        y = rng.normal(0, 0.01, size=40)
        r = screen_metrics(y, np.full(40, 0.01), q=0.05)
        taus.append(r.tau)
        n_sel.append(r.selection.n_selected)
    assert np.median(taus) < 0.003
    assert np.mean(n_sel) < 0.2


def test_no_shrinkage_option_gives_one_sided_p_values():
    y, s = np.array([0.02, -0.01]), np.array([0.01, 0.01])
    r = screen_metrics(y, s, tau=float("inf"))
    np.testing.assert_allclose(r.lfsr, norm.cdf(-np.abs(y) / s))
    np.testing.assert_allclose(r.posterior_means, y)
    r0 = screen_metrics(y, s, tau=0.0)
    np.testing.assert_allclose(r0.lfsr, 0.5)
    assert r0.selection.n_selected == 0


def test_false_sign_rate_calibrated_when_prior_is_right():
    rng = np.random.default_rng(0)
    fsp = []
    for _ in range(200):
        m = 60
        theta = rng.normal(0, 0.02, m)
        s = rng.uniform(0.01, 0.03, m)
        y = theta + s * rng.normal(size=m)
        r = screen_metrics(y, s, q=0.1)
        sel = r.selection.selected
        fsp.append(np.mean(np.sign(theta[sel]) != np.sign(r.posterior_means[sel])) if sel.any() else 0.0)
    assert np.mean(fsp) <= 0.12


def test_validation():
    with pytest.raises(ValueError):
        bayesian_fdr_select([0.1, 1.2])
    with pytest.raises(ValueError):
        bayesian_fdr_select([0.1], q=0)
    with pytest.raises(ValueError):
        bayesian_fdr_select([0.1, 0.2], names=["a"])
    with pytest.raises(ValueError):
        bayesian_fdr_select([0.1, 0.2], names=["a", "a"])
    with pytest.raises(ValueError):
        screen_metrics([0.1, 0.2], [0.1])
    with pytest.raises(ValueError):
        screen_metrics([0.1], [0.0])
    with pytest.raises(ValueError):
        screen_metrics([0.1], [0.1], tau=-1)
    with pytest.raises(ValueError):
        lfsr_normal([0.1], [-1])
    with pytest.raises(ValueError):
        lfsr_from_draws(np.empty((0, 2)))
