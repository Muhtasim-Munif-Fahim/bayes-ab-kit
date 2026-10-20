"""Tests for delta-method ratio metrics."""

from __future__ import annotations

import csv
import math

import numpy as np
import pytest
from scipy.stats import norm

from bayes_ab_kit import (
    compare_ratio_metrics,
    ratio_summary,
    ratio_summary_from_moments,
)
from bayes_ab_kit.cli import main


def _users(rng, n, mean_ctr=0.1, concentration=10.0):
    """Clustered clicks/views: heterogeneous per-user CTR and view counts."""
    views = rng.geometric(0.2, n)
    ctr = rng.beta(mean_ctr * concentration, (1 - mean_ctr) * concentration, n)
    return rng.binomial(views, ctr).astype(float), views.astype(float)


def test_delta_variance_matches_formula_and_moments():
    rng = np.random.default_rng(0)
    x, y = _users(rng, 500)
    s = ratio_summary(x, y)
    n = x.size
    mx, my = x.mean(), y.mean()
    cov = np.cov(x, y, ddof=1)
    expected = (cov[0, 0] / my**2 - 2 * mx * cov[0, 1] / my**3 + mx**2 * cov[1, 1] / my**4) / n
    assert s.ratio == pytest.approx(x.sum() / y.sum())
    assert s.variance == pytest.approx(expected, rel=1e-10)
    m = ratio_summary_from_moments(n, x.sum(), y.sum(), (x * x).sum(), (y * y).sum(), (x * y).sum())
    assert m.variance == pytest.approx(s.variance, rel=1e-8)
    assert m.ratio == pytest.approx(s.ratio)
    assert s.naive_se == pytest.approx(math.sqrt(s.ratio * (1 - s.ratio) / y.sum()))
    assert s.design_effect == pytest.approx((s.se / s.naive_se) ** 2)
    assert s.to_dict()["n"] == 500


def test_delta_se_agrees_with_bootstrap():
    rng = np.random.default_rng(1)
    x, y = _users(rng, 4000)
    s = ratio_summary(x, y)
    idx = rng.integers(0, x.size, size=(1500, x.size))
    boot = x[idx].sum(axis=1) / y[idx].sum(axis=1)
    assert s.se == pytest.approx(boot.std(ddof=1), rel=0.1)
    assert s.design_effect > 1.5  # clustering inflates the variance


def test_iid_bernoulli_units_give_design_effect_near_one():
    rng = np.random.default_rng(2)
    y = np.ones(20000)
    x = (rng.random(20000) < 0.3).astype(float)
    s = ratio_summary(x, y)
    assert s.design_effect == pytest.approx(1.0, rel=0.01)


def test_aa_false_positive_rate_delta_vs_naive():
    rng = np.random.default_rng(3)
    trials, delta_hits, naive_hits = 400, 0, 0
    for _ in range(trials):
        a = ratio_summary(*_users(rng, 600))
        b = ratio_summary(*_users(rng, 600))
        res = compare_ratio_metrics(a, b)
        delta_hits += res.p_value < 0.05
        naive_hits += res.naive_p_value < 0.05
    assert 0.02 <= delta_hits / trials <= 0.085
    assert naive_hits / trials > 0.12  # pooled-view SE is far too small


def test_ci_coverage_of_true_difference():
    rng = np.random.default_rng(4)
    covered = 0
    trials = 300
    for _ in range(trials):
        a = ratio_summary(*_users(rng, 800, mean_ctr=0.10))
        b = ratio_summary(*_users(rng, 800, mean_ctr=0.12))
        lo, hi = compare_ratio_metrics(a, b).difference_interval
        covered += lo <= 0.02 <= hi
    assert 0.9 <= covered / trials <= 0.98


def test_comparison_identities_and_orientation():
    rng = np.random.default_rng(5)
    a = ratio_summary(*_users(rng, 3000, mean_ctr=0.10))
    b = ratio_summary(*_users(rng, 3000, mean_ctr=0.11))
    res = compare_ratio_metrics(a, b, ci=0.9)
    assert res.prob_b_better == pytest.approx(norm.cdf(res.z))
    assert res.p_value == pytest.approx(2 * norm.sf(abs(res.z)))
    lo, hi = res.difference_interval
    assert hi - lo == pytest.approx(2 * norm.ppf(0.95) * res.difference_se)
    # E[max(A-B,0)] - E[max(B-A,0)] = E[A-B] = -diff
    assert res.expected_loss_choose_b - res.expected_loss_choose_a == pytest.approx(-res.difference)
    assert res.relative_lift == pytest.approx(b.ratio / a.ratio - 1)
    llo, lhi = res.relative_lift_interval
    assert llo < res.relative_lift < lhi
    flipped = compare_ratio_metrics(a, b, ci=0.9, higher_is_better=False)
    assert flipped.prob_b_better == pytest.approx(1 - res.prob_b_better)
    assert flipped.expected_loss_choose_a == pytest.approx(res.expected_loss_choose_b)
    assert res.to_dict()["a"]["n"] == 3000


def test_validation():
    with pytest.raises(ValueError, match="same length"):
        ratio_summary([1, 2], [1, 2, 3])
    with pytest.raises(ValueError, match="at least 2"):
        ratio_summary([1], [1])
    with pytest.raises(ValueError, match="positive"):
        ratio_summary([0, 0], [0, 0])
    with pytest.raises(ValueError, match="non-negative"):
        ratio_summary([1, 1], [-1, 3])
    with pytest.raises(ValueError, match="inconsistent"):
        ratio_summary_from_moments(10, 10.0, 10.0, 1.0, 10.0, 10.0)
    with pytest.raises(ValueError, match="ci"):
        s = ratio_summary([1, 2, 3], [2, 3, 4])
        compare_ratio_metrics(s, s, ci=1.0)
    with pytest.raises(TypeError):
        compare_ratio_metrics((1, 2), (3, 4))
    revenue = ratio_summary([10.0, 25.0, 4.0], [1, 3, 1])  # revenue per session: ratio > 1
    assert revenue.naive_se is None and revenue.design_effect is None


def test_cli_ratio(tmp_path, capsys):
    rng = np.random.default_rng(6)
    path = tmp_path / "users.csv"
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["arm", "numerator", "denominator"])
        for arm, ctr in (("A", 0.10), ("B", 0.13)):
            x, y = _users(rng, 1500, mean_ctr=ctr)
            writer.writerows((arm, int(c), int(v)) for c, v in zip(x, y))
    assert main(["ratio", "--data", str(path)]) == 0
    out = capsys.readouterr().out
    assert "design effect" in out and "p-value (naive pooled)" in out and "P(B better)" in out
    with pytest.raises(SystemExit):
        main(["ratio", "--data", str(path), "--control", "Z"])
    bad = tmp_path / "bad.csv"
    bad.write_text("arm,clicks\nA,1\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        main(["ratio", "--data", str(bad)])
