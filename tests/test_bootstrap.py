"""Tests for the Bayesian bootstrap."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import trim_mean

from bayes_ab_kit import (
    BOOTSTRAP_STATISTICS,
    bayesian_bootstrap,
    bayesian_bootstrap_ratio,
    compare_bayesian_bootstrap,
    weighted_quantile,
)
from bayes_ab_kit.bootstrap import _batch_statistic
from bayes_ab_kit.cli import main


@pytest.fixture
def sample():
    return np.random.default_rng(0).lognormal(0.0, 1.0, 400)


def test_mean_posterior_matches_exact_moments(sample):
    post = bayesian_bootstrap(sample, "mean", n_draws=60000, seed=1)
    n = sample.size
    exact_var = ((sample - sample.mean()) ** 2).sum() / (n * (n + 1))
    assert post.mean == pytest.approx(sample.mean(), rel=2e-3)
    assert post.sd ** 2 == pytest.approx(exact_var, rel=0.03)
    assert post.point_estimate == pytest.approx(sample.mean())


def test_uniform_weights_reproduce_classical_statistics(sample):
    xs = np.sort(sample)
    u = np.full((1, sample.size), 1.0 / sample.size)
    for q in (0.1, 0.5, 0.9, 0.99):
        expected = np.quantile(sample, q, method="inverted_cdf")
        assert _batch_statistic(xs, u, "quantile", q, 0.0)[0] == pytest.approx(expected)
        assert weighted_quantile(sample, np.ones(sample.size), q) == pytest.approx(expected)
    assert _batch_statistic(xs, u, "trimmed_mean", 0.5, 0.1)[0] == pytest.approx(trim_mean(sample, 0.1))
    assert _batch_statistic(xs, u, "std", 0.5, 0.1)[0] == pytest.approx(sample.std())


def test_vectorised_quantile_matches_scalar_weighted_quantile(sample):
    rng = np.random.default_rng(4)
    order = np.argsort(sample)
    for _ in range(20):
        w = rng.dirichlet(np.ones(sample.size))
        for q in (0.25, 0.5, 0.9):
            vec = _batch_statistic(sample[order], w[order][None, :], "quantile", q, 0.0)[0]
            assert vec == weighted_quantile(sample, w, q)


def test_all_statistics_give_sensible_intervals(sample):
    for stat in BOOTSTRAP_STATISTICS:
        post = bayesian_bootstrap(sample, stat, q=0.9, n_draws=3000, seed=2)
        lo, hi = post.interval
        assert lo <= post.point_estimate <= hi
        assert post.draws.shape == (3000,)
    med = bayesian_bootstrap(sample, "median", seed=3)
    assert med.statistic == "median"
    assert bayesian_bootstrap(sample, "quantile", q=0.9).statistic == "quantile(0.9)"


def test_chunking_and_seed_reproducibility(sample):
    a = bayesian_bootstrap(sample, "median", n_draws=500, seed=9, chunk_size=400 * 7)
    b = bayesian_bootstrap(sample, "median", n_draws=500, seed=9, chunk_size=400 * 7)
    np.testing.assert_array_equal(a.draws, b.draws)
    c = bayesian_bootstrap(sample, "mean", n_draws=1000, seed=9, chunk_size=10)  # 1 row per chunk
    assert c.draws.size == 1000


def test_callable_statistic(sample):
    def weighted_geomean(x, w):
        return float(np.exp(w @ np.log(x)))

    post = bayesian_bootstrap(sample, weighted_geomean, n_draws=500, seed=1)
    assert post.statistic == "weighted_geomean"
    assert post.point_estimate == pytest.approx(np.exp(np.log(sample).mean()))
    lo, hi = post.interval
    assert lo < post.point_estimate < hi


def test_ratio_posterior():
    rng = np.random.default_rng(1)
    views = rng.poisson(10, 2000) + 1
    clicks = rng.binomial(views, 0.2)
    post = bayesian_bootstrap_ratio(clicks, views, n_draws=4000, seed=1)
    assert post.point_estimate == pytest.approx(clicks.sum() / views.sum())
    lo, hi = post.interval
    assert lo < 0.2 < hi
    # Close to the delta-method SE.
    from bayes_ab_kit import ratio_summary

    assert post.sd == pytest.approx(ratio_summary(clicks, views).se, rel=0.1)
    with pytest.raises(ValueError, match="same length"):
        bayesian_bootstrap_ratio([1, 2, 3], [1, 2])


def test_compare_detects_shift_and_orientation():
    rng = np.random.default_rng(5)
    a = rng.lognormal(0.0, 0.8, 600)
    b = rng.lognormal(0.25, 0.8, 600)
    res = compare_bayesian_bootstrap(a, b, "median", seed=1)
    assert res.prob_b_better > 0.99
    assert res.difference_interval[0] > 0
    assert res.relative_lift == pytest.approx(np.median(b) / np.median(a) - 1, rel=0.05)
    assert res.expected_loss_choose_a > res.expected_loss_choose_b
    lower = compare_bayesian_bootstrap(a, b, "median", seed=1, higher_is_better=False)
    assert lower.prob_b_better == pytest.approx(1 - res.prob_b_better)
    assert lower.expected_loss_choose_b == pytest.approx(res.expected_loss_choose_a)
    same = compare_bayesian_bootstrap(a, a.copy(), "quantile", q=0.9, seed=2)
    assert 0.2 < same.prob_b_better < 0.8
    d = same.to_dict()
    assert d["a"]["n_draws"] == 4000


def test_compare_accepts_posteriors_and_handles_nonpositive():
    rng = np.random.default_rng(6)
    pa = bayesian_bootstrap(rng.normal(0, 1, 100), "mean", seed=1)
    pb = bayesian_bootstrap(rng.normal(0.5, 1, 100), "mean", seed=2)
    res = compare_bayesian_bootstrap(pa, pb)
    assert res.relative_lift is None and res.relative_lift_interval is None
    short = bayesian_bootstrap(rng.normal(0, 1, 100), "mean", n_draws=100, seed=3)
    with pytest.raises(ValueError, match="same number"):
        compare_bayesian_bootstrap(pa, short)


def test_validation():
    with pytest.raises(ValueError, match="at least two"):
        bayesian_bootstrap([1.0])
    with pytest.raises(ValueError, match="finite"):
        bayesian_bootstrap([1.0, np.nan])
    with pytest.raises(ValueError, match="unknown statistic"):
        bayesian_bootstrap([1.0, 2.0], "mode")
    with pytest.raises(ValueError, match="trim"):
        bayesian_bootstrap([1.0, 2.0], "trimmed_mean", trim=0.5)
    with pytest.raises(ValueError, match="ci"):
        bayesian_bootstrap([1.0, 2.0], ci=1.0)
    with pytest.raises(ValueError, match="q must"):
        weighted_quantile(np.array([1.0, 2.0]), np.ones(2), 1.5)


def test_cli_bootstrap(tmp_path, capsys):
    rng = np.random.default_rng(2)
    path = tmp_path / "values.csv"
    lines = ["arm,value"]
    lines += [f"A,{v:.4f}" for v in rng.lognormal(0, 0.5, 300)]
    lines += [f"B,{v:.4f}" for v in rng.lognormal(0.3, 0.5, 300)]
    path.write_text("\n".join(lines) + "\n")
    assert main(["bootstrap", "--data", str(path), "--statistic", "quantile", "--q", "0.9", "--draws", "1000"]) == 0
    out = capsys.readouterr().out
    assert "quantile(0.9)" in out and "P(B better)" in out
    bad = tmp_path / "bad.csv"
    bad.write_text("arm,x\nA,1\n")
    with pytest.raises(SystemExit) as exc:
        main(["bootstrap", "--data", str(bad)])
    assert exc.value.code == 2
