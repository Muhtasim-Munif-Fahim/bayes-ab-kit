import numpy as np
import pytest

from bayes_ab_kit import compare_robust_means, robust_posterior
from bayes_ab_kit.cli import main


def test_fixed_large_nu_recovers_normal_posterior():
    rng = np.random.default_rng(0)
    y = rng.normal(10.0, 2.0, 400)
    post = robust_posterior(y, nu=1e6, n_draws=4000, seed=1)
    se = y.std(ddof=1) / np.sqrt(y.size)
    assert post.mu == pytest.approx(y.mean(), abs=0.2 * se)
    assert post.mu_draws.std() == pytest.approx(se, rel=0.1)
    assert post.sigma == pytest.approx(y.std(ddof=1), rel=0.05)
    assert np.all(post.nu_draws == 1e6)


def test_t_posterior_matches_quadrature_for_fixed_nu():
    # The Gibbs marginal of mu (fixed nu, Jeffreys sigma prior) must match a
    # brute-force grid integration over (mu, sigma).
    from scipy import stats

    rng = np.random.default_rng(5)
    y = stats.t.rvs(3, loc=1.0, scale=0.5, size=40, random_state=rng)
    post = robust_posterior(y, nu=3.0, n_draws=20000, burn_in=2000, seed=2)
    mus = np.linspace(y.min(), y.max(), 400)
    sig = np.geomspace(0.05, 5.0, 400)
    M, S = np.meshgrid(mus, sig, indexing="ij")
    ll = stats.t.logpdf(y[None, None, :], 3.0, loc=M[..., None], scale=S[..., None]).sum(-1)
    # p(sigma^2) ~ 1/sigma^2  =>  p(sigma) d sigma ~ d log sigma, uniform on the geometric grid
    logp = ll
    w = np.exp(logp - logp.max()).sum(axis=1)
    w /= w.sum()
    mean = float((w * mus).sum())
    sd = float(np.sqrt((w * (mus - mean) ** 2).sum()))
    assert post.mu == pytest.approx(mean, abs=0.05 * sd + 0.01)
    assert post.mu_draws.std() == pytest.approx(sd, rel=0.08)


def test_outliers_do_not_drag_robust_location():
    rng = np.random.default_rng(1)
    y = np.concatenate([rng.normal(10.0, 2.0, 380), rng.normal(80.0, 5.0, 20)])
    post = robust_posterior(y, seed=2)
    assert y.mean() > 13.0
    lo, hi = post.mu_interval
    assert lo < 10.0 < hi or abs(post.mu - 10.0) < 0.3
    assert post.nu_median < 5 and post.prob_heavy_tailed > 0.95


def test_learns_heavy_tails_and_normal_tails():
    rng = np.random.default_rng(3)
    heavy = robust_posterior(rng.standard_t(2, 600), seed=4)
    light = robust_posterior(rng.normal(size=3000), seed=4)
    assert heavy.nu_median < 4
    assert light.nu_median > 20 and light.prob_heavy_tailed < 0.1


def test_comparison_resists_outliers_that_flip_the_raw_mean():
    rng = np.random.default_rng(4)
    a = rng.normal(10.0, 2.0, 300)
    b = rng.normal(11.0, 2.0, 300)
    b[:5] = -200.0  # a handful of refunds flips the raw mean difference
    assert b.mean() < a.mean()
    res = compare_robust_means(a, b, seed=0)
    assert res.prob_b_better > 0.99
    assert res.difference == pytest.approx(1.0, abs=0.4)
    assert res.effect_size > 0.1
    assert res.relative_lift == pytest.approx(0.1, abs=0.04)
    assert res.expected_loss_choose_b < res.expected_loss_choose_a
    lower = compare_robust_means(a, b, seed=0, higher_is_better=False)
    assert lower.prob_b_better == pytest.approx(1 - res.prob_b_better)
    d = res.to_dict()
    assert d["a"]["n"] == 300 and "effect_size_interval" in d


def test_reproducible_and_reuses_posteriors():
    rng = np.random.default_rng(6)
    a, b = rng.normal(size=50), rng.normal(0.5, 1.0, 50)
    r1 = compare_robust_means(a, b, n_draws=500, burn_in=100, seed=9)
    r2 = compare_robust_means(a, b, n_draws=500, burn_in=100, seed=9)
    assert r1.difference == r2.difference
    r3 = compare_robust_means(r1.a, r1.b)
    assert r3.difference == pytest.approx(r1.difference)
    with pytest.raises(ValueError, match="same number"):
        compare_robust_means(r1.a, robust_posterior(b, n_draws=100, burn_in=10, seed=1))
    lift = compare_robust_means(a - 10, b, n_draws=200, burn_in=50, seed=1)
    assert lift.relative_lift is None


def test_validation():
    with pytest.raises(ValueError, match="at least 3"):
        robust_posterior([1.0, 2.0])
    with pytest.raises(ValueError, match="finite"):
        robust_posterior([1.0, np.nan, 2.0])
    with pytest.raises(ValueError, match="all be equal"):
        robust_posterior([1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="nu"):
        robust_posterior([1.0, 2.0, 3.0], nu=0)
    with pytest.raises(ValueError, match="ci"):
        robust_posterior([1.0, 2.0, 3.0], ci=1.0)
    with pytest.raises(ValueError, match="n_draws"):
        robust_posterior([1.0, 2.0, 3.0], n_draws=0)


def test_cli_robust(tmp_path, capsys):
    rng = np.random.default_rng(2)
    path = tmp_path / "values.csv"
    lines = ["arm,value"]
    lines += [f"A,{v:.4f}" for v in rng.standard_t(3, 200) + 5]
    lines += [f"B,{v:.4f}" for v in rng.standard_t(3, 200) + 5.5]
    path.write_text("\n".join(lines) + "\n")
    assert main(["robust", "--data", str(path), "--draws", "800", "--burn-in", "200"]) == 0
    out = capsys.readouterr().out
    assert "Robust Student-t" in out and "P(B better)" in out and "effect size" in out
    assert main(["robust", "--data", str(path), "--nu", "4", "--draws", "300", "--burn-in", "50"]) == 0
    assert "nu fixed" in capsys.readouterr().out
    with pytest.raises(SystemExit) as exc:
        main(["robust", "--data", str(path), "--control", "Z"])
    assert exc.value.code == 2
