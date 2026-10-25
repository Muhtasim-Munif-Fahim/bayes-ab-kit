import numpy as np
import pytest

from bayes_ab_kit import (
    DirichletMultinomialPosterior,
    compare_categorical,
    probability_of_superiority,
)
from bayes_ab_kit.cli import main


def test_posterior_update_and_closed_form_moments():
    post = DirichletMultinomialPosterior(5, prior=1.0)
    post.update([5, 10, 20, 40, 25])
    assert post.n == 100
    assert np.allclose(post.alpha, [6, 11, 21, 41, 26])
    assert post.share_means().sum() == pytest.approx(1.0)
    draws = post.sample(np.random.default_rng(1), 200_000) @ post.scores
    assert post.mean_score() == pytest.approx(draws.mean(), abs=2e-3)
    assert post.mean_score_var() == pytest.approx(draws.var(), rel=0.02)


def test_share_intervals_contain_mean():
    post = DirichletMultinomialPosterior.from_counts([30, 50, 20], scores=[-1, 0, 1])
    iv = post.share_intervals(0.9)
    m = post.share_means()
    assert np.all(iv[:, 0] < m) and np.all(m < iv[:, 1])
    assert post.mean_score() == pytest.approx((20 + 1 - 30 - 1) / 103)


def test_validation():
    with pytest.raises(ValueError):
        DirichletMultinomialPosterior(1)
    with pytest.raises(ValueError):
        DirichletMultinomialPosterior(3, prior=[1, 0, 1])
    with pytest.raises(ValueError):
        DirichletMultinomialPosterior(3, scores=[1, 2])
    with pytest.raises(ValueError):
        DirichletMultinomialPosterior.from_counts([1, -2, 3])
    with pytest.raises(ValueError):
        DirichletMultinomialPosterior.from_counts([1, 2]).update([1, 2, 3])
    with pytest.raises(ValueError, match="same number"):
        compare_categorical([1, 2], [1, 2, 3])


def test_probability_of_superiority_exact_cases():
    a = [0.0, 1.0, 0.0]
    assert probability_of_superiority(a, [0, 0, 1.0]) == pytest.approx(1.0)
    assert probability_of_superiority(a, [1.0, 0, 0]) == pytest.approx(0.0)
    assert probability_of_superiority(a, a) == pytest.approx(0.5)
    rng = np.random.default_rng(0)
    pa, pb = rng.dirichlet(np.ones(4)), rng.dirichlet(np.ones(4))
    brute = sum(pa[i] * pb[j] * (1.0 if j > i else 0.5 if j == i else 0.0)
                for i in range(4) for j in range(4))
    assert probability_of_superiority(pa, pb) == pytest.approx(brute)
    assert probability_of_superiority(pa, pb) + probability_of_superiority(pb, pa) == pytest.approx(1.0)
    batch = probability_of_superiority(np.vstack([pa, pb]), np.vstack([pb, pa]))
    assert batch.shape == (2,)


def test_identical_arms_are_a_coin_flip():
    res = compare_categorical([10, 20, 30, 20, 10], [10, 20, 30, 20, 10], seed=3)
    assert res.prob_b_better == pytest.approx(0.5, abs=0.02)
    assert res.superiority == pytest.approx(0.5, abs=0.01)
    assert res.mean_diff == pytest.approx(0.0)
    assert res.mean_diff_interval[0] < 0 < res.mean_diff_interval[1]


def test_clear_improvement_and_direction_flip():
    a, b = [40, 30, 15, 10, 5], [5, 10, 15, 30, 40]
    res = compare_categorical(a, b, n_samples=20_000)
    assert res.prob_b_better > 0.999 and res.winner == "B"
    assert res.superiority > 0.7 and res.prob_superiority_above_half > 0.999
    assert res.expected_loss_choose_b < 1e-6 < res.expected_loss_choose_a
    assert res.prob_share_b_higher[-1] > 0.99 and res.prob_share_b_higher[0] < 0.01
    flipped = compare_categorical(a, b, higher_is_better=False, n_samples=20_000)
    assert flipped.prob_b_better < 0.001 and flipped.winner == "A"
    assert flipped.superiority == pytest.approx(1 - res.superiority, abs=0.01)


def test_custom_scores_change_mean_not_superiority():
    a, b = [30, 50, 20], [22, 50, 28]
    r1 = compare_categorical(a, b, scores=[-1, 0, 1], seed=7)
    r2 = compare_categorical(a, b, scores=[0, 10, 100], seed=7)
    assert r1.superiority == pytest.approx(r2.superiority)
    assert r1.mean_diff != pytest.approx(r2.mean_diff)
    assert compare_categorical(a, b, seed=7) == compare_categorical(a, b, seed=7)


def test_cli_categorical(capsys):
    assert main(["categorical", "--counts-a", "5,10,20,40,25", "--counts-b", "3,7,18,42,30",
                 "--samples", "5000"]) == 0
    out = capsys.readouterr().out
    assert "P(superiority)" in out and "mean diff (B - A)" in out
    assert main(["categorical", "--counts-a", "30,50,20", "--counts-b", "22,50,28",
                 "--scores=-1,0,1", "--samples", "2000"]) == 0
    with pytest.raises(SystemExit) as exc:
        main(["categorical", "--counts-a", "1,2", "--counts-b", "1,x"])
    assert exc.value.code == 2
