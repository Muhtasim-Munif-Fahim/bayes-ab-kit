import numpy as np
import pytest

from bayes_ab_kit.posteriors import BetaBinomialPosterior
from bayes_ab_kit.sampling import (
    make_rng,
    posterior_draws,
    summarize_draws,
)


def test_same_seed_gives_identical_streams():
    assert np.array_equal(make_rng(11).random(10), make_rng(11).random(10))


def test_different_seeds_diverge():
    assert not np.array_equal(make_rng(1).random(10), make_rng(2).random(10))


def test_none_seed_is_accepted():
    rng = make_rng(None)
    assert isinstance(rng.random(), float)


def test_negative_seed_rejected():
    with pytest.raises(ValueError):
        make_rng(-5)


def test_posterior_draws_match_posterior_mean():
    post = BetaBinomialPosterior.from_counts(40, 200)
    rng = make_rng(99)
    draws = posterior_draws(post, rng, size=50_000)
    assert draws.shape == (50_000,)
    assert draws.mean() == pytest.approx(post.mean(), abs=0.005)


def test_zero_size_rejected():
    post = BetaBinomialPosterior.from_counts(4, 20)
    with pytest.raises(ValueError):
        posterior_draws(post, make_rng(3), size=0)


def test_summarize_matches_numpy_statistics():
    draws = np.linspace(0.0, 1.0, 101)
    summary = summarize_draws(draws, ci=0.9)
    assert summary.mean == pytest.approx(np.mean(draws))
    assert summary.std == pytest.approx(np.std(draws, ddof=1))
    lo, hi = np.quantile(draws, [0.05, 0.95])
    assert summary.lower == pytest.approx(lo)
    assert summary.upper == pytest.approx(hi)
    assert summary.n == 101


def test_summary_interval_orders_correctly():
    post = BetaBinomialPosterior.from_counts(25, 300)
    summary = summarize_draws(posterior_draws(post, make_rng(5), 4000))
    assert summary.lower <= summary.mean <= summary.upper


def test_summarize_rejects_bad_input():
    with pytest.raises(ValueError):
        summarize_draws(np.array([]))
    with pytest.raises(ValueError):
        summarize_draws(np.ones(5), ci=0)