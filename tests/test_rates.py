"""Tests for Gamma-Poisson rate comparisons."""

from __future__ import annotations

import numpy as np
import pytest

from bayes_ab_kit import (
    GammaPoissonPosterior,
    compare_rates,
    prob_rate_b_exceeds_a,
    rate_ratio_interval,
)
from bayes_ab_kit.cli import main


def test_posterior_conjugate_update_and_mean():
    post = GammaPoissonPosterior.from_counts(30, 10.0, shape0=2.0, rate0=1.0)
    assert post.shape == pytest.approx(32.0)
    assert post.rate == pytest.approx(11.0)
    assert post.mean() == pytest.approx(32.0 / 11.0)
    lo, hi = post.credible_interval(0.9)
    assert lo < post.mean() < hi


def test_exact_probability_matches_monte_carlo():
    a = GammaPoissonPosterior.from_counts(120, 1000.0)
    b = GammaPoissonPosterior.from_counts(145, 1000.0)
    rng = np.random.default_rng(3)
    mc = np.mean(b.sample(rng, 400_000) > a.sample(rng, 400_000))
    assert prob_rate_b_exceeds_a(a, b) == pytest.approx(mc, abs=0.004)


def test_symmetry_identical_arms_is_one_half():
    a = GammaPoissonPosterior.from_counts(50, 200.0)
    b = GammaPoissonPosterior.from_counts(50, 200.0)
    assert prob_rate_b_exceeds_a(a, b) == pytest.approx(0.5, abs=1e-12)


def test_rate_ratio_interval_matches_simulation():
    a = GammaPoissonPosterior.from_counts(80, 500.0)
    b = GammaPoissonPosterior.from_counts(110, 450.0)
    lo, med, hi = rate_ratio_interval(a, b, 0.9)
    rng = np.random.default_rng(9)
    ratio = b.sample(rng, 400_000) / a.sample(rng, 400_000)
    q = np.quantile(ratio, [0.05, 0.5, 0.95])
    assert (lo, med, hi) == pytest.approx(tuple(q), rel=0.01)


def test_compare_rates_lower_is_better_flips_direction():
    up = compare_rates(40, 1000.0, 70, 1000.0, seed=1)
    down = compare_rates(40, 1000.0, 70, 1000.0, higher_is_better=False, seed=1)
    assert up.prob_b_better > 0.99
    assert down.prob_b_better == pytest.approx(1.0 - up.prob_b_better)
    assert up.expected_loss_choose_b < up.expected_loss_choose_a
    assert down.expected_loss_choose_a < down.expected_loss_choose_b
    assert up.rate_ratio_interval[0] > 1.0


def test_winner_requires_threshold():
    res = compare_rates(40, 1000.0, 70, 1000.0, loss_threshold=1e-3, seed=2)
    assert res.winner == "B"
    assert compare_rates(40, 1000.0, 41, 1000.0, loss_threshold=1e-5).winner is None
    assert compare_rates(40, 1000.0, 70, 1000.0).winner is None


def test_predictive_events_is_negative_binomial_with_right_mean():
    post = GammaPoissonPosterior.from_counts(20, 10.0, shape0=1.0, rate0=1.0)
    dist = post.predictive_events(5.0)
    assert dist.mean() == pytest.approx(post.mean() * 5.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(events_a=-1, exposure_a=1.0, events_b=1, exposure_b=1.0),
        dict(events_a=1, exposure_a=0.0, events_b=1, exposure_b=1.0),
        dict(events_a=0, exposure_a=0.0, events_b=1, exposure_b=1.0),
        dict(events_a=1, exposure_a=-2.0, events_b=1, exposure_b=1.0),
    ],
)
def test_invalid_inputs_raise(kwargs):
    with pytest.raises(ValueError):
        compare_rates(**kwargs)


def test_cli_rates(capsys):
    assert main([
        "rates", "--events-a", "40", "--exposure-a", "1000",
        "--events-b", "70", "--exposure-b", "1000",
        "--loss-threshold", "0.001",
    ]) == 0
    out = capsys.readouterr().out
    assert "P(B better)" in out
    assert "rate ratio B/A" in out
    winner_line = next(line for line in out.splitlines() if line.startswith("winner"))
    assert winner_line.rstrip().endswith(": B")


def test_cli_rates_lower_is_better(capsys):
    main([
        "rates", "--events-a", "40", "--exposure-a", "1000",
        "--events-b", "70", "--exposure-b", "1000", "--lower-is-better",
    ])
    assert "lower rate is better" in capsys.readouterr().out
