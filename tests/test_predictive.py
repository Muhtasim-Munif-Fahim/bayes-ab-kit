"""Tests for predictive probability of success."""

from __future__ import annotations

import pytest

from bayes_ab_kit import predictive_probability_of_success
from bayes_ab_kit.cli import main
from bayes_ab_kit.predictive import PredictiveSuccessResult


def test_clear_winner_with_remaining_traffic_has_high_pps():
    # B already well ahead; more traffic should keep success likely.
    result = predictive_probability_of_success(
        80,
        1000,
        140,
        1000,
        remaining_a=2000,
        remaining_b=2000,
        threshold=0.95,
        n_simulations=1500,
        decision_samples=1500,
        seed=7,
    )
    assert isinstance(result, PredictiveSuccessResult)
    assert result.predictive_probability > 0.85
    assert result.continue_recommended
    assert result.futility_probability < 0.1


def test_null_data_has_low_pps_at_strict_threshold():
    result = predictive_probability_of_success(
        100,
        1000,
        102,
        1000,
        remaining_a=500,
        threshold=0.95,
        n_simulations=1500,
        decision_samples=1500,
        seed=11,
    )
    assert result.predictive_probability < 0.25
    assert result.prob_already_success == 0.0


def test_zero_remaining_equals_current_decision():
    # No more data: PPS collapses to whether we already pass the threshold.
    result = predictive_probability_of_success(
        200,
        1000,
        280,
        1000,
        remaining_a=0,
        remaining_b=0,
        threshold=0.95,
        n_simulations=800,
        decision_samples=800,
        seed=3,
    )
    assert result.prob_already_success == 1.0
    assert result.predictive_probability == pytest.approx(1.0, abs=0.02)


def test_validation_errors():
    with pytest.raises(ValueError, match="conversions cannot exceed"):
        predictive_probability_of_success(10, 5, 1, 10, remaining_a=10)
    with pytest.raises(ValueError, match="threshold"):
        predictive_probability_of_success(1, 10, 2, 10, remaining_a=10, threshold=0.4)
    with pytest.raises(ValueError, match="remaining_a"):
        predictive_probability_of_success(1, 10, 2, 10, remaining_a=-1)


def test_remaining_b_override():
    result = predictive_probability_of_success(
        50,
        500,
        70,
        500,
        remaining_a=100,
        remaining_b=500,
        n_simulations=500,
        decision_samples=500,
        seed=1,
    )
    assert result.remaining_a == 100
    assert result.remaining_b == 500


def test_cli_pps(capsys):
    rc = main(
        [
            "pps",
            "--conversions-a",
            "80",
            "--trials-a",
            "1000",
            "--conversions-b",
            "140",
            "--trials-b",
            "1000",
            "--remaining",
            "1000",
            "--sims",
            "800",
            "--decision-samples",
            "800",
            "--seed",
            "5",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "predictive P(success)" in out
    assert "continue recommended" in out
