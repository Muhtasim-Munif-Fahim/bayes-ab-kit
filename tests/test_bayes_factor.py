"""Tests for Beta-Binomial Bayes factors and sequential early stopping."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.special import betaln

from bayes_ab_kit.bayes_factor import (
    bayes_factor_10,
    bayes_factor_decision,
    kass_raftery_label,
    log_bayes_factor_10,
    log_marginal_h0,
    log_marginal_h1,
    simulate_bayes_factor_stopping,
)
from bayes_ab_kit.decisions import Decision
from bayes_ab_kit.posteriors import BetaBinomialPosterior


def test_log_marginal_matches_hand_computation():
    # Uniform prior Beta(1,1); H1 log marg omits binomial coeffs.
    sa, na, sb, nb = 2, 5, 1, 4
    expected_h1 = (
        betaln(1 + sa, 1 + (na - sa))
        + betaln(1 + sb, 1 + (nb - sb))
        - 2 * betaln(1, 1)
    )
    expected_h0 = betaln(1 + sa + sb, 1 + (na - sa) + (nb - sb)) - betaln(1, 1)
    assert log_marginal_h1(sa, na, sb, nb) == pytest.approx(expected_h1)
    assert log_marginal_h0(sa, na, sb, nb) == pytest.approx(expected_h0)
    assert log_bayes_factor_10(sa, na, sb, nb) == pytest.approx(expected_h1 - expected_h0)


def test_identical_arms_favor_shared_null():
    bf = bayes_factor_10(50, 1000, 50, 1000)
    assert bf < 1.0
    assert "H0" in kass_raftery_label(bf)


def test_large_uplift_favors_alternative():
    bf = bayes_factor_10(80, 1000, 200, 1000)
    assert bf > 10.0
    assert "H1" in kass_raftery_label(bf)


def test_bayes_factor_decision_ships_leader_on_strong_evidence():
    a = BetaBinomialPosterior.from_counts(40, 1000)
    b = BetaBinomialPosterior.from_counts(120, 1000)
    result = bayes_factor_decision(a, b)
    assert result.bf10 > 10
    assert result.decision is Decision.SHIP_B
    assert result.log_bf10 == pytest.approx(math.log(result.bf10))


def test_bayes_factor_decision_keeps_running_when_weak():
    a = BetaBinomialPosterior.from_counts(5, 40)
    b = BetaBinomialPosterior.from_counts(8, 40)
    result = bayes_factor_decision(a, b)
    assert 0.1 < result.bf10 < 10
    assert result.decision is Decision.KEEP_RUNNING


def test_near_identical_large_n_is_practical_equivalence():
    a = BetaBinomialPosterior.from_counts(50, 1000)
    b = BetaBinomialPosterior.from_counts(55, 1000)
    result = bayes_factor_decision(a, b)
    assert result.bf10 <= 0.1
    assert result.decision is Decision.PRACTICAL_EQUIVALENCE


def test_kass_raftery_boundaries():
    assert kass_raftery_label(150) == "decisive for H1"
    assert kass_raftery_label(12) == "strong for H1"
    assert kass_raftery_label(4) == "positive for H1"
    assert kass_raftery_label(1.5) == "barely worth mentioning for H1"
    assert kass_raftery_label(0.05) == "strong for H0"
    with pytest.raises(ValueError):
        kass_raftery_label(0.0)


def test_rejects_invalid_counts():
    with pytest.raises(ValueError):
        bayes_factor_10(5, 3, 1, 10)
    with pytest.raises(ValueError):
        bayes_factor_10(-1, 10, 1, 10)
    with pytest.raises(ValueError):
        log_marginal_h1(1, 10, 1, 10, alpha0=0.0)


def test_mismatched_priors_raise():
    a = BetaBinomialPosterior.from_counts(10, 100, alpha0=1.0, beta0=1.0)
    b = BetaBinomialPosterior.from_counts(12, 100, alpha0=2.0, beta0=2.0)
    with pytest.raises(ValueError, match="prior"):
        bayes_factor_decision(a, b)


def test_sequential_null_rarely_stops_for_h1():
    result = simulate_bayes_factor_stopping(
        true_rate_a=0.10,
        true_rate_b=0.10,
        per_look=200,
        looks=4,
        log_bf_threshold=math.log(10.0),
        n_simulations=400,
        seed=11,
    )
    assert result.decisive_for_h1_rate < 0.15
    assert 0.0 <= result.stop_rate <= 1.0
    assert result.mean_looks_at_stop <= 4.0


def test_sequential_uplift_stops_more_often_for_h1():
    null = simulate_bayes_factor_stopping(
        true_rate_a=0.10,
        true_rate_b=0.10,
        per_look=300,
        looks=5,
        n_simulations=300,
        seed=3,
    )
    alt = simulate_bayes_factor_stopping(
        true_rate_a=0.10,
        true_rate_b=0.16,
        per_look=300,
        looks=5,
        n_simulations=300,
        seed=3,
    )
    assert alt.decisive_for_h1_rate > null.decisive_for_h1_rate
    assert alt.decisive_for_h1_rate > 0.2


def test_sequential_rejects_bad_args():
    with pytest.raises(ValueError):
        simulate_bayes_factor_stopping(true_rate_a=0.0, true_rate_b=0.1)
    with pytest.raises(ValueError):
        simulate_bayes_factor_stopping(
            true_rate_a=0.1, true_rate_b=0.1, log_bf_threshold=0.0
        )


def test_cli_bayes_factor(capsys):
    from bayes_ab_kit.cli import main

    assert main([
        "bayes-factor",
        "--conversions-a", "40",
        "--trials-a", "1000",
        "--conversions-b", "120",
        "--trials-b", "1000",
    ]) == 0
    out = capsys.readouterr().out
    assert "BF10" in out
    assert "Kass-Raftery" in out
