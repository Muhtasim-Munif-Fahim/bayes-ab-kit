import numpy as np
import pytest

from bayes_ab_kit.decisions import Decision
from bayes_ab_kit.posteriors import BetaBinomialPosterior
from bayes_ab_kit.risk import RiskResult, expected_loss, risk_decision


def b_variant(conversions, trials):
    return BetaBinomialPosterior.from_counts(conversions, trials)


def test_symmetric_variants_have_equal_losses():
    a = b_variant(60, 500)
    loss = expected_loss(a, a, n_samples=30_000)
    assert 0 < loss < 0.01


def test_loss_of_dominant_variant_is_tiny():
    weak = b_variant(40, 1000)
    strong = b_variant(150, 1000)
    assert expected_loss(strong, weak, n_samples=50_000) < 0.002
    assert expected_loss(weak, strong, n_samples=50_000) > 0.02


def test_loss_decreases_as_evidence_grows_for_same_rate_gap():
    small = expected_loss(b_variant(55, 1000), b_variant(45, 1000), n_samples=40_000)
    large = expected_loss(b_variant(550, 10_000), b_variant(450, 10_000), n_samples=40_000)
    assert large < small * 0.5


def test_expected_loss_is_nonnegative_and_deterministic():
    a = b_variant(70, 900)
    b = b_variant(85, 900)
    first = expected_loss(a, b, n_samples=20_000)
    second = expected_loss(a, b, n_samples=20_000)
    assert first >= 0
    assert first == second


def test_risk_result_shape():
    result = risk_decision(b_variant(90, 1000), b_variant(140, 1000), threshold=0.005)
    assert isinstance(result, RiskResult)
    assert result.leader == "b"
    assert 0 <= result.prob_leader_beats_follower <= 1


def test_clear_winner_ships_under_default_threshold():
    result = risk_decision(b_variant(35, 1000), b_variant(120, 1000))
    assert result.decision is Decision.SHIP_B
    assert result.expected_loss_of_leader < result.threshold


def test_ambiguous_data_keeps_running():
    result = risk_decision(b_variant(48, 1000), b_variant(52, 1000), threshold=0.001)
    assert result.decision is Decision.KEEP_RUNNING


def test_looser_threshold_ships_sooner():
    tight = risk_decision(b_variant(52, 1000), b_variant(64, 1000), threshold=0.0005)
    loose = risk_decision(b_variant(52, 1000), b_variant(64, 1000), threshold=0.02)
    assert tight.decision is Decision.KEEP_RUNNING
    assert loose.decision is Decision.SHIP_B


def test_leading_a_is_recognized():
    result = risk_decision(b_variant(130, 1000), b_variant(60, 1000), threshold=0.005)
    assert result.leader == "a"
    assert result.decision is Decision.SHIP_A


def test_invalid_threshold_rejected():
    with pytest.raises(ValueError):
        risk_decision(b_variant(5, 100), b_variant(6, 100), threshold=0.0)


def test_invalid_sample_count_rejected():
    with pytest.raises(ValueError):
        risk_decision(b_variant(5, 100), b_variant(6, 100), n_samples=-1)


def test_mc_estimates_are_stable_across_seeds_in_distribution():
    a = b_variant(80, 2000)
    b = b_variant(95, 2000)
    losses = [
        expected_loss(b, a, n_samples=20_000, seed=s) for s in (1, 2, 3)
    ]
    spread = max(losses) - min(losses)
    assert spread < 0.001