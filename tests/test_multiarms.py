import numpy as np
import pytest

from bayes_ab_kit.decisions import prob_b_beats_a_mc
from bayes_ab_kit.multiarms import (
    ArmBestShare,
    MultiArmBestResult,
    _choose_leader,
    _shares_of_being_best,
    probability_of_being_best,
)
from bayes_ab_kit.posteriors import BetaBinomialPosterior


def b_variant(conversions, trials):
    return BetaBinomialPosterior.from_counts(conversions, trials)


def test_identical_arms_share_probability_equally():
    arms = {name: b_variant(50, 1000) for name in ("A", "B", "C")}
    result = probability_of_being_best(arms, n_samples=40_000)
    probs = list(result.probabilities.values())
    assert sum(probs) == pytest.approx(1.0, abs=1e-12)
    for p in probs:
        assert p == pytest.approx(1 / 3, abs=0.03)


def test_dominant_arm_is_almost_surely_best():
    result = probability_of_being_best(
        {
            "control": b_variant(40, 1000),
            "weak": b_variant(45, 1000),
            "winner": b_variant(150, 1000),
        },
        n_samples=30_000,
    )
    assert result.leader == "winner"
    assert result.probabilities["winner"] > 0.95
    assert result.probabilities["control"] < 0.05
    assert result.probabilities["weak"] < 0.05


def test_weakest_arm_has_smallest_share():
    result = probability_of_being_best(
        {
            "low": b_variant(30, 800),
            "mid": b_variant(70, 800),
            "high": b_variant(110, 800),
        },
        n_samples=25_000,
    )
    probs = result.probabilities
    assert probs["low"] < probs["mid"] < probs["high"]
    assert result.leader == "high"


def test_four_arms_still_sum_to_one():
    result = probability_of_being_best(
        {
            "A": b_variant(20, 400),
            "B": b_variant(24, 400),
            "C": b_variant(28, 400),
            "D": b_variant(32, 400),
        },
        n_samples=20_000,
    )
    assert len(result.arms) == 4
    assert sum(result.probabilities.values()) == pytest.approx(1.0, abs=1e-12)


def test_result_is_deterministic_under_seed():
    arms = {
        "A": b_variant(80, 900),
        "B": b_variant(90, 900),
        "C": b_variant(85, 900),
    }
    first = probability_of_being_best(arms, n_samples=12_000, seed=11)
    second = probability_of_being_best(arms, n_samples=12_000, seed=11)
    assert first == second
    assert isinstance(first, MultiArmBestResult)
    assert all(isinstance(arm, ArmBestShare) for arm in first.arms)


def test_preserves_input_order():
    result = probability_of_being_best(
        {
            "gamma": b_variant(10, 100),
            "alpha": b_variant(12, 100),
            "beta": b_variant(11, 100),
        },
        n_samples=5_000,
    )
    assert [arm.name for arm in result.arms] == ["gamma", "alpha", "beta"]


def test_two_arm_case_matches_pairwise_mc():
    a = b_variant(40, 500)
    b = b_variant(55, 500)
    pairwise = prob_b_beats_a_mc(a, b, n_samples=20_000, seed=7)
    multi = probability_of_being_best(
        {"A": a, "B": b}, n_samples=20_000, seed=7
    )
    # Pairwise MC uses a strict greater-than; multi-arm splits ties, so the
    # estimates agree up to the (vanishing) tie rate.
    assert multi.probabilities["B"] == pytest.approx(pairwise, abs=0.01)
    assert multi.probabilities["A"] + multi.probabilities["B"] == pytest.approx(1.0)


def test_carries_counts_and_posterior_means():
    post = b_variant(95, 1000)
    result = probability_of_being_best(
        {"A": post, "B": b_variant(100, 1000), "C": b_variant(90, 1000)},
        n_samples=4_000,
    )
    arm_a = result.arms[0]
    assert arm_a.conversions == 95
    assert arm_a.trials == 1000
    assert arm_a.posterior_mean == pytest.approx(post.mean())


def test_leader_is_the_arm_with_highest_p_best():
    result = probability_of_being_best(
        {
            "close": b_variant(80, 1000),
            "ahead": b_variant(120, 1000),
            "lag": b_variant(60, 1000),
        },
        n_samples=15_000,
    )
    top = max(result.arms, key=lambda arm: arm.prob_best)
    assert result.leader == top.name


def test_exact_ties_are_split_equally():
    shares = _shares_of_being_best(np.ones((8, 3)))
    assert shares == pytest.approx(np.array([1 / 3, 1 / 3, 1 / 3]))


def test_leader_tie_breaks_on_posterior_mean_then_name():
    even = (
        ArmBestShare("B", 10, 100, posterior_mean=0.2, prob_best=0.5),
        ArmBestShare("A", 12, 100, posterior_mean=0.4, prob_best=0.5),
    )
    assert _choose_leader(even) == "A"
    equal_mean = (
        ArmBestShare("M", 10, 100, posterior_mean=0.3, prob_best=0.5),
        ArmBestShare("Z", 10, 100, posterior_mean=0.3, prob_best=0.5),
    )
    assert _choose_leader(equal_mean) == "Z"


def test_rejects_fewer_than_two_arms():
    with pytest.raises(ValueError, match="at least two"):
        probability_of_being_best({"A": b_variant(5, 50)})


def test_rejects_empty_name():
    with pytest.raises(ValueError, match="arm name"):
        probability_of_being_best(
            {"A": b_variant(5, 50), "  ": b_variant(6, 50), "C": b_variant(7, 50)}
        )


def test_rejects_non_posterior_arm():
    with pytest.raises(ValueError, match="BetaBinomialPosterior"):
        probability_of_being_best({"A": b_variant(5, 50), "B": "nope", "C": b_variant(7, 50)})


def test_rejects_nonpositive_sample_count():
    arms = {"A": b_variant(5, 50), "B": b_variant(6, 50), "C": b_variant(7, 50)}
    with pytest.raises(ValueError, match="n_samples"):
        probability_of_being_best(arms, n_samples=0)
    with pytest.raises(ValueError, match="n_samples"):
        probability_of_being_best(arms, n_samples=-10)
