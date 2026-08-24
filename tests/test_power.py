import numpy as np
import pytest

from bayes_ab_kit.power import SampleSizePlan, power_curve, required_sample_size, simulate_power


def test_plan_matches_reference_computation():
    plan = required_sample_size(0.10, 0.12)
    from scipy import stats as st

    p_bar = 0.11
    expected = (
        st.norm.ppf(0.975) * np.sqrt(2 * p_bar * (1 - p_bar))
        + st.norm.ppf(0.80) * np.sqrt(0.10 * 0.90 + 0.12 * 0.88)
    ) ** 2 / (0.02**2)
    assert plan.n_per_arm == int(np.ceil(expected))


def test_smaller_effect_needs_more_visitors():
    small = required_sample_size(0.10, 0.11).n_per_arm
    large = required_sample_size(0.10, 0.20).n_per_arm
    assert small > large


def test_higher_power_needs_more_visitors():
    low = required_sample_size(0.10, 0.15, power=0.70).n_per_arm
    high = required_sample_size(0.10, 0.15, power=0.95).n_per_arm
    assert high > low


def test_one_sided_requires_fewer_than_two_sided():
    one = required_sample_size(0.10, 0.14, alternative="one-sided").n_per_arm
    two = required_sample_size(0.10, 0.14, alternative="two-sided").n_per_arm
    assert one < two


def test_result_is_ceiled_integer_with_fields():
    plan = required_sample_size(0.05, 0.06, alpha=0.01, power=0.9)
    assert isinstance(plan, SampleSizePlan)
    assert float(plan.n_per_arm).is_integer() and plan.n_per_arm > 0
    assert plan.alpha == pytest.approx(0.01)
    assert plan.alternative == "two-sided"


def test_simulated_power_lands_near_target_at_planned_n():
    plan = required_sample_size(0.12, 0.16, power=0.80)
    empirical = simulate_power(
        0.12, 0.16, n_per_arm=plan.n_per_arm, n_sims=8000, seed=101
    )
    assert 0.70 < empirical < 0.89


def test_power_increases_with_n_and_zero_effect_stays_at_alpha():
    weak_n = simulate_power(0.10, 0.15, n_per_arm=400, n_sims=6000, seed=7)
    strong_n = simulate_power(0.10, 0.15, n_per_arm=1600, n_sims=6000, seed=8)
    assert strong_n > weak_n
    null_power = simulate_power(0.10, 0.10, n_per_arm=1000, n_sims=8000, seed=9)
    assert abs(null_power - 0.05) < 0.015


def test_curve_is_monotone_non_decreasing():
    curve = power_curve(0.08, 0.13, ns_per_arm=(300, 700, 1400), n_sims=2500, seed=12)
    powers = [p for _, p in curve]
    assert powers[0] <= powers[1] <= powers[2]
    assert all(n == int(n) for n, _ in curve)


def test_simulation_deterministic_under_seed():
    a = simulate_power(0.1, 0.14, n_per_arm=900, n_sims=2000, seed=3)
    b = simulate_power(0.1, 0.14, n_per_arm=900, n_sims=2000, seed=3)
    assert a == b


def test_invalid_inputs_rejected():
    with pytest.raises(ValueError):
        required_sample_size(1.5, 0.2)
    with pytest.raises(ValueError):
        required_sample_size(0.1, 0.1)
    with pytest.raises(ValueError):
        required_sample_size(0.1, 0.2, alpha=0)
    with pytest.raises(ValueError):
        required_sample_size(0.1, 0.2, alternative="both")
    with pytest.raises(ValueError):
        simulate_power(0.1, 0.2, n_per_arm=0)
    with pytest.raises(ValueError):
        simulate_power(-0.1, 0.2, n_per_arm=100)


def test_extreme_rates_rejected_gracefully():
    for bad in ((0.0, 0.2), (0.2, 0.0), (1.0, 0.5), (0.5, 1.0)):
        with pytest.raises(ValueError):
            required_sample_size(*bad)


def test_numpy_rates_accepted_and_converted():
    plan = required_sample_size(np.float64(0.10), np.float64(0.125))
    assert plan.n_per_arm > 0