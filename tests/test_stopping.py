import numpy as np
import pytest

from bayes_ab_kit.stopping import (
    StoppingPlanResult,
    required_max_looks_for_power,
    simulate_stopping_plan,
)


def test_strong_effect_stops_early_and_saves_sample():
    res = simulate_stopping_plan(
        p_a=0.10, p_b=0.20, per_look_n=300, max_looks=10, n_sims=1500, seed=8
    )
    assert res.stop_rate > 0.9
    assert res.expected_sample_size_per_arm < 10 * 300 * 0.6
    assert res.mean_stop_look is not None and res.mean_stop_look < 10


def test_null_plan_stops_rarely_with_high_threshold():
    res = simulate_stopping_plan(
        p_a=0.12, p_b=0.12, per_look_n=250, max_looks=8, threshold=0.995, n_sims=1200, seed=15
    )
    assert res.stop_rate < 0.10


def test_expected_size_grows_with_smaller_effect():
    big = simulate_stopping_plan(0.10, 0.22, per_look_n=200, max_looks=12, n_sims=900, seed=21)
    small = simulate_stopping_plan(0.10, 0.13, per_look_n=200, max_looks=12, n_sims=900, seed=22)
    assert small.expected_sample_size_per_arm > big.expected_sample_size_per_arm


def test_unstoppable_runs_count_full_horizon():
    res = simulate_stopping_plan(
        p_a=0.30, p_b=0.31, per_look_n=100, max_looks=4, threshold=0.999, n_sims=400, seed=3
    )
    floor = res.per_look_n
    ceiling = res.max_looks * res.per_look_n
    assert floor <= res.expected_sample_size_per_arm <= ceiling


def test_result_is_deterministic_under_seed():
    a = simulate_stopping_plan(0.1, 0.18, per_look_n=150, max_looks=6, n_sims=500, seed=77)
    b = simulate_stopping_plan(0.1, 0.18, per_look_n=150, max_looks=6, n_sims=500, seed=77)
    assert a == b


def test_result_fields_consistent():
    res = simulate_stopping_plan(0.05, 0.09, per_look_n=120, max_looks=7, n_sims=800, seed=41)
    assert isinstance(res, StoppingPlanResult)
    assert 0 <= res.stopped <= res.n_sims
    assert np.isclose(res.stop_rate, res.stopped / res.n_sims)
    if res.stopped == 0:
        assert res.mean_stop_look is None
        assert res.expected_sample_size_per_arm == pytest.approx(7 * 120)


def test_power_search_finds_feasible_plan():
    looks = required_max_looks_for_power(
        p_a=0.10,
        p_b=0.25,
        target_power=0.9,
        per_look_n=200,
        max_search_looks=25,
        n_sims=600,
        seed=5,
    )
    assert looks is not None and 1 <= looks <= 25


def test_power_search_returns_none_when_unreachable():
    looks = required_max_looks_for_power(
        p_a=0.499,
        p_b=0.501,
        target_power=0.95,
        per_look_n=50,
        max_search_looks=3,
        n_sims=300,
        seed=6,
    )
    assert looks is None


def test_invalid_inputs_rejected():
    with pytest.raises(ValueError):
        simulate_stopping_plan(p_a=0.0, p_b=0.2, per_look_n=100, max_looks=3)
    with pytest.raises(ValueError):
        simulate_stopping_plan(p_a=0.1, p_b=1.2, per_look_n=100, max_looks=3)
    with pytest.raises(ValueError):
        simulate_stopping_plan(p_a=0.1, p_b=0.2, per_look_n=-5, max_looks=3)
    with pytest.raises(ValueError):
        simulate_stopping_plan(p_a=0.1, p_b=0.2, per_look_n=100, max_looks=3, threshold=0.4)
    with pytest.raises(ValueError):
        required_max_looks_for_power(0.1, 0.2, target_power=0)