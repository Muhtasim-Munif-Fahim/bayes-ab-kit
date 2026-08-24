import numpy as np
import pytest

from bayes_ab_kit.sequential import (
    PeekSimulationResult,
    fixed_horizon_false_stop_rate,
    simulate_null_peeking,
)


def test_fixed_horizon_size_close_to_nominal_alpha():
    rate = fixed_horizon_false_stop_rate(
        p_true=0.10, total_n_per_arm=800, n_sims=6000, seed=7
    )
    assert abs(rate - 0.05) < 0.012


def test_peeking_inflates_false_positive_rate():
    peeked = simulate_null_peeking(
        p_true=0.10, per_look_n=200, n_looks=5, n_sims=4000, seed=11
    ).false_stop_rate
    fixed = fixed_horizon_false_stop_rate(
        p_true=0.10, total_n_per_arm=1000, n_sims=4000, seed=12
    )
    assert peeked > fixed


def test_more_looks_increase_risk_monotonically():
    rates = [
        simulate_null_peeking(p_true=0.08, per_look_n=250, n_looks=k, n_sims=3000, seed=21).false_stop_rate
        for k in (2, 4, 8)
    ]
    assert rates[0] < rates[1] < rates[2]


def test_result_fields_are_consistent():
    res = simulate_null_peeking(p_true=0.05, per_look_n=150, n_looks=6, n_sims=1500, seed=31)
    assert isinstance(res, PeekSimulationResult)
    assert len(res.histogram) == 6
    assert sum(res.histogram) == res.false_stops
    assert res.false_stops <= res.n_sims
    if res.false_stops:
        assert 1 <= res.mean_first_significant_look <= 6
    else:
        assert res.mean_first_significant_look is None


def test_simulation_is_deterministic_under_seed():
    a = simulate_null_peeking(0.1, 100, 4, n_sims=500, seed=99)
    b = simulate_null_peeking(0.1, 100, 4, n_sims=500, seed=99)
    assert a == b


def test_histogram_counts_match_first_look_behaviour():
    res = simulate_null_peeking(p_true=0.5, per_look_n=400, n_looks=3, n_sims=1200, seed=5)
    assert all(h >= 0 for h in res.histogram)


def test_invalid_inputs_rejected():
    with pytest.raises(ValueError):
        simulate_null_peeking(p_true=0.0, per_look_n=100, n_looks=3)
    with pytest.raises(ValueError):
        simulate_null_peeking(p_true=1.0, per_look_n=100, n_looks=3)
    with pytest.raises(ValueError):
        simulate_null_peeking(p_true=0.1, per_look_n=0, n_looks=3)
    with pytest.raises(ValueError):
        simulate_null_peeking(p_true=0.1, per_look_n=50, n_looks=2, alpha=1.5)
    with pytest.raises(ValueError):
        simulate_null_peeking(p_true=0.1, per_look_n=50, n_looks=0)


def test_alpha_zero_prevents_all_stops():
    res = simulate_null_peeking(
        p_true=0.1, per_look_n=200, n_looks=5, n_sims=400, alpha=1e-12, seed=3
    )
    assert res.false_stops == 0
    assert res.false_stop_rate == pytest.approx(0.0)


def test_rate_is_fraction_of_sims():
    res = simulate_null_peeking(p_true=0.2, per_look_n=180, n_looks=4, n_sims=2500, seed=44)
    assert np.isclose(res.false_stop_rate, res.false_stops / 2500)