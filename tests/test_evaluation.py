import numpy as np
import pytest

from bayes_ab_kit.decisions import Decision
from bayes_ab_kit.evaluation import VariantData, arpu_draws, evaluate_variants


def make_variant(name, conversions, trials, seed, n_orders=400, mu=3.0):
    orders = np.random.default_rng(seed).lognormal(mean=mu, sigma=0.6, size=n_orders)
    return VariantData(name=name, conversions=conversions, trials=trials, order_values=orders)


def test_identical_variants_stay_inconclusive():
    a = make_variant("A", 100, 2000, seed=1)
    b = make_variant("B", 100, 2000, seed=2)
    result = evaluate_variants(a, b, n_samples=20_000)
    assert 0.2 < result.prob_b_beats_a < 0.8
    assert result.decision is Decision.KEEP_RUNNING


def test_better_rate_and_revenue_ship_b():
    a = make_variant("A", 60, 2000, seed=5, mu=2.8)
    b = make_variant("B", 140, 2000, seed=6, mu=3.4)
    result = evaluate_variants(a, b, n_samples=30_000)
    assert result.decision is Decision.SHIP_B
    assert result.diff_lower > 0
    assert result.prob_b_beats_a > 0.95


def test_worse_b_ships_a():
    a = make_variant("A", 150, 2000, seed=9, mu=3.4)
    b = make_variant("B", 55, 2000, seed=10, mu=2.7)
    result = evaluate_variants(a, b, n_samples=30_000)
    assert result.decision is Decision.SHIP_A
    assert result.diff_upper < 0


def test_draws_positive_deterministic_and_centered():
    v = make_variant("A", 120, 2400, seed=17)
    d1 = arpu_draws(v, np.random.default_rng(42), n_samples=5000)
    d2 = arpu_draws(v, np.random.default_rng(42), n_samples=5000)
    assert np.array_equal(d1, d2)
    assert np.all(d1 > 0)


def test_summary_interval_brackets_mean():
    a = make_variant("A", 90, 1800, seed=23)
    b = make_variant("B", 96, 1800, seed=24)
    result = evaluate_variants(a, b, ci=0.9, n_samples=20_000)
    for s in (result.summary_a, result.summary_b):
        assert s.lower < s.mean < s.upper


def test_arpu_scales_with_order_size():
    small_orders = make_variant("A", 100, 2000, seed=31, mu=2.5)
    big_orders = make_variant("B", 100, 2000, seed=32, mu=4.5)
    result = evaluate_variants(small_orders, big_orders, n_samples=20_000)
    assert result.summary_b.mean > result.summary_a.mean * 3


def test_zero_conversions_yield_tiny_but_valid_draws():
    v = VariantData(
        name="A",
        conversions=0,
        trials=500,
        order_values=np.random.default_rng(3).lognormal(size=50),
    )
    draws = arpu_draws(v, np.random.default_rng(0), n_samples=3000)
    assert np.all(draws >= 0)
    assert draws.mean() < 0.01


def test_invalid_credible_level_rejected():
    a = make_variant("A", 10, 200, seed=1)
    b = make_variant("B", 11, 200, seed=2)
    with pytest.raises(ValueError):
        evaluate_variants(a, b, ci=1.2)


def test_empty_orders_rejected():
    bad = VariantData(name="X", conversions=1, trials=10, order_values=np.array([]))
    ok = make_variant("Y", 2, 10, seed=4)
    with pytest.raises(ValueError):
        arpu_draws(bad, np.random.default_rng(0), n_samples=100)
    with pytest.raises(ValueError):
        evaluate_variants(bad, ok, n_samples=1000)


def test_blank_name_rejected():
    with pytest.raises(ValueError):
        VariantData(name="", conversions=1, trials=2, order_values=np.ones(3))


def test_nonpositive_sample_count_rejected():
    a = make_variant("A", 10, 200, seed=1)
    b = make_variant("B", 12, 200, seed=2)
    with pytest.raises(ValueError):
        evaluate_variants(a, b, n_samples=0)
def test_zero_conversion_variant_with_orders_is_valid_but_rare():
    v = VariantData(
        name="A",
        conversions=0,
        trials=200,
        order_values=np.array([5.0]),
    )
    draws = arpu_draws(v, np.random.default_rng(1), n_samples=1000)
    assert draws.mean() < 0.15


def test_single_order_value_supported_end_to_end():
    a = make_variant("A", 80, 1000, seed=51)
    b = VariantData("B", 90, 1000, order_values=np.full(3, 42.0))
    result = evaluate_variants(a, b, n_samples=5000, sigma_log=0.6)
    assert result.summary_b.lower >= 0
def test_zero_spread_orders_surface_clear_error_without_sigma():
    a = make_variant("A", 80, 1000, seed=52)
    b = VariantData("B", 90, 1000, order_values=np.full(4, 30.0))
    with pytest.raises(ValueError, match="no dispersion"):
        evaluate_variants(a, b, n_samples=2000)