import numpy as np
import pytest

from bayes_ab_kit.synthdata import (
    VariantObservations,
    VariantSpec,
    aa_pair_specs,
    generate_experiment,
    generate_variant,
    uplift_pair_specs,
)


def test_counts_respect_traffic_and_order_length():
    spec = VariantSpec("A", visitors=1000, conversion_rate=0.2)
    obs = generate_variant(spec, np.random.default_rng(1))
    assert isinstance(obs, VariantObservations)
    assert 0 <= obs.conversions <= 1000
    assert obs.trials == 1000
    assert len(obs.order_values) == obs.conversions


def test_orders_are_positive_when_conversions_exist():
    spec = VariantSpec("A", visitors=800, conversion_rate=0.5)
    obs = generate_variant(spec, np.random.default_rng(2))
    assert np.all(obs.order_values > 0)


def test_zero_rate_gives_no_conversions_or_orders():
    spec = VariantSpec("A", visitors=500, conversion_rate=0.0)
    obs = generate_variant(spec, np.random.default_rng(3))
    assert obs.conversions == 0
    assert obs.order_values.size == 0


def test_experiment_is_reproducible_by_seed():
    specs = list(uplift_pair_specs(2000, 0.10, 0.3))
    run_a = generate_experiment(specs, seed=42)
    run_b = generate_experiment(specs, seed=42)
    for x, y in zip(run_a, run_b):
        assert x.conversions == y.conversions
        assert np.array_equal(x.order_values, y.order_values)


def test_different_seeds_diverge():
    specs = list(uplift_pair_specs(2000, 0.10, 0.3))
    run_a = generate_experiment(specs, seed=1)
    run_b = generate_experiment(specs, seed=2)
    assert run_a[0].conversions != run_b[0].conversions or not np.array_equal(
        run_a[0].order_values, run_b[0].order_values
    )


def test_observed_rates_cluster_around_truth():
    specs = [VariantSpec("A", visitors=200_000, conversion_rate=0.15)]
    (obs,) = generate_experiment(specs, seed=9)
    assert obs.conversions / obs.trials == pytest.approx(0.15, abs=0.01)


def test_uplift_pair_raises_treatment_rate():
    a_spec, b_spec = uplift_pair_specs(10_000, 0.10, 0.5, log_revenue_uplift=0.2)
    assert b_spec.conversion_rate == pytest.approx(0.15)
    assert b_spec.log_revenue_mean == pytest.approx(a_spec.log_revenue_mean + 0.2)
    assert a_spec.conversion_rate == pytest.approx(0.10)


def test_aa_pair_specs_are_identical():
    a_spec, b_spec = aa_pair_specs(3000, 0.08)
    assert a_spec.name != b_spec.name
    assert (a_spec.visitors, a_spec.conversion_rate) == (
        b_spec.visitors,
        b_spec.conversion_rate,
    )
    assert a_spec.log_revenue_mean == b_spec.log_revenue_mean


def test_duplicate_names_rejected():
    dup = [VariantSpec("A", 100, 0.1), VariantSpec("A", 100, 0.1)]
    with pytest.raises(ValueError):
        generate_experiment(dup)


def test_empty_spec_list_rejected():
    with pytest.raises(ValueError):
        generate_experiment([])


def test_invalid_spec_parameters_rejected():
    with pytest.raises(ValueError):
        generate_variant(VariantSpec("A", -5, 0.1), np.random.default_rng(0))
    with pytest.raises(ValueError):
        generate_variant(VariantSpec("A", 100, 1.5), np.random.default_rng(0))
    with pytest.raises(ValueError):
        VariantSpec("A", 100, 0.1, log_revenue_sigma=0.0)


def test_blank_name_rejected():
    with pytest.raises(ValueError):
        VariantSpec("", 100, 0.1)