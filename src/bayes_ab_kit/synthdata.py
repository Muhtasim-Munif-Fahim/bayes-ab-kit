"""Synthetic experiment data for demos and method validation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VariantSpec:
    """Ground-truth parameters used to fabricate one variant's traffic."""

    name: str
    visitors: int
    conversion_rate: float
    log_revenue_mean: float = 3.0
    log_revenue_sigma: float = 0.7

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("variant name must be non-empty")
        if self.visitors < 0:
            raise ValueError("visitor count must be non-negative")
        if not 0 <= self.conversion_rate <= 1:
            raise ValueError("conversion_rate must lie inside [0, 1]")
        if self.log_revenue_sigma <= 0:
            raise ValueError("log_revenue_sigma must be positive")


@dataclass(frozen=True)
class VariantObservations:
    """Simulated observations for one variant."""

    name: str
    conversions: int
    trials: int
    order_values: np.ndarray


def generate_variant(
    spec: VariantSpec,
    rng: np.random.Generator,
) -> VariantObservations:
    """Simulate one variant: binomial conversions plus lognormal order values."""
    conversions = int(rng.binomial(spec.visitors, spec.conversion_rate))
    orders = (
        rng.lognormal(
            mean=spec.log_revenue_mean,
            sigma=spec.log_revenue_sigma,
            size=conversions,
        )
        if conversions
        else np.empty(0)
    )
    return VariantObservations(
        name=spec.name,
        conversions=conversions,
        trials=spec.visitors,
        order_values=orders,
    )


def generate_experiment(
    specs: list[VariantSpec],
    seed: int | None = 20260824,
) -> list[VariantObservations]:
    """Simulate every arm of an experiment from one seeded generator."""
    if not specs:
        raise ValueError("at least one variant specification is required")
    names = [s.name for s in specs]
    if len(set(names)) != len(names):
        raise ValueError("variant names must be unique")
    rng = np.random.default_rng(seed)
    return [generate_variant(s, rng) for s in specs]


def aa_pair_specs(
    visitors: int,
    conversion_rate: float,
    log_revenue_mean: float = 3.0,
    log_revenue_sigma: float = 0.7,
) -> tuple[VariantSpec, VariantSpec]:
    """Two identical arms, useful for validating analysis pipelines."""
    return (
        VariantSpec("A", visitors, conversion_rate, log_revenue_mean, log_revenue_sigma),
        VariantSpec("B", visitors, conversion_rate, log_revenue_mean, log_revenue_sigma),
    )


def uplift_pair_specs(
    visitors_per_arm: int,
    baseline_rate: float,
    relative_uplift: float,
    log_revenue_mean_a: float = 3.0,
    log_revenue_uplift: float = 0.0,
    log_revenue_sigma: float = 0.7,
) -> tuple[VariantSpec, VariantSpec]:
    """Control plus a treatment with multiplicative conversion uplift."""
    return (
        VariantSpec("A", visitors_per_arm, baseline_rate, log_revenue_mean_a, log_revenue_sigma),
        VariantSpec(
            "B",
            visitors_per_arm,
            min(baseline_rate * (1.0 + relative_uplift), 1.0),
            log_revenue_mean_a + log_revenue_uplift,
            log_revenue_sigma,
        ),
    )


__all__ = [
    "VariantObservations",
    "VariantSpec",
    "aa_pair_specs",
    "generate_experiment",
    "generate_variant",
    "uplift_pair_specs",
]