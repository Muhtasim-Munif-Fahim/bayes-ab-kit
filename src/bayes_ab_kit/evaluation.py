"""Joint evaluation of conversion and revenue per visitor (ARPU)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .decisions import Decision
from .posteriors import BetaBinomialPosterior
from .revenue import LognormalRevenueModel


@dataclass(frozen=True)
class VariantData:
    """Observed experiment data for one variant."""

    name: str
    conversions: int
    trials: int
    order_values: np.ndarray

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("variant name must be non-empty")


@dataclass(frozen=True)
class ArpuSummary:
    """Posterior summary of a single variant's expected revenue per visitor."""

    name: str
    mean: float
    lower: float
    upper: float


@dataclass(frozen=True)
class ArpuEvaluation:
    """Two-variant ARPU comparison with a credibility-based decision."""

    summary_a: ArpuSummary
    summary_b: ArpuSummary
    prob_b_beats_a: float
    diff_lower: float
    diff_upper: float
    ci: float
    decision: Decision


def _validate_ci(ci: float) -> None:
    if not 0 < ci < 1:
        raise ValueError("credible level must lie strictly inside (0, 1)")


def arpu_draws(
    variant: VariantData,
    rng: np.random.Generator,
    n_samples: int,
) -> np.ndarray:
    """Monte Carlo draws of revenue per visitor for one variant."""
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if variant.order_values.size == 0:
        raise ValueError(f"variant {variant.name!r} has no order values")
    rate_draws = BetaBinomialPosterior.from_counts(
        variant.conversions, variant.trials
    ).sample(rng, size=n_samples)
    revenue_model = LognormalRevenueModel.from_revenues(variant.order_values)
    value_draws = revenue_model.expected_value_draws(rng, size=n_samples)
    return rate_draws * value_draws


def evaluate_variants(
    variant_a: VariantData,
    variant_b: VariantData,
    ci: float = 0.95,
    n_samples: int = 50_000,
    seed: int | None = 20260824,
) -> ArpuEvaluation:
    """Compare two variants on expected revenue per visitor."""
    _validate_ci(ci)
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    rng = np.random.default_rng(seed)
    draws_a = arpu_draws(variant_a, rng, n_samples)
    draws_b = arpu_draws(variant_b, rng, n_samples)

    def summary(draws: np.ndarray, name: str) -> ArpuSummary:
        tail = (1.0 - ci) / 2.0
        lo, hi = np.quantile(draws, [tail, 1.0 - tail])
        return ArpuSummary(name=name, mean=float(np.mean(draws)), lower=float(lo), upper=float(hi))

    summary_a = summary(draws_a, variant_a.name)
    summary_b = summary(draws_b, variant_b.name)

    diff = draws_b - draws_a
    tail = (1.0 - ci) / 2.0
    diff_lo, diff_hi = np.quantile(diff, [tail, 1.0 - tail])
    prob = float(np.mean(diff > 0))

    if diff_lo > 0:
        decision = Decision.SHIP_B
    elif diff_hi < 0:
        decision = Decision.SHIP_A
    else:
        decision = Decision.KEEP_RUNNING

    return ArpuEvaluation(
        summary_a=summary_a,
        summary_b=summary_b,
        prob_b_beats_a=prob,
        diff_lower=float(diff_lo),
        diff_upper=float(diff_hi),
        ci=ci,
        decision=decision,
    )