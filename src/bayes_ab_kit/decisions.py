"""Superiority decision rules comparing two conversion-rate posteriors.

ROPE (practical-equivalence) rules live in :mod:`bayes_ab_kit.rope`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy import integrate, stats

from .posteriors import BetaBinomialPosterior


class Decision(str, Enum):
    """Outcome of a two-variant comparison."""

    SHIP_A = "ship_a"
    SHIP_B = "ship_b"
    PRACTICAL_EQUIVALENCE = "practical_equivalence"
    KEEP_RUNNING = "keep_running"


@dataclass(frozen=True)
class ComparisonResult:
    """Full output of a credibility-based variant comparison."""

    prob_b_beats_a: float
    diff_ci_lower: float
    diff_ci_upper: float
    ci: float
    decision: Decision


def _validate_level(ci: float) -> None:
    if not 0 < ci < 1:
        raise ValueError("credible level must lie strictly inside (0, 1)")


def prob_b_beats_a_quad(
    posterior_a: BetaBinomialPosterior,
    posterior_b: BetaBinomialPosterior,
) -> float:
    """P(rate_B > rate_A) by adaptive quadrature (no sampling noise)."""
    def integrand(p: np.ndarray) -> np.ndarray:
        return stats.beta.cdf(p, posterior_a.alpha, posterior_a.beta) * stats.beta.pdf(
            p, posterior_b.alpha, posterior_b.beta
        )

    value, _ = integrate.quad(integrand, 0.0, 1.0)
    return float(value)


def prob_b_beats_a_mc(
    posterior_a: BetaBinomialPosterior,
    posterior_b: BetaBinomialPosterior,
    n_samples: int = 100_000,
    seed: int | None = 20260824,
) -> float:
    """P(rate_B > rate_A) by Monte Carlo with a seeded generator."""
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    rng = np.random.default_rng(seed)
    draws_a = rng.beta(posterior_a.alpha, posterior_a.beta, size=n_samples)
    draws_b = rng.beta(posterior_b.alpha, posterior_b.beta, size=n_samples)
    return float(np.mean(draws_b > draws_a))


def difference_credible_interval(
    posterior_a: BetaBinomialPosterior,
    posterior_b: BetaBinomialPosterior,
    ci: float = 0.95,
    n_samples: int = 100_000,
    seed: int | None = 20260824,
) -> tuple[float, float]:
    """Equal-tailed credible interval for ``rate_B - rate_A`` via Monte Carlo."""
    _validate_level(ci)
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    rng = np.random.default_rng(seed)
    draws_a = rng.beta(posterior_a.alpha, posterior_a.beta, size=n_samples)
    draws_b = rng.beta(posterior_b.alpha, posterior_b.beta, size=n_samples)
    diff = draws_b - draws_a
    tail = (1.0 - ci) / 2.0
    lower, upper = np.quantile(diff, [tail, 1.0 - tail])
    return float(lower), float(upper)


def superiority_decision(
    posterior_a: BetaBinomialPosterior,
    posterior_b: BetaBinomialPosterior,
    ci: float = 0.95,
    n_samples: int = 100_000,
    seed: int | None = 20260824,
) -> ComparisonResult:
    """Decide whether one variant beats the other at a credible level.

    Ships the losing variant only when the whole credible interval for the
    rate difference lies on its side of zero; otherwise recommends continuing.
    """
    _validate_level(ci)
    lower, upper = difference_credible_interval(
        posterior_a, posterior_b, ci=ci, n_samples=n_samples, seed=seed
    )
    prob = prob_b_beats_a_quad(posterior_a, posterior_b)
    if lower > 0:
        decision = Decision.SHIP_B
    elif upper < 0:
        decision = Decision.SHIP_A
    else:
        decision = Decision.KEEP_RUNNING
    return ComparisonResult(
        prob_b_beats_a=prob,
        diff_ci_lower=lower,
        diff_ci_upper=upper,
        ci=ci,
        decision=decision,
    )