"""Posterior-sampling utilities built on seeded NumPy generators."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .posteriors import BetaBinomialPosterior


def make_rng(seed: int | None = 20260824) -> np.random.Generator:
    """Create a generator; a fixed seed makes downstream results reproducible."""
    if seed is not None and seed < 0:
        raise ValueError("seed must be non-negative or None")
    return np.random.default_rng(seed)


def posterior_draws(
    posterior: BetaBinomialPosterior,
    rng: np.random.Generator,
    size: int,
) -> np.ndarray:
    """Draw ``size`` parameter values from a Beta-Binomial posterior."""
    if size <= 0:
        raise ValueError("size must be positive")
    return posterior.sample(rng, size=size)


@dataclass(frozen=True)
class SampleSummary:
    """Descriptive statistics of a batch of posterior draws."""

    mean: float
    std: float
    lower: float
    upper: float
    ci: float
    n: int


def summarize_draws(draws: np.ndarray, ci: float = 0.95) -> SampleSummary:
    """Summarize Monte Carlo draws with mean, spread, and equal-tailed interval."""
    if draws.size == 0:
        raise ValueError("draws must be non-empty")
    if not 0 < ci < 1:
        raise ValueError("credible level must lie strictly inside (0, 1)")
    tail = (1.0 - ci) / 2.0
    lower, upper = np.quantile(draws, [tail, 1.0 - tail])
    return SampleSummary(
        mean=float(np.mean(draws)),
        std=float(np.std(draws, ddof=1)),
        lower=float(lower),
        upper=float(upper),
        ci=ci,
        n=int(draws.size),
    )