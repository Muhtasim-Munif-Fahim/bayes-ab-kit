"""Beta-Binomial posteriors for binary-outcome experiment metrics."""

from __future__ import annotations

import numpy as np
from scipy import stats


class BetaBinomialPosterior:
    """Posterior over a conversion probability from Bernoulli observations.

    Uses the Beta distribution as a conjugate prior. Starting from
    ``Beta(alpha0, beta0)``, observing ``successes`` conversions out of
    ``successes + failures`` visitors updates the posterior to
    ``Beta(alpha0 + successes, beta0 + failures)``.
    """

    def __init__(self, alpha0: float = 1.0, beta0: float = 1.0) -> None:
        if alpha0 <= 0 or beta0 <= 0:
            raise ValueError("prior hyperparameters must be positive")
        self.alpha0 = float(alpha0)
        self.beta0 = float(beta0)
        self.successes = 0
        self.failures = 0

    @classmethod
    def from_counts(
        cls,
        conversions: int,
        trials: int,
        alpha0: float = 1.0,
        beta0: float = 1.0,
    ) -> "BetaBinomialPosterior":
        """Build a posterior directly from observed counts."""
        if trials < 0:
            raise ValueError("trial count must be non-negative")
        if not 0 <= conversions <= trials:
            raise ValueError("conversion count must lie in [0, trials]")
        post = cls(alpha0=alpha0, beta0=beta0)
        post.update(successes=conversions, failures=trials - conversions)
        return post

    @property
    def alpha(self) -> float:
        return self.alpha0 + self.successes

    @property
    def beta(self) -> float:
        return self.beta0 + self.failures

    @property
    def trials(self) -> int:
        return self.successes + self.failures

    def update(self, successes: int, failures: int) -> "BetaBinomialPosterior":
        """Condition on an additional batch of observations."""
        if successes < 0 or failures < 0:
            raise ValueError("counts must be non-negative")
        self.successes += successes
        self.failures += failures
        return self

    def mean(self) -> float:
        return float(stats.beta.mean(self.alpha, self.beta))

    def variance(self) -> float:
        return float(stats.beta.var(self.alpha, self.beta))

    def mode(self) -> float | None:
        """Posterior mode; undefined while either parameter is <= 1."""
        if self.alpha > 1 and self.beta > 1:
            return (self.alpha - 1) / (self.alpha + self.beta - 2)
        return None

    def quantile(self, q: float) -> float:
        if not 0 <= q <= 1:
            raise ValueError("quantile level must lie in [0, 1]")
        return float(stats.beta.ppf(q, self.alpha, self.beta))

    def credible_interval(self, ci: float = 0.95) -> tuple[float, float]:
        """Equal-tailed credible interval with the given coverage."""
        if not 0 < ci < 1:
            raise ValueError("credible level must lie strictly inside (0, 1)")
        tail = (1.0 - ci) / 2.0
        lower = float(stats.beta.ppf(tail, self.alpha, self.beta))
        upper = float(stats.beta.ppf(1.0 - tail, self.alpha, self.beta))
        return lower, upper

    def pdf(self, x: float | np.ndarray) -> float | np.ndarray:
        return stats.beta.pdf(x, self.alpha, self.beta)

    def cdf(self, x: float | np.ndarray) -> float | np.ndarray:
        return stats.beta.cdf(x, self.alpha, self.beta)

    def prob_above(self, value: float) -> float:
        """Posterior probability that the conversion rate exceeds ``value``."""
        if value <= 0:
            return 1.0
        if value >= 1:
            return 0.0
        return float(1.0 - stats.beta.cdf(value, self.alpha, self.beta))

    def sample(self, rng: np.random.Generator, size: int | tuple[int, ...]) -> np.ndarray:
        """Draw samples using the supplied generator (reproducible via seed)."""
        return rng.beta(self.alpha, self.beta, size=size)

    def __repr__(self) -> str:
        return (
            f"BetaBinomialPosterior(alpha={self.alpha:.4g}, beta={self.beta:.4g}, "
            f"trials={self.trials})"
        )
