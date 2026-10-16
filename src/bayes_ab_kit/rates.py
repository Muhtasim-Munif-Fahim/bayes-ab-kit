"""Gamma-Poisson posteriors for count / rate metrics (events per exposure).

Many experiment metrics are counts accumulated over an exposure: purchases per
user-day, crashes per 1,000 sessions, support tickets per account-month. The
conjugate model for such data is

* likelihood: ``events ~ Poisson(rate * exposure)``
* prior:      ``rate ~ Gamma(shape0, rate0)``
* posterior:  ``rate ~ Gamma(shape0 + events, rate0 + exposure)``

Two independent Gamma posteriors admit closed forms for the quantities a
decision needs:

* ``P(rate_B > rate_A)``: with ``G_a ~ Gamma(alpha_a, 1)`` and
  ``G_b ~ Gamma(alpha_b, 1)``, ``U = G_b / (G_a + G_b) ~ Beta(alpha_b,
  alpha_a)``. Since ``rate_B > rate_A`` iff ``U > c / (1 + c)`` with
  ``c = beta_b / beta_a``, the probability is a regularized incomplete beta.
* The rate ratio ``rate_B / rate_A = (beta_a / beta_b) * U / (1 - U)`` is a
  scaled beta-prime variable, so its credible interval comes from Beta
  quantiles without simulation.

Expected loss (the expected shortfall of picking the wrong arm, in rate units)
is estimated by Monte Carlo from seeded posterior draws. ``higher_is_better``
flips the direction for "bad" counts such as errors or churn events.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import special, stats

from .sampling import make_rng


class GammaPoissonPosterior:
    """Posterior over a Poisson event rate from ``(events, exposure)`` data."""

    def __init__(self, shape0: float = 0.5, rate0: float = 1e-6) -> None:
        if shape0 <= 0 or rate0 <= 0:
            raise ValueError("prior hyperparameters must be positive")
        self.shape0 = float(shape0)
        self.rate0 = float(rate0)
        self.events = 0
        self.exposure = 0.0

    @classmethod
    def from_counts(
        cls,
        events: int,
        exposure: float,
        shape0: float = 0.5,
        rate0: float = 1e-6,
    ) -> "GammaPoissonPosterior":
        """Build a posterior directly from observed totals."""
        post = cls(shape0=shape0, rate0=rate0)
        post.update(events=events, exposure=exposure)
        return post

    @property
    def shape(self) -> float:
        return self.shape0 + self.events

    @property
    def rate(self) -> float:
        return self.rate0 + self.exposure

    def update(self, events: int, exposure: float) -> "GammaPoissonPosterior":
        """Condition on an additional batch of events over ``exposure``."""
        if isinstance(events, bool) or int(events) != events or events < 0:
            raise ValueError("events must be a non-negative integer")
        if not np.isfinite(exposure) or exposure < 0:
            raise ValueError("exposure must be a non-negative finite number")
        if events > 0 and exposure == 0:
            raise ValueError("events require positive exposure")
        self.events += int(events)
        self.exposure += float(exposure)
        return self

    def mean(self) -> float:
        return self.shape / self.rate

    def var(self) -> float:
        return self.shape / self.rate**2

    def credible_interval(self, ci: float = 0.95) -> tuple[float, float]:
        if not 0 < ci < 1:
            raise ValueError("ci must lie in (0, 1)")
        tail = (1.0 - ci) / 2.0
        dist = stats.gamma(a=self.shape, scale=1.0 / self.rate)
        return float(dist.ppf(tail)), float(dist.ppf(1.0 - tail))

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        if size <= 0:
            raise ValueError("size must be positive")
        return rng.gamma(self.shape, 1.0 / self.rate, size=size)

    def predictive_events(self, exposure: float) -> stats.rv_discrete:
        """Posterior predictive count over a new ``exposure`` (negative binomial)."""
        if exposure <= 0:
            raise ValueError("exposure must be positive")
        p = self.rate / (self.rate + exposure)
        return stats.nbinom(self.shape, p)


def prob_rate_b_exceeds_a(post_a: GammaPoissonPosterior, post_b: GammaPoissonPosterior) -> float:
    """Exact ``P(rate_B > rate_A)`` for independent Gamma posteriors."""
    c = post_b.rate / post_a.rate
    threshold = c / (1.0 + c)
    return float(special.betaincc(post_b.shape, post_a.shape, threshold))


def rate_ratio_interval(
    post_a: GammaPoissonPosterior,
    post_b: GammaPoissonPosterior,
    ci: float = 0.95,
) -> tuple[float, float, float]:
    """Exact ``(lower, median, upper)`` of ``rate_B / rate_A``."""
    if not 0 < ci < 1:
        raise ValueError("ci must lie in (0, 1)")
    tail = (1.0 - ci) / 2.0
    scale = post_a.rate / post_b.rate
    u = stats.beta(post_b.shape, post_a.shape).ppf([tail, 0.5, 1.0 - tail])
    ratio = scale * u / (1.0 - u)
    return float(ratio[0]), float(ratio[1]), float(ratio[2])


@dataclass(frozen=True)
class RateComparisonResult:
    """Summary of a two-arm Gamma-Poisson rate comparison."""

    events_a: int
    exposure_a: float
    events_b: int
    exposure_b: float
    rate_a_mean: float
    rate_b_mean: float
    rate_a_interval: tuple[float, float]
    rate_b_interval: tuple[float, float]
    prob_b_better: float
    rate_ratio_median: float
    rate_ratio_interval: tuple[float, float]
    expected_loss_choose_a: float
    expected_loss_choose_b: float
    higher_is_better: bool
    ci: float
    loss_threshold: float | None

    @property
    def winner(self) -> str | None:
        """Arm whose expected loss is under ``loss_threshold``, if any."""
        if self.loss_threshold is None:
            return None
        best = "B" if self.expected_loss_choose_b <= self.expected_loss_choose_a else "A"
        loss = self.expected_loss_choose_b if best == "B" else self.expected_loss_choose_a
        return best if loss <= self.loss_threshold else None


def compare_rates(
    events_a: int,
    exposure_a: float,
    events_b: int,
    exposure_b: float,
    *,
    shape0: float = 0.5,
    rate0: float = 1e-6,
    higher_is_better: bool = True,
    ci: float = 0.95,
    loss_threshold: float | None = None,
    n_samples: int = 50_000,
    seed: int | None = 20261016,
) -> RateComparisonResult:
    """Compare event rates of arms A and B under independent Gamma priors.

    ``prob_b_better`` is ``P(rate_B > rate_A)`` when ``higher_is_better`` and
    ``P(rate_B < rate_A)`` otherwise. Expected losses are in rate units
    (events per unit exposure) and use the same orientation.
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if loss_threshold is not None and loss_threshold < 0:
        raise ValueError("loss_threshold must be non-negative")
    post_a = GammaPoissonPosterior.from_counts(events_a, exposure_a, shape0, rate0)
    post_b = GammaPoissonPosterior.from_counts(events_b, exposure_b, shape0, rate0)
    if post_a.exposure == 0 or post_b.exposure == 0:
        raise ValueError("each arm needs positive exposure")

    p_b_higher = prob_rate_b_exceeds_a(post_a, post_b)
    prob_b_better = p_b_higher if higher_is_better else 1.0 - p_b_higher
    lo, med, hi = rate_ratio_interval(post_a, post_b, ci)

    rng = make_rng(seed)
    draws_a = post_a.sample(rng, n_samples)
    draws_b = post_b.sample(rng, n_samples)
    diff = draws_b - draws_a if higher_is_better else draws_a - draws_b
    loss_b = float(np.mean(np.maximum(-diff, 0.0)))
    loss_a = float(np.mean(np.maximum(diff, 0.0)))

    return RateComparisonResult(
        events_a=post_a.events,
        exposure_a=post_a.exposure,
        events_b=post_b.events,
        exposure_b=post_b.exposure,
        rate_a_mean=post_a.mean(),
        rate_b_mean=post_b.mean(),
        rate_a_interval=post_a.credible_interval(ci),
        rate_b_interval=post_b.credible_interval(ci),
        prob_b_better=float(prob_b_better),
        rate_ratio_median=med,
        rate_ratio_interval=(lo, hi),
        expected_loss_choose_a=loss_a,
        expected_loss_choose_b=loss_b,
        higher_is_better=bool(higher_is_better),
        ci=float(ci),
        loss_threshold=None if loss_threshold is None else float(loss_threshold),
    )


__all__ = [
    "GammaPoissonPosterior",
    "RateComparisonResult",
    "compare_rates",
    "prob_rate_b_exceeds_a",
    "rate_ratio_interval",
]
