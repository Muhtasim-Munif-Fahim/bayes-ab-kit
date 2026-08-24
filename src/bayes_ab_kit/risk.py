"""Expected-loss decision rules for choosing between two variants."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .decisions import Decision
from .posteriors import BetaBinomialPosterior


def _draws(posterior: BetaBinomialPosterior, rng: np.random.Generator, size: int) -> np.ndarray:
    return rng.beta(posterior.alpha, posterior.beta, size=size)


def expected_loss(
    chosen: BetaBinomialPosterior,
    other: BetaBinomialPosterior,
    n_samples: int = 100_000,
    seed: int | None = 20260824,
) -> float:
    """Expected improvement lost by shipping ``chosen`` instead of ``other``.

    Computed as ``E[max(rate_other - rate_chosen, 0)]`` by Monte Carlo. A
    value near zero means the chosen variant is very unlikely to be much
    worse than the alternative.
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    rng = np.random.default_rng(seed)
    diff = _draws(other, rng, n_samples) - _draws(chosen, rng, n_samples)
    return float(np.mean(np.maximum(diff, 0.0)))


@dataclass(frozen=True)
class RiskResult:
    """Outcome of a threshold-based expected-loss comparison."""

    leader: str
    expected_loss_of_leader: float
    threshold: float
    prob_leader_beats_follower: float
    decision: Decision


def risk_decision(
    posterior_a: BetaBinomialPosterior,
    posterior_b: BetaBinomialPosterior,
    threshold: float = 0.0025,
    n_samples: int = 100_000,
    seed: int | None = 20260824,
) -> RiskResult:
    """Ship the leading variant once its expected loss falls below ``threshold``.

    The leader is the variant with the higher posterior mean. If its expected
    loss is under the tolerance the experiment can stop in its favour;
    otherwise the recommendation is to keep collecting data.
    """
    if not 0 < threshold <= 1:
        raise ValueError("threshold must lie inside (0, 1]")
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    rng = np.random.default_rng(seed)
    draws_a = _draws(posterior_a, rng, n_samples)
    draws_b = _draws(posterior_b, rng, n_samples)

    if posterior_b.mean() >= posterior_a.mean():
        leader, follower_draws, leader_draws = "b", draws_a, draws_b
    else:
        leader, follower_draws, leader_draws = "a", draws_b, draws_a

    loss = float(np.mean(np.maximum(follower_draws - leader_draws, 0.0)))
    prob_leader_best = float(np.mean(leader_draws > follower_draws))

    if loss < threshold:
        decision = Decision.SHIP_A if leader == "a" else Decision.SHIP_B
    else:
        decision = Decision.KEEP_RUNNING

    return RiskResult(
        leader=leader,
        expected_loss_of_leader=loss,
        threshold=threshold,
        prob_leader_beats_follower=prob_leader_best,
        decision=decision,
    )