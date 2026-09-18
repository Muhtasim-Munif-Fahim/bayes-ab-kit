"""Monte Carlo probability of being best for three or more conversion arms."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from .posteriors import BetaBinomialPosterior


@dataclass(frozen=True)
class ArmBestShare:
    """One arm's posterior probability of having the highest conversion rate."""

    name: str
    conversions: int
    trials: int
    posterior_mean: float
    prob_best: float


@dataclass(frozen=True)
class MultiArmBestResult:
    """P(each arm is best) estimated from independent Beta posterior draws.

    ``arms`` preserves input order. ``leader`` is the arm with the largest
    share; ties fall back to higher posterior mean, then name.
    """

    arms: tuple[ArmBestShare, ...]
    n_samples: int
    leader: str

    @property
    def probabilities(self) -> dict[str, float]:
        return {arm.name: arm.prob_best for arm in self.arms}


def _shares_of_being_best(draws: np.ndarray) -> np.ndarray:
    """Per-arm P(best) from a ``(n_samples, n_arms)`` draw matrix.

    Exact ties (rare for continuous Beta draws) are split equally so the
    shares always sum to one.
    """
    maxima = np.max(draws, axis=1, keepdims=True)
    is_max = draws == maxima
    n_tied = is_max.sum(axis=1, keepdims=True)
    return (is_max / n_tied).mean(axis=0)


def _choose_leader(shares: tuple[ArmBestShare, ...]) -> str:
    return max(
        shares,
        key=lambda arm: (arm.prob_best, arm.posterior_mean, arm.name),
    ).name


def probability_of_being_best(
    arms: Mapping[str, BetaBinomialPosterior],
    n_samples: int = 100_000,
    seed: int | None = 20260824,
) -> MultiArmBestResult:
    """Estimate P(each conversion arm is best) by Monte Carlo.

    Each arm is a named :class:`~bayes_ab_kit.posteriors.BetaBinomialPosterior`.
    Independent draws are taken from every posterior; an arm is best on a
    draw when its sampled rate is at least as large as every other arm's.
    Intended for three or more arms; two arms are accepted so the estimator
    can be checked against :func:`~bayes_ab_kit.decisions.prob_b_beats_a_mc`.
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if len(arms) < 2:
        raise ValueError("at least two arms are required")

    items: list[tuple[str, BetaBinomialPosterior]] = []
    seen: set[str] = set()
    for raw_name, posterior in arms.items():
        name = str(raw_name).strip()
        if not name:
            raise ValueError("arm name must be non-empty")
        if name in seen:
            raise ValueError(f"duplicate arm name: {name}")
        if not isinstance(posterior, BetaBinomialPosterior):
            raise ValueError("each arm must be a BetaBinomialPosterior")
        seen.add(name)
        items.append((name, posterior))

    rng = np.random.default_rng(seed)
    draws = np.column_stack(
        [posterior.sample(rng, size=n_samples) for _, posterior in items]
    )
    probs = _shares_of_being_best(draws)

    shares = tuple(
        ArmBestShare(
            name=name,
            conversions=posterior.successes,
            trials=posterior.trials,
            posterior_mean=posterior.mean(),
            prob_best=float(prob),
        )
        for (name, posterior), prob in zip(items, probs, strict=True)
    )
    return MultiArmBestResult(
        arms=shares,
        n_samples=n_samples,
        leader=_choose_leader(shares),
    )


__all__ = [
    "ArmBestShare",
    "MultiArmBestResult",
    "probability_of_being_best",
]
