"""ROPE decisions and expected-loss stopping for two conversion-rate posteriors."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .decisions import Decision
from .posteriors import BetaBinomialPosterior
from .risk import risk_decision


def _validate_ci(ci: float) -> None:
    if not 0 < ci < 1:
        raise ValueError("credible level must lie strictly inside (0, 1)")


def _parse_rope(rope: float | tuple[float, float]) -> tuple[float, float]:
    if isinstance(rope, (tuple, list)):
        if len(rope) != 2:
            raise ValueError("rope interval must have two endpoints")
        lower, upper = float(rope[0]), float(rope[1])
        if not lower < upper:
            raise ValueError("rope lower bound must be less than the upper bound")
        return lower, upper
    width = float(rope)
    if width <= 0:
        raise ValueError("rope half-width must be positive")
    return -width, width


@dataclass(frozen=True)
class RopeResult:
    """Win / loss / practical-equivalence verdict for ``rate_B - rate_A``.

    The equal-tailed credible interval is compared with the region of practical
    equivalence (ROPE). Fully above the ROPE ships B, fully below ships A,
    fully inside declares the variants equivalent, and any overlap with a
    ROPE boundary means keep collecting data.
    """

    rope_lower: float
    rope_upper: float
    prob_in_rope: float
    prob_below_rope: float
    prob_above_rope: float
    diff_ci_lower: float
    diff_ci_upper: float
    ci: float
    decision: Decision


@dataclass(frozen=True)
class ExpectedLossStopResult:
    """Whether the leading variant's expected loss is small enough to stop."""

    leader: str
    expected_loss: float
    threshold: float
    should_stop: bool
    decision: Decision


def rope_decision(
    posterior_a: BetaBinomialPosterior,
    posterior_b: BetaBinomialPosterior,
    rope: float | tuple[float, float] = 0.01,
    ci: float = 0.95,
    n_samples: int = 100_000,
    seed: int | None = 20260824,
) -> RopeResult:
    """Decide win, loss, or practical equivalence from two conversion posteriors.

    ``rope`` is either a positive half-width (the interval ``[-rope, rope]``)
    or an explicit ``(lower, upper)`` interval on the conversion-rate
    difference ``rate_B - rate_A``.
    """
    _validate_ci(ci)
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    rope_lower, rope_upper = _parse_rope(rope)

    rng = np.random.default_rng(seed)
    draws_a = rng.beta(posterior_a.alpha, posterior_a.beta, size=n_samples)
    draws_b = rng.beta(posterior_b.alpha, posterior_b.beta, size=n_samples)
    diff = draws_b - draws_a

    prob_in_rope = float(np.mean((diff >= rope_lower) & (diff <= rope_upper)))
    prob_below_rope = float(np.mean(diff < rope_lower))
    prob_above_rope = float(np.mean(diff > rope_upper))

    tail = (1.0 - ci) / 2.0
    ci_lower, ci_upper = (float(v) for v in np.quantile(diff, [tail, 1.0 - tail]))

    if ci_lower > rope_upper:
        decision = Decision.SHIP_B
    elif ci_upper < rope_lower:
        decision = Decision.SHIP_A
    elif ci_lower >= rope_lower and ci_upper <= rope_upper:
        decision = Decision.PRACTICAL_EQUIVALENCE
    else:
        decision = Decision.KEEP_RUNNING

    return RopeResult(
        rope_lower=rope_lower,
        rope_upper=rope_upper,
        prob_in_rope=prob_in_rope,
        prob_below_rope=prob_below_rope,
        prob_above_rope=prob_above_rope,
        diff_ci_lower=ci_lower,
        diff_ci_upper=ci_upper,
        ci=ci,
        decision=decision,
    )


def expected_loss_stop(
    posterior_a: BetaBinomialPosterior,
    posterior_b: BetaBinomialPosterior,
    threshold: float = 0.0025,
    n_samples: int = 100_000,
    seed: int | None = 20260824,
) -> ExpectedLossStopResult:
    """Stop when the posterior-mean leader's expected loss is below ``threshold``.

    Uses :func:`bayes_ab_kit.risk.risk_decision` for the loss calculation.
    ``should_stop`` is true only when that loss is inside the tolerance, in
    which case ``decision`` ships the leader; otherwise the experiment should
    keep running.
    """
    result = risk_decision(
        posterior_a,
        posterior_b,
        threshold=threshold,
        n_samples=n_samples,
        seed=seed,
    )
    return ExpectedLossStopResult(
        leader=result.leader,
        expected_loss=result.expected_loss_of_leader,
        threshold=result.threshold,
        should_stop=result.decision is not Decision.KEEP_RUNNING,
        decision=result.decision,
    )


__all__ = [
    "ExpectedLossStopResult",
    "RopeResult",
    "expected_loss_stop",
    "rope_decision",
]
