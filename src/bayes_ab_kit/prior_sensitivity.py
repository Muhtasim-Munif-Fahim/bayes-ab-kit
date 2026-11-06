"""Prior tipping-point / sensitivity analysis for Beta-Binomial A/B tests.

A conversion decision that looks decisive under a flat ``Beta(1, 1)`` prior can
flip under a sceptical one. This module answers two practical questions:

1. Along a ray of symmetric priors ``Beta(s, s)`` (strength ``2s``), at what
   strength does ``P(B > A)`` cross a decision threshold?
2. For a grid of prior means at fixed strength, how does ``P(B > A)`` move?

Both reuse the exact quadrature in
:func:`bayes_ab_kit.decisions.prob_b_beats_a_quad`, so there is no Monte Carlo
noise in the sensitivity curve.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .decisions import prob_b_beats_a_quad
from .posteriors import BetaBinomialPosterior

__all__ = [
    "PriorSensitivityPoint",
    "TippingPointResult",
    "prior_mean_sweep",
    "symmetric_prior_curve",
    "tipping_point_strength",
]


@dataclass(frozen=True)
class PriorSensitivityPoint:
    alpha0: float
    beta0: float
    prior_mean: float
    prior_strength: float
    prob_b_beats_a: float


@dataclass(frozen=True)
class TippingPointResult:
    """Smallest symmetric prior strength where ``P(B>A)`` crosses ``threshold``."""

    conversions_a: int
    trials_a: int
    conversions_b: int
    trials_b: int
    threshold: float
    direction: str  # "below" or "above"
    tipping_strength: Optional[float]  # 2 * s for Beta(s,s); None if never crossed
    tipping_s: Optional[float]
    baseline_prob: float
    curve: Tuple[PriorSensitivityPoint, ...]


def _counts(conversions: int, trials: int, label: str) -> Tuple[int, int]:
    if trials < 0 or conversions < 0 or conversions > trials:
        raise ValueError(f"{label}: conversions must lie in [0, trials]")
    return int(conversions), int(trials)


def _prob(ca, ta, cb, tb, a0, b0) -> float:
    post_a = BetaBinomialPosterior.from_counts(ca, ta, alpha0=a0, beta0=b0)
    post_b = BetaBinomialPosterior.from_counts(cb, tb, alpha0=a0, beta0=b0)
    return float(prob_b_beats_a_quad(post_a, post_b))


def symmetric_prior_curve(
    conversions_a: int,
    trials_a: int,
    conversions_b: int,
    trials_b: int,
    strengths: Sequence[float] = (2.0, 4.0, 10.0, 20.0, 50.0, 100.0, 200.0),
) -> List[PriorSensitivityPoint]:
    """Evaluate ``P(B>A)`` under symmetric ``Beta(s,s)`` priors.

    ``strength`` is ``alpha0 + beta0 = 2s``. Both arms share the same prior.
    """
    ca, ta = _counts(conversions_a, trials_a, "A")
    cb, tb = _counts(conversions_b, trials_b, "B")
    points: List[PriorSensitivityPoint] = []
    for strength in strengths:
        strength = float(strength)
        if strength <= 0:
            raise ValueError("strengths must be positive")
        s = strength / 2.0
        p = _prob(ca, ta, cb, tb, s, s)
        points.append(
            PriorSensitivityPoint(
                alpha0=s,
                beta0=s,
                prior_mean=0.5,
                prior_strength=strength,
                prob_b_beats_a=p,
            )
        )
    return points


def tipping_point_strength(
    conversions_a: int,
    trials_a: int,
    conversions_b: int,
    trials_b: int,
    *,
    threshold: float = 0.95,
    direction: str = "below",
    max_strength: float = 1_000.0,
    n_grid: int = 64,
) -> TippingPointResult:
    """Find the smallest ``Beta(s,s)`` strength where ``P(B>A)`` crosses ``threshold``.

    ``direction="below"`` (default) looks for the first strength where the
    probability *drops below* ``threshold`` — the usual sceptical-prior
    tipping point for a "ship B" call. ``direction="above"`` is the mirror
    case for a currently-weak result that becomes decisive under a stronger
    prior concentrated at 0.5 (rare; mainly for diagnostics).

    Returns ``tipping_strength=None`` when the curve never crosses inside
    ``[2, max_strength]``.
    """
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must lie in (0, 1)")
    if direction not in ("below", "above"):
        raise ValueError("direction must be 'below' or 'above'")
    if max_strength < 2.0:
        raise ValueError("max_strength must be at least 2")
    if n_grid < 8:
        raise ValueError("n_grid must be at least 8")

    ca, ta = _counts(conversions_a, trials_a, "A")
    cb, tb = _counts(conversions_b, trials_b, "B")

    strengths = np.geomspace(2.0, float(max_strength), int(n_grid))
    curve = tuple(symmetric_prior_curve(ca, ta, cb, tb, strengths=strengths))
    baseline = curve[0].prob_b_beats_a

    tipping_s = tipping_strength = None
    for point in curve:
        p = point.prob_b_beats_a
        crossed = p < threshold if direction == "below" else p > threshold
        if crossed:
            tipping_strength = point.prior_strength
            tipping_s = point.alpha0
            break

    return TippingPointResult(
        conversions_a=ca,
        trials_a=ta,
        conversions_b=cb,
        trials_b=tb,
        threshold=float(threshold),
        direction=direction,
        tipping_strength=tipping_strength,
        tipping_s=tipping_s,
        baseline_prob=baseline,
        curve=curve,
    )


def prior_mean_sweep(
    conversions_a: int,
    trials_a: int,
    conversions_b: int,
    trials_b: int,
    *,
    strength: float = 10.0,
    means: Sequence[float] = (0.1, 0.25, 0.5, 0.75, 0.9),
) -> List[PriorSensitivityPoint]:
    """Sweep prior means at fixed strength ``alpha0 + beta0``.

    For each mean ``m``, uses ``Beta(m * strength, (1 - m) * strength)``.
    """
    ca, ta = _counts(conversions_a, trials_a, "A")
    cb, tb = _counts(conversions_b, trials_b, "B")
    strength = float(strength)
    if strength <= 0:
        raise ValueError("strength must be positive")
    points: List[PriorSensitivityPoint] = []
    for m in means:
        m = float(m)
        if not 0.0 < m < 1.0:
            raise ValueError("means must lie strictly in (0, 1)")
        a0 = m * strength
        b0 = (1.0 - m) * strength
        p = _prob(ca, ta, cb, tb, a0, b0)
        points.append(
            PriorSensitivityPoint(
                alpha0=a0,
                beta0=b0,
                prior_mean=m,
                prior_strength=strength,
                prob_b_beats_a=p,
            )
        )
    return points
