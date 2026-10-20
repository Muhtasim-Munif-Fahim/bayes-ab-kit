"""Ratio metrics with delta-method variance (Deng, Knoblich & Lu, 2018).

Many product metrics are ratios whose denominator is not the randomization
unit: click-through rate (clicks / page views) or revenue per session, with
users randomized. Pooling every page view into one Bernoulli sample
(``sqrt(r (1 - r) / total_views)``) ignores the correlation between views
from the same user. Its standard error is usually far too small, so A/A
tests "win" much more often than ``alpha``.

The correct unit is the user. With per-user numerator ``X_i`` and
denominator ``Y_i``, the metric is ``r = mean(X) / mean(Y)`` and the first-order
delta method gives

    Var(r) ~ (1/n) * [ s_xx / mu_y^2 - 2 mu_x s_xy / mu_y^3 + mu_x^2 s_yy / mu_y^4 ]

This module provides that estimate (from raw arrays or from aggregated
moments, as you would get from a SQL ``GROUP BY``), the naive pooled
standard error and the resulting design effect, and an A/B comparison. The
comparison reports the absolute difference with a frequentist CI and
p-value, the relative lift via the delta method on ``log r``, and the
large-sample flat-prior posterior ``N(diff, se^2)``. That posterior gives
``P(B > A)`` and closed-form expected losses, consistent with the rest of
the kit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

__all__ = [
    "RatioComparisonResult",
    "RatioSummary",
    "compare_ratio_metrics",
    "ratio_summary",
    "ratio_summary_from_moments",
]


def _validate_ci(ci: float) -> float:
    if not 0.0 < float(ci) < 1.0:
        raise ValueError("ci must be in (0, 1)")
    return float(ci)


@dataclass(frozen=True)
class RatioSummary:
    """Delta-method summary of one arm's ratio metric ``sum(X) / sum(Y)``.

    ``naive_se`` is the pooled-Bernoulli standard error. It is defined only
    when ``0 <= ratio <= 1``, i.e. for a "successes per opportunity" metric.
    ``design_effect = (se / naive_se)^2`` is how many times the naive
    variance understates the truth.
    """

    n: int
    sum_numerator: float
    sum_denominator: float
    mean_numerator: float
    mean_denominator: float
    ratio: float
    variance: float
    se: float
    naive_se: float | None
    design_effect: float | None

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def ratio_summary_from_moments(
    n: int,
    sum_x: float,
    sum_y: float,
    sum_xx: float,
    sum_yy: float,
    sum_xy: float,
) -> RatioSummary:
    """Delta-method ratio summary from per-unit sufficient statistics.

    ``sum_xx = sum(X_i^2)``, ``sum_yy = sum(Y_i^2)`` and ``sum_xy = sum(X_i Y_i)``
    over the ``n`` randomization units. Sample (co)variances use ``n - 1``.
    """
    if isinstance(n, bool) or int(n) != n or int(n) < 2:
        raise ValueError("n must be an integer >= 2 (randomization units)")
    n = int(n)
    vals = [float(v) for v in (sum_x, sum_y, sum_xx, sum_yy, sum_xy)]
    if not all(math.isfinite(v) for v in vals):
        raise ValueError("moments must be finite")
    sum_x, sum_y, sum_xx, sum_yy, sum_xy = vals
    if sum_y <= 0.0:
        raise ValueError("sum of denominators must be positive")
    mx, my = sum_x / n, sum_y / n
    sxx = (sum_xx - n * mx * mx) / (n - 1)
    syy = (sum_yy - n * my * my) / (n - 1)
    sxy = (sum_xy - n * mx * my) / (n - 1)
    if sxx < -1e-9 * max(1.0, abs(sum_xx)) or syy < -1e-9 * max(1.0, abs(sum_yy)):
        raise ValueError("moments are inconsistent (negative variance)")
    sxx, syy = max(sxx, 0.0), max(syy, 0.0)
    r = mx / my
    var = (sxx / my**2 - 2.0 * mx * sxy / my**3 + mx**2 * syy / my**4) / n
    var = max(var, 0.0)
    se = math.sqrt(var)
    naive_se = deff = None
    if 0.0 <= r <= 1.0:
        naive_se = math.sqrt(r * (1.0 - r) / sum_y)
        deff = (var / naive_se**2) if naive_se > 0 else None
    return RatioSummary(
        n=n,
        sum_numerator=sum_x,
        sum_denominator=sum_y,
        mean_numerator=mx,
        mean_denominator=my,
        ratio=r,
        variance=var,
        se=se,
        naive_se=naive_se,
        design_effect=deff,
    )


def ratio_summary(numerator, denominator) -> RatioSummary:
    """Delta-method ratio summary from per-unit numerator and denominator arrays."""
    x = np.asarray(numerator, dtype=float).ravel()
    y = np.asarray(denominator, dtype=float).ravel()
    if x.shape != y.shape:
        raise ValueError("numerator and denominator must have the same length")
    if x.size < 2:
        raise ValueError("need at least 2 randomization units")
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        raise ValueError("numerator and denominator must be finite")
    if np.any(y < 0):
        raise ValueError("denominators must be non-negative")
    n = x.size
    mx, my = x.mean(), y.mean()
    if my <= 0:
        raise ValueError("sum of denominators must be positive")
    # Centered moments for numerical stability, then reuse the moment path.
    dx, dy = x - mx, y - my
    return ratio_summary_from_moments(
        n,
        float(x.sum()),
        float(y.sum()),
        float(n * mx * mx + (dx * dx).sum()),
        float(n * my * my + (dy * dy).sum()),
        float(n * mx * my + (dx * dy).sum()),
    )


@dataclass(frozen=True)
class RatioComparisonResult:
    """A/B comparison of a ratio metric (B relative to A)."""

    a: RatioSummary
    b: RatioSummary
    ci: float
    difference: float
    difference_se: float
    difference_interval: tuple[float, float]
    z: float
    p_value: float
    relative_lift: float
    relative_lift_interval: tuple[float, float]
    prob_b_better: float
    expected_loss_choose_a: float
    expected_loss_choose_b: float
    naive_difference_se: float | None
    naive_p_value: float | None
    higher_is_better: bool

    def to_dict(self) -> dict:
        out = dict(self.__dict__)
        out["a"] = self.a.to_dict()
        out["b"] = self.b.to_dict()
        return out


def compare_ratio_metrics(
    a: RatioSummary,
    b: RatioSummary,
    *,
    ci: float = 0.95,
    higher_is_better: bool = True,
) -> RatioComparisonResult:
    """Compare two arms' ratio metrics with delta-method standard errors.

    The posterior quantities use the large-sample flat-prior approximation
    ``r_B - r_A ~ N(diff, se^2)``. Expected loss of choosing an arm is the
    expected shortfall versus the other arm, in metric units, oriented by
    ``higher_is_better``.
    """
    ci = _validate_ci(ci)
    if not isinstance(a, RatioSummary) or not isinstance(b, RatioSummary):
        raise TypeError("a and b must be RatioSummary objects (see ratio_summary)")
    z_crit = float(norm.ppf(0.5 + ci / 2.0))
    diff = b.ratio - a.ratio
    se = math.sqrt(a.variance + b.variance)
    if se <= 0.0:
        raise ValueError("both arms have zero variance; the comparison is degenerate")
    z = diff / se
    p = float(2.0 * norm.sf(abs(z)))
    if a.ratio > 0 and b.ratio > 0:
        log_lift = math.log(b.ratio / a.ratio)
        se_log = math.sqrt(a.variance / a.ratio**2 + b.variance / b.ratio**2)
        lift = math.expm1(log_lift)
        lift_ci = (math.expm1(log_lift - z_crit * se_log), math.expm1(log_lift + z_crit * se_log))
    else:
        lift = float("nan")
        lift_ci = (float("nan"), float("nan"))
    d = diff if higher_is_better else -diff  # advantage of B in "good" units
    pdf, cdf = float(norm.pdf(d / se)), float(norm.cdf(d / se))
    loss_b = se * pdf - d * (1.0 - cdf)  # E[max(A_good - B_good, 0)]
    loss_a = se * pdf + d * cdf  # E[max(B_good - A_good, 0)]
    naive_se = naive_p = None
    if a.naive_se is not None and b.naive_se is not None:
        naive_se = math.sqrt(a.naive_se**2 + b.naive_se**2)
        if naive_se > 0:
            naive_p = float(2.0 * norm.sf(abs(diff) / naive_se))
    return RatioComparisonResult(
        a=a,
        b=b,
        ci=ci,
        difference=diff,
        difference_se=se,
        difference_interval=(diff - z_crit * se, diff + z_crit * se),
        z=z,
        p_value=p,
        relative_lift=lift,
        relative_lift_interval=lift_ci,
        prob_b_better=cdf,
        expected_loss_choose_a=max(loss_a, 0.0),
        expected_loss_choose_b=max(loss_b, 0.0),
        naive_difference_se=naive_se,
        naive_p_value=naive_p,
        higher_is_better=bool(higher_is_better),
    )
