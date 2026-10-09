"""CUPED covariate adjustment for experiment metrics (Deng et al., 2013).

CUPED (Controlled-experiment Using Pre-Experiment Data) replaces each
unit's outcome ``Y`` with the adjusted metric

    Y_cv = Y - θ (X - E[X])

where ``X`` is a pre-experiment covariate (or any mean-zeroable control
variate) and ``θ = Cov(Y, X) / Var(X)``. The adjusted metric is unbiased
for ``E[Y]`` whenever ``E[X]`` is known or estimated from the same sample,
and its variance is ``Var(Y) (1 - ρ²)`` when ``θ`` is the population
regression coefficient.

This module is the continuous-metric counterpart of the Beta-Binomial
conversion tools: use it to shrink A/B revenue / latency / engagement
variance before feeding means into :mod:`bayes_ab_kit.revenue` or a
t-test.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CupedFit:
    """Estimated CUPED coefficient and variance-reduction diagnostics."""

    theta: float
    mean_x: float
    mean_y: float
    var_y: float
    var_x: float
    cov_yx: float
    correlation: float
    variance_reduction: float

    def to_dict(self) -> dict:
        return {
            "theta": self.theta,
            "mean_x": self.mean_x,
            "mean_y": self.mean_y,
            "var_y": self.var_y,
            "var_x": self.var_x,
            "cov_yx": self.cov_yx,
            "correlation": self.correlation,
            "variance_reduction": self.variance_reduction,
        }


@dataclass(frozen=True)
class CupedResult:
    """Adjusted outcomes plus the fit used to produce them."""

    adjusted: np.ndarray
    fit: CupedFit
    var_adjusted: float
    n: int

    def to_dict(self) -> dict:
        return {
            "adjusted": self.adjusted.tolist(),
            "fit": self.fit.to_dict(),
            "var_adjusted": self.var_adjusted,
            "n": self.n,
        }


def _as_1d(name: str, values) -> np.ndarray:
    arr = np.asarray(values, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def estimate_cuped_theta(y, x) -> CupedFit:
    """Estimate ``θ = Cov(Y, X) / Var(X)`` and variance-reduction stats."""
    y_arr = _as_1d("y", y)
    x_arr = _as_1d("x", x)
    if y_arr.shape != x_arr.shape:
        raise ValueError("y and x must have the same length")
    if y_arr.size < 2:
        raise ValueError("need at least two observations to estimate CUPED")

    mean_y = float(y_arr.mean())
    mean_x = float(x_arr.mean())
    y_c = y_arr - mean_y
    x_c = x_arr - mean_x
    # Use population (ddof=0) moments so θ matches the classic CUPED formula
    # on the same sample used for adjustment.
    var_y = float(np.mean(y_c * y_c))
    var_x = float(np.mean(x_c * x_c))
    cov_yx = float(np.mean(y_c * x_c))
    if var_x <= 0.0:
        raise ValueError("x must have positive variance")
    theta = cov_yx / var_x
    if var_y <= 0.0:
        corr = 0.0
        reduction = 0.0
    else:
        corr = cov_yx / float(np.sqrt(var_y * var_x))
        reduction = float(corr * corr)
    return CupedFit(
        theta=float(theta),
        mean_x=mean_x,
        mean_y=mean_y,
        var_y=var_y,
        var_x=var_x,
        cov_yx=cov_yx,
        correlation=float(corr),
        variance_reduction=reduction,
    )


def cuped_adjust(y, x, *, theta: float | None = None, mean_x: float | None = None) -> CupedResult:
    """Return CUPED-adjusted outcomes ``Y - θ (X - mean_x)``.

    When ``theta`` is omitted it is estimated from ``(y, x)``. When
    ``mean_x`` is omitted the sample mean of ``x`` is used (so the
    adjusted vector has the same mean as ``y``).
    """
    y_arr = _as_1d("y", y)
    x_arr = _as_1d("x", x)
    if y_arr.shape != x_arr.shape:
        raise ValueError("y and x must have the same length")

    if theta is None:
        fit = estimate_cuped_theta(y_arr, x_arr)
        used_theta = fit.theta
        used_mean_x = fit.mean_x if mean_x is None else float(mean_x)
        if mean_x is not None:
            # Recompute diagnostics with the provided centring mean but keep θ.
            adjusted = y_arr - used_theta * (x_arr - used_mean_x)
            var_adj = float(np.mean((adjusted - adjusted.mean()) ** 2))
            fit = CupedFit(
                theta=used_theta,
                mean_x=used_mean_x,
                mean_y=fit.mean_y,
                var_y=fit.var_y,
                var_x=fit.var_x,
                cov_yx=fit.cov_yx,
                correlation=fit.correlation,
                variance_reduction=fit.variance_reduction,
            )
        else:
            adjusted = y_arr - used_theta * (x_arr - used_mean_x)
            var_adj = float(np.mean((adjusted - adjusted.mean()) ** 2))
    else:
        used_theta = float(theta)
        if not np.isfinite(used_theta):
            raise ValueError("theta must be finite")
        used_mean_x = float(x_arr.mean()) if mean_x is None else float(mean_x)
        fit = estimate_cuped_theta(y_arr, x_arr)
        # Preserve user θ but report the data-driven diagnostics alongside.
        fit = CupedFit(
            theta=used_theta,
            mean_x=used_mean_x,
            mean_y=fit.mean_y,
            var_y=fit.var_y,
            var_x=fit.var_x,
            cov_yx=fit.cov_yx,
            correlation=fit.correlation,
            variance_reduction=fit.variance_reduction,
        )
        adjusted = y_arr - used_theta * (x_arr - used_mean_x)
        var_adj = float(np.mean((adjusted - adjusted.mean()) ** 2))

    return CupedResult(
        adjusted=np.asarray(adjusted, dtype=float),
        fit=fit,
        var_adjusted=var_adj,
        n=int(y_arr.size),
    )


def cuped_variance_reduction(y, x) -> float:
    """Return estimated ``ρ² = Corr(Y, X)²`` (fractional variance cut)."""
    return estimate_cuped_theta(y, x).variance_reduction
