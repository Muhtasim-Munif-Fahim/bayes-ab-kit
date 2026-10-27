"""Bayesian random-effects meta-analysis of experiment results.

Teams often run the same kind of change many times (or the same experiment in
several markets). Each experiment ``i`` reports a lift estimate ``y_i`` with
standard error ``s_i``. The normal hierarchical model (Gelman et al., *Bayesian
Data Analysis*, 3rd ed., §5.4–5.5) is::

    y_i | theta_i ~ N(theta_i, s_i^2)
    theta_i | mu, tau ~ N(mu, tau^2)
    p(mu) flat,  tau ~ prior (uniform, half-normal or half-Cauchy)

Given ``tau``, everything is Gaussian, so the posterior is computed exactly by
integrating ``tau`` over a fine grid (no MCMC):

* ``p(tau | y) ∝ p(tau) V_mu^{1/2} prod (s_i^2 + tau^2)^{-1/2}
  exp(-(y_i - mu_hat)^2 / (2 (s_i^2 + tau^2)))``
* ``mu | tau, y ~ N(mu_hat, V_mu)`` with precision-weighted ``mu_hat``
* ``theta_i | tau, y`` is normal with mean ``B_i mu_hat + (1 - B_i) y_i``,
  where ``B_i = s_i^2 / (s_i^2 + tau^2)`` is the pooling factor.

The result gives the overall effect ``mu``, the heterogeneity ``tau``, partially
pooled (shrunk) per-experiment effects — which correct the winner's curse of
picking the best-looking experiment — and the predictive distribution of the
effect in a *new* experiment, ``N(mu_hat, V_mu + tau^2)`` mixed over ``tau``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

__all__ = ["MetaAnalysisResult", "meta_analysis", "TAU_PRIORS"]

TAU_PRIORS = ("uniform", "half-normal", "half-cauchy")


@dataclass(frozen=True)
class MetaAnalysisResult:
    """Posterior summaries of a normal random-effects meta-analysis."""

    names: tuple[str, ...]
    estimates: np.ndarray
    std_errors: np.ndarray
    ci: float
    threshold: float
    tau_prior: str
    mu_mean: float
    mu_sd: float
    mu_interval: tuple[float, float]
    prob_mu_above: float
    tau_mean: float
    tau_median: float
    tau_interval: tuple[float, float]
    theta_means: np.ndarray
    theta_sds: np.ndarray
    theta_intervals: np.ndarray
    prob_theta_above: np.ndarray
    pooling: np.ndarray
    predictive_mean: float
    predictive_sd: float
    predictive_interval: tuple[float, float]
    prob_new_above: float
    tau_grid: np.ndarray = field(repr=False)
    tau_weights: np.ndarray = field(repr=False)


def _mixture_quantile(p: float, w: np.ndarray, m: np.ndarray, s: np.ndarray) -> float:
    """Quantile of the normal mixture ``sum w_k N(m_k, s_k^2)``."""

    s = np.maximum(s, 1e-300)
    lo = float(np.min(m - 12 * s))
    hi = float(np.max(m + 12 * s))
    if hi - lo < 1e-15:
        return float(m[0])
    return float(brentq(lambda x: float(w @ norm.cdf((x - m) / s)) - p, lo, hi, xtol=1e-12))


def _mixture_summary(w, m, v, ci, threshold):
    mean = float(w @ m)
    var = float(w @ (v + m**2) - mean**2)
    s = np.sqrt(np.maximum(v, 0.0))
    a = (1 - ci) / 2
    interval = (_mixture_quantile(a, w, m, s), _mixture_quantile(1 - a, w, m, s))
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(s > 0, (m - threshold) / np.where(s > 0, s, 1), np.sign(m - threshold) * np.inf)
    prob = float(w @ norm.cdf(z))
    return mean, float(np.sqrt(max(var, 0.0))), interval, prob


def meta_analysis(
    estimates: Sequence[float],
    std_errors: Sequence[float],
    *,
    names: Sequence[str] | None = None,
    tau_prior: str = "half-cauchy",
    tau_scale: float | None = None,
    ci: float = 0.95,
    threshold: float = 0.0,
    grid_size: int = 2000,
    tau_max: float | None = None,
) -> MetaAnalysisResult:
    """Exact grid-based Bayesian random-effects meta-analysis.

    Args:
        estimates: per-experiment effect estimates ``y_i`` (e.g. absolute lifts).
        std_errors: their standard errors ``s_i`` (> 0).
        names: optional experiment labels.
        tau_prior: ``"uniform"`` (flat on ``[0, tau_max]``, Gelman's default;
            needs at least 3 experiments; with only a handful its heavy tail
            makes results sensitive to ``tau_max``), ``"half-normal"`` or
            ``"half-cauchy"`` (default) with scale ``tau_scale``.
        tau_scale: prior scale; defaults to the median standard error.
        ci: credible / predictive interval level.
        threshold: report posterior probabilities of exceeding this value.
        grid_size: number of ``tau`` grid points.
        tau_max: upper end of the ``tau`` grid (default: generous multiple of
            the spread of the data and the standard errors).
    """

    y = np.asarray(estimates, dtype=float).reshape(-1)
    s = np.asarray(std_errors, dtype=float).reshape(-1)
    k = y.size
    if k < 2:
        raise ValueError("need at least 2 experiments")
    if s.size != k:
        raise ValueError("estimates and std_errors must have the same length")
    if not (np.all(np.isfinite(y)) and np.all(np.isfinite(s))):
        raise ValueError("estimates and std_errors must be finite")
    if np.any(s <= 0):
        raise ValueError("std_errors must be positive")
    if not 0 < ci < 1:
        raise ValueError("ci must lie strictly inside (0, 1)")
    if tau_prior not in TAU_PRIORS:
        raise ValueError(f"tau_prior must be one of {TAU_PRIORS}")
    if tau_prior == "uniform" and k < 3:
        raise ValueError("the uniform tau prior needs at least 3 experiments for a proper posterior")
    if grid_size < 50:
        raise ValueError("grid_size must be >= 50")
    labels = tuple(str(n) for n in names) if names is not None else tuple(f"exp{i + 1}" for i in range(k))
    if len(labels) != k:
        raise ValueError("names must have one label per experiment")
    scale = float(np.median(s)) if tau_scale is None else float(tau_scale)
    if scale <= 0:
        raise ValueError("tau_scale must be positive")
    if tau_max is None:
        spread = float(np.ptp(y))
        tau_max = 10.0 * max(spread, float(np.max(s)), scale if tau_prior != "uniform" else 0.0)
    if tau_max <= 0:
        raise ValueError("tau_max must be positive")

    tau = np.linspace(0.0, float(tau_max), int(grid_size))
    var = s[None, :] ** 2 + tau[:, None] ** 2           # (G, k)
    prec = 1.0 / var
    v_mu = 1.0 / prec.sum(axis=1)                       # (G,)
    mu_hat = (prec * y[None, :]).sum(axis=1) * v_mu
    log_post = (0.5 * np.log(v_mu) - 0.5 * np.log(var).sum(axis=1)
                - 0.5 * ((y[None, :] - mu_hat[:, None]) ** 2 * prec).sum(axis=1))
    if tau_prior == "half-normal":
        log_post += -0.5 * (tau / scale) ** 2
    elif tau_prior == "half-cauchy":
        log_post += -np.log1p((tau / scale) ** 2)
    # trapezoid weights on the uniform grid
    trap = np.ones_like(tau)
    trap[0] = trap[-1] = 0.5
    w = np.exp(log_post - log_post.max()) * trap
    w /= w.sum()

    mu_mean, mu_sd, mu_int, p_mu = _mixture_summary(w, mu_hat, v_mu, ci, threshold)

    cdf = np.cumsum(w)
    a = (1 - ci) / 2

    def tau_q(p: float) -> float:
        return float(np.interp(p, cdf, tau))

    B = s[None, :] ** 2 / var                            # pooling factor (G, k)
    m_theta = B * mu_hat[:, None] + (1 - B) * y[None, :]
    v_theta = s[None, :] ** 2 * (1 - B) + B**2 * v_mu[:, None]
    theta_means = np.empty(k)
    theta_sds = np.empty(k)
    theta_int = np.empty((k, 2))
    p_theta = np.empty(k)
    for i in range(k):
        mi, si, ii, pi = _mixture_summary(w, m_theta[:, i], v_theta[:, i], ci, threshold)
        theta_means[i], theta_sds[i], theta_int[i], p_theta[i] = mi, si, ii, pi

    pred_mean, pred_sd, pred_int, p_new = _mixture_summary(w, mu_hat, v_mu + tau**2, ci, threshold)

    return MetaAnalysisResult(
        names=labels, estimates=y, std_errors=s, ci=float(ci), threshold=float(threshold),
        tau_prior=tau_prior, mu_mean=mu_mean, mu_sd=mu_sd, mu_interval=mu_int, prob_mu_above=p_mu,
        tau_mean=float(w @ tau), tau_median=tau_q(0.5), tau_interval=(tau_q(a), tau_q(1 - a)),
        theta_means=theta_means, theta_sds=theta_sds, theta_intervals=theta_int,
        prob_theta_above=p_theta, pooling=w @ B, predictive_mean=pred_mean, predictive_sd=pred_sd,
        predictive_interval=pred_int, prob_new_above=p_new, tau_grid=tau, tau_weights=w,
    )
