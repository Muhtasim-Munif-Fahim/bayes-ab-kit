"""Multiplicity control for many metrics or segments, the Bayesian way.

A dashboard with dozens of metrics (or segments) will show some "wins" by
chance. Instead of Bonferroni / BH p-value corrections, this module works
with posterior error probabilities:

* **local false sign rate** (Stephens 2017): ``lfsr_j = min(P(theta_j >= 0),
  P(theta_j <= 0))`` -- the posterior probability that we would get the
  *sign* of effect ``j`` wrong if we declared it. Computed from posterior
  draws (:func:`lfsr_from_draws`) or a normal posterior
  (:func:`lfsr_normal`).
* **Bayesian FDR selection** (Newton et al. 2004; Müller et al. 2004):
  declare the ``k`` discoveries with the smallest local error probabilities,
  with ``k`` the largest value whose *average* local error is ``<= q``.
  The posterior expected false-discovery (here: false-sign) proportion of
  the selected set is then ``<= q`` (:func:`bayesian_fdr_select`).
* **Empirical-Bayes metric screen** (:func:`screen_metrics`): given
  per-metric estimates and standard errors, fit the normal prior
  ``theta_j ~ N(0, tau^2)`` by marginal maximum likelihood, shrink every
  estimate and apply the selection rule to the resulting lfsr. Shrinkage
  is what makes the multiplicity correction: with many noisy metrics,
  ``tau`` is small and isolated large estimates are pulled toward zero.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import norm

__all__ = [
    "BayesianFdrResult",
    "MetricScreenResult",
    "bayesian_fdr_select",
    "lfsr_from_draws",
    "lfsr_normal",
    "screen_metrics",
]


def lfsr_from_draws(draws) -> np.ndarray:
    """Local false sign rate of each column of posterior ``draws``.

    ``draws`` has shape ``(n_draws,)`` or ``(n_draws, n_effects)``. Ties at
    exactly zero count against both signs, as in Stephens (2017).
    """
    D = np.asarray(draws, dtype=float)
    if D.ndim == 1:
        D = D[:, None]
    if D.ndim != 2 or D.shape[0] == 0:
        raise ValueError("draws must be a non-empty 1-D or 2-D array")
    if not np.all(np.isfinite(D)):
        raise ValueError("draws must be finite")
    p_ge = np.mean(D >= 0, axis=0)
    p_le = np.mean(D <= 0, axis=0)
    return np.minimum(p_ge, p_le)


def lfsr_normal(means, sds) -> np.ndarray:
    """lfsr of normal posteriors ``N(mean_j, sd_j^2)``: ``Phi(-|mean| / sd)``."""
    m = np.atleast_1d(np.asarray(means, dtype=float))
    s = np.atleast_1d(np.asarray(sds, dtype=float))
    if m.shape != s.shape:
        raise ValueError("means and sds must have the same shape")
    if np.any(s <= 0) or not np.all(np.isfinite(m)):
        raise ValueError("sds must be positive and means finite")
    return norm.cdf(-np.abs(m) / s)


@dataclass(frozen=True)
class BayesianFdrResult:
    """Outcome of Bayesian FDR selection over local error probabilities."""

    names: tuple[str, ...]
    local_error: np.ndarray
    q_values: np.ndarray
    selected: np.ndarray
    q: float
    n_selected: int
    expected_false_discoveries: float
    expected_fdr: float
    threshold: float

    @property
    def selected_names(self) -> tuple[str, ...]:
        return tuple(n for n, s in zip(self.names, self.selected) if s)

    def summary(self) -> str:
        return (
            f"{self.n_selected}/{len(self.names)} selected at q={self.q:g}: "
            f"expected FDR {self.expected_fdr:.4f}, "
            f"expected false discoveries {self.expected_false_discoveries:.3f}"
        )


def _names(names, n):
    if names is None:
        return tuple(f"m{i}" for i in range(n))
    names = tuple(str(x) for x in names)
    if len(names) != n:
        raise ValueError("names must have one entry per effect")
    if len(set(names)) != n:
        raise ValueError("names must be unique")
    return names


def bayesian_fdr_select(local_error, q: float = 0.05, names: Sequence[str] | None = None) -> BayesianFdrResult:
    """Select discoveries so the posterior expected FDR is at most ``q``.

    Parameters
    ----------
    local_error : array-like
        Posterior probability that each declaration would be wrong -- an
        lfsr, or a local fdr ``P(theta_j = 0 | data)``.
    q : float
        Target posterior expected false discovery rate, in (0, 1).

    Returns
    -------
    BayesianFdrResult
        ``q_values[j]`` is the smallest target at which effect ``j`` is
        selected (the running mean of sorted local errors).
    """
    e = np.atleast_1d(np.asarray(local_error, dtype=float))
    if e.ndim != 1 or e.size == 0:
        raise ValueError("local_error must be a non-empty 1-D array")
    if np.any(~np.isfinite(e)) or np.any(e < 0) or np.any(e > 1):
        raise ValueError("local_error must lie in [0, 1]")
    if not 0 < q < 1:
        raise ValueError("q must be in (0, 1)")
    n = e.size
    order = np.argsort(e, kind="stable")
    running = np.cumsum(e[order]) / np.arange(1, n + 1)
    qv = np.empty(n)
    qv[order] = running  # running mean of a sorted sequence is nondecreasing
    ok = np.nonzero(running <= q + 1e-15)[0]
    k = int(ok[-1] + 1) if ok.size else 0
    selected = np.zeros(n, dtype=bool)
    selected[order[:k]] = True
    efd = float(e[selected].sum())
    return BayesianFdrResult(
        names=_names(names, n),
        local_error=e,
        q_values=qv,
        selected=selected,
        q=float(q),
        n_selected=k,
        expected_false_discoveries=efd,
        expected_fdr=efd / k if k else 0.0,
        threshold=float(e[order[k - 1]]) if k else 0.0,
    )


@dataclass(frozen=True)
class MetricScreenResult:
    """Empirical-Bayes shrinkage plus lfsr-based selection across metrics."""

    names: tuple[str, ...]
    estimates: np.ndarray
    std_errors: np.ndarray
    tau: float
    posterior_means: np.ndarray
    posterior_sds: np.ndarray
    shrinkage: np.ndarray
    lfsr: np.ndarray
    selection: BayesianFdrResult

    @property
    def selected_names(self) -> tuple[str, ...]:
        return self.selection.selected_names

    def table(self) -> list[dict]:
        """One row per metric, sorted by lfsr."""
        rows = []
        for j in np.argsort(self.lfsr, kind="stable"):
            rows.append(
                {
                    "metric": self.names[j],
                    "estimate": float(self.estimates[j]),
                    "std_error": float(self.std_errors[j]),
                    "posterior_mean": float(self.posterior_means[j]),
                    "posterior_sd": float(self.posterior_sds[j]),
                    "lfsr": float(self.lfsr[j]),
                    "q_value": float(self.selection.q_values[j]),
                    "selected": bool(self.selection.selected[j]),
                }
            )
        return rows

    def summary(self) -> str:
        return f"tau = {self.tau:.4g}; " + self.selection.summary()


def _fit_tau(y, s):
    def nll(log_tau):
        v = s**2 + np.exp(2 * log_tau)
        return 0.5 * np.sum(np.log(v) + y**2 / v)

    scale = float(max(np.max(np.abs(y)), np.max(s)))
    lo, hi = np.log(scale * 1e-6), np.log(scale * 10)
    res = minimize_scalar(nll, bounds=(lo, hi), method="bounded", options={"xatol": 1e-8})
    tau = float(np.exp(res.x))
    # boundary check: tau -> 0 if the likelihood is maximised at the bottom
    if nll(lo) <= res.fun + 1e-12:
        tau = 0.0
    return tau


def screen_metrics(
    estimates,
    std_errors,
    names: Sequence[str] | None = None,
    q: float = 0.05,
    tau: float | None = None,
) -> MetricScreenResult:
    """Screen many metric lifts with empirical-Bayes shrinkage and lfsr.

    Parameters
    ----------
    estimates, std_errors : array-like
        Per-metric lift estimates and their standard errors.
    q : float
        Target posterior expected false sign rate of the selected set.
    tau : float, optional
        Prior SD of true effects. ``None`` fits it by marginal maximum
        likelihood (``y_j ~ N(0, s_j^2 + tau^2)``); ``float("inf")`` means
        no shrinkage (lfsr = one-sided p-value).
    """
    y = np.atleast_1d(np.asarray(estimates, dtype=float))
    s = np.atleast_1d(np.asarray(std_errors, dtype=float))
    if y.ndim != 1 or y.size == 0 or y.shape != s.shape:
        raise ValueError("estimates and std_errors must be 1-D arrays of the same length")
    if not (np.all(np.isfinite(y)) and np.all(np.isfinite(s)) and np.all(s > 0)):
        raise ValueError("estimates must be finite and std_errors positive")
    if tau is None:
        tau = _fit_tau(y, s)
    tau = float(tau)
    if tau < 0:
        raise ValueError("tau must be non-negative")
    if np.isinf(tau):
        b = np.zeros_like(s)
        pm, psd = y.copy(), s.copy()
    elif tau == 0:
        b = np.ones_like(s)
        pm, psd = np.zeros_like(y), np.zeros_like(s)
    else:
        b = s**2 / (s**2 + tau**2)
        pm = (1 - b) * y
        psd = np.sqrt((1 - b)) * s
    lf = np.full(y.shape, 0.5) if tau == 0 else lfsr_normal(pm, psd)
    sel = bayesian_fdr_select(lf, q=q, names=names)
    return MetricScreenResult(
        names=sel.names,
        estimates=y,
        std_errors=s,
        tau=tau,
        posterior_means=pm,
        posterior_sds=psd,
        shrinkage=b,
        lfsr=lf,
        selection=sel,
    )
