"""Mixture sequential probability ratio test (mSPRT): always-valid A/B inference.

Classic fixed-horizon p-values are only valid at a single, pre-planned look;
checking them after every batch inflates the false-positive rate (see
``sequential.simulate_null_peeking``). The mSPRT (Robbins 1970; Johari,
Koomen, Pekelis & Walsh, "Always Valid Inference", Operations Research 2022)
replaces the p-value with a Bayes-factor-style mixture likelihood ratio that
is a non-negative martingale under the null, so by Ville's inequality

    P_H0( sup_t Lambda_t >= 1/alpha ) <= alpha.

With an approximately normal effect estimate ``delta_hat ~ N(theta, V)`` and a
normal mixing distribution ``theta ~ N(theta0, tau^2)`` the mixture ratio has
the closed form

    Lambda = sqrt(V / (V + tau^2)) * exp( tau^2 (delta_hat - theta0)^2 / (2 V (V + tau^2)) ).

The always-valid p-value is ``min(1, 1 / max_{s<=t} Lambda_s)`` and inverting
the test gives a confidence sequence, intervals that cover the true effect
at every look simultaneously with probability at least ``1 - alpha``. The
mixing scale ``tau`` is the effect size the test is tuned to detect quickly;
validity holds for any ``tau > 0``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .sampling import make_rng


def _validate_alpha(alpha: float) -> float:
    alpha_f = float(alpha)
    if not 0.0 < alpha_f < 1.0:
        raise ValueError("alpha must lie strictly inside (0, 1)")
    return alpha_f


def _validate_tau(tau: float) -> float:
    tau_f = float(tau)
    if not (math.isfinite(tau_f) and tau_f > 0.0):
        raise ValueError("tau must be a positive, finite mixing scale")
    return tau_f


def msprt_log_likelihood_ratio(
    delta_hat: float, variance: float, tau: float, theta0: float = 0.0
) -> float:
    """Log of the normal-mixture likelihood ratio ``Lambda`` for one look.

    ``delta_hat`` is the current effect estimate, ``variance`` its sampling
    variance ``V`` and ``tau`` the standard deviation of the normal mixing
    distribution centred on the null value ``theta0``.
    """
    tau_f = _validate_tau(tau)
    v = float(variance)
    if not (math.isfinite(v) and v > 0.0):
        raise ValueError("variance must be positive and finite")
    t2 = tau_f * tau_f
    dev = float(delta_hat) - float(theta0)
    return 0.5 * math.log(v / (v + t2)) + t2 * dev * dev / (2.0 * v * (v + t2))


def msprt_radius(variance: float, tau: float, alpha: float) -> float:
    """Half-width of the level-``alpha`` mSPRT confidence interval at one look.

    It is the set of ``theta0`` with ``Lambda(theta0) < 1/alpha``, i.e.
    ``|delta_hat - theta0| < sqrt(2 V (V + tau^2) / tau^2 * (log(1/alpha) + log(sqrt((V + tau^2) / V))))``.
    """
    alpha_f = _validate_alpha(alpha)
    tau_f = _validate_tau(tau)
    v = float(variance)
    if not (math.isfinite(v) and v > 0.0):
        raise ValueError("variance must be positive and finite")
    t2 = tau_f * tau_f
    bound = math.log(1.0 / alpha_f) + 0.5 * math.log((v + t2) / v)
    return math.sqrt(2.0 * v * (v + t2) / t2 * bound)


@dataclass(frozen=True)
class MsprtLook:
    """State of the always-valid test after one look."""

    look: int
    delta: float
    std_error: float
    log_lr: float
    p_value: float
    ci_low: float
    ci_high: float


@dataclass(frozen=True)
class MsprtResult:
    """Always-valid p-values and confidence sequence over a run of looks."""

    looks: tuple[MsprtLook, ...]
    alpha: float
    tau: float
    theta0: float
    stopped_at: int | None

    @property
    def rejected(self) -> bool:
        """True once the always-valid p-value has reached ``alpha``."""
        return self.stopped_at is not None

    @property
    def p_value(self) -> float:
        return self.looks[-1].p_value

    @property
    def confidence_interval(self) -> tuple[float, float]:
        last = self.looks[-1]
        return last.ci_low, last.ci_high


def msprt_monitor(
    deltas: Sequence[float],
    variances: Sequence[float],
    *,
    tau: float,
    alpha: float = 0.05,
    theta0: float = 0.0,
) -> MsprtResult:
    """Run the normal-mixture mSPRT over a sequence of looks.

    ``deltas[t]`` is the cumulative effect estimate at look ``t`` and
    ``variances[t]`` its variance. The p-value is the running minimum of
    ``1 / Lambda_t`` (capped at 1). The confidence sequence is the running
    intersection of the per-look intervals; should that intersection ever be
    empty (a probability-at-most-``alpha`` event) the previous interval is kept.
    ``stopped_at`` is the first look whose p-value is at most ``alpha``.
    """
    alpha_f = _validate_alpha(alpha)
    tau_f = _validate_tau(tau)
    d = np.asarray(deltas, dtype=float)
    v = np.asarray(variances, dtype=float)
    if d.ndim != 1 or v.shape != d.shape:
        raise ValueError("deltas and variances must be 1-D sequences of the same length")
    if d.size == 0:
        raise ValueError("need at least one look")
    if not np.all(np.isfinite(d)):
        raise ValueError("deltas must be finite")
    looks: list[MsprtLook] = []
    running_max = -math.inf
    low, high = -math.inf, math.inf
    stopped: int | None = None
    for t in range(d.size):
        log_lr = msprt_log_likelihood_ratio(d[t], v[t], tau_f, theta0)
        running_max = max(running_max, log_lr)
        p = float(min(1.0, math.exp(-running_max)))
        r = msprt_radius(v[t], tau_f, alpha_f)
        new_low, new_high = max(low, d[t] - r), min(high, d[t] + r)
        if new_low <= new_high:
            low, high = new_low, new_high
        looks.append(
            MsprtLook(
                look=t,
                delta=float(d[t]),
                std_error=float(math.sqrt(v[t])),
                log_lr=float(log_lr),
                p_value=p,
                ci_low=float(low),
                ci_high=float(high),
            )
        )
        if stopped is None and p <= alpha_f:
            stopped = t
    return MsprtResult(
        looks=tuple(looks), alpha=alpha_f, tau=tau_f, theta0=float(theta0), stopped_at=stopped
    )


def _proportion_estimates(
    conversions_a: np.ndarray, trials_a: np.ndarray, conversions_b: np.ndarray, trials_b: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Difference ``p_B - p_A`` and its plug-in variance (Wilson-style smoothing)."""
    delta = conversions_b / trials_b - conversions_a / trials_a
    # smoothed rates keep the variance positive at 0 or n conversions
    sa = (conversions_a + 0.5) / (trials_a + 1.0)
    sb = (conversions_b + 0.5) / (trials_b + 1.0)
    var = sa * (1.0 - sa) / trials_a + sb * (1.0 - sb) / trials_b
    return delta, var


def msprt_proportions(
    cumulative: Sequence[Sequence[int]],
    *,
    alpha: float = 0.05,
    tau: float = 0.02,
    theta0: float = 0.0,
) -> MsprtResult:
    """Always-valid test of ``p_B - p_A`` over cumulative conversion counts.

    ``cumulative[t] = (conversions_a, trials_a, conversions_b, trials_b)``
    holds running totals at look ``t``; totals must never decrease. The
    effect estimate is the observed rate difference with the unpooled
    plug-in variance (rates smoothed by ``(x + 0.5) / (n + 1)``). ``tau``
    defaults to 0.02, tuned for absolute lifts of about two points.
    """
    rows = [tuple(row) for row in cumulative]
    if not rows:
        raise ValueError("need at least one look")
    arr = np.zeros((len(rows), 4))
    previous = np.zeros(4)
    for t, row in enumerate(rows):
        if len(row) != 4:
            raise ValueError("each look must be (conversions_a, trials_a, conversions_b, trials_b)")
        vals = []
        for value in row:
            if isinstance(value, bool) or int(value) != value or value < 0:
                raise ValueError("counts must be non-negative integers")
            vals.append(int(value))
        ca, na, cb, nb = vals
        if na == 0 or nb == 0:
            raise ValueError("each arm needs at least one trial at every look")
        if ca > na or cb > nb:
            raise ValueError("conversions cannot exceed trials")
        cur = np.array(vals, dtype=float)
        if np.any(cur < previous):
            raise ValueError("cumulative counts must be non-decreasing")
        previous = cur
        arr[t] = cur
    delta, var = _proportion_estimates(arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3])
    return msprt_monitor(delta, var, tau=tau, alpha=alpha, theta0=theta0)


@dataclass(frozen=True)
class MsprtSimulationResult:
    """A/A peeking comparison of mSPRT against a naive repeated z-test."""

    rate: float
    per_look: int
    looks: int
    simulations: int
    alpha: float
    tau: float
    msprt_false_positive_rate: float
    naive_false_positive_rate: float


def simulate_msprt_false_positives(
    rate: float = 0.1,
    per_look: int = 500,
    looks: int = 20,
    *,
    alpha: float = 0.05,
    tau: float = 0.02,
    simulations: int = 2000,
    seed: int | None = 20261018,
) -> MsprtSimulationResult:
    """Share of A/A experiments that ever reject when checked after every batch.

    Both arms convert at ``rate``. The naive policy stops the first time a
    two-sided z-test (same plug-in variance) has ``p < alpha``; the mSPRT
    stops the first time its always-valid p-value is at most ``alpha``.
    """
    alpha_f = _validate_alpha(alpha)
    tau_f = _validate_tau(tau)
    if not 0.0 < rate < 1.0:
        raise ValueError("rate must lie strictly inside (0, 1)")
    if per_look < 1 or looks < 1 or simulations < 1:
        raise ValueError("per_look, looks and simulations must be positive")
    rng = make_rng(seed)
    conv_a = np.cumsum(rng.binomial(per_look, rate, size=(simulations, looks)), axis=1)
    conv_b = np.cumsum(rng.binomial(per_look, rate, size=(simulations, looks)), axis=1)
    trials = per_look * np.arange(1, looks + 1, dtype=float)
    delta, var = _proportion_estimates(conv_a, trials, conv_b, trials)
    t2 = tau_f * tau_f
    log_lr = 0.5 * np.log(var / (var + t2)) + t2 * delta**2 / (2.0 * var * (var + t2))
    msprt_hits = np.any(log_lr >= math.log(1.0 / alpha_f), axis=1)
    z_crit = _normal_quantile(1.0 - alpha_f / 2.0)
    naive_hits = np.any(np.abs(delta) / np.sqrt(var) > z_crit, axis=1)
    return MsprtSimulationResult(
        rate=float(rate),
        per_look=int(per_look),
        looks=int(looks),
        simulations=int(simulations),
        alpha=alpha_f,
        tau=tau_f,
        msprt_false_positive_rate=float(msprt_hits.mean()),
        naive_false_positive_rate=float(naive_hits.mean()),
    )


def _normal_quantile(p: float) -> float:
    """Standard normal quantile by bisection on ``erf`` (no SciPy needed)."""
    lo, hi = -10.0, 10.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if 0.5 * (1.0 + math.erf(mid / math.sqrt(2.0))) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
