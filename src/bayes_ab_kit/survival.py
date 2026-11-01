"""Bayesian time-to-event comparison with right censoring (Weibull model).

Many experiment outcomes are *durations*: time to first purchase, time to
churn, session length, time until a ticket is resolved. When the experiment
ends, some units have not had the event yet, so their time is only a lower
bound (right-censored). Dropping or truncating those units biases the result.

We model each arm with a Weibull distribution in the proportional-hazards
parameterisation

* survival ``S(t) = exp(-lam * t**k)``, hazard ``h(t) = lam * k * t**(k-1)``.

For a *fixed* shape ``k`` the scale ``lam`` is conjugate: with a
``Gamma(a0, b0)`` prior and data ``(t_i, d_i)`` (``d_i = 1`` for an observed
event, ``0`` for censored) the posterior is

    lam | k ~ Gamma(a0 + D, b0 + sum_i t_i**k),   D = sum_i d_i.

Integrating ``lam`` out gives the exact marginal likelihood of ``k``

    log p(data | k) = D log k + (k - 1) sum_{d_i=1} log t_i
                      + log Gamma(a0 + D) - (a0 + D) log(b0 + sum t_i**k)
                      + a0 log b0 - log Gamma(a0),

so the posterior over ``k`` is computed exactly on a grid (log-uniform prior
by default). There is no MCMC: draws come from picking ``k`` from the grid
posterior and then ``lam`` from its Gamma conditional. ``k = 1`` is the
exponential model (constant hazard). ``k > 1`` means the hazard rises over
time and ``k < 1`` means it falls.

Comparisons use posterior draws of the **median time to event**
``(log 2 / lam) ** (1/k)`` and of the **restricted mean survival time**
up to a horizon ``tau``

    RMST(tau) = int_0^tau S(t) dt = lam**(-1/k) Gamma(1 + 1/k) P(1/k, lam tau**k)

where ``P`` is the regularized lower incomplete gamma function. RMST stays
well defined when the hazards cross, so no proportional-hazards assumption
between arms is needed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import special

from .sampling import make_rng

_LOG2 = float(np.log(2.0))


def _validate_data(times, events) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(times, dtype=float).ravel()
    if t.size == 0:
        raise ValueError("times must not be empty")
    if not np.all(np.isfinite(t)) or np.any(t <= 0):
        raise ValueError("times must be positive finite numbers")
    if events is None:
        d = np.ones_like(t)
    else:
        d = np.asarray(events, dtype=float).ravel()
        if d.shape != t.shape:
            raise ValueError("events must have the same length as times")
        if not np.all((d == 0) | (d == 1)):
            raise ValueError("events must be 0 (censored) or 1 (observed)")
    if d.sum() == 0:
        raise ValueError("at least one observed (uncensored) event is required")
    return t, d


def default_shape_grid(n: int = 400, low: float = 0.1, high: float = 10.0) -> np.ndarray:
    """Log-spaced grid of Weibull shapes (uniform prior in ``log k``)."""
    if n < 2:
        raise ValueError("shape grid needs at least 2 points")
    if not 0 < low < high:
        raise ValueError("need 0 < low < high")
    return np.geomspace(low, high, int(n))


class WeibullPosterior:
    """Posterior over Weibull ``(k, lam)`` from right-censored durations.

    ``shape_grid`` is the support of the (discrete, uniform) prior over the
    shape ``k``; ``a0, b0`` are the Gamma prior on ``lam`` given ``k``. The
    defaults are weakly informative. Because ``lam`` carries the time
    units through ``t**k``, they work best when times are in "natural" units
    (not, say, milliseconds for day-long durations).
    """

    def __init__(
        self,
        times,
        events=None,
        *,
        shape_grid=None,
        a0: float = 1e-3,
        b0: float = 1e-3,
    ) -> None:
        if a0 <= 0 or b0 <= 0:
            raise ValueError("prior hyperparameters a0 and b0 must be positive")
        self.times, self.events = _validate_data(times, events)
        grid = default_shape_grid() if shape_grid is None else np.asarray(shape_grid, dtype=float).ravel()
        if grid.size < 1 or np.any(~np.isfinite(grid)) or np.any(grid <= 0):
            raise ValueError("shape_grid must contain positive finite values")
        self.shape_grid = grid
        self.a0 = float(a0)
        self.b0 = float(b0)

        n_events = float(self.events.sum())
        sum_log_t_events = float(np.sum(self.events * np.log(self.times)))
        log_t = np.log(self.times)
        # sum_i t_i**k for every grid k, computed in log space for stability
        log_tk = grid[:, None] * log_t[None, :]
        self.exposure = np.exp(special.logsumexp(log_tk, axis=1))
        self.post_shape = self.a0 + n_events
        self.post_rate = self.b0 + self.exposure
        loglik = (
            n_events * np.log(grid)
            + (grid - 1.0) * sum_log_t_events
            + special.gammaln(self.post_shape)
            - self.post_shape * np.log(self.post_rate)
            + self.a0 * np.log(self.b0)
            - special.gammaln(self.a0)
        )
        self.log_marginal = loglik
        w = np.exp(loglik - loglik.max())
        self.shape_weights = w / w.sum()

    @property
    def n(self) -> int:
        return int(self.times.size)

    @property
    def n_events(self) -> int:
        return int(self.events.sum())

    def shape_mean(self) -> float:
        return float(np.sum(self.shape_weights * self.shape_grid))

    def shape_interval(self, ci: float = 0.95) -> tuple[float, float]:
        _check_ci(ci)
        cdf = np.cumsum(self.shape_weights)
        tail = (1.0 - ci) / 2.0
        lo = self.shape_grid[min(int(np.searchsorted(cdf, tail)), cdf.size - 1)]
        hi = self.shape_grid[min(int(np.searchsorted(cdf, 1.0 - tail)), cdf.size - 1)]
        return float(lo), float(hi)

    def sample(self, rng: np.random.Generator, size: int) -> tuple[np.ndarray, np.ndarray]:
        """Joint draws ``(k, lam)`` from the exact posterior."""
        if size <= 0:
            raise ValueError("size must be positive")
        idx = rng.choice(self.shape_grid.size, size=size, p=self.shape_weights)
        lam = rng.gamma(self.post_shape, 1.0 / self.post_rate[idx])
        return self.shape_grid[idx], lam

    def survival(self, t, rng: np.random.Generator | None = None, size: int = 4000, ci: float = 0.95):
        """Posterior mean and ``ci`` band of ``S(t)`` at times ``t``.

        Returns ``(mean, lower, upper)`` arrays with the shape of ``t``.
        """
        _check_ci(ci)
        tt = np.atleast_1d(np.asarray(t, dtype=float))
        if np.any(tt < 0):
            raise ValueError("t must be non-negative")
        k, lam = self.sample(make_rng(0) if rng is None else rng, size)
        surv = np.exp(-lam[:, None] * tt[None, :] ** k[:, None])
        tail = (1.0 - ci) / 2.0
        return (
            surv.mean(axis=0),
            np.quantile(surv, tail, axis=0),
            np.quantile(surv, 1.0 - tail, axis=0),
        )


def _check_ci(ci: float) -> None:
    if not 0 < ci < 1:
        raise ValueError("ci must lie in (0, 1)")


def weibull_median(k, lam) -> np.ndarray:
    """Median time to event ``(log 2 / lam) ** (1/k)``."""
    k = np.asarray(k, dtype=float)
    lam = np.asarray(lam, dtype=float)
    return (_LOG2 / lam) ** (1.0 / k)


def weibull_rmst(k, lam, tau: float) -> np.ndarray:
    """Restricted mean survival time up to ``tau`` (closed form)."""
    if not np.isfinite(tau) or tau <= 0:
        raise ValueError("tau must be a positive finite number")
    k = np.asarray(k, dtype=float)
    lam = np.asarray(lam, dtype=float)
    a = 1.0 / k
    return lam ** (-a) * special.gamma(1.0 + a) * special.gammainc(a, lam * tau**k)


def simulate_weibull_survival(
    n: int,
    shape: float,
    scale: float,
    censor_time: float | None = None,
    seed: int | None = 20261101,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw ``n`` Weibull durations (``S(t) = exp(-(t/scale)**shape)``).

    Units still event-free at ``censor_time`` are administratively censored.
    Returns ``(times, events)``.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if shape <= 0 or scale <= 0:
        raise ValueError("shape and scale must be positive")
    rng = make_rng(seed)
    t = scale * rng.weibull(shape, size=int(n))
    if censor_time is None:
        return t, np.ones(int(n), dtype=int)
    if censor_time <= 0:
        raise ValueError("censor_time must be positive")
    events = (t <= censor_time).astype(int)
    return np.minimum(t, censor_time), events


@dataclass(frozen=True)
class SurvivalArmSummary:
    n: int
    n_events: int
    shape_mean: float
    shape_interval: tuple[float, float]
    median_mean: float
    median_interval: tuple[float, float]
    rmst_mean: float
    rmst_interval: tuple[float, float]


@dataclass(frozen=True)
class SurvivalComparisonResult:
    """Two-arm Weibull time-to-event comparison.

    ``prob_b_better`` and the expected losses are based on RMST up to
    ``tau``. ``longer_is_better`` says whether a longer time to event is good
    (retention, time to churn) or bad (time to convert, time to resolve).
    """

    a: SurvivalArmSummary
    b: SurvivalArmSummary
    tau: float
    rmst_difference: float
    rmst_difference_interval: tuple[float, float]
    median_ratio: float
    median_ratio_interval: tuple[float, float]
    prob_b_better: float
    prob_b_longer_median: float
    expected_loss_choose_a: float
    expected_loss_choose_b: float
    longer_is_better: bool
    ci: float


def _interval(x: np.ndarray, ci: float) -> tuple[float, float]:
    tail = (1.0 - ci) / 2.0
    lo, hi = np.quantile(x, [tail, 1.0 - tail])
    return float(lo), float(hi)


def _summarize(post: WeibullPosterior, k, lam, tau: float, ci: float):
    med = weibull_median(k, lam)
    rmst = weibull_rmst(k, lam, tau)
    summary = SurvivalArmSummary(
        n=post.n,
        n_events=post.n_events,
        shape_mean=post.shape_mean(),
        shape_interval=post.shape_interval(ci),
        median_mean=float(np.mean(med)),
        median_interval=_interval(med, ci),
        rmst_mean=float(np.mean(rmst)),
        rmst_interval=_interval(rmst, ci),
    )
    return summary, med, rmst


def compare_survival(
    times_a,
    events_a,
    times_b,
    events_b,
    *,
    tau: float | None = None,
    longer_is_better: bool = True,
    ci: float = 0.95,
    shape_grid=None,
    a0: float = 1e-3,
    b0: float = 1e-3,
    n_samples: int = 40_000,
    seed: int | None = 20261101,
) -> SurvivalComparisonResult:
    """Compare two arms' time-to-event distributions under Weibull models.

    ``tau`` defaults to the smaller of the two arms' largest observed times,
    the usual choice for RMST, so neither arm is extrapolated past its data.
    """
    _check_ci(ci)
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    post_a = WeibullPosterior(times_a, events_a, shape_grid=shape_grid, a0=a0, b0=b0)
    post_b = WeibullPosterior(times_b, events_b, shape_grid=shape_grid, a0=a0, b0=b0)
    if tau is None:
        tau = float(min(post_a.times.max(), post_b.times.max()))
    if not np.isfinite(tau) or tau <= 0:
        raise ValueError("tau must be a positive finite number")

    rng = make_rng(seed)
    ka, la = post_a.sample(rng, n_samples)
    kb, lb = post_b.sample(rng, n_samples)
    sum_a, med_a, rmst_a = _summarize(post_a, ka, la, tau, ci)
    sum_b, med_b, rmst_b = _summarize(post_b, kb, lb, tau, ci)

    diff = rmst_b - rmst_a
    gain = diff if longer_is_better else -diff
    ratio = med_b / med_a
    p_longer_median = float(np.mean(med_b > med_a))
    return SurvivalComparisonResult(
        a=sum_a,
        b=sum_b,
        tau=float(tau),
        rmst_difference=float(np.mean(diff)),
        rmst_difference_interval=_interval(diff, ci),
        median_ratio=float(np.median(ratio)),
        median_ratio_interval=_interval(ratio, ci),
        prob_b_better=float(np.mean(gain > 0)),
        prob_b_longer_median=p_longer_median,
        expected_loss_choose_a=float(np.mean(np.maximum(gain, 0.0))),
        expected_loss_choose_b=float(np.mean(np.maximum(-gain, 0.0))),
        longer_is_better=bool(longer_is_better),
        ci=float(ci),
    )


__all__ = [
    "SurvivalArmSummary",
    "SurvivalComparisonResult",
    "WeibullPosterior",
    "compare_survival",
    "default_shape_grid",
    "simulate_weibull_survival",
    "weibull_median",
    "weibull_rmst",
]
