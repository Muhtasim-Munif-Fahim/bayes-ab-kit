"""Borrowing historical control data with (normalized) power priors.

A historical experiment with ``x0`` conversions out of ``n0`` visitors can
inform the current control arm, but only to the extent that the two are
compatible. The *power prior* (Ibrahim & Chen 2000) raises the historical
likelihood to a power ``a0 in [0, 1]``::

    pi(p | D0, a0) ∝ L(p | D0)^a0 * Beta(p; alpha0, beta0)
                   = Beta(alpha0 + a0 * x0, beta0 + a0 * (n0 - x0))

so ``a0 = 0`` ignores the history, ``a0 = 1`` pools it fully, and
``a0 * n0`` is the number of historical visitors effectively borrowed.

Instead of fixing ``a0``, the *normalized* power prior (Duan, Ye & Smith
2006) puts a ``Beta(c, d)`` prior on ``a0`` and lets the data decide: the
marginal posterior of ``a0`` is proportional to

    p(a0) * B(alpha0 + a0 x0 + x, beta0 + a0 (n0 - x0) + n - x)
          / B(alpha0 + a0 x0, beta0 + a0 (n0 - x0)),

which downweights the history automatically under prior-data conflict.
The posterior of ``p`` is then a mixture of Beta distributions over an
``a0`` grid. The empirical-Bayes choice ``a0_hat = argmax`` of the same
marginal likelihood (Gravestock & Held 2017) is also reported.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize, special, stats

from .posteriors import BetaBinomialPosterior

__all__ = [
    "PowerPriorResult",
    "HistoricalControlComparison",
    "power_prior_posterior",
    "normalized_power_prior",
    "compare_with_historical_control",
]


def _check_counts(x, n, label):
    if isinstance(x, bool) or isinstance(n, bool):
        raise TypeError(f"{label} counts must be integers")
    if int(n) != n or int(x) != x:
        raise ValueError(f"{label} counts must be integers")
    if n < 0 or not 0 <= x <= n:
        raise ValueError(f"{label} conversions must lie in [0, trials]")
    return int(x), int(n)


def _check_prior(alpha0, beta0):
    if not (np.isfinite(alpha0) and np.isfinite(beta0)) or alpha0 <= 0 or beta0 <= 0:
        raise ValueError("prior hyperparameters must be positive")


def power_prior_posterior(
    conversions: int,
    trials: int,
    hist_conversions: int,
    hist_trials: int,
    a0: float,
    alpha0: float = 1.0,
    beta0: float = 1.0,
) -> BetaBinomialPosterior:
    """Conjugate posterior for the current arm under a fixed power ``a0``."""
    x, n = _check_counts(conversions, trials, "current")
    x0, n0 = _check_counts(hist_conversions, hist_trials, "historical")
    _check_prior(alpha0, beta0)
    if not np.isfinite(a0) or not 0.0 <= a0 <= 1.0:
        raise ValueError("a0 must lie in [0, 1]")
    return BetaBinomialPosterior.from_counts(
        x, n, alpha0=alpha0 + a0 * x0, beta0=beta0 + a0 * (n0 - x0)
    )


@dataclass(frozen=True)
class PowerPriorResult:
    """Posterior from :func:`normalized_power_prior`."""

    a0_grid: np.ndarray
    a0_weights: np.ndarray
    a0_mean: float
    a0_eb: float
    hist_trials: int
    alphas: np.ndarray
    betas: np.ndarray

    @property
    def borrowed_sample_size(self) -> float:
        """Posterior-expected number of historical visitors borrowed, ``E[a0] * n0``."""
        return self.a0_mean * self.hist_trials

    def mean(self) -> float:
        return float(np.sum(self.a0_weights * self.alphas / (self.alphas + self.betas)))

    def cdf(self, p: float) -> float:
        return float(np.sum(self.a0_weights * stats.beta.cdf(p, self.alphas, self.betas)))

    def quantile(self, q: float) -> float:
        if not 0.0 < q < 1.0:
            raise ValueError("quantile level must lie in (0, 1)")
        return float(optimize.brentq(lambda p: self.cdf(p) - q, 0.0, 1.0, xtol=1e-12))

    def credible_interval(self, ci: float = 0.95) -> tuple[float, float]:
        if not 0.0 < ci < 1.0:
            raise ValueError("ci must lie in (0, 1)")
        tail = (1.0 - ci) / 2.0
        return self.quantile(tail), self.quantile(1.0 - tail)

    def sample(self, size: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """Draw conversion rates from the Beta-mixture posterior."""
        rng = np.random.default_rng() if rng is None else rng
        idx = rng.choice(self.a0_weights.size, size=size, p=self.a0_weights)
        return rng.beta(self.alphas[idx], self.betas[idx])


def _log_marginal(a0, x, n, x0, n0, alpha0, beta0):
    a_prior = alpha0 + a0 * x0
    b_prior = beta0 + a0 * (n0 - x0)
    return special.betaln(a_prior + x, b_prior + n - x) - special.betaln(a_prior, b_prior)


def normalized_power_prior(
    conversions: int,
    trials: int,
    hist_conversions: int,
    hist_trials: int,
    alpha0: float = 1.0,
    beta0: float = 1.0,
    a0_prior: tuple[float, float] = (1.0, 1.0),
    grid_size: int = 401,
) -> PowerPriorResult:
    """Posterior for the current arm with ``a0 ~ Beta(*a0_prior)`` learned from data."""
    x, n = _check_counts(conversions, trials, "current")
    x0, n0 = _check_counts(hist_conversions, hist_trials, "historical")
    _check_prior(alpha0, beta0)
    c, d = (float(v) for v in a0_prior)
    if not (c > 0 and d > 0):
        raise ValueError("a0_prior parameters must be positive")
    if isinstance(grid_size, bool) or not isinstance(grid_size, (int, np.integer)) or grid_size < 3:
        raise ValueError("grid_size must be an integer >= 3")

    # midpoint grid on (0, 1) so Beta(c, d) priors with c or d < 1 stay finite
    grid = (np.arange(grid_size) + 0.5) / grid_size
    log_m = _log_marginal(grid, x, n, x0, n0, alpha0, beta0)
    log_w = log_m + stats.beta.logpdf(grid, c, d)
    log_w -= special.logsumexp(log_w)
    weights = np.exp(log_w)
    weights /= weights.sum()

    res = optimize.minimize_scalar(
        lambda a: -_log_marginal(a, x, n, x0, n0, alpha0, beta0),
        bounds=(0.0, 1.0),
        method="bounded",
        options={"xatol": 1e-6},
    )
    a0_eb = float(res.x)
    # bounded search never evaluates the endpoints exactly; check them
    ends = {0.0: _log_marginal(0.0, x, n, x0, n0, alpha0, beta0),
            1.0: _log_marginal(1.0, x, n, x0, n0, alpha0, beta0)}
    best_end = max(ends, key=ends.get)
    if ends[best_end] >= -res.fun:
        a0_eb = best_end

    alphas = alpha0 + grid * x0 + x
    betas = beta0 + grid * (n0 - x0) + n - x
    return PowerPriorResult(
        a0_grid=grid,
        a0_weights=weights,
        a0_mean=float(np.sum(weights * grid)),
        a0_eb=a0_eb,
        hist_trials=n0,
        alphas=alphas,
        betas=betas,
    )


@dataclass(frozen=True)
class HistoricalControlComparison:
    """Treatment vs. a control arm that borrows historical data."""

    control: PowerPriorResult
    treatment: BetaBinomialPosterior
    prob_treatment_better: float
    expected_lift: float
    lift_interval: tuple[float, float]
    prob_treatment_better_no_borrowing: float


def compare_with_historical_control(
    control_conversions: int,
    control_trials: int,
    treatment_conversions: int,
    treatment_trials: int,
    hist_conversions: int,
    hist_trials: int,
    alpha0: float = 1.0,
    beta0: float = 1.0,
    a0_prior: tuple[float, float] = (1.0, 1.0),
    ci: float = 0.95,
    n_samples: int = 200_000,
    seed: int | None = 20261112,
) -> HistoricalControlComparison:
    """Compare a treatment to a control augmented by a normalized power prior.

    Returns ``P(p_T > p_C)``, the posterior mean and interval of the
    absolute lift ``p_T - p_C``, and the same probability without
    borrowing for reference.
    """
    if isinstance(n_samples, bool) or not isinstance(n_samples, (int, np.integer)) or n_samples <= 0:
        raise ValueError("n_samples must be a positive integer")
    if not 0.0 < ci < 1.0:
        raise ValueError("ci must lie in (0, 1)")
    control = normalized_power_prior(
        control_conversions, control_trials, hist_conversions, hist_trials,
        alpha0=alpha0, beta0=beta0, a0_prior=a0_prior,
    )
    xt, nt = _check_counts(treatment_conversions, treatment_trials, "treatment")
    treatment = BetaBinomialPosterior.from_counts(xt, nt, alpha0=alpha0, beta0=beta0)
    rng = np.random.default_rng(seed)
    pc = control.sample(int(n_samples), rng)
    pt = rng.beta(treatment.alpha, treatment.beta, size=int(n_samples))
    lift = pt - pc
    plain_c = BetaBinomialPosterior.from_counts(
        control_conversions, control_trials, alpha0=alpha0, beta0=beta0
    )
    pc0 = rng.beta(plain_c.alpha, plain_c.beta, size=int(n_samples))
    tail = (1.0 - ci) / 2.0
    lo, hi = np.quantile(lift, [tail, 1.0 - tail])
    return HistoricalControlComparison(
        control=control,
        treatment=treatment,
        prob_treatment_better=float(np.mean(lift > 0)),
        expected_lift=float(lift.mean()),
        lift_interval=(float(lo), float(hi)),
        prob_treatment_better_no_borrowing=float(np.mean(pt > pc0)),
    )
