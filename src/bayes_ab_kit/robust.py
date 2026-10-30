"""Robust Bayesian comparison of means with a Student-t likelihood.

Revenue per user, latency and session length often have heavy tails. A few
extreme values can move a normal-model (or plain bootstrap) mean far enough
to flip a decision. Following Kruschke's BEST (2013), each arm is modelled as

    y_i ~ Student-t(nu, mu, sigma)

so outliers are absorbed by heavy tails instead of dragging ``mu``. The
degrees of freedom ``nu`` are learned from the data: a large ``nu`` means the
data look normal, and a small ``nu`` means heavy tails.

Inference is an exact Gibbs sampler that uses the scale-mixture form of the
t distribution, ``y_i | lam_i ~ N(mu, sigma^2 / lam_i)`` with
``lam_i ~ Gamma(nu/2, rate=nu/2)``:

* ``lam_i | . ~ Gamma((nu + 1)/2, rate=(nu + (y_i - mu)^2 / sigma^2) / 2)``
* ``mu | . ~ Normal`` (conjugate, with a wide normal prior centred on the
  sample median),
* ``sigma^2 | . ~ InvGamma(n/2, sum lam_i (y_i - mu)^2 / 2)`` (Jeffreys prior
  ``p(sigma^2) ~ 1/sigma^2``),
* ``nu | lam`` is drawn exactly from its full conditional on a fine grid,
  under BEST's shifted-exponential prior ``nu - 1 ~ Exp(mean 29)``. Pass a
  fixed ``nu`` to turn this step off.

:func:`robust_posterior` fits one arm. :func:`compare_robust_means` fits both
arms and reports the difference of locations, the relative lift, the
standardised effect size ``(mu_B - mu_A) / sqrt((sigma_A^2 + sigma_B^2)/2)``,
``P(B better)`` and the expected loss of each decision. These follow the same
conventions as :mod:`bayes_ab_kit.bootstrap`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

import numpy as np
from scipy.special import gammaln

__all__ = [
    "RobustPosterior",
    "RobustComparison",
    "robust_posterior",
    "compare_robust_means",
]

_NU_GRID = np.unique(np.concatenate([np.linspace(1.0, 10.0, 91), np.geomspace(10.0, 300.0, 120)]))


@dataclass
class RobustPosterior:
    """Posterior draws of a Student-t model for one arm."""

    n: int
    mu_draws: np.ndarray
    sigma_draws: np.ndarray
    nu_draws: np.ndarray
    ci: float

    @property
    def mu(self) -> float:
        return float(np.mean(self.mu_draws))

    @property
    def mu_interval(self) -> Tuple[float, float]:
        tail = (1.0 - self.ci) / 2.0
        lo, hi = np.quantile(self.mu_draws, [tail, 1.0 - tail])
        return float(lo), float(hi)

    @property
    def sigma(self) -> float:
        return float(np.mean(self.sigma_draws))

    @property
    def nu_median(self) -> float:
        return float(np.median(self.nu_draws))

    @property
    def prob_heavy_tailed(self) -> float:
        """Posterior probability that ``nu < 10`` (clearly heavier than normal tails)."""
        return float(np.mean(self.nu_draws < 10.0))

    def to_dict(self) -> dict:
        lo, hi = self.mu_interval
        return {
            "n": self.n,
            "mu": self.mu,
            "mu_interval": [lo, hi],
            "sigma": self.sigma,
            "nu_median": self.nu_median,
            "prob_heavy_tailed": self.prob_heavy_tailed,
            "ci": self.ci,
            "n_draws": int(self.mu_draws.size),
        }


@dataclass
class RobustComparison:
    """Two-arm comparison of Student-t locations (B minus A)."""

    a: RobustPosterior
    b: RobustPosterior
    difference: float
    difference_interval: Tuple[float, float]
    relative_lift: Optional[float]
    relative_lift_interval: Optional[Tuple[float, float]]
    effect_size: float
    effect_size_interval: Tuple[float, float]
    prob_b_better: float
    expected_loss_choose_a: float
    expected_loss_choose_b: float
    higher_is_better: bool
    ci: float

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k not in ("a", "b")}
        d["a"] = self.a.to_dict()
        d["b"] = self.b.to_dict()
        return d


def _validate_ci(ci: float) -> float:
    if not 0.0 < float(ci) < 1.0:
        raise ValueError("ci must be in (0, 1)")
    return float(ci)


def _nu_log_prior(grid: np.ndarray) -> np.ndarray:
    return -(grid - 1.0) / 29.0


def robust_posterior(
    values: Sequence[float],
    *,
    nu: Optional[float] = None,
    n_draws: int = 4000,
    burn_in: int = 1000,
    prior_scale: float = 100.0,
    ci: float = 0.95,
    seed: Optional[int] = None,
) -> RobustPosterior:
    """Gibbs-sample the Student-t posterior of one arm.

    Parameters
    ----------
    values:
        Observed values (at least 3, finite, not all equal).
    nu:
        Fix the degrees of freedom (``> 0``). Use ``None`` (the default) to
        learn them. A very large ``nu`` gives back the normal model.
    n_draws, burn_in:
        Kept draws and discarded warm-up iterations.
    prior_scale:
        The prior on ``mu`` is ``Normal(median, (prior_scale * MAD-scale)^2)``,
        which is effectively flat at the data's scale.
    """

    ci = _validate_ci(ci)
    y = np.asarray(values, dtype=float).ravel()
    if y.size < 3:
        raise ValueError("need at least 3 observations per arm")
    if not np.all(np.isfinite(y)):
        raise ValueError("values must be finite")
    if isinstance(n_draws, bool) or int(n_draws) < 1 or int(n_draws) != n_draws:
        raise ValueError("n_draws must be a positive integer")
    if isinstance(burn_in, bool) or int(burn_in) < 0 or int(burn_in) != burn_in:
        raise ValueError("burn_in must be a non-negative integer")
    if nu is not None and not (np.isfinite(nu) and nu > 0):
        raise ValueError("nu must be a positive number or None")
    if prior_scale <= 0:
        raise ValueError("prior_scale must be positive")
    n = y.size
    med = float(np.median(y))
    scale = 1.4826 * float(np.median(np.abs(y - med)))
    if scale <= 0:
        scale = float(np.std(y))
    if scale <= 0:
        raise ValueError("values must not all be equal")
    m0, v0 = med, (prior_scale * scale) ** 2

    rng = np.random.default_rng(seed)
    mu, sig2 = med, scale**2
    cur_nu = 5.0 if nu is None else float(nu)
    grid = _NU_GRID
    log_prior = _nu_log_prior(grid)
    half = grid / 2.0
    const = n * (half * np.log(half) - gammaln(half))
    total = int(burn_in) + int(n_draws)
    mus = np.empty(int(n_draws))
    sigs = np.empty(int(n_draws))
    nus = np.empty(int(n_draws))
    for it in range(total):
        r2 = (y - mu) ** 2
        lam = rng.gamma((cur_nu + 1.0) / 2.0, 2.0 / (cur_nu + r2 / sig2))
        prec = 1.0 / v0 + lam.sum() / sig2
        mean = (m0 / v0 + (lam * y).sum() / sig2) / prec
        mu = rng.normal(mean, 1.0 / np.sqrt(prec))
        ss = float((lam * (y - mu) ** 2).sum())
        sig2 = (ss / 2.0) / rng.gamma(n / 2.0)
        if nu is None:
            slog, ssum = float(np.log(lam).sum()), float(lam.sum())
            logp = const + (half - 1.0) * slog - half * ssum + log_prior
            p = np.exp(logp - logp.max())
            cur_nu = float(grid[rng.choice(grid.size, p=p / p.sum())])
        k = it - int(burn_in)
        if k >= 0:
            mus[k], sigs[k], nus[k] = mu, np.sqrt(sig2), cur_nu
    return RobustPosterior(n=n, mu_draws=mus, sigma_draws=sigs, nu_draws=nus, ci=ci)


def compare_robust_means(
    a: Union[Sequence[float], RobustPosterior],
    b: Union[Sequence[float], RobustPosterior],
    *,
    higher_is_better: bool = True,
    nu: Optional[float] = None,
    n_draws: int = 4000,
    burn_in: int = 1000,
    ci: float = 0.95,
    seed: Optional[int] = None,
) -> RobustComparison:
    """Compare Student-t locations of control ``a`` and treatment ``b``.

    ``a``/``b`` are raw samples or existing :class:`RobustPosterior` objects
    with the same number of draws. Independent seeds for the two chains are
    derived from ``seed``. The relative lift ``mu_B / mu_A - 1`` is reported
    only when every draw of ``mu_A`` is positive.
    """

    ci = _validate_ci(ci)
    ss = np.random.SeedSequence(seed)
    sa, sb = (int(s.generate_state(1)[0]) for s in ss.spawn(2))
    kw = dict(nu=nu, n_draws=n_draws, burn_in=burn_in, ci=ci)
    pa = a if isinstance(a, RobustPosterior) else robust_posterior(a, seed=sa, **kw)
    pb = b if isinstance(b, RobustPosterior) else robust_posterior(b, seed=sb, **kw)
    if pa.mu_draws.size != pb.mu_draws.size:
        raise ValueError("posteriors must have the same number of draws")
    tail = (1.0 - ci) / 2.0
    diff = pb.mu_draws - pa.mu_draws
    dlo, dhi = np.quantile(diff, [tail, 1.0 - tail])
    pooled = np.sqrt((pa.sigma_draws**2 + pb.sigma_draws**2) / 2.0)
    eff = diff / pooled
    elo, ehi = np.quantile(eff, [tail, 1.0 - tail])
    gain = diff if higher_is_better else -diff
    prob = float(np.mean(gain > 0) + 0.5 * np.mean(gain == 0))
    if np.all(pa.mu_draws > 0):
        lift = pb.mu_draws / pa.mu_draws - 1.0
        llo, lhi = np.quantile(lift, [tail, 1.0 - tail])
        rel: Optional[float] = float(pb.mu / pa.mu - 1.0)
        rel_int: Optional[Tuple[float, float]] = (float(llo), float(lhi))
    else:
        rel, rel_int = None, None
    return RobustComparison(
        a=pa, b=pb,
        difference=float(np.mean(diff)), difference_interval=(float(dlo), float(dhi)),
        relative_lift=rel, relative_lift_interval=rel_int,
        effect_size=float(np.mean(eff)), effect_size_interval=(float(elo), float(ehi)),
        prob_b_better=prob,
        expected_loss_choose_a=float(np.mean(np.maximum(gain, 0.0))),
        expected_loss_choose_b=float(np.mean(np.maximum(-gain, 0.0))),
        higher_is_better=higher_is_better, ci=ci,
    )
