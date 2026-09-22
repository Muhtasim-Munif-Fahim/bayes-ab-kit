"""Empirical-Bayes hierarchical Beta-Binomial shrinkage for many variants.

Each arm has its own conversion rate, and those rates share a Beta prior.
The prior hyperparameters are learned by maximising the beta-binomial
marginal likelihood (empirical Bayes), then each arm is updated with that
shared prior. Small samples are pulled toward the prior mean; large samples
stay near their observed rate.

The fitted prior is treated as known. Credible intervals are equal-tailed
Beta intervals and do not include uncertainty in the hyperparameters.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import exp, log

import numpy as np
from scipy.optimize import minimize
from scipy.special import digamma, gammaln

from .posteriors import BetaBinomialPosterior

# Numerical bounds for the shared prior mean and concentration κ = α + β.
# κ → 0 is a diffuse prior; a large κ pools every arm at the prior mean.
_MU_MIN = 1e-4
_MU_MAX = 1.0 - _MU_MIN
_KAPPA_MIN = 0.05
_KAPPA_MAX = 1e6


@dataclass(frozen=True)
class ShrunkArm:
    """One arm after shrinkage toward the shared Beta prior.

    ``posterior_mean`` is the mean of ``Beta(α + conversions, β + failures)``.
    ``prior_weight`` is ``(α + β) / (trials + α + β)``: the fraction of that
    mean borrowed from the grand mean rather than the raw rate. Optional
    ``ci_lower`` / ``ci_upper`` are the equal-tailed credible interval; both
    are ``None`` when no interval was requested.
    """

    name: str
    conversions: int
    trials: int
    raw_rate: float
    posterior_mean: float
    prior_weight: float
    ci_lower: float | None = None
    ci_upper: float | None = None


@dataclass(frozen=True)
class HierarchicalBetaResult:
    """Shared empirical-Bayes Beta prior and the shrunk arm posteriors.

    ``arms`` preserves input order. ``prior_mean`` is the shrinkage target
    ``α / (α + β)``. ``posteriors()`` rebuilds independent Beta-Binomial
    posteriors that use this shared prior, so they can be passed to ROPE,
    P(best), and uplift helpers. Those downstream calls still treat the
    prior as fixed.
    """

    arms: tuple[ShrunkArm, ...]
    prior_alpha: float
    prior_beta: float
    prior_mean: float
    prior_strength: float
    ci: float | None

    def posteriors(self) -> dict[str, BetaBinomialPosterior]:
        """Beta-Binomial posteriors conditioned on the fitted shared prior."""
        return {
            arm.name: BetaBinomialPosterior.from_counts(
                arm.conversions,
                arm.trials,
                alpha0=self.prior_alpha,
                beta0=self.prior_beta,
            )
            for arm in self.arms
        }


def _validate_ci(ci: float) -> None:
    if not 0 < ci < 1:
        raise ValueError("credible level must lie strictly inside (0, 1)")


def _require_count(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{label} must be an integer")
    parsed = int(value)
    if parsed < 0:
        raise ValueError(f"{label} must be non-negative")
    return parsed


def _coerce_counts(value: object) -> tuple[int, int]:
    if isinstance(value, BetaBinomialPosterior):
        conversions, trials = value.successes, value.trials
    elif isinstance(value, (tuple, list)) and len(value) == 2:
        conversions = _require_count(value[0], "conversions")
        trials = _require_count(value[1], "trials")
    else:
        raise ValueError(
            "each arm must be (conversions, trials) or a BetaBinomialPosterior"
        )
    if trials <= 0:
        raise ValueError("trial count must be positive")
    if conversions > trials:
        raise ValueError("conversion count must lie in [0, trials]")
    return conversions, trials


def _parse_arms(
    arms: Mapping[str, tuple[int, int] | BetaBinomialPosterior],
) -> list[tuple[str, int, int]]:
    if not isinstance(arms, Mapping):
        raise ValueError("arms must be a mapping of name to counts or posteriors")
    if len(arms) < 2:
        raise ValueError("at least two arms are required")

    parsed: list[tuple[str, int, int]] = []
    seen: set[str] = set()
    for raw_name, value in arms.items():
        name = str(raw_name).strip()
        if not name:
            raise ValueError("arm name must be non-empty")
        if name in seen:
            raise ValueError(f"duplicate arm name: {name}")
        seen.add(name)
        conversions, trials = _coerce_counts(value)
        parsed.append((name, conversions, trials))
    return parsed


def _beta_binomial_loglik(
    alpha: float,
    beta: float,
    conversions: np.ndarray,
    trials: np.ndarray,
) -> float:
    """Sum of beta-binomial log PMFs, including the binomial coefficient."""
    failures = trials - conversions
    log_binom = gammaln(trials + 1.0) - gammaln(conversions + 1.0) - gammaln(failures + 1.0)
    log_beta_ratio = (
        gammaln(alpha + conversions)
        + gammaln(beta + failures)
        - gammaln(alpha + beta + trials)
        - gammaln(alpha)
        - gammaln(beta)
        + gammaln(alpha + beta)
    )
    return float(np.sum(log_binom + log_beta_ratio))


def _log_marginal_and_grad(
    mu: float,
    kappa: float,
    conversions: np.ndarray,
    trials: np.ndarray,
) -> tuple[float, float, float]:
    """Beta-binomial log marginal and derivatives in ``(μ, κ)``.

    ``μ = α / (α + β)`` and ``κ = α + β``. The binomial coefficient does
    not depend on the prior, so it is omitted from the gradient.
    """
    alpha = mu * kappa
    beta_param = (1.0 - mu) * kappa
    failures = trials - conversions
    k = conversions.size

    ll = _beta_binomial_loglik(alpha, beta_param, conversions, trials)
    d_alpha = float(
        np.sum(digamma(alpha + conversions) - digamma(kappa + trials))
        - k * (digamma(alpha) - digamma(kappa))
    )
    d_beta = float(
        np.sum(digamma(beta_param + failures) - digamma(kappa + trials))
        - k * (digamma(beta_param) - digamma(kappa))
    )
    d_mu = (d_alpha - d_beta) * kappa
    d_kappa = d_alpha * mu + d_beta * (1.0 - mu)
    return ll, d_mu, d_kappa


def _moment_start(conversions: np.ndarray, trials: np.ndarray) -> tuple[float, float]:
    """Method-of-moments ``(μ, κ)`` used to start the marginal MLE."""
    total = float(trials.sum())
    mu = float(np.clip(conversions.sum() / total, _MU_MIN, _MU_MAX))
    rates = conversions / trials
    weighted_var = float(np.sum(trials * (rates - mu) ** 2) / total)
    binomial_floor = conversions.size / total
    mu_var = mu * (1.0 - mu)
    ratio = weighted_var / mu_var
    if ratio <= binomial_floor or binomial_floor >= 1.0:
        return mu, _KAPPA_MAX
    rho = (ratio - binomial_floor) / (1.0 - binomial_floor)
    rho = float(np.clip(rho, 1e-12, 1.0 - 1.0 / (_KAPPA_MAX + 1.0)))
    kappa = float(np.clip(1.0 / rho - 1.0, _KAPPA_MIN, _KAPPA_MAX))
    return mu, kappa


def _fit_shared_beta(
    conversions: np.ndarray,
    trials: np.ndarray,
) -> tuple[float, float]:
    """Maximum-likelihood ``(α, β)`` of the shared Beta prior."""
    mu_mom, kappa_mom = _moment_start(conversions, trials)
    rates = conversions / trials
    mu_unweighted = float(np.clip(rates.mean(), _MU_MIN, _MU_MAX))
    starts = [(mu_mom, kappa_mom), (mu_unweighted, kappa_mom)]
    for kappa in (0.5, 2.0, 10.0, 100.0, 1_000.0, _KAPPA_MAX):
        starts.append((mu_mom, kappa))

    bounds = [(_MU_MIN, _MU_MAX), (log(_KAPPA_MIN), log(_KAPPA_MAX))]

    def objective(x: np.ndarray) -> tuple[float, np.ndarray]:
        mu = float(x[0])
        kappa = exp(float(x[1]))
        ll, d_mu, d_kappa = _log_marginal_and_grad(mu, kappa, conversions, trials)
        gradient = np.array([-d_mu, -d_kappa * kappa], dtype=float)
        return -ll, gradient

    best_fun = np.inf
    best_mu = mu_mom
    best_kappa = kappa_mom
    for mu0, kappa0 in starts:
        x0 = np.array([mu0, log(kappa0)], dtype=float)
        fitted = minimize(
            objective,
            x0,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={"ftol": 1e-14, "gtol": 1e-8, "maxiter": 250},
        )
        if not np.isfinite(fitted.fun):
            continue
        if fitted.fun < best_fun:
            best_fun = float(fitted.fun)
            best_mu = float(fitted.x[0])
            best_kappa = exp(float(fitted.x[1]))

    if not np.isfinite(best_fun):
        raise ValueError("empirical Bayes fit failed to converge")

    alpha = best_mu * best_kappa
    beta_param = (1.0 - best_mu) * best_kappa
    return alpha, beta_param


def hierarchical_beta_shrinkage(
    arms: Mapping[str, tuple[int, int] | BetaBinomialPosterior],
    ci: float | None = 0.95,
) -> HierarchicalBetaResult:
    """Shrink conversion arms toward a shared Beta prior learned from them.

    ``arms`` maps a variant name to ``(conversions, trials)`` or to a
    :class:`~bayes_ab_kit.posteriors.BetaBinomialPosterior`. Posterior
    objects contribute only their observed counts; any prior already stored
    on them is ignored and replaced by the shared empirical-Bayes prior.

    Pass ``ci=None`` to skip credible intervals. Otherwise each arm carries
    an equal-tailed interval with that coverage.
    """
    if ci is not None:
        _validate_ci(ci)
    records = _parse_arms(arms)
    conversions = np.array([item[1] for item in records], dtype=float)
    trials = np.array([item[2] for item in records], dtype=float)
    alpha, beta_param = _fit_shared_beta(conversions, trials)
    strength = alpha + beta_param
    prior_mean = alpha / strength

    shrunk: list[ShrunkArm] = []
    for name, conv, n_trials in records:
        posterior = BetaBinomialPosterior.from_counts(
            conv, n_trials, alpha0=alpha, beta0=beta_param
        )
        lower: float | None = None
        upper: float | None = None
        if ci is not None:
            lower, upper = posterior.credible_interval(ci)
        shrunk.append(
            ShrunkArm(
                name=name,
                conversions=conv,
                trials=n_trials,
                raw_rate=conv / n_trials,
                posterior_mean=posterior.mean(),
                prior_weight=strength / (n_trials + strength),
                ci_lower=lower,
                ci_upper=upper,
            )
        )

    return HierarchicalBetaResult(
        arms=tuple(shrunk),
        prior_alpha=alpha,
        prior_beta=beta_param,
        prior_mean=prior_mean,
        prior_strength=strength,
        ci=ci,
    )


__all__ = [
    "HierarchicalBetaResult",
    "ShrunkArm",
    "hierarchical_beta_shrinkage",
]
