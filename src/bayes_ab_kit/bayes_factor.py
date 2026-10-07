"""Bayes factors for Beta-Binomial A/B tests with sequential early stopping.

Compares a shared-rate null ``H0: θ_A = θ_B = θ ~ Beta(α0, β0)`` against an
independent-rates alternative ``H1: θ_A, θ_B ~iid Beta(α0, β0)``. The
marginal likelihood under each hypothesis is available in closed form via
the Beta function, so the Bayes factor ``BF10 = p(data | H1) / p(data | H0)``
is exact (Kass and Raftery, 1995; Rouder et al., 2009).

A sequential monitor peeks at accumulating counts and stops early once
``|log BF10|`` crosses a threshold (default ``log(10) ≈ 2.3``, "strong"
evidence on the Kass–Raftery scale).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import betaln

from .decisions import Decision
from .posteriors import BetaBinomialPosterior


def _require_nonneg_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{label} must be a non-negative integer")
    ivalue = int(value)
    if ivalue < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return ivalue


def _require_counts(conversions: object, trials: object, label: str) -> tuple[int, int]:
    conversions_i = _require_nonneg_int(conversions, f"{label} conversions")
    trials_i = _require_nonneg_int(trials, f"{label} trials")
    if conversions_i > trials_i:
        raise ValueError(f"{label} conversions cannot exceed trials")
    return conversions_i, trials_i


def log_marginal_h1(
    conversions_a: int,
    trials_a: int,
    conversions_b: int,
    trials_b: int,
    *,
    alpha0: float = 1.0,
    beta0: float = 1.0,
) -> float:
    """Log marginal likelihood under independent Beta rates (H1).

    Omits binomial coefficients shared with H0, which cancel in the Bayes
    factor. Requires ``alpha0 > 0`` and ``beta0 > 0``.
    """
    if alpha0 <= 0 or beta0 <= 0:
        raise ValueError("prior hyperparameters must be positive")
    sa, na = _require_counts(conversions_a, trials_a, "arm a")
    sb, nb = _require_counts(conversions_b, trials_b, "arm b")
    fa, fb = na - sa, nb - sb
    return float(
        betaln(alpha0 + sa, beta0 + fa)
        + betaln(alpha0 + sb, beta0 + fb)
        - 2.0 * betaln(alpha0, beta0)
    )


def log_marginal_h0(
    conversions_a: int,
    trials_a: int,
    conversions_b: int,
    trials_b: int,
    *,
    alpha0: float = 1.0,
    beta0: float = 1.0,
) -> float:
    """Log marginal likelihood under a shared Beta rate (H0)."""
    if alpha0 <= 0 or beta0 <= 0:
        raise ValueError("prior hyperparameters must be positive")
    sa, na = _require_counts(conversions_a, trials_a, "arm a")
    sb, nb = _require_counts(conversions_b, trials_b, "arm b")
    fa, fb = na - sa, nb - sb
    return float(
        betaln(alpha0 + sa + sb, beta0 + fa + fb) - betaln(alpha0, beta0)
    )


def log_bayes_factor_10(
    conversions_a: int,
    trials_a: int,
    conversions_b: int,
    trials_b: int,
    *,
    alpha0: float = 1.0,
    beta0: float = 1.0,
) -> float:
    """Natural log of ``BF10 = p(data|H1) / p(data|H0)``."""
    return log_marginal_h1(
        conversions_a, trials_a, conversions_b, trials_b, alpha0=alpha0, beta0=beta0
    ) - log_marginal_h0(
        conversions_a, trials_a, conversions_b, trials_b, alpha0=alpha0, beta0=beta0
    )


def bayes_factor_10(
    conversions_a: int,
    trials_a: int,
    conversions_b: int,
    trials_b: int,
    *,
    alpha0: float = 1.0,
    beta0: float = 1.0,
) -> float:
    """``BF10`` comparing independent rates (H1) to a shared rate (H0)."""
    return float(
        np.exp(
            log_bayes_factor_10(
                conversions_a,
                trials_a,
                conversions_b,
                trials_b,
                alpha0=alpha0,
                beta0=beta0,
            )
        )
    )


def kass_raftery_label(bf10: float) -> str:
    """Interpret ``BF10`` on the Kass–Raftery (1995) scale.

    Categories: ``decisive`` (|BF|≥100), ``strong`` (≥10), ``positive`` (≥3),
    ``barely worth mentioning`` (≥1), else the reciprocal categories for
    evidence favoring H0. Uses ``BF10`` itself (not log).
    """
    if not np.isfinite(bf10) or bf10 <= 0:
        raise ValueError("bf10 must be a positive finite number")
    if bf10 >= 100:
        return "decisive for H1"
    if bf10 >= 10:
        return "strong for H1"
    if bf10 >= 3:
        return "positive for H1"
    if bf10 >= 1:
        return "barely worth mentioning for H1"
    inv = 1.0 / bf10
    if inv >= 100:
        return "decisive for H0"
    if inv >= 10:
        return "strong for H0"
    if inv >= 3:
        return "positive for H0"
    return "barely worth mentioning for H0"


@dataclass(frozen=True)
class BayesFactorResult:
    """Point-in-time Bayes factor for a two-arm conversion experiment."""

    bf10: float
    log_bf10: float
    label: str
    conversions_a: int
    trials_a: int
    conversions_b: int
    trials_b: int
    alpha0: float
    beta0: float

    @property
    def decision(self) -> Decision:
        """Map strong-or-better evidence onto ship / keep-running."""
        if self.bf10 >= 10.0:
            # Prefer the arm with the higher posterior mean under H1 priors.
            post_a = BetaBinomialPosterior.from_counts(
                self.conversions_a, self.trials_a, alpha0=self.alpha0, beta0=self.beta0
            )
            post_b = BetaBinomialPosterior.from_counts(
                self.conversions_b, self.trials_b, alpha0=self.alpha0, beta0=self.beta0
            )
            return Decision.SHIP_B if post_b.mean() >= post_a.mean() else Decision.SHIP_A
        if self.bf10 <= 0.1:
            return Decision.PRACTICAL_EQUIVALENCE
        return Decision.KEEP_RUNNING


def bayes_factor_decision(
    posterior_a: BetaBinomialPosterior,
    posterior_b: BetaBinomialPosterior,
    *,
    alpha0: float | None = None,
    beta0: float | None = None,
) -> BayesFactorResult:
    """Compute ``BF10`` from two fitted Beta-Binomial posteriors.

    Prior hyperparameters default to those stored on ``posterior_a`` (and must
    match ``posterior_b`` when both are provided explicitly via the optional
    overrides).
    """
    a0 = float(posterior_a.alpha0 if alpha0 is None else alpha0)
    b0 = float(posterior_a.beta0 if beta0 is None else beta0)
    if alpha0 is None and (
        posterior_a.alpha0 != posterior_b.alpha0 or posterior_a.beta0 != posterior_b.beta0
    ):
        raise ValueError("posteriors must share the same prior hyperparameters")
    log_bf = log_bayes_factor_10(
        posterior_a.successes,
        posterior_a.trials,
        posterior_b.successes,
        posterior_b.trials,
        alpha0=a0,
        beta0=b0,
    )
    bf = float(np.exp(log_bf))
    return BayesFactorResult(
        bf10=bf,
        log_bf10=float(log_bf),
        label=kass_raftery_label(bf),
        conversions_a=int(posterior_a.successes),
        trials_a=int(posterior_a.trials),
        conversions_b=int(posterior_b.successes),
        trials_b=int(posterior_b.trials),
        alpha0=a0,
        beta0=b0,
    )


@dataclass(frozen=True)
class SequentialBayesFactorResult:
    """Operating characteristics of a sequential Bayes-factor monitor."""

    n_simulations: int
    looks: int
    per_look: int
    log_bf_threshold: float
    true_rate_a: float
    true_rate_b: float
    stop_rate: float
    mean_looks_at_stop: float
    mean_log_bf_at_stop: float
    decisive_for_h1_rate: float
    decisive_for_h0_rate: float


def simulate_bayes_factor_stopping(
    *,
    true_rate_a: float,
    true_rate_b: float,
    per_look: int = 500,
    looks: int = 5,
    log_bf_threshold: float = float(np.log(10.0)),
    alpha0: float = 1.0,
    beta0: float = 1.0,
    n_simulations: int = 2000,
    seed: int | None = 20261007,
) -> SequentialBayesFactorResult:
    """Simulate early stopping when ``|log BF10|`` exceeds a threshold.

    Each simulation draws ``per_look`` Bernoulli outcomes per arm at every
    look, accumulates counts, and stops at the first look where
    ``|log BF10| >= log_bf_threshold``. Under a shared null
    (``true_rate_a == true_rate_b``) the ``stop_rate`` for H1 is the false
    early-stop rate; under an uplift it estimates power to stop for H1.
    """
    if not 0.0 < true_rate_a < 1.0 or not 0.0 < true_rate_b < 1.0:
        raise ValueError("true rates must lie strictly inside (0, 1)")
    if per_look < 1 or looks < 1:
        raise ValueError("per_look and looks must be positive integers")
    if log_bf_threshold <= 0:
        raise ValueError("log_bf_threshold must be positive")
    if alpha0 <= 0 or beta0 <= 0:
        raise ValueError("prior hyperparameters must be positive")
    if n_simulations < 1:
        raise ValueError("n_simulations must be positive")

    rng = np.random.default_rng(seed)
    stops = 0
    looks_at_stop: list[int] = []
    log_bfs: list[float] = []
    h1_stops = 0
    h0_stops = 0

    for _ in range(n_simulations):
        sa = sb = na = nb = 0
        stopped = False
        final_log_bf = 0.0
        final_look = looks
        for look in range(1, looks + 1):
            sa += int(rng.binomial(per_look, true_rate_a))
            sb += int(rng.binomial(per_look, true_rate_b))
            na += per_look
            nb += per_look
            log_bf = log_bayes_factor_10(sa, na, sb, nb, alpha0=alpha0, beta0=beta0)
            final_log_bf = log_bf
            final_look = look
            if abs(log_bf) >= log_bf_threshold:
                stopped = True
                if log_bf >= log_bf_threshold:
                    h1_stops += 1
                else:
                    h0_stops += 1
                break
        if stopped:
            stops += 1
            looks_at_stop.append(final_look)
            log_bfs.append(final_log_bf)
        else:
            looks_at_stop.append(looks)
            log_bfs.append(final_log_bf)

    return SequentialBayesFactorResult(
        n_simulations=n_simulations,
        looks=looks,
        per_look=per_look,
        log_bf_threshold=float(log_bf_threshold),
        true_rate_a=float(true_rate_a),
        true_rate_b=float(true_rate_b),
        stop_rate=stops / n_simulations,
        mean_looks_at_stop=float(np.mean(looks_at_stop)),
        mean_log_bf_at_stop=float(np.mean(log_bfs)),
        decisive_for_h1_rate=h1_stops / n_simulations,
        decisive_for_h0_rate=h0_stops / n_simulations,
    )


__all__ = [
    "BayesFactorResult",
    "SequentialBayesFactorResult",
    "bayes_factor_10",
    "bayes_factor_decision",
    "kass_raftery_label",
    "log_bayes_factor_10",
    "log_marginal_h0",
    "log_marginal_h1",
    "simulate_bayes_factor_stopping",
]
