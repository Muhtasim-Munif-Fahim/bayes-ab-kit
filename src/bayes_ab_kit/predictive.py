"""Posterior predictive probability of eventual experiment success.

At an interim look you already have counts for arms A and B, and a plan to
collect more traffic. The predictive probability of success (PPS) answers:
"if the true rates are drawn from today's posteriors, and we run out the
remaining sample, how often would we declare B superior to A?"

The procedure is Monte Carlo:

1. Draw ``(p_a, p_b)`` from the current Beta posteriors.
2. Simulate the remaining binomial outcomes under those rates.
3. Update the posteriors with the simulated counts.
4. Score ``P(p_b > p_a | final data)`` with a shared Monte Carlo draw.
5. Count the share of worlds where that probability meets ``threshold``.

A low PPS is a futility signal; a high PPS supports continuing or stopping
for success once the current posterior already clears the threshold.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .decisions import prob_b_beats_a_quad
from .posteriors import BetaBinomialPosterior
from .sampling import make_rng


@dataclass(frozen=True)
class PredictiveSuccessResult:
    """Operating characteristics of a planned remaining sample size."""

    conversions_a: int
    trials_a: int
    conversions_b: int
    trials_b: int
    remaining_a: int
    remaining_b: int
    threshold: float
    n_simulations: int
    predictive_probability: float
    mean_final_prob_b_beats_a: float
    prob_already_success: float
    futility_probability: float
    futility_threshold: float

    @property
    def continue_recommended(self) -> bool:
        """True when PPS clears a conventional 0.5 bar and futility does not."""

        return (
            self.predictive_probability >= 0.5
            and self.futility_probability < 0.5
        )


def _validate_nonneg_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return int(value)


def _batch_prob_b_beats_a(
    alpha_a: np.ndarray,
    beta_a: np.ndarray,
    alpha_b: np.ndarray,
    beta_b: np.ndarray,
    rng: np.random.Generator,
    decision_samples: int,
) -> np.ndarray:
    """Vectorized Monte Carlo P(B > A) for many Beta posteriors at once."""

    draws_a = rng.beta(alpha_a[:, None], beta_a[:, None], size=(alpha_a.size, decision_samples))
    draws_b = rng.beta(alpha_b[:, None], beta_b[:, None], size=(alpha_b.size, decision_samples))
    return np.mean(draws_b > draws_a, axis=1)


def predictive_probability_of_success(
    conversions_a: int,
    trials_a: int,
    conversions_b: int,
    trials_b: int,
    *,
    remaining_a: int,
    remaining_b: int | None = None,
    threshold: float = 0.95,
    futility_threshold: float = 0.10,
    alpha0: float = 1.0,
    beta0: float = 1.0,
    n_simulations: int = 4000,
    decision_samples: int = 4_000,
    seed: int | None = 20261014,
) -> PredictiveSuccessResult:
    """Estimate P(declare B > A after collecting the remaining sample).

    Parameters
    ----------
    conversions_a, trials_a, conversions_b, trials_b:
        Observed counts so far.
    remaining_a, remaining_b:
        Planned additional trials per arm. ``remaining_b`` defaults to
        ``remaining_a``.
    threshold:
        Final ``P(B > A)`` required to call success (default 0.95).
    futility_threshold:
        Bar below which a final ``P(B > A)`` counts as futile (default 0.10).
    alpha0, beta0:
        Shared Beta prior hyperparameters.
    n_simulations:
        Number of posterior-predictive worlds.
    decision_samples:
        Monte Carlo draws used to score each world's final ``P(B > A)``.
    seed:
        RNG seed for reproducibility.
    """
    conversions_a = _validate_nonneg_int(conversions_a, "conversions_a")
    trials_a = _validate_nonneg_int(trials_a, "trials_a")
    conversions_b = _validate_nonneg_int(conversions_b, "conversions_b")
    trials_b = _validate_nonneg_int(trials_b, "trials_b")
    if conversions_a > trials_a or conversions_b > trials_b:
        raise ValueError("conversions cannot exceed trials")
    remaining_a = _validate_nonneg_int(remaining_a, "remaining_a")
    if remaining_b is None:
        remaining_b = remaining_a
    else:
        remaining_b = _validate_nonneg_int(remaining_b, "remaining_b")
    if not 0.5 <= threshold < 1.0:
        raise ValueError("threshold must lie in [0.5, 1)")
    if not 0.0 < futility_threshold <= 0.5:
        raise ValueError("futility_threshold must lie in (0, 0.5]")
    if alpha0 <= 0 or beta0 <= 0:
        raise ValueError("prior hyperparameters must be positive")
    if n_simulations <= 0 or decision_samples <= 0:
        raise ValueError("simulation counts must be positive")

    post_a = BetaBinomialPosterior.from_counts(
        conversions_a, trials_a, alpha0=alpha0, beta0=beta0
    )
    post_b = BetaBinomialPosterior.from_counts(
        conversions_b, trials_b, alpha0=alpha0, beta0=beta0
    )

    rng = make_rng(seed)
    p_a = post_a.sample(rng, n_simulations)
    p_b = post_b.sample(rng, n_simulations)

    if remaining_a > 0:
        future_a = rng.binomial(remaining_a, p_a)
    else:
        future_a = np.zeros(n_simulations, dtype=int)
    if remaining_b > 0:
        future_b = rng.binomial(remaining_b, p_b)
    else:
        future_b = np.zeros(n_simulations, dtype=int)

    final_succ_a = conversions_a + future_a
    final_fail_a = (trials_a - conversions_a) + (remaining_a - future_a)
    final_succ_b = conversions_b + future_b
    final_fail_b = (trials_b - conversions_b) + (remaining_b - future_b)

    alpha_a = alpha0 + final_succ_a.astype(float)
    beta_a = beta0 + final_fail_a.astype(float)
    alpha_b = alpha0 + final_succ_b.astype(float)
    beta_b = beta0 + final_fail_b.astype(float)

    final_probs = _batch_prob_b_beats_a(
        alpha_a, beta_a, alpha_b, beta_b, rng, decision_samples
    )
    current_prob = prob_b_beats_a_quad(post_a, post_b)

    pps = float(np.mean(final_probs >= threshold))
    futility = float(np.mean(final_probs < futility_threshold))
    return PredictiveSuccessResult(
        conversions_a=conversions_a,
        trials_a=trials_a,
        conversions_b=conversions_b,
        trials_b=trials_b,
        remaining_a=remaining_a,
        remaining_b=remaining_b,
        threshold=float(threshold),
        n_simulations=int(n_simulations),
        predictive_probability=pps,
        mean_final_prob_b_beats_a=float(np.mean(final_probs)),
        prob_already_success=float(current_prob >= threshold),
        futility_probability=futility,
        futility_threshold=float(futility_threshold),
    )


__all__ = [
    "PredictiveSuccessResult",
    "predictive_probability_of_success",
]
