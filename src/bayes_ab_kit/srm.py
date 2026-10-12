"""Sample ratio mismatch (SRM) checks for experiment traffic allocation.

A sample ratio mismatch happens when the observed split of units across arms
differs from the planned split (for example 50.8% / 49.2% with 200k users on
a 50/50 test). It usually means an assignment, logging or filtering bug, and
it invalidates the metric comparison, so it should be checked first.

Two tests are provided for the planned shares ``p0``:

* a frequentist Pearson chi-square goodness-of-fit test, and
* a Bayesian Dirichlet-multinomial Bayes factor. It compares
  ``H1: p ~ Dirichlet(a)`` with ``H0: p = p0``, and its marginal likelihoods
  are available in closed form. Under ``H0``, ``BF10`` evaluated on the
  growing counts is a non-negative martingale with mean one, so Ville's
  inequality gives ``P(sup_t BF10_t >= 1/alpha) <= alpha``. The sequential
  monitor can therefore be checked after every batch without inflating the
  false-alarm rate (Lindon & Malek, 2020, "Anytime-valid inference for
  multinomial count data").
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.special import gammaln
from scipy.stats import chi2

DEFAULT_SRM_ALPHA = 0.001


def _validate_counts(counts: Sequence[int]) -> np.ndarray:
    values = list(counts)
    if len(values) < 2:
        raise ValueError("need counts for at least two arms")
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
            raise ValueError("counts must be non-negative integers")
    arr = np.asarray(values, dtype=float)
    if arr.sum() <= 0:
        raise ValueError("counts must not all be zero")
    return arr


def _validate_shares(expected_shares: Sequence[float] | None, k: int) -> np.ndarray:
    if expected_shares is None:
        return np.full(k, 1.0 / k)
    shares = np.asarray(list(expected_shares), dtype=float)
    if shares.shape != (k,):
        raise ValueError("expected_shares must have one entry per arm")
    if not np.all(np.isfinite(shares)) or np.any(shares <= 0):
        raise ValueError("expected_shares must be positive")
    return shares / shares.sum()


def _validate_alpha(alpha: float) -> float:
    if isinstance(alpha, bool) or not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    return float(alpha)


def _prior(prior_concentration: float | Sequence[float], k: int) -> np.ndarray:
    prior = np.broadcast_to(np.asarray(prior_concentration, dtype=float), (k,)).copy()
    if not np.all(np.isfinite(prior)) or np.any(prior <= 0):
        raise ValueError("prior_concentration must be positive")
    return prior


def srm_log_bayes_factor(
    counts: Sequence[int],
    expected_shares: Sequence[float] | None = None,
    *,
    prior_concentration: float | Sequence[float] = 1.0,
) -> float:
    """Natural log of ``BF10`` for a Dirichlet allocation vs the planned shares.

    ``log BF10 = log B(a + n) - log B(a) - sum(n_i log p0_i)``, where ``B`` is
    the multivariate Beta function. The multinomial coefficient is the same
    under both hypotheses and cancels.
    """
    n = _validate_counts(counts)
    p0 = _validate_shares(expected_shares, n.size)
    a = _prior(prior_concentration, n.size)
    log_h1 = (
        gammaln(a.sum())
        - gammaln(a).sum()
        + gammaln(a + n).sum()
        - gammaln(a.sum() + n.sum())
    )
    log_h0 = float(np.sum(n * np.log(p0)))
    return float(log_h1 - log_h0)


@dataclass(frozen=True)
class SrmResult:
    """Outcome of a one-shot sample ratio mismatch check."""

    arms: tuple[str, ...]
    counts: tuple[int, ...]
    expected_shares: tuple[float, ...]
    observed_shares: tuple[float, ...]
    chi_square: float
    dof: int
    p_value: float
    log_bf10: float
    bf10: float
    posterior_mean_shares: tuple[float, ...]
    alpha: float

    @property
    def mismatch(self) -> bool:
        """True when the chi-square p-value is below ``alpha``."""
        return self.p_value < self.alpha

    @property
    def bayesian_mismatch(self) -> bool:
        """True when ``BF10 >= 1 / alpha``; this rule is safe under peeking."""
        return self.log_bf10 >= -np.log(self.alpha)


def srm_check(
    counts: Sequence[int],
    expected_shares: Sequence[float] | None = None,
    *,
    arms: Sequence[str] | None = None,
    alpha: float = DEFAULT_SRM_ALPHA,
    prior_concentration: float | Sequence[float] = 1.0,
) -> SrmResult:
    """Check observed unit counts per arm against the planned allocation.

    ``expected_shares`` defaults to an even split and is renormalised to sum
    to one. ``alpha`` defaults to 0.001: SRM checks run on every experiment,
    so a strict threshold keeps false alarms rare.
    """
    n = _validate_counts(counts)
    p0 = _validate_shares(expected_shares, n.size)
    a = _prior(prior_concentration, n.size)
    alpha_f = _validate_alpha(alpha)
    if arms is None:
        arm_names = tuple(chr(ord("A") + i) if i < 26 else f"arm{i}" for i in range(n.size))
    else:
        arm_names = tuple(str(x) for x in arms)
        if len(arm_names) != n.size:
            raise ValueError("arms must have one name per count")
    total = float(n.sum())
    expected = total * p0
    stat = float(np.sum((n - expected) ** 2 / expected))
    dof = int(n.size - 1)
    log_bf = srm_log_bayes_factor(n.astype(int).tolist(), p0, prior_concentration=a)
    posterior = (a + n) / (a.sum() + total)
    return SrmResult(
        arms=arm_names,
        counts=tuple(int(x) for x in n),
        expected_shares=tuple(float(x) for x in p0),
        observed_shares=tuple(float(x) for x in n / total),
        chi_square=stat,
        dof=dof,
        p_value=float(chi2.sf(stat, dof)),
        log_bf10=log_bf,
        bf10=float(np.exp(min(log_bf, 700.0))),
        posterior_mean_shares=tuple(float(x) for x in posterior),
        alpha=alpha_f,
    )


@dataclass(frozen=True)
class SequentialSrmResult:
    """Anytime-valid SRM monitor over a sequence of cumulative counts."""

    log_bf10_path: tuple[float, ...]
    anytime_p_values: tuple[float, ...]
    alpha: float
    stopped_at: int | None

    @property
    def mismatch(self) -> bool:
        return self.stopped_at is not None


def sequential_srm_monitor(
    cumulative_counts: Sequence[Sequence[int]],
    expected_shares: Sequence[float] | None = None,
    *,
    alpha: float = DEFAULT_SRM_ALPHA,
    prior_concentration: float | Sequence[float] = 1.0,
) -> SequentialSrmResult:
    """Track ``BF10`` across looks and alarm the first time it reaches ``1/alpha``.

    ``cumulative_counts[t]`` holds the per-arm totals at look ``t``. They must
    never decrease. The anytime-valid p-value at look ``t`` is
    ``min(1, 1 / max_{s<=t} BF10_s)``. ``stopped_at`` is the 0-based index of
    the first look where that p-value is at most ``alpha``, or ``None``.
    """
    alpha_f = _validate_alpha(alpha)
    looks = [list(c) for c in cumulative_counts]
    if not looks:
        raise ValueError("need at least one look")
    k = len(looks[0])
    previous = np.zeros(k)
    path: list[float] = []
    pvals: list[float] = []
    running_max = -np.inf
    stopped: int | None = None
    for t, look in enumerate(looks):
        arr = _validate_counts(look)
        if arr.size != k:
            raise ValueError("every look must have the same number of arms")
        if np.any(arr < previous):
            raise ValueError("cumulative counts must be non-decreasing")
        previous = arr
        log_bf = srm_log_bayes_factor(look, expected_shares, prior_concentration=prior_concentration)
        path.append(log_bf)
        running_max = max(running_max, log_bf)
        pvals.append(float(min(1.0, np.exp(-running_max))))
        if stopped is None and pvals[-1] <= alpha_f:
            stopped = t
    return SequentialSrmResult(
        log_bf10_path=tuple(path),
        anytime_p_values=tuple(pvals),
        alpha=alpha_f,
        stopped_at=stopped,
    )
