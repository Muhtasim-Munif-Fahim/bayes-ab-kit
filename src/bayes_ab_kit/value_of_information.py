"""Expected value of information for Beta-Binomial experiments.

Decision-theoretic stopping asks whether *more data is worth its cost*.

* **EVPI** (expected value of perfect information) is the expected gain in
  conversion rate from learning the true rates before choosing an arm:
  ``E[max_k p_k] - max_k E[p_k]``. It is an upper bound on what any further
  experimentation can deliver and equals the expected loss of shipping the
  current leader.
* **EVSI** (expected value of sample information) is the same quantity for
  a finite additional sample of ``n`` visitors per arm: it is the
  preposterior expected improvement of the decision made *after* seeing
  the extra data, ``E_y[max_k E[p_k | y]] - max_k E[p_k]``.
* **ENBS** (expected net benefit of sampling) converts EVSI into a sample
  size recommendation by trading the gain applied to ``horizon`` future
  visitors against a per-visitor sampling cost.

All estimators use the identity ``E_y[E[p_k|y]] = E[p_k]`` so the Monte Carlo
estimate is the mean of a non-negative quantity (``max_k m_k - m_{k*}`` with
``k*`` the current leader), which keeps EVSI/EVPI non-negative and greatly
reduces Monte Carlo noise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .posteriors import BetaBinomialPosterior


def _as_arms(posteriors) -> tuple[list[str], np.ndarray, np.ndarray]:
    if isinstance(posteriors, Mapping):
        names = [str(k) for k in posteriors]
        items = list(posteriors.values())
    else:
        items = list(posteriors)
        names = [chr(ord("a") + i) if i < 26 else f"arm_{i}" for i in range(len(items))]
    if len(items) < 2:
        raise ValueError("at least two posteriors are required")
    for post in items:
        if not isinstance(post, BetaBinomialPosterior):
            raise TypeError("posteriors must be BetaBinomialPosterior instances")
    alpha = np.array([p.alpha for p in items], dtype=float)
    beta = np.array([p.beta for p in items], dtype=float)
    return names, alpha, beta


def _check_samples(n_samples: int) -> int:
    if isinstance(n_samples, bool) or not isinstance(n_samples, (int, np.integer)) or n_samples <= 0:
        raise ValueError("n_samples must be a positive integer")
    return int(n_samples)


def evpi(
    posteriors: Sequence[BetaBinomialPosterior] | Mapping[str, BetaBinomialPosterior],
    n_samples: int = 200_000,
    seed: int | None = 20261109,
) -> float:
    """Expected value of perfect information, in conversion-rate units."""
    n_samples = _check_samples(n_samples)
    _, alpha, beta = _as_arms(posteriors)
    rng = np.random.default_rng(seed)
    means = alpha / (alpha + beta)
    leader = int(np.argmax(means))
    draws = rng.beta(alpha, beta, size=(n_samples, alpha.size))
    return float(np.mean(draws.max(axis=1) - draws[:, leader]))


def evsi(
    posteriors: Sequence[BetaBinomialPosterior] | Mapping[str, BetaBinomialPosterior],
    n_per_arm: int,
    n_samples: int = 200_000,
    seed: int | None = 20261109,
) -> float:
    """Expected value of observing ``n_per_arm`` more visitors on every arm.

    Simulates the preposterior: true rates are drawn from the current
    posteriors, future conversions from ``Binomial(n_per_arm, p_k)``, and the
    arm with the best updated posterior mean is chosen. Returns the
    expected gain over choosing the current leader now.
    """
    if isinstance(n_per_arm, bool) or not isinstance(n_per_arm, (int, np.integer)) or n_per_arm < 0:
        raise ValueError("n_per_arm must be a non-negative integer")
    n_samples = _check_samples(n_samples)
    _, alpha, beta = _as_arms(posteriors)
    if n_per_arm == 0:
        return 0.0
    rng = np.random.default_rng(seed)
    means = alpha / (alpha + beta)
    leader = int(np.argmax(means))
    rates = rng.beta(alpha, beta, size=(n_samples, alpha.size))
    conversions = rng.binomial(int(n_per_arm), rates)
    updated = (alpha + conversions) / (alpha + beta + n_per_arm)
    return float(np.mean(updated.max(axis=1) - updated[:, leader]))


@dataclass(frozen=True)
class SampleSizeValue:
    """ENBS evaluation at one candidate additional sample size."""

    n_per_arm: int
    evsi: float
    gain: float
    cost: float
    enbs: float


@dataclass(frozen=True)
class ValueOfInformationResult:
    """Outcome of :func:`optimal_additional_sample`."""

    names: tuple[str, ...]
    leader: str
    evpi: float
    max_value: float
    horizon: float
    cost_per_visitor: float
    curve: tuple[SampleSizeValue, ...]
    best_n_per_arm: int
    best_enbs: float

    @property
    def keep_running(self) -> bool:
        """``True`` when some additional sample has positive net benefit."""
        return self.best_n_per_arm > 0 and self.best_enbs > 0


def optimal_additional_sample(
    posteriors: Sequence[BetaBinomialPosterior] | Mapping[str, BetaBinomialPosterior],
    horizon: float,
    cost_per_visitor: float = 0.0,
    candidate_n: Sequence[int] | None = None,
    value_per_conversion: float = 1.0,
    n_samples: int = 100_000,
    seed: int | None = 20261109,
) -> ValueOfInformationResult:
    """Pick the extra per-arm sample size maximising expected net benefit.

    ``ENBS(n) = value_per_conversion * horizon * EVSI(n) - cost_per_visitor * K * n``
    where ``horizon`` is the number of future visitors who will receive the
    shipped arm and ``K`` the number of arms. ``n = 0`` (stop now) is always
    a candidate with ``ENBS = 0``. The same random seed is used for every
    candidate (common random numbers) so the curve is smooth.
    """
    if not np.isfinite(horizon) or horizon <= 0:
        raise ValueError("horizon must be positive")
    if not np.isfinite(cost_per_visitor) or cost_per_visitor < 0:
        raise ValueError("cost_per_visitor must be non-negative")
    if not np.isfinite(value_per_conversion) or value_per_conversion <= 0:
        raise ValueError("value_per_conversion must be positive")
    names, alpha, beta = _as_arms(posteriors)
    k = alpha.size
    if candidate_n is None:
        base = alpha.min() + beta.min()
        candidate_n = sorted({int(round(base * f)) for f in (0.1, 0.25, 0.5, 1, 2, 4, 8)} - {0})
    grid = sorted({int(n) for n in candidate_n} | {0})
    if any(n < 0 for n in grid):
        raise ValueError("candidate_n must be non-negative")

    curve = []
    for n in grid:
        value = evsi(posteriors, n, n_samples=n_samples, seed=seed)
        gain = value_per_conversion * horizon * value
        cost = cost_per_visitor * k * n
        curve.append(SampleSizeValue(n_per_arm=n, evsi=value, gain=gain, cost=cost, enbs=gain - cost))
    best = max(curve, key=lambda row: (row.enbs, -row.n_per_arm))
    means = alpha / (alpha + beta)
    perfect = evpi(posteriors, n_samples=n_samples, seed=seed)
    return ValueOfInformationResult(
        names=tuple(names),
        leader=names[int(np.argmax(means))],
        evpi=perfect,
        max_value=value_per_conversion * horizon * perfect,
        horizon=float(horizon),
        cost_per_visitor=float(cost_per_visitor),
        curve=tuple(curve),
        best_n_per_arm=best.n_per_arm,
        best_enbs=best.enbs,
    )
