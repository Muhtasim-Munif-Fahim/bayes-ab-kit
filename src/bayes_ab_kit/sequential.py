"""Simulations of sequential-testing risk for repeatedly checked experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _one_sided_pvalue(
    succ_a: np.ndarray,
    n_a: int,
    succ_b: np.ndarray,
    n_b: int,
) -> np.ndarray:
    """Pooled two-proportion z-test, one-sided in favour of B."""
    p_a = succ_a / n_a
    p_b = succ_b / n_b
    pooled = (succ_a + succ_b) / (n_a + n_b)
    se = np.sqrt(pooled * (1.0 - pooled) * (1.0 / n_a + 1.0 / n_b))
    se = np.where(se <= 0, 1e-12, se)
    z = (p_b - p_a) / se
    from scipy import stats as st

    return st.norm.sf(z)


@dataclass(frozen=True)
class PeekSimulationResult:
    """Aggregate behaviour of a repeated-significance testing plan."""

    n_sims: int
    n_looks: int
    per_look_n: int
    alpha: float
    false_stops: int
    false_stop_rate: float
    mean_first_significant_look: float | None
    histogram: tuple[int, ...]


def simulate_null_peeking(
    p_true: float,
    per_look_n: int,
    n_looks: int,
    n_sims: int = 2000,
    alpha: float = 0.05,
    seed: int | None = 20260824,
) -> PeekSimulationResult:
    """Simulate A/A experiments checked after every block of ``per_look_n`` visitors.

    Both arms share the true conversion probability ``p_true``, so stopping at
    nominal significance level ``alpha`` is always a false stop. Returns the
    empirical false-stop rate plus where in the schedule those stops occurred.
    """
    if not 0 < p_true < 1:
        raise ValueError("p_true must lie strictly inside (0, 1)")
    if per_look_n <= 0 or n_looks <= 0 or n_sims <= 0:
        raise ValueError("sample counts must be positive")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly inside (0, 1)")

    rng = np.random.default_rng(seed)
    succ_a = np.zeros(n_sims)
    succ_b = np.zeros(n_sims)
    first_significant = np.full(n_sims, -1, dtype=int)

    for look in range(1, n_looks + 1):
        succ_a += rng.binomial(per_look_n, p_true, size=n_sims)
        succ_b += rng.binomial(per_look_n, p_true, size=n_sims)
        pending = np.flatnonzero(first_significant == -1)
        if pending.size == 0:
            break
        n = look * per_look_n
        pvals = _one_sided_pvalue(succ_a[pending], n, succ_b[pending], n)
        stopped = pending[pvals < alpha]
        first_significant[stopped] = look

    stopped_mask = first_significant > 0
    false_stops = int(stopped_mask.sum())
    looks = np.arange(1, n_looks + 1)
    histogram = tuple(int((first_significant == k).sum()) for k in looks)

    return PeekSimulationResult(
        n_sims=int(n_sims),
        n_looks=int(n_looks),
        per_look_n=int(per_look_n),
        alpha=float(alpha),
        false_stops=false_stops,
        false_stop_rate=false_stops / n_sims,
        mean_first_significant_look=(
            float(first_significant[stopped_mask].mean()) if false_stops else None
        ),
        histogram=histogram,
    )


def fixed_horizon_false_stop_rate(
    p_true: float,
    total_n_per_arm: int,
    n_sims: int = 5000,
    alpha: float = 0.05,
    seed: int | None = 20260824,
) -> float:
    """Empirical one-shot test size under the null (single look, no peeking)."""
    result = simulate_null_peeking(
        p_true=p_true,
        per_look_n=total_n_per_arm,
        n_looks=1,
        n_sims=n_sims,
        alpha=alpha,
        seed=seed,
    )
    return result.false_stop_rate