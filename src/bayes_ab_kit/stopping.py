"""Expected sample size and early-stopping behaviour of Bayesian peeking plans."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _prob_b_superior(succ_a: np.ndarray, n: int, succ_b: np.ndarray) -> np.ndarray:
    """P(p_B > p_A) using a Normal approximation to each Beta posterior."""
    post_a_mean = succ_a / n
    post_b_mean = succ_b / n
    var_a = post_a_mean * (1.0 - post_a_mean) / n
    var_b = post_b_mean * (1.0 - post_b_mean) / n
    sd = np.sqrt(var_a + var_b)
    from scipy import stats as st

    return st.norm.cdf((post_b_mean - post_a_mean) / sd)


@dataclass(frozen=True)
class StoppingPlanResult:
    """Operating characteristics of a Bayesian early-stopping plan."""

    n_sims: int
    per_look_n: int
    max_looks: int
    threshold: float
    stopped: int
    stop_rate: float
    expected_sample_size_per_arm: float
    mean_stop_look: float | None


def simulate_stopping_plan(
    p_a: float,
    p_b: float,
    per_look_n: int,
    max_looks: int,
    threshold: float = 0.975,
    n_sims: int = 2000,
    seed: int | None = 20260824,
) -> StoppingPlanResult:
    """Simulate an experiment monitored after every block of ``per_look_n`` visitors.

    Monitoring stops in favour of B once ``P(p_B > p_A)`` (Normal-approximated
    Beta posteriors) exceeds ``threshold``. Reports how often the plan stops,
    how many visitors per arm that costs on average, and the mean stopping
    look among stopped runs.
    """
    for label, value in (("p_a", p_a), ("p_b", p_b)):
        if not 0 < value < 1:
            raise ValueError(f"{label} must lie strictly inside (0, 1)")
    if per_look_n <= 0 or max_looks <= 0 or n_sims <= 0:
        raise ValueError("sample counts must be positive")
    if not 0.5 < threshold < 1:
        raise ValueError("threshold must lie inside (0.5, 1)")

    rng = np.random.default_rng(seed)
    succ_a = np.zeros(n_sims)
    succ_b = np.zeros(n_sims)
    stop_look = np.full(n_sims, max_looks, dtype=int)

    for look in range(1, max_looks + 1):
        pending = np.flatnonzero(stop_look == max_looks)
        if pending.size == 0:
            break
        succ_a[pending] += rng.binomial(per_look_n, p_a, size=pending.size)
        succ_b[pending] += rng.binomial(per_look_n, p_b, size=pending.size)
        n = look * per_look_n
        probs = _prob_b_superior(succ_a[pending], n, succ_b[pending])
        crossed = pending[probs > threshold]
        stop_look[crossed] = look

    stopped_mask = stop_look < max_looks
    stopped_count = int(stopped_mask.sum())
    return StoppingPlanResult(
        n_sims=int(n_sims),
        per_look_n=int(per_look_n),
        max_looks=int(max_looks),
        threshold=float(threshold),
        stopped=stopped_count,
        stop_rate=stopped_count / n_sims,
        expected_sample_size_per_arm=float(np.mean(stop_look * per_look_n)),
        mean_stop_look=(
            float(np.mean(stop_look[stopped_mask])) if stopped_count else None
        ),
    )


def required_max_looks_for_power(
    p_a: float,
    p_b: float,
    target_power: float = 0.80,
    per_look_n: int = 500,
    max_search_looks: int = 40,
    n_sims: int = 1500,
    threshold: float = 0.975,
    seed: int | None = 20260824,
) -> int | None:
    """Smallest simulated plan length whose detection rate reaches ``target_power``.

    Scans plan lengths in order and returns the first that achieves the
    target, or ``None`` when even ``max_search_looks`` falls short.
    """
    if not 0 < target_power <= 1:
        raise ValueError("target_power must lie inside (0, 1]")
    for looks in range(1, max_search_looks + 1):
        result = simulate_stopping_plan(
            p_a=p_a,
            p_b=p_b,
            per_look_n=per_look_n,
            max_looks=looks,
            threshold=threshold,
            n_sims=n_sims,
            seed=seed,
        )
        if result.stop_rate >= target_power:
            return looks
    return None


__all__ = [
    "StoppingPlanResult",
    "required_max_looks_for_power",
    "simulate_stopping_plan",
]