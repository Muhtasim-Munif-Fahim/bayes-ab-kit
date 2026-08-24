"""Pre-experiment planning: sample size requirements and simulated power."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _z(level: float) -> float:
    from scipy import stats as st

    return float(st.norm.ppf(level))


@dataclass(frozen=True)
class SampleSizePlan:
    """Analytic sample-size requirement for detecting a minimum detectable effect."""

    baseline_rate: float
    expected_rate: float
    alpha: float
    power: float
    alternative: str
    n_per_arm: int


def required_sample_size(
    baseline_rate: float,
    expected_rate: float,
    alpha: float = 0.05,
    power: float = 0.80,
    alternative: str = "two-sided",
) -> SampleSizePlan:
    """Visitors per arm needed to detect ``baseline -> expected`` with the
    requested significance level and power (Normal approximation)."""
    if not 0 < baseline_rate < 1 or not 0 < expected_rate < 1:
        raise ValueError("rates must lie strictly inside (0, 1)")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly inside (0, 1)")
    if not 0 < power < 1:
        raise ValueError("power must lie strictly inside (0, 1)")
    if alternative not in {"two-sided", "one-sided"}:
        raise ValueError("alternative must be 'two-sided' or 'one-sided'")
    if baseline_rate == expected_rate:
        raise ValueError("expected rate must differ from the baseline")

    p_bar = (baseline_rate + expected_rate) / 2.0
    if alternative == "two-sided":
        z_alpha = _z(1.0 - alpha / 2.0)
    else:
        z_alpha = _z(1.0 - alpha)
    z_beta = _z(power)

    numerator = (
        z_alpha * np.sqrt(2.0 * p_bar * (1.0 - p_bar))
        + z_beta
        * np.sqrt(
            baseline_rate * (1.0 - baseline_rate)
            + expected_rate * (1.0 - expected_rate)
        )
    ) ** 2
    denominator = (expected_rate - baseline_rate) ** 2
    n_per_arm = int(np.ceil(numerator / denominator))
    return SampleSizePlan(
        baseline_rate=float(baseline_rate),
        expected_rate=float(expected_rate),
        alpha=float(alpha),
        power=float(power),
        alternative=alternative,
        n_per_arm=n_per_arm,
    )


def simulate_power(
    p_a: float,
    p_b: float,
    n_per_arm: int,
    n_sims: int = 4000,
    alpha: float = 0.05,
    seed: int | None = 20260824,
) -> float:
    """Empirical detection probability of a one-shot one-sided z-test."""
    if not 0 < p_a < 1 or not 0 < p_b < 1:
        raise ValueError("rates must lie strictly inside (0, 1)")
    if n_per_arm <= 0 or n_sims <= 0:
        raise ValueError("sample counts must be positive")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly inside (0, 1)")

    rng = np.random.default_rng(seed)
    conv_a = rng.binomial(n_per_arm, p_a, size=n_sims)
    conv_b = rng.binomial(n_per_arm, p_b, size=n_sims)
    diff = conv_b / n_per_arm - conv_a / n_per_arm
    pooled = (conv_a + conv_b) / (2 * n_per_arm)
    se = np.sqrt(pooled * (1.0 - pooled) * (2.0 / n_per_arm))
    se = np.where(se <= 0, 1e-12, se)
    from scipy import stats as st

    z = diff / se
    return float(np.mean(st.norm.sf(z) < alpha))


def power_curve(
    p_a: float,
    p_b: float,
    ns_per_arm: list[int] | tuple[int, ...],
    n_sims: int = 3000,
    alpha: float = 0.05,
    seed: int | None = 20260824,
) -> list[tuple[int, float]]:
    """Empirical power at each candidate sample size."""
    curve = []
    for n in ns_per_arm:
        power = simulate_power(p_a, p_b, n_per_arm=n, n_sims=n_sims, alpha=alpha, seed=seed)
        curve.append((int(n), power))
    return curve


__all__ = ["SampleSizePlan", "power_curve", "required_sample_size", "simulate_power"]