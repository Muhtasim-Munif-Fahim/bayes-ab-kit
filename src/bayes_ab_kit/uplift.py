"""Posterior probability a variant beats control, with expected uplift.

Absolute uplift is ``rate_variant - rate_control``. Relative uplift is
``(rate_variant - rate_control) / rate_control``. Expected absolute
uplift and expected relative uplift have closed forms under independent
Beta-Binomial posteriors; equal-tailed credible intervals use the same
seeded Monte Carlo convention as :mod:`bayes_ab_kit.decisions` and
:mod:`bayes_ab_kit.rope`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from .decisions import difference_credible_interval, prob_b_beats_a_quad
from .posteriors import BetaBinomialPosterior


def _validate_level(ci: float) -> None:
    if not 0 < ci < 1:
        raise ValueError("credible level must lie strictly inside (0, 1)")


def expected_absolute_uplift(
    control: BetaBinomialPosterior,
    variant: BetaBinomialPosterior,
) -> float:
    """``E[rate_variant - rate_control]`` from Beta posterior means."""
    return variant.mean() - control.mean()


def expected_relative_uplift(
    control: BetaBinomialPosterior,
    variant: BetaBinomialPosterior,
) -> float:
    """``E[(rate_variant - rate_control) / rate_control]``.

    Independence of the two Beta posteriors gives
    ``E[variant / control] = E[variant] E[1 / control]``. For
    ``control ~ Beta(α, β)`` with ``α > 1``,
    ``E[1 / control] = (α + β - 1) / (α - 1)``. When ``α <= 1`` the
    expectation diverges and this returns ``+inf``.
    """
    if control.alpha <= 1.0:
        return float("inf")
    mean_inv_control = (control.alpha + control.beta - 1.0) / (control.alpha - 1.0)
    return variant.mean() * mean_inv_control - 1.0


def _relative_uplift_interval(
    control: BetaBinomialPosterior,
    variant: BetaBinomialPosterior,
    ci: float,
    n_samples: int,
    seed: int | None,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws_control = rng.beta(control.alpha, control.beta, size=n_samples)
    draws_variant = rng.beta(variant.alpha, variant.beta, size=n_samples)
    relative = (draws_variant - draws_control) / draws_control
    tail = (1.0 - ci) / 2.0
    lower, upper = np.quantile(relative, [tail, 1.0 - tail])
    return float(lower), float(upper)


@dataclass(frozen=True)
class UpliftResult:
    """P(variant > control), expected uplift, and equal-tailed intervals.

    ``expected_uplift`` is the posterior mean of the absolute rate
    difference. ``expected_relative_uplift`` is the posterior mean of the
    relative difference; it is ``+inf`` when the control Beta shape
    ``alpha`` is at most 1 (so ``E[1 / rate_control]`` diverges).
    """

    prob_beats_control: float
    expected_uplift: float
    expected_relative_uplift: float
    uplift_ci_lower: float
    uplift_ci_upper: float
    relative_uplift_ci_lower: float
    relative_uplift_ci_upper: float
    ci: float


@dataclass(frozen=True)
class VariantVsControl:
    """One non-control arm's uplift summary against the designated control."""

    name: str
    conversions: int
    trials: int
    posterior_mean: float
    uplift: UpliftResult


@dataclass(frozen=True)
class ControlUpliftResult:
    """Per-variant uplift versus a named control arm.

    ``variants`` preserves input order and excludes the control. Pairwise
    summaries match :func:`variant_vs_control` at the same ``ci``,
    ``n_samples``, and ``seed``.
    """

    control_name: str
    control_conversions: int
    control_trials: int
    control_mean: float
    variants: tuple[VariantVsControl, ...]
    ci: float
    n_samples: int


def variant_vs_control(
    control: BetaBinomialPosterior,
    variant: BetaBinomialPosterior,
    ci: float = 0.95,
    n_samples: int = 100_000,
    seed: int | None = 20260824,
) -> UpliftResult:
    """Summarise a variant against a control conversion posterior.

    Returns P(variant > control) by quadrature (no sampling noise),
    conjugate expected absolute and relative uplift, and Monte Carlo
    equal-tailed credible intervals. The absolute-interval draws use the
    same generator sequence as
    :func:`~bayes_ab_kit.decisions.difference_credible_interval` and
    :func:`~bayes_ab_kit.rope.rope_decision`, so the interval can be
    reused with those APIs at a matching seed and sample count.
    """
    if not isinstance(control, BetaBinomialPosterior) or not isinstance(
        variant, BetaBinomialPosterior
    ):
        raise ValueError("control and variant must be BetaBinomialPosterior")
    _validate_level(ci)
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")

    abs_lower, abs_upper = difference_credible_interval(
        control, variant, ci=ci, n_samples=n_samples, seed=seed
    )
    rel_lower, rel_upper = _relative_uplift_interval(
        control, variant, ci=ci, n_samples=n_samples, seed=seed
    )
    return UpliftResult(
        prob_beats_control=prob_b_beats_a_quad(control, variant),
        expected_uplift=expected_absolute_uplift(control, variant),
        expected_relative_uplift=expected_relative_uplift(control, variant),
        uplift_ci_lower=abs_lower,
        uplift_ci_upper=abs_upper,
        relative_uplift_ci_lower=rel_lower,
        relative_uplift_ci_upper=rel_upper,
        ci=ci,
    )


def variants_vs_control(
    arms: Mapping[str, BetaBinomialPosterior],
    control: str,
    ci: float = 0.95,
    n_samples: int = 100_000,
    seed: int | None = 20260824,
) -> ControlUpliftResult:
    """Compare every non-control conversion arm against a named control.

    ``arms`` is the same mapping shape as
    :func:`~bayes_ab_kit.multiarms.probability_of_being_best`. Each
    variant is scored with :func:`variant_vs_control`; this is a
    control-referenced uplift diagnostic, not a P(best) ranking.
    """
    _validate_level(ci)
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if not isinstance(arms, Mapping) or len(arms) < 2:
        raise ValueError("at least two arms are required")

    control_key = str(control).strip()
    if not control_key:
        raise ValueError("control name must be non-empty")

    items: list[tuple[str, BetaBinomialPosterior]] = []
    seen: set[str] = set()
    for raw_name, posterior in arms.items():
        name = str(raw_name).strip()
        if not name:
            raise ValueError("arm name must be non-empty")
        if name in seen:
            raise ValueError(f"duplicate arm name: {name}")
        if not isinstance(posterior, BetaBinomialPosterior):
            raise ValueError("each arm must be a BetaBinomialPosterior")
        seen.add(name)
        items.append((name, posterior))

    if control_key not in seen:
        raise ValueError(f"control arm not found: {control_key}")

    control_post = next(post for name, post in items if name == control_key)
    variants = tuple(
        VariantVsControl(
            name=name,
            conversions=posterior.successes,
            trials=posterior.trials,
            posterior_mean=posterior.mean(),
            uplift=variant_vs_control(
                control_post,
                posterior,
                ci=ci,
                n_samples=n_samples,
                seed=seed,
            ),
        )
        for name, posterior in items
        if name != control_key
    )
    if not variants:
        raise ValueError("at least one non-control variant is required")

    return ControlUpliftResult(
        control_name=control_key,
        control_conversions=control_post.successes,
        control_trials=control_post.trials,
        control_mean=control_post.mean(),
        variants=variants,
        ci=ci,
        n_samples=n_samples,
    )


__all__ = [
    "ControlUpliftResult",
    "UpliftResult",
    "VariantVsControl",
    "expected_absolute_uplift",
    "expected_relative_uplift",
    "variant_vs_control",
    "variants_vs_control",
]
