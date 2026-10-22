"""Bayesian bootstrap for arbitrary metrics (Rubin, 1981).

Conjugate models cover conversions, counts and (log)normal revenue, but
many decision metrics are medians, tail quantiles (p90 latency), trimmed
means or ratios whose distribution is awkward to model. The Bayesian
bootstrap treats the observed values as the support of the population and
places a flat Dirichlet prior on their probabilities. Each posterior draw
reweights the sample with ``w ~ Dirichlet(1, ..., 1)`` and evaluates the
*weighted* statistic, so the draws approximate the posterior of the
population statistic without any parametric assumption.

* :func:`bayesian_bootstrap` returns posterior draws (and a summary) for a
  built-in statistic — ``mean``, ``median``, ``quantile``,
  ``trimmed_mean``, ``std`` — or any callable ``f(values, weights)``.
* :func:`bayesian_bootstrap_ratio` handles ``sum(X) / sum(Y)`` metrics
  (clicks / views per user) by sharing the weights across numerator and
  denominator of the same unit.
* :func:`compare_bayesian_bootstrap` compares two independent arms:
  difference and relative-lift credible intervals, ``P(B better)`` and the
  expected loss of each decision, matching the conventions of the rest of
  the kit.

For the mean the draws are exact: ``E = mean(x)`` and
``Var = sum((x - mean)^2) / (n (n + 1))``. Draws are generated in chunks
so large samples do not need an ``n_draws x n`` matrix in memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple, Union

import numpy as np

__all__ = [
    "BootstrapPosterior",
    "BootstrapComparison",
    "STATISTICS",
    "weighted_quantile",
    "bayesian_bootstrap",
    "bayesian_bootstrap_ratio",
    "compare_bayesian_bootstrap",
]

STATISTICS = ("mean", "median", "quantile", "trimmed_mean", "std")

Statistic = Union[str, Callable[[np.ndarray, np.ndarray], float]]


def _validate_ci(ci: float) -> float:
    if not 0.0 < float(ci) < 1.0:
        raise ValueError("ci must be in (0, 1)")
    return float(ci)


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float, *, presorted: bool = False) -> float:
    """Quantile of the weighted empirical distribution (inverse CDF, lower).

    Returns the smallest value ``v`` with ``sum(w[x <= v]) >= q * sum(w)``,
    the natural quantile of a discrete distribution on the sample.
    """

    if not 0.0 <= q <= 1.0:
        raise ValueError("q must be in [0, 1]")
    x = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    if not presorted:
        order = np.argsort(x, kind="mergesort")
        x, w = x[order], w[order]
    cw = np.cumsum(w)
    target = q * cw[-1]
    idx = int(np.searchsorted(cw, target - 1e-12 * cw[-1], side="left"))
    return float(x[min(idx, x.size - 1)])


def _batch_statistic(x_sorted: np.ndarray, W: np.ndarray, statistic: str, q: float, trim: float) -> np.ndarray:
    """Vectorised weighted statistic for each row of ``W`` (weights sum to 1)."""

    if statistic == "mean":
        return W @ x_sorted
    if statistic == "std":
        m = W @ x_sorted
        m2 = W @ (x_sorted ** 2)
        return np.sqrt(np.maximum(m2 - m ** 2, 0.0))
    cw = np.cumsum(W, axis=1)
    if statistic in ("median", "quantile"):
        qq = 0.5 if statistic == "median" else q
        idx = (cw < qq - 1e-12).sum(axis=1)
        return x_sorted[np.minimum(idx, x_sorted.size - 1)]
    if statistic == "trimmed_mean":
        # Weighted trimmed mean: keep the probability mass in [trim, 1 - trim].
        lo, hi = trim, 1.0 - trim
        prev = cw - W
        mass = np.clip(np.minimum(cw, hi) - np.maximum(prev, lo), 0.0, None)
        return (mass @ x_sorted) / (hi - lo)
    raise ValueError(f"unknown statistic {statistic!r}")


@dataclass(frozen=True)
class BootstrapPosterior:
    """Posterior draws of a statistic under the Bayesian bootstrap."""

    statistic: str
    n: int
    draws: np.ndarray
    point_estimate: float
    mean: float
    sd: float
    ci: float
    interval: Tuple[float, float]

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "draws"}
        d["n_draws"] = int(self.draws.size)
        return d


def _summarise(name: str, n: int, draws: np.ndarray, point: float, ci: float) -> BootstrapPosterior:
    tail = (1.0 - ci) / 2.0
    lo, hi = np.quantile(draws, [tail, 1.0 - tail])
    return BootstrapPosterior(statistic=name, n=n, draws=draws, point_estimate=float(point),
                              mean=float(draws.mean()), sd=float(draws.std(ddof=1)) if draws.size > 1 else 0.0,
                              ci=ci, interval=(float(lo), float(hi)))


def _check_values(values, name: str = "values") -> np.ndarray:
    x = np.asarray(values, dtype=float).reshape(-1)
    if x.size < 2:
        raise ValueError(f"{name} needs at least two observations")
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{name} must be finite")
    return x


def bayesian_bootstrap(
    values: Sequence[float],
    statistic: Statistic = "mean",
    *,
    q: float = 0.5,
    trim: float = 0.1,
    n_draws: int = 4000,
    ci: float = 0.95,
    seed: Optional[int] = None,
    chunk_size: int = 1_000_000,
) -> BootstrapPosterior:
    """Posterior draws of ``statistic`` for a sample under the Bayesian bootstrap.

    ``statistic`` is one of :data:`STATISTICS` or a callable
    ``f(values, weights) -> float`` (weights sum to one). ``q`` is used by
    ``"quantile"``, ``trim`` (fraction cut from each tail, in ``[0, 0.5)``)
    by ``"trimmed_mean"``. ``chunk_size`` bounds the number of Dirichlet
    weights held in memory at once.
    """

    ci = _validate_ci(ci)
    x = _check_values(values)
    if n_draws < 2:
        raise ValueError("n_draws must be at least 2")
    if not 0.0 <= trim < 0.5:
        raise ValueError("trim must be in [0, 0.5)")
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must be in [0, 1]")
    rng = np.random.default_rng(seed)
    n = x.size
    uniform = np.full(n, 1.0 / n)
    if callable(statistic):
        name = getattr(statistic, "__name__", "custom")
        draws = np.empty(n_draws)
        for i in range(n_draws):
            w = rng.dirichlet(np.ones(n))
            draws[i] = float(statistic(x, w))
        point = float(statistic(x, uniform))
        return _summarise(name, n, draws, point, ci)
    if statistic not in STATISTICS:
        raise ValueError(f"unknown statistic {statistic!r}; choose from {', '.join(STATISTICS)} or pass a callable")
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    rows = max(1, chunk_size // n)
    out = []
    remaining = n_draws
    while remaining > 0:
        m = min(rows, remaining)
        g = rng.standard_exponential((m, n))
        W = g / g.sum(axis=1, keepdims=True)  # Dirichlet(1,...,1); order is irrelevant
        out.append(_batch_statistic(xs, W, statistic, q, trim))
        remaining -= m
    draws = np.concatenate(out)
    point = float(_batch_statistic(xs, uniform[None, :], statistic, q, trim)[0])
    label = f"quantile({q:g})" if statistic == "quantile" else statistic
    return _summarise(label, n, draws, point, ci)


def bayesian_bootstrap_ratio(
    numerator: Sequence[float],
    denominator: Sequence[float],
    *,
    n_draws: int = 4000,
    ci: float = 0.95,
    seed: Optional[int] = None,
    chunk_size: int = 1_000_000,
) -> BootstrapPosterior:
    """Posterior of ``sum(X) / sum(Y)`` with unit-level weights shared by X and Y."""

    ci = _validate_ci(ci)
    xn = _check_values(numerator, "numerator")
    yd = _check_values(denominator, "denominator")
    if xn.size != yd.size:
        raise ValueError("numerator and denominator must have the same length")
    if yd.sum() <= 0:
        raise ValueError("denominator must have a positive sum")
    rng = np.random.default_rng(seed)
    n = xn.size
    rows = max(1, chunk_size // n)
    out = []
    remaining = n_draws
    while remaining > 0:
        m = min(rows, remaining)
        W = rng.standard_exponential((m, n))
        out.append((W @ xn) / (W @ yd))
        remaining -= m
    return _summarise("ratio", n, np.concatenate(out), xn.sum() / yd.sum(), ci)


@dataclass(frozen=True)
class BootstrapComparison:
    """Two-arm comparison of a statistic under the Bayesian bootstrap."""

    statistic: str
    a: BootstrapPosterior
    b: BootstrapPosterior
    difference: float
    difference_interval: Tuple[float, float]
    relative_lift: Optional[float]
    relative_lift_interval: Optional[Tuple[float, float]]
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


def compare_bayesian_bootstrap(
    a: Union[Sequence[float], BootstrapPosterior],
    b: Union[Sequence[float], BootstrapPosterior],
    statistic: Statistic = "mean",
    *,
    higher_is_better: bool = True,
    q: float = 0.5,
    trim: float = 0.1,
    n_draws: int = 4000,
    ci: float = 0.95,
    seed: Optional[int] = None,
) -> BootstrapComparison:
    """Compare a statistic between control ``a`` and treatment ``b``.

    ``a``/``b`` are raw samples (bootstrapped here with independent seeds
    derived from ``seed``) or existing :class:`BootstrapPosterior` objects
    with the same number of draws. The difference is ``B - A``; the
    relative lift ``B / A - 1`` is reported only when every draw of A is
    positive. ``expected_loss_choose_X`` is the posterior expected shortfall
    from picking X (in the metric's units, oriented by ``higher_is_better``).
    """

    ci = _validate_ci(ci)
    if isinstance(a, BootstrapPosterior) and isinstance(b, BootstrapPosterior):
        pa, pb = a, b
        if pa.draws.size != pb.draws.size:
            raise ValueError("posteriors must have the same number of draws")
    else:
        ss = np.random.SeedSequence(seed)
        sa, sb = (int(s.generate_state(1)[0]) for s in ss.spawn(2))
        kw = dict(q=q, trim=trim, n_draws=n_draws, ci=ci)
        pa = a if isinstance(a, BootstrapPosterior) else bayesian_bootstrap(a, statistic, seed=sa, **kw)
        pb = b if isinstance(b, BootstrapPosterior) else bayesian_bootstrap(b, statistic, seed=sb, **kw)
        if pa.draws.size != pb.draws.size:
            raise ValueError("posteriors must have the same number of draws")
    diff = pb.draws - pa.draws
    tail = (1.0 - ci) / 2.0
    dlo, dhi = np.quantile(diff, [tail, 1.0 - tail])
    gain = diff if higher_is_better else -diff
    prob = float(np.mean(gain > 0) + 0.5 * np.mean(gain == 0))
    loss_a = float(np.mean(np.maximum(gain, 0.0)))
    loss_b = float(np.mean(np.maximum(-gain, 0.0)))
    if np.all(pa.draws > 0):
        lift = pb.draws / pa.draws - 1.0
        llo, lhi = np.quantile(lift, [tail, 1.0 - tail])
        rel = float(pb.point_estimate / pa.point_estimate - 1.0) if pa.point_estimate > 0 else float(np.median(lift))
        rel_int: Optional[Tuple[float, float]] = (float(llo), float(lhi))
    else:
        rel, rel_int = None, None
    return BootstrapComparison(statistic=pa.statistic, a=pa, b=pb,
                               difference=float(pb.point_estimate - pa.point_estimate),
                               difference_interval=(float(dlo), float(dhi)), relative_lift=rel,
                               relative_lift_interval=rel_int, prob_b_better=prob,
                               expected_loss_choose_a=loss_a, expected_loss_choose_b=loss_b,
                               higher_is_better=higher_is_better, ci=ci)
