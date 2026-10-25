"""Dirichlet-multinomial posteriors for categorical and ordinal outcomes.

Star ratings, NPS buckets, plan choices and survey answers are neither
binary conversions nor continuous metrics. The conjugate model for counts
``n = (n_1, ..., n_K)`` over ``K`` categories is

* likelihood: ``n ~ Multinomial(N, p)``
* prior:      ``p ~ Dirichlet(a_1, ..., a_K)``
* posterior:  ``p ~ Dirichlet(a_1 + n_1, ..., a_K + n_K)``

Attaching a numeric score ``s_k`` to every category (1..5 stars, NPS
-1/0/+1) turns the share vector into a mean score ``mu = s . p`` whose
posterior mean and variance are closed form (``alpha0 = sum alpha``)::

    E[mu]   = s . alpha / alpha0
    Var[mu] = (sum s_k^2 alpha_k / alpha0 - E[mu]^2) / (alpha0 + 1)

Comparisons between two arms use seeded Dirichlet draws: the posterior of
the mean-score difference, ``P(mu_B > mu_A)``, expected loss, the
posterior of each category share difference, and — for ordinal scales —
the *probability of superiority* ``P(Y_B > Y_A) + 0.5 P(Y_B = Y_A)`` of a
random B response over a random A response (the Bayesian analogue of the
Mann-Whitney / Vargha-Delaney A effect size), which uses only the order of
the categories and not the scores.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .sampling import make_rng

__all__ = [
    "DirichletMultinomialPosterior",
    "CategoricalComparisonResult",
    "compare_categorical",
    "probability_of_superiority",
]


def _as_counts(counts: Sequence[int]) -> np.ndarray:
    arr = np.asarray(counts)
    if arr.ndim != 1 or arr.size < 2:
        raise ValueError("counts must be a 1-D sequence with at least two categories")
    if np.any(arr < 0) or not np.all(np.equal(np.mod(arr, 1), 0)):
        raise ValueError("counts must be non-negative integers")
    return arr.astype(np.int64)


class DirichletMultinomialPosterior:
    """Posterior over category shares from multinomial counts."""

    def __init__(
        self,
        n_categories: int,
        prior: float | Sequence[float] = 1.0,
        scores: Sequence[float] | None = None,
    ) -> None:
        if int(n_categories) != n_categories or n_categories < 2:
            raise ValueError("n_categories must be an integer >= 2")
        k = int(n_categories)
        alpha = np.broadcast_to(np.asarray(prior, dtype=float), (k,)).copy()
        if np.any(alpha <= 0) or not np.all(np.isfinite(alpha)):
            raise ValueError("prior concentrations must be positive and finite")
        self.prior = alpha
        if scores is None:
            scores = np.arange(1, k + 1, dtype=float)
        s = np.asarray(scores, dtype=float)
        if s.shape != (k,) or not np.all(np.isfinite(s)):
            raise ValueError("scores must be finite with one value per category")
        self.scores = s
        self.counts = np.zeros(k, dtype=np.int64)

    @classmethod
    def from_counts(
        cls,
        counts: Sequence[int],
        prior: float | Sequence[float] = 1.0,
        scores: Sequence[float] | None = None,
    ) -> "DirichletMultinomialPosterior":
        arr = _as_counts(counts)
        post = cls(arr.size, prior=prior, scores=scores)
        post.update(arr)
        return post

    @property
    def n_categories(self) -> int:
        return int(self.prior.size)

    @property
    def alpha(self) -> np.ndarray:
        return self.prior + self.counts

    @property
    def n(self) -> int:
        return int(self.counts.sum())

    def update(self, counts: Sequence[int]) -> "DirichletMultinomialPosterior":
        arr = _as_counts(counts)
        if arr.size != self.n_categories:
            raise ValueError("counts must have one entry per category")
        self.counts = self.counts + arr
        return self

    def share_means(self) -> np.ndarray:
        a = self.alpha
        return a / a.sum()

    def share_intervals(self, ci: float = 0.95) -> np.ndarray:
        """Equal-tailed marginal intervals; each share is ``Beta(a_k, alpha0 - a_k)``."""
        from scipy import stats

        if not 0 < ci < 1:
            raise ValueError("ci must lie in (0, 1)")
        a = self.alpha
        tail = (1 - ci) / 2
        lo = stats.beta.ppf(tail, a, a.sum() - a)
        hi = stats.beta.ppf(1 - tail, a, a.sum() - a)
        return np.column_stack([lo, hi])

    def mean_score(self) -> float:
        return float(self.scores @ self.share_means())

    def mean_score_var(self) -> float:
        a = self.alpha
        a0 = a.sum()
        m = self.scores @ a / a0
        second = (self.scores**2) @ a / a0
        return float((second - m**2) / (a0 + 1.0))

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        """Draw ``(size, K)`` share vectors from the posterior."""
        if size <= 0:
            raise ValueError("size must be positive")
        return rng.dirichlet(self.alpha, size=size)


def probability_of_superiority(p_a: np.ndarray, p_b: np.ndarray) -> np.ndarray:
    """``P(Y_B > Y_A) + 0.5 P(Y_B = Y_A)`` for ordered categories.

    ``p_a`` and ``p_b`` are share vectors (or ``(m, K)`` batches) over the
    same ordered categories. Returns a scalar or a length-``m`` array.
    """

    pa = np.atleast_2d(np.asarray(p_a, dtype=float))
    pb = np.atleast_2d(np.asarray(p_b, dtype=float))
    if pa.shape != pb.shape:
        raise ValueError("share vectors must have the same shape")
    cdf_a_below = np.cumsum(pa, axis=1) - pa  # P(Y_A < k)
    out = np.sum(pb * (cdf_a_below + 0.5 * pa), axis=1)
    return out if np.ndim(p_a) > 1 else out[0]


@dataclass(frozen=True)
class CategoricalComparisonResult:
    """Two-arm Dirichlet-multinomial comparison."""

    counts_a: tuple[int, ...]
    counts_b: tuple[int, ...]
    scores: tuple[float, ...]
    mean_score_a: float
    mean_score_b: float
    mean_score_a_interval: tuple[float, float]
    mean_score_b_interval: tuple[float, float]
    mean_diff: float
    mean_diff_interval: tuple[float, float]
    prob_b_better: float
    expected_loss_choose_a: float
    expected_loss_choose_b: float
    share_diff_means: tuple[float, ...]
    prob_share_b_higher: tuple[float, ...]
    superiority: float
    superiority_interval: tuple[float, float]
    prob_superiority_above_half: float
    higher_is_better: bool
    ci: float

    @property
    def winner(self) -> str:
        return "B" if self.prob_b_better >= 0.5 else "A"


def compare_categorical(
    counts_a: Sequence[int],
    counts_b: Sequence[int],
    *,
    scores: Sequence[float] | None = None,
    prior: float | Sequence[float] = 1.0,
    higher_is_better: bool = True,
    ci: float = 0.95,
    n_samples: int = 50_000,
    seed: int | None = 20261025,
) -> CategoricalComparisonResult:
    """Compare two arms on a categorical / ordinal outcome.

    ``scores`` default to ``1..K`` (e.g. star ratings). ``higher_is_better``
    controls which direction of the mean score counts as better (and flips
    the probability of superiority accordingly).
    """

    if not 0 < ci < 1:
        raise ValueError("ci must lie in (0, 1)")
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    post_a = DirichletMultinomialPosterior.from_counts(counts_a, prior=prior, scores=scores)
    post_b = DirichletMultinomialPosterior.from_counts(counts_b, prior=prior, scores=scores)
    if post_a.n_categories != post_b.n_categories:
        raise ValueError("both arms need the same number of categories")
    rng = make_rng(seed)
    pa = post_a.sample(rng, n_samples)
    pb = post_b.sample(rng, n_samples)
    s = post_a.scores
    mu_a = pa @ s
    mu_b = pb @ s
    diff = mu_b - mu_a
    sign = 1.0 if higher_is_better else -1.0
    tail = (1 - ci) / 2
    q = lambda x: (float(np.quantile(x, tail)), float(np.quantile(x, 1 - tail)))  # noqa: E731
    sup = probability_of_superiority(pa, pb)
    if not higher_is_better:
        sup = 1.0 - sup
    gain = sign * diff
    return CategoricalComparisonResult(
        counts_a=tuple(int(c) for c in post_a.counts),
        counts_b=tuple(int(c) for c in post_b.counts),
        scores=tuple(float(v) for v in s),
        mean_score_a=post_a.mean_score(),
        mean_score_b=post_b.mean_score(),
        mean_score_a_interval=q(mu_a),
        mean_score_b_interval=q(mu_b),
        mean_diff=float(post_b.mean_score() - post_a.mean_score()),
        mean_diff_interval=q(diff),
        prob_b_better=float(np.mean(gain > 0)),
        expected_loss_choose_a=float(np.mean(np.maximum(gain, 0.0))),
        expected_loss_choose_b=float(np.mean(np.maximum(-gain, 0.0))),
        share_diff_means=tuple(float(v) for v in post_b.share_means() - post_a.share_means()),
        prob_share_b_higher=tuple(float(v) for v in np.mean(pb > pa, axis=0)),
        superiority=float(np.mean(sup)),
        superiority_interval=q(sup),
        prob_superiority_above_half=float(np.mean(sup > 0.5)),
        higher_is_better=higher_is_better,
        ci=ci,
    )
