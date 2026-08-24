"""Bayesian models for continuous revenue metrics.

Covers two common shapes:

* ``NormalRevenueModel`` — order values treated as Gaussian with a fixed
  standard deviation and a conjugate Normal prior on the mean.
* ``LognormalRevenueModel`` — strictly positive order values whose logarithm
  is Gaussian; useful for skewed spend distributions.
"""

from __future__ import annotations

import numpy as np

from .sampling import make_rng


def _validate_ci(ci: float) -> None:
    if not 0 < ci < 1:
        raise ValueError("credible level must lie strictly inside (0, 1)")


class NormalRevenueModel:
    """Conjugate Normal posterior over a mean with known observation noise."""

    def __init__(
        self,
        mu0: float = 0.0,
        tau0: float = 1000.0,
        known_sigma: float = 1.0,
    ) -> None:
        if known_sigma <= 0:
            raise ValueError("known_sigma must be positive")
        if tau0 <= 0:
            raise ValueError("prior scale tau0 must be positive")
        self.mu0 = float(mu0)
        self.tau0_sq = float(tau0) ** 2
        self.known_sigma_sq = float(known_sigma) ** 2
        self.n = 0
        self.sum_x = 0.0

    @classmethod
    def from_samples(
        cls,
        values: np.ndarray,
        known_sigma: float,
        mu0: float = 0.0,
        tau0: float = 1000.0,
    ) -> "NormalRevenueModel":
        model = cls(mu0=mu0, tau0=tau0, known_sigma=known_sigma)
        arr = np.asarray(values, dtype=float).ravel()
        if arr.size == 0:
            raise ValueError("values must be non-empty")
        model.n = int(arr.size)
        model.sum_x = float(arr.sum())
        return model

    @property
    def posterior_precision(self) -> float:
        return 1.0 / self.tau0_sq + self.n / self.known_sigma_sq

    @property
    def posterior_var(self) -> float:
        return 1.0 / self.posterior_precision

    def mean(self) -> float:
        weighted = self.mu0 / self.tau0_sq + self.sum_x / self.known_sigma_sq
        return weighted * self.posterior_var

    def sd(self) -> float:
        return float(np.sqrt(self.posterior_var))

    def credible_interval(self, ci: float = 0.95) -> tuple[float, float]:
        _validate_ci(ci)
        tail = (1.0 - ci) / 2.0
        from scipy import stats as st

        m = self.mean()
        sd = self.sd()
        return (
            m + float(st.norm.ppf(tail)) * sd,
            m + float(st.norm.ppf(1.0 - tail)) * sd,
        )

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        if size <= 0:
            raise ValueError("size must be positive")
        return rng.normal(self.mean(), self.sd(), size=size)


class LognormalRevenueModel:
    """Lognormal model for positive order values with a conjugate prior on ``mu``.

    Assumes ``log(value) ~ Normal(mu, sigma_log^2)`` where ``sigma_log`` is a
    fixed dispersion parameter (by default the sample standard deviation of
    the logged values). Uncertainty about ``mu`` is handled with the same
    Normal-Normal conjugacy used above.
    """

    def __init__(self, mu_posterior: NormalRevenueModel, sigma_log: float) -> None:
        if sigma_log <= 0:
            raise ValueError("sigma_log must be positive")
        self.mu_posterior = mu_posterior
        self.sigma_log = float(sigma_log)

    @classmethod
    def from_revenues(
        cls,
        values: np.ndarray,
        mu0: float = 0.0,
        tau0: float = 100.0,
        sigma_log: float | None = None,
    ) -> "LognormalRevenueModel":
        arr = np.asarray(values, dtype=float).ravel()
        if arr.size == 0:
            raise ValueError("values must be non-empty")
        if np.any(arr <= 0):
            raise ValueError("lognormal model requires strictly positive values")
        logs = np.log(arr)
        if sigma_log is None:
            if arr.size == 1:
                sigma_log = 1.0
            else:
                spread = float(np.std(logs, ddof=1))
                if spread <= 0:
                    raise ValueError(
                        "order values show no dispersion on the log scale; "
                        "pass sigma_log explicitly"
                    )
                sigma_log = spread
        mu_model = NormalRevenueModel.from_samples(
            logs, known_sigma=sigma_log, mu0=mu0, tau0=tau0
        )
        return cls(mu_posterior=mu_model, sigma_log=sigma_log)

    @property
    def n(self) -> int:
        return self.mu_posterior.n

    def expected_value(self) -> float:
        """Analytic posterior expectation ``E[exp(mu + sigma_log^2 / 2)]``."""
        m = self.mu_posterior.mean()
        v = self.mu_posterior.posterior_var
        return float(np.exp(m + (self.sigma_log**2 + v) / 2.0))

    def expected_value_draws(self, rng: np.random.Generator, size: int) -> np.ndarray:
        """Monte Carlo draws of the expected revenue per payer."""
        mu_draws = self.mu_posterior.sample(rng, size=size)
        return np.exp(mu_draws + self.sigma_log**2 / 2.0)

    def credible_interval_for_mu(self, ci: float = 0.95) -> tuple[float, float]:
        return self.mu_posterior.credible_interval(ci)


__all__ = ["LognormalRevenueModel", "NormalRevenueModel", "make_rng"]