"""Tests for CUPED covariate adjustment."""

from __future__ import annotations

import numpy as np
import pytest

from bayes_ab_kit.cuped import (
    cuped_adjust,
    cuped_variance_reduction,
    estimate_cuped_theta,
)


def test_perfect_covariate_zeroes_variance():
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    y = 3.0 * x + 1.0
    result = cuped_adjust(y, x)
    assert result.fit.variance_reduction == pytest.approx(1.0, abs=1e-9)
    assert result.var_adjusted == pytest.approx(0.0, abs=1e-9)
    assert result.adjusted.mean() == pytest.approx(y.mean(), abs=1e-9)


def test_uncorrelated_covariate_no_reduction():
    rng = np.random.default_rng(1)
    y = rng.normal(size=500)
    x = rng.normal(size=500)
    fit = estimate_cuped_theta(y, x)
    assert abs(fit.correlation) < 0.15
    assert fit.variance_reduction < 0.05


def test_partial_correlation_reduces_variance():
    rng = np.random.default_rng(2)
    x = rng.normal(size=400)
    noise = rng.normal(size=400)
    y = x + noise
    result = cuped_adjust(y, x)
    assert result.var_adjusted < result.fit.var_y
    assert result.fit.variance_reduction == pytest.approx(0.5, abs=0.05)


def test_mean_preserved():
    rng = np.random.default_rng(3)
    y = rng.normal(loc=5.0, size=100)
    x = y + rng.normal(scale=0.5, size=100)
    result = cuped_adjust(y, x)
    assert result.adjusted.mean() == pytest.approx(y.mean(), abs=1e-10)


def test_fixed_theta():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    x = np.array([0.0, 1.0, 2.0, 3.0])
    result = cuped_adjust(y, x, theta=0.5)
    expected = y - 0.5 * (x - x.mean())
    assert np.allclose(result.adjusted, expected)
    assert result.fit.theta == 0.5


def test_cuped_variance_reduction_matches_fit():
    rng = np.random.default_rng(4)
    x = rng.normal(size=100)
    y = 0.7 * x + rng.normal(size=100)
    assert cuped_variance_reduction(y, x) == pytest.approx(
        estimate_cuped_theta(y, x).variance_reduction
    )


def test_to_dict():
    y = np.array([1.0, 2.0, 3.0])
    x = np.array([1.0, 1.5, 2.0])
    d = cuped_adjust(y, x).to_dict()
    assert "adjusted" in d and "fit" in d


def test_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        cuped_adjust([1.0, 2.0], [1.0])


def test_constant_x_raises():
    with pytest.raises(ValueError, match="positive variance"):
        estimate_cuped_theta([1.0, 2.0, 3.0], [5.0, 5.0, 5.0])
