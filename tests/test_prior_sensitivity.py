import pytest

from bayes_ab_kit.prior_sensitivity import (
    prior_mean_sweep,
    symmetric_prior_curve,
    tipping_point_strength,
)


def test_stronger_symmetric_prior_pulls_probability_toward_half():
    # Clear win for B under flat prior; stronger Beta(s,s) shrinks P(B>A).
    curve = symmetric_prior_curve(50, 1000, 80, 1000, strengths=(2.0, 200.0, 2000.0))
    probs = [p.prob_b_beats_a for p in curve]
    assert probs[0] > 0.95
    assert probs[0] > probs[1] > probs[2]
    assert probs[-1] < probs[0]


def test_tipping_point_finds_crossing():
    res = tipping_point_strength(50, 1000, 80, 1000, threshold=0.95, max_strength=5000)
    assert res.baseline_prob > 0.95
    assert res.tipping_strength is not None
    assert res.tipping_strength > 2.0
    # At the tipping strength, P should be at or below threshold.
    tip_point = next(p for p in res.curve if p.prior_strength == res.tipping_strength)
    assert tip_point.prob_b_beats_a <= 0.95 + 1e-9


def test_no_tipping_when_evidence_overwhelming():
    # Huge sample: even Beta(500,500) won't pull below 0.95.
    res = tipping_point_strength(500, 10000, 800, 10000, threshold=0.95, max_strength=200)
    assert res.baseline_prob > 0.99
    assert res.tipping_strength is None


def test_prior_mean_sweep_extremes():
    points = prior_mean_sweep(10, 100, 20, 100, strength=20.0, means=(0.2, 0.5, 0.8))
    assert len(points) == 3
    assert points[0].prior_mean == 0.2
    assert all(0.0 < p.prob_b_beats_a < 1.0 for p in points)


def test_rejects_bad_args():
    with pytest.raises(ValueError):
        tipping_point_strength(5, 10, 6, 10, threshold=1.5)
    with pytest.raises(ValueError):
        prior_mean_sweep(5, 10, 6, 10, means=(0.0, 1.0))
