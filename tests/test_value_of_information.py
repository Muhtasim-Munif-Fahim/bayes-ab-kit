import numpy as np
import pytest

from bayes_ab_kit import (
    BetaBinomialPosterior,
    evpi,
    evsi,
    expected_loss,
    optimal_additional_sample,
)


def _pair():
    a = BetaBinomialPosterior.from_counts(48, 1000)
    b = BetaBinomialPosterior.from_counts(55, 1000)
    return a, b


def test_evpi_equals_expected_loss_of_leader():
    a, b = _pair()
    value = evpi([a, b], n_samples=400_000, seed=1)
    loss = expected_loss(b, a, n_samples=400_000, seed=2)
    assert value == pytest.approx(loss, rel=0.03)


def test_evsi_zero_monotone_and_bounded_by_evpi():
    a, b = _pair()
    perfect = evpi([a, b], seed=3)
    values = [evsi([a, b], n, seed=3) for n in (0, 100, 1000, 10_000, 200_000)]
    assert values[0] == 0.0
    assert all(v >= 0 for v in values)
    assert all(np.diff(values) > -2e-5)
    assert values[-1] <= perfect + 2e-4
    assert values[-1] == pytest.approx(perfect, rel=0.08)


def test_evsi_matches_exact_two_arm_enumeration():
    # Small case: enumerate beta-binomial predictive exactly.
    from scipy.stats import betabinom

    a = BetaBinomialPosterior.from_counts(3, 10)
    b = BetaBinomialPosterior.from_counts(5, 12)
    n = 8
    ya = np.arange(n + 1)
    pa = betabinom.pmf(ya, n, a.alpha, a.beta)
    pb = betabinom.pmf(ya, n, b.alpha, b.beta)
    ma = (a.alpha + ya) / (a.alpha + a.beta + n)
    mb = (b.alpha + ya) / (b.alpha + b.beta + n)
    exact = float(np.sum(pa[:, None] * pb[None, :] * np.maximum(ma[:, None], mb[None, :])))
    exact -= max(a.mean(), b.mean())
    assert evsi([a, b], n, n_samples=400_000, seed=4) == pytest.approx(exact, abs=5e-4)


def test_identical_certain_arms_have_no_value():
    a = BetaBinomialPosterior.from_counts(50_000, 1_000_000)
    b = BetaBinomialPosterior.from_counts(80_000, 1_000_000)
    assert evpi([a, b], seed=0) < 1e-9
    assert evsi([a, b], 1000, seed=0) < 1e-9


def test_optimal_sample_trades_gain_against_cost():
    a, b = _pair()
    cheap = optimal_additional_sample([a, b], horizon=1_000_000, cost_per_visitor=0.001,
                                      candidate_n=[0, 500, 2000, 8000], n_samples=50_000)
    assert cheap.keep_running
    assert cheap.best_n_per_arm > 0
    assert cheap.leader == "b"
    pricey = optimal_additional_sample([a, b], horizon=1_000, cost_per_visitor=1.0,
                                       candidate_n=[0, 500, 2000], n_samples=50_000)
    assert pricey.best_n_per_arm == 0
    assert not pricey.keep_running
    row = cheap.curve[1]
    assert row.enbs == pytest.approx(row.gain - row.cost)
    assert cheap.max_value == pytest.approx(1_000_000 * cheap.evpi)


def test_default_grid_and_mapping_names():
    a, b = _pair()
    c = BetaBinomialPosterior.from_counts(40, 1000)
    res = optimal_additional_sample({"ctl": a, "new": b, "alt": c}, horizon=50_000, n_samples=20_000)
    assert res.names == ("ctl", "new", "alt")
    assert res.leader == "new"
    assert res.curve[0].n_per_arm == 0 and len(res.curve) > 3


@pytest.mark.parametrize(
    "call",
    [
        lambda a, b: evsi([a, b], -1),
        lambda a, b: evpi([a]),
        lambda a, b: evpi([a, b], n_samples=0),
        lambda a, b: optimal_additional_sample([a, b], horizon=0),
        lambda a, b: optimal_additional_sample([a, b], horizon=10, cost_per_visitor=-1),
    ],
)
def test_validation(call):
    a, b = _pair()
    with pytest.raises(ValueError):
        call(a, b)


def test_type_check():
    with pytest.raises(TypeError):
        evpi([1, 2])
