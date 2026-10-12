import math

import numpy as np
import pytest
from scipy.special import betaln
from scipy.stats import chisquare

from bayes_ab_kit import sequential_srm_monitor, srm_check, srm_log_bayes_factor
from bayes_ab_kit.cli import main


def test_chi_square_matches_scipy():
    counts = [5100, 4900, 10_300]
    shares = [0.25, 0.25, 0.5]
    result = srm_check(counts, shares)
    ref = chisquare(counts, f_exp=np.array(shares) * sum(counts))
    assert result.chi_square == pytest.approx(ref.statistic)
    assert result.p_value == pytest.approx(ref.pvalue)
    assert result.dof == 2


def test_two_arm_bayes_factor_matches_beta_function():
    # Dirichlet(1, 1) = Beta(1, 1); marginal is B(n1 + 1, n2 + 1) / B(1, 1).
    n1, n2 = 37, 63
    expected = betaln(n1 + 1, n2 + 1) - (n1 * math.log(0.5) + n2 * math.log(0.5))
    assert srm_log_bayes_factor([n1, n2]) == pytest.approx(expected)


def test_balanced_split_is_not_flagged_and_favours_h0():
    result = srm_check([50_120, 49_880])
    assert not result.mismatch
    assert not result.bayesian_mismatch
    assert result.log_bf10 < 0  # evidence for the planned 50/50 split


def test_classic_srm_is_flagged():
    # 50.8% / 49.2% on 200k users.
    result = srm_check([101_600, 98_400], arms=["control", "treatment"])
    assert result.mismatch
    assert result.bayesian_mismatch
    assert result.arms == ("control", "treatment")
    assert result.observed_shares[0] == pytest.approx(0.508)


def test_expected_shares_are_normalised_and_unequal_plans_work():
    a = srm_check([9_000, 1_000], [9, 1])
    b = srm_check([9_000, 1_000], [0.9, 0.1])
    assert a.expected_shares == pytest.approx((0.9, 0.1))
    assert a.chi_square == pytest.approx(b.chi_square)
    assert a.chi_square == pytest.approx(0.0)
    assert not a.mismatch


def test_posterior_mean_shares():
    result = srm_check([3, 7], prior_concentration=1.0)
    assert result.posterior_mean_shares == pytest.approx((4 / 12, 8 / 12))


def test_sequential_monitor_alarms_on_drifting_allocation():
    rng = np.random.default_rng(0)
    looks = np.cumsum(rng.multinomial(1000, [0.53, 0.47], size=30), axis=0)
    result = sequential_srm_monitor(looks.tolist(), alpha=0.001)
    assert result.mismatch
    assert result.stopped_at is not None
    assert result.anytime_p_values[result.stopped_at] <= 0.001
    # Anytime-valid p-values never increase.
    assert all(x >= y for x, y in zip(result.anytime_p_values, result.anytime_p_values[1:]))


def test_sequential_monitor_controls_false_alarms_under_peeking():
    rng = np.random.default_rng(42)
    alpha = 0.05
    sims, looks = 400, 25
    alarms = 0
    for _ in range(sims):
        path = np.cumsum(rng.multinomial(200, [0.5, 0.5], size=looks), axis=0)
        if sequential_srm_monitor(path.tolist(), alpha=alpha).mismatch:
            alarms += 1
    # Ville's inequality bounds the rate by alpha; allow Monte Carlo slack.
    assert alarms / sims <= alpha + 0.03


def test_validation():
    with pytest.raises(ValueError, match="two arms"):
        srm_check([10])
    with pytest.raises(ValueError, match="non-negative"):
        srm_check([10, -1])
    with pytest.raises(ValueError, match="one entry per arm"):
        srm_check([10, 10], [0.5, 0.3, 0.2])
    with pytest.raises(ValueError, match="alpha"):
        srm_check([10, 10], alpha=1.5)
    with pytest.raises(ValueError, match="non-decreasing"):
        sequential_srm_monitor([[10, 10], [9, 12]])


def test_cli_srm_reports_mismatch(capsys):
    rc = main(["srm", "--arm", "control:101600", "--arm", "treatment:98400"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "| control" in out
    assert "SRM (p < 0.001)" in out and ": yes" in out


def test_cli_srm_bad_arm_spec_exits_2():
    with pytest.raises(SystemExit) as exc:
        main(["srm", "--arm", "control", "--arm", "b:10"])
    assert exc.value.code == 2
