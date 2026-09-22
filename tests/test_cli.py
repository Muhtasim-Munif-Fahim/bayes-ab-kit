import pytest

from bayes_ab_kit.cli import build_parser, main


def test_convert_clear_winner_prints_report(capsys):
    rc = main(
        [
            "convert",
            "--conversions-a", "40", "--trials-a", "1000",
            "--conversions-b", "95", "--trials-b", "1000",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "# Conversion test" in out
    assert "ship_b" in out
    assert "E[uplift] (B - A)" in out
    assert "E[relative uplift]" in out


def test_convert_writes_report_file(tmp_path, capsys):
    target = tmp_path / "rep.md"
    rc = main(
        [
            "convert",
            "--conversions-a", "50", "--trials-a", "800",
            "--conversions-b", "70", "--trials-b", "800",
            "--out", str(target),
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert f"report written to {target}" in out
    assert target.exists() and "Per-variant summary" in target.read_text(encoding="utf-8")


def test_convert_respects_prior_flags():
    parser = build_parser()
    args = parser.parse_args(
        ["convert", "--conversions-a", "1", "--trials-a", "10",
         "--conversions-b", "2", "--trials-b", "10",
         "--prior-alpha", "2", "--prior-beta", "3", "--samples", "2000"]
    )
    assert args.prior_alpha == 2 and args.prior_beta == 3


def test_power_prints_plan(capsys):
    rc = main(["power", "--baseline", "0.10", "--expected", "0.13"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "visitors per arm" in out
    digits = [ln for ln in out.splitlines() if ln.startswith("visitors per arm")]
    n_value = int(digits[0].split(":")[1])
    assert n_value > 0


def test_peek_reports_false_stop_rate(capsys):
    rc = main(["peek", "--rate", "0.08", "--per-look", "120", "--looks", "4", "--sims", "300"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "empirical false-stop rate" in out


def test_invalid_rate_exits_with_code_two(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["peek", "--rate", "1.7", "--per-look", "100", "--looks", "2"])
    err = capsys.readouterr().err
    assert excinfo.value.code == 2
    assert "error" in err


def test_missing_required_arguments_exit_nonzero():
    with pytest.raises(SystemExit):
        main(["convert"])


def test_unknown_command_rejected():
    with pytest.raises(SystemExit):
        main(["frobnicate"])


def test_version_flag():
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0


def test_no_command_shows_help_error():
    with pytest.raises(SystemExit) as excinfo:
        main([])
    assert excinfo.value.code == 2


def test_rope_clear_winner_prints_decision_and_loss(capsys):
    rc = main(
        [
            "rope",
            "--conversions-a", "40", "--trials-a", "1000",
            "--conversions-b", "95", "--trials-b", "1000",
            "--samples", "20000",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "ROPE decision              : ship_b" in out
    assert "stop for expected loss     : yes" in out
    assert "expected-loss decision     : ship_b" in out


def test_rope_equivalence_and_custom_thresholds(capsys):
    rc = main(
        [
            "rope",
            "--conversions-a", "10000", "--trials-a", "100000",
            "--conversions-b", "10020", "--trials-b", "100000",
            "--rope", "0.01",
            "--loss-threshold", "0.01",
            "--samples", "20000",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "practical_equivalence" in out
    assert "stop for expected loss     : yes" in out


def test_best_three_arms_prints_table_and_leader(capsys):
    rc = main(
        [
            "best",
            "--arm", "A:40:1000",
            "--arm", "B:95:1000",
            "--arm", "C:50:1000",
            "--samples", "20000",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "P(best)" in out
    assert "leader                     : B" in out
    assert "P(leader is best)" in out
    assert "Monte Carlo samples        : 20000" in out


def test_best_parser_collects_repeated_arm_flags():
    parser = build_parser()
    args = parser.parse_args(
        ["best", "--arm", "A:1:10", "--arm", "B:2:10", "--arm", "C:3:10", "--samples", "1000"]
    )
    assert args.arm == ["A:1:10", "B:2:10", "C:3:10"]
    assert args.samples == 1000


def test_best_requires_three_arms(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["best", "--arm", "A:10:100", "--arm", "B:12:100"])
    err = capsys.readouterr().err
    assert excinfo.value.code == 2
    assert "at least three" in err


def test_best_rejects_malformed_arm_spec(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["best", "--arm", "A:10:100", "--arm", "B:12:100", "--arm", "not-a-spec"])
    err = capsys.readouterr().err
    assert excinfo.value.code == 2
    assert "NAME:CONVERSIONS:TRIALS" in err


def test_best_rejects_duplicate_names(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                "best",
                "--arm", "A:10:100",
                "--arm", "A:12:100",
                "--arm", "C:8:100",
            ]
        )
    err = capsys.readouterr().err
    assert excinfo.value.code == 2
    assert "duplicate arm name" in err


def test_best_rejects_conversions_outside_trials(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                "best",
                "--arm", "A:10:100",
                "--arm", "B:12:100",
                "--arm", "C:200:100",
            ]
        )
    err = capsys.readouterr().err
    assert excinfo.value.code == 2
    assert "conversion count" in err


def test_rope_invalid_half_width_exits_with_code_two(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                "rope",
                "--conversions-a", "10", "--trials-a", "100",
                "--conversions-b", "12", "--trials-b", "100",
                "--rope", "0",
            ]
        )
    err = capsys.readouterr().err
    assert excinfo.value.code == 2
    assert "error" in err


def test_uplift_two_arms_prints_probability_and_expectations(capsys):
    rc = main(
        [
            "uplift",
            "--control", "A:40:1000",
            "--arm", "B:95:1000",
            "--samples", "20000",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "P(> control)" in out
    assert "E[uplift]" in out
    assert "E[rel. uplift]" in out
    assert "control                    : A" in out
    assert "Monte Carlo samples        : 20000" in out
    table_body = out.split("control                    :")[0]
    assert "| B " in table_body
    assert "| A " not in table_body


def test_uplift_multi_arm_prints_each_variant(capsys):
    rc = main(
        [
            "uplift",
            "--control", "A:40:1000",
            "--arm", "B:95:1000",
            "--arm", "C:50:1000",
            "--samples", "15000",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "| B " in out
    assert "| C " in out
    assert "control                    : A" in out


def test_uplift_parser_collects_control_and_arms():
    parser = build_parser()
    args = parser.parse_args(
        [
            "uplift",
            "--control", "A:1:10",
            "--arm", "B:2:10",
            "--arm", "C:3:10",
            "--samples", "1000",
            "--ci", "0.9",
        ]
    )
    assert args.control == "A:1:10"
    assert args.arm == ["B:2:10", "C:3:10"]
    assert args.samples == 1000
    assert args.ci == 0.9


def test_uplift_rejects_duplicate_control_name(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                "uplift",
                "--control", "A:10:100",
                "--arm", "A:12:100",
            ]
        )
    err = capsys.readouterr().err
    assert excinfo.value.code == 2
    assert "duplicate arm name" in err


def test_uplift_rejects_malformed_spec(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["uplift", "--control", "A:10:100", "--arm", "not-a-spec"])
    err = capsys.readouterr().err
    assert excinfo.value.code == 2
    assert "NAME:CONVERSIONS:TRIALS" in err


def test_shrink_prints_posterior_means_and_shared_prior(capsys):
    rc = main(
        [
            "shrink",
            "--arm", "noisy:8:20",
            "--arm", "control:500:5000",
            "--arm", "winner:2000:5000",
            "--ci", "0.9",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "posterior mean" in out
    assert "prior weight" in out
    assert "90% CI" in out
    assert "prior mean (grand mean)" in out
    assert "shared prior alpha" in out
    assert "| noisy " in out
    assert "| control " in out
    assert "| winner " in out
    noisy_line = next(line for line in out.splitlines() if line.startswith("| noisy"))
    cells = [cell.strip() for cell in noisy_line.strip("|").split("|")]
    raw_rate = float(cells[3])
    posterior_mean = float(cells[4])
    assert raw_rate == pytest.approx(0.4)
    assert posterior_mean < raw_rate


def test_shrink_requires_two_arms(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["shrink", "--arm", "A:10:100"])
    err = capsys.readouterr().err
    assert excinfo.value.code == 2
    assert "at least two" in err


def test_shrink_rejects_duplicate_names(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["shrink", "--arm", "A:10:100", "--arm", "A:12:100"])
    err = capsys.readouterr().err
    assert excinfo.value.code == 2
    assert "duplicate arm name" in err


def test_shrink_rejects_conversions_outside_trials(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["shrink", "--arm", "A:10:100", "--arm", "B:200:100"])
    err = capsys.readouterr().err
    assert excinfo.value.code == 2
    assert "conversion count" in err
