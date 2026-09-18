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
