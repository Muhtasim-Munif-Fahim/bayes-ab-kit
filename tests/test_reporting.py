import numpy as np
import pytest

from bayes_ab_kit.reporting import (
    ComparisonBlock,
    ConversionReport,
    VariantLine,
    markdown_table,
    render_conversion_report,
    write_report,
)


def line(name="A", conv=80, trials=1000, lo=0.066, mean=0.081, hi=0.098):
    return VariantLine(name, conv, trials, mean, lo, hi)


def make_report(**overrides):
    fields = dict(
        title="Checkout funnel test",
        alpha0=1.0,
        beta0=1.0,
        variants=[line("A"), line("B", 110, 1000, 0.094, 0.111, 0.130)],
        comparison=ComparisonBlock(
            prob_b_beats_a=0.987,
            diff_lower=0.0042,
            diff_upper=0.0558,
            decision="ship_b",
        ),
    )
    fields.update(overrides)
    return ConversionReport(**fields)


def test_table_has_piped_header_and_divider():
    md = markdown_table(["a", "b"], [["1", "2"]])
    lines = md.splitlines()
    assert lines[0].startswith("|")
    assert set(lines[1]) <= {"|", "-"}
    assert lines[2].count("|") == lines[0].count("|") == 3


def test_table_pads_columns_to_uniform_width():
    md = markdown_table(["x"], [["longer-cell"]])
    header, _, row = md.splitlines()
    assert len(header) == len(row)


def test_table_rejects_mismatched_widths():
    with pytest.raises(ValueError):
        markdown_table(["a", "b"], [["1"]])
    with pytest.raises(ValueError):
        markdown_table([], [])
    with pytest.raises(ValueError):
        markdown_table(["a"], [])


def test_report_contains_sections_and_numbers():
    md = render_conversion_report(make_report())
    assert "# Checkout funnel test" in md
    assert "## Per-variant summary" in md
    assert "## Comparison" in md
    assert "0.987" in md
    assert "ship_b" in md
    assert "| A " in md and "| B " in md


def test_report_includes_notes_when_present():
    block = ComparisonBlock(0.99, 0.01, 0.05, "ship_b", notes=["Check SRM before shipping."])
    md = render_conversion_report(make_report(comparison=block))
    assert "## Notes" in md
    assert "- Check SRM before shipping." in md


def test_render_is_deterministic():
    first = render_conversion_report(make_report())
    second = render_conversion_report(make_report())
    assert first == second


def test_report_requires_two_variants_and_title():
    with pytest.raises(ValueError):
        make_report(variants=[line("A")])
    with pytest.raises(ValueError):
        make_report(title="")


def test_write_report_creates_file_with_content(tmp_path):
    target = tmp_path / "nested" / "report.md"
    md = render_conversion_report(make_report())
    written = write_report(md, target)
    assert written == target
    assert target.read_text(encoding="utf-8").strip().startswith("# Checkout funnel")


def test_write_report_refuses_empty_text(tmp_path):
    with pytest.raises(ValueError):
        write_report("   \n", tmp_path / "empty.md")