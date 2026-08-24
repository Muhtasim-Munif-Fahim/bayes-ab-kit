"""Markdown rendering of experiment analyses."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    """Render a GitHub-flavoured Markdown table."""
    if not headers:
        raise ValueError("headers must be non-empty")
    if not rows:
        raise ValueError("rows must be non-empty")
    for row in rows:
        if len(row) != len(headers):
            raise ValueError("row width does not match header count")

    widths = [
        max(len(h), *(len(str(r[i])) for r in rows))
        for i, h in enumerate(headers)
    ]

    def fmt(cells: list[str]) -> str:
        return "| " + " | ".join(str(c).ljust(widths[i]) for i, c in enumerate(cells)) + " |"

    divider = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    return "\n".join([fmt(headers), divider, *(fmt(r) for r in rows)])


@dataclass(frozen=True)
class VariantLine:
    """One row of a per-variant conversion summary."""

    name: str
    conversions: int
    trials: int
    rate_mean: float
    rate_lower: float
    rate_upper: float


@dataclass(frozen=True)
class ComparisonBlock:
    """Headline comparison statistics for the report body."""

    prob_b_beats_a: float
    diff_lower: float
    diff_upper: float
    decision: str
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ConversionReport:
    """Everything needed to render a two-arm conversion analysis."""

    title: str
    alpha0: float
    beta0: float
    variants: list[VariantLine]
    comparison: ComparisonBlock

    def __post_init__(self) -> None:
        if not self.title:
            raise ValueError("title must be non-empty")
        if len(self.variants) < 2:
            raise ValueError("at least two variants are required")


def render_conversion_report(report: ConversionReport) -> str:
    """Render a complete Markdown document for a conversion experiment."""
    parts = [f"# {report.title}", ""]
    parts += [
        "Analysis uses Beta-Binomial posteriors "
        f"with prior Beta({report.alpha0:g}, {report.beta0:g}).",
        "",
    ]

    table = markdown_table(
        ["variant", "conversions", "visitors", "rate", "95% CI"],
        [
            [
                v.name,
                str(v.conversions),
                str(v.trials),
                f"{v.rate_mean:.4f}",
                f"[{v.rate_lower:.4f}, {v.rate_upper:.4f}]",
            ]
            for v in report.variants
        ],
    )
    parts += ["## Per-variant summary", "", table, ""]

    c = report.comparison
    parts += [
        "## Comparison",
        "",
        f"- P(B beats A): `{c.prob_b_beats_a:.4f}`",
        f"- 95% CI for rate difference (B - A): `[{c.diff_lower:+.4f}, {c.diff_upper:+.4f}]`",
        f"- Decision: **{c.decision}**",
    ]
    if c.notes:
        parts.append("")
        parts.append("## Notes")
        parts.append("")
        parts += [f"- {note}" for note in c.notes]
    parts.append("")
    return "\n".join(parts)


def write_report(markdown: str, path: str | Path) -> Path:
    """Write rendered Markdown to disk and return the resolved path."""
    if not markdown.strip():
        raise ValueError("refusing to write an empty report")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(markdown, encoding="utf-8", newline="\n")
    return target


__all__ = [
    "ComparisonBlock",
    "ConversionReport",
    "VariantLine",
    "markdown_table",
    "render_conversion_report",
    "write_report",
]