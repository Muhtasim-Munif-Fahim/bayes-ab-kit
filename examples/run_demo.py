"""End-to-end demo of bayes-ab-kit on a synthetic two-arm experiment.

Run from the repository root:

    python examples/run_demo.py [output_path]

Produces a Markdown report (default: examples/demo_report.md).
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np

from bayes_ab_kit.decisions import superiority_decision
from bayes_ab_kit.evaluation import VariantData, evaluate_variants
from bayes_ab_kit.posteriors import BetaBinomialPosterior
from bayes_ab_kit.power import required_sample_size
from bayes_ab_kit.reporting import (
    ComparisonBlock,
    ConversionReport,
    VariantLine,
    render_conversion_report,
    write_report,
)
from bayes_ab_kit.risk import expected_loss
from bayes_ab_kit.sequential import simulate_null_peeking
from bayes_ab_kit.stopping import simulate_stopping_plan
from bayes_ab_kit.synthdata import generate_experiment, uplift_pair_specs


def main(out_path: Path = Path(__file__).parent / "demo_report.md") -> Path:
    specs = uplift_pair_specs(
        visitors_per_arm=2500,
        baseline_rate=0.10,
        relative_uplift=0.25,
        log_revenue_uplift=0.15,
    )
    arms = {arm.name: arm for arm in generate_experiment(specs, seed=20260824)}
    control, treatment = arms["A"], arms["B"]

    post_a = BetaBinomialPosterior.from_counts(control.conversions, control.trials)
    post_b = BetaBinomialPosterior.from_counts(treatment.conversions, treatment.trials)

    comparison = superiority_decision(post_a, post_b, ci=0.95)
    loss_if_ship_b = expected_loss(post_b, post_a, n_samples=50_000)

    arpu = evaluate_variants(
        VariantData("A", control.conversions, control.trials, control.order_values),
        VariantData("B", treatment.conversions, treatment.trials, treatment.order_values),
        n_samples=40_000,
    )

    plan = required_sample_size(0.10, 0.125, alpha=0.05, power=0.80)
    peek_risk = simulate_null_peeking(
        p_true=0.10, per_look_n=500, n_looks=5, n_sims=1500
    )
    stopping = simulate_stopping_plan(
        p_a=0.10, p_b=0.125, per_look_n=500, max_looks=5, threshold=0.975, n_sims=1500
    )

    report = ConversionReport(
        title="Synthetic checkout experiment: A vs B",
        alpha0=post_a.alpha0,
        beta0=post_a.beta0,
        variants=[
            VariantLine(
                "A", control.conversions, control.trials,
                post_a.mean(), *post_a.credible_interval(),
            ),
            VariantLine(
                "B", treatment.conversions, treatment.trials,
                post_b.mean(), *post_b.credible_interval(),
            ),
        ],
        comparison=ComparisonBlock(
            prob_b_beats_a=comparison.prob_b_beats_a,
            diff_lower=comparison.diff_ci_lower,
            diff_upper=comparison.diff_ci_upper,
            decision=comparison.decision.value,
            notes=[
                f"Expected loss if shipping B: {loss_if_ship_b:.5f} (threshold 0.0025).",
                (
                    f"ARPU posterior means: A={arpu.summary_a.mean:.4f}, "
                    f"B={arpu.summary_b.mean:.4f}; decision={arpu.decision.value}."
                ),
                f"Planned sample size for 12.5% target at 80% power: {plan.n_per_arm} per arm.",
                (
                    f"Peeking A/A false-stop rate over 5 looks: "
                    f"{peek_risk.false_stop_rate:.3f} at nominal 0.05."
                ),
                (
                    f"Bayesian stopping plan detection rate: {stopping.stop_rate:.3f}, "
                    f"expected visitors per arm: {stopping.expected_sample_size_per_arm:.0f}."
                ),
            ],
        ),
    )
    markdown = render_conversion_report(report)
    return write_report(markdown, out_path)


if __name__ == "__main__":
    override = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    written = main(override) if override else main()
    print(f"report written to {written}")