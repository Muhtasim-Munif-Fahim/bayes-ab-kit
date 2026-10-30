"""Command-line interface for bayes-ab-kit."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from . import __version__
from .decisions import superiority_decision
from .hierarchical import hierarchical_beta_shrinkage
from .msprt import msprt_proportions, simulate_msprt_false_positives
from .multiarms import probability_of_being_best
from .power import required_sample_size
from .posteriors import BetaBinomialPosterior
from .reporting import ComparisonBlock, ConversionReport, VariantLine, markdown_table, render_conversion_report, write_report
from .bayes_factor import bayes_factor_decision, simulate_bayes_factor_stopping
from .rope import expected_loss_stop, rope_decision
from .sequential import simulate_null_peeking
from .srm import DEFAULT_SRM_ALPHA, srm_check
from .predictive import predictive_probability_of_success
from .rates import compare_rates
from .ratio import compare_ratio_metrics, ratio_summary
from .uplift import variant_vs_control, variants_vs_control


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bayes-ab",
        description="Bayesian A/B testing and experiment analysis toolkit",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_convert = sub.add_parser(
        "convert", help="compare conversion rates between two variants"
    )
    p_convert.add_argument("--conversions-a", type=int, required=True)
    p_convert.add_argument("--trials-a", type=int, required=True)
    p_convert.add_argument("--conversions-b", type=int, required=True)
    p_convert.add_argument("--trials-b", type=int, required=True)
    p_convert.add_argument("--ci", type=float, default=0.95)
    p_convert.add_argument("--prior-alpha", type=float, default=1.0)
    p_convert.add_argument("--prior-beta", type=float, default=1.0)
    p_convert.add_argument("--samples", type=int, default=100_000)
    p_convert.add_argument("--seed", type=int, default=20260824)
    p_convert.add_argument("--out", help="write the Markdown report to this path")

    p_power = sub.add_parser("power", help="pre-test sample size calculation")
    p_power.add_argument("--baseline", type=float, required=True)
    p_power.add_argument("--expected", type=float, required=True)
    p_power.add_argument("--alpha", type=float, default=0.05)
    p_power.add_argument("--target-power", type=float, default=0.80)
    p_power.add_argument("--alternative", choices=["two-sided", "one-sided"], default="two-sided")

    p_peek = sub.add_parser("peek", help="estimate false-stop risk of a peeking plan")
    p_peek.add_argument("--rate", type=float, required=True, help="shared true rate under the null")
    p_peek.add_argument("--per-look", type=int, required=True)
    p_peek.add_argument("--looks", type=int, required=True)
    p_peek.add_argument("--alpha", type=float, default=0.05)
    p_peek.add_argument("--sims", type=int, default=2000)
    p_peek.add_argument("--seed", type=int, default=20260824)

    p_rope = sub.add_parser(
        "rope",
        help="ROPE win/loss/equivalence decision and expected-loss stopping",
    )
    p_rope.add_argument("--conversions-a", type=int, required=True)
    p_rope.add_argument("--trials-a", type=int, required=True)
    p_rope.add_argument("--conversions-b", type=int, required=True)
    p_rope.add_argument("--trials-b", type=int, required=True)
    p_rope.add_argument(
        "--rope",
        type=float,
        default=0.01,
        help="symmetric ROPE half-width on rate_B - rate_A (default: 0.01)",
    )
    p_rope.add_argument(
        "--loss-threshold",
        type=float,
        default=0.0025,
        help="stop if the leader's expected loss is below this value",
    )
    p_rope.add_argument("--ci", type=float, default=0.95)
    p_rope.add_argument("--prior-alpha", type=float, default=1.0)
    p_rope.add_argument("--prior-beta", type=float, default=1.0)
    p_rope.add_argument("--samples", type=int, default=100_000)
    p_rope.add_argument("--seed", type=int, default=20260824)

    p_best = sub.add_parser(
        "best",
        help="Monte Carlo P(each arm is best) for three or more conversion arms",
    )
    p_best.add_argument(
        "--arm",
        action="append",
        required=True,
        metavar="NAME:CONVERSIONS:TRIALS",
        help="variant as name:conversions:trials (repeat at least three times)",
    )
    p_best.add_argument("--prior-alpha", type=float, default=1.0)
    p_best.add_argument("--prior-beta", type=float, default=1.0)
    p_best.add_argument("--samples", type=int, default=100_000)
    p_best.add_argument("--seed", type=int, default=20260824)

    p_uplift = sub.add_parser(
        "uplift",
        help="P(variant > control), expected uplift, and credible intervals",
    )
    p_uplift.add_argument(
        "--control",
        required=True,
        metavar="NAME:CONVERSIONS:TRIALS",
        help="control arm as name:conversions:trials",
    )
    p_uplift.add_argument(
        "--arm",
        action="append",
        required=True,
        metavar="NAME:CONVERSIONS:TRIALS",
        help="variant as name:conversions:trials (repeat for extra arms)",
    )
    p_uplift.add_argument("--ci", type=float, default=0.95)
    p_uplift.add_argument("--prior-alpha", type=float, default=1.0)
    p_uplift.add_argument("--prior-beta", type=float, default=1.0)
    p_uplift.add_argument("--samples", type=int, default=100_000)
    p_uplift.add_argument("--seed", type=int, default=20260824)

    p_shrink = sub.add_parser(
        "shrink",
        help="empirical-Bayes hierarchical Beta shrinkage across conversion arms",
    )
    p_shrink.add_argument(
        "--arm",
        action="append",
        required=True,
        metavar="NAME:CONVERSIONS:TRIALS",
        help="variant as name:conversions:trials (repeat at least twice)",
    )
    p_shrink.add_argument("--ci", type=float, default=0.95)

    bf = sub.add_parser(
        "bayes-factor",
        help="Beta-Binomial Bayes factor BF10 with Kass-Raftery label",
    )
    bf.add_argument("--conversions-a", type=int, required=True)
    bf.add_argument("--trials-a", type=int, required=True)
    bf.add_argument("--conversions-b", type=int, required=True)
    bf.add_argument("--trials-b", type=int, required=True)
    bf.add_argument("--prior-alpha", type=float, default=1.0)
    bf.add_argument("--prior-beta", type=float, default=1.0)

    bf_stop = sub.add_parser(
        "bf-stop",
        help="Simulate sequential early stopping on |log BF10|",
    )
    bf_stop.add_argument("--rate-a", type=float, required=True)
    bf_stop.add_argument("--rate-b", type=float, required=True)
    bf_stop.add_argument("--per-look", type=int, default=500)
    bf_stop.add_argument("--looks", type=int, default=5)
    bf_stop.add_argument("--log-bf-threshold", type=float, default=None,
                         help="default log(10) ≈ 2.302 (strong evidence)")
    bf_stop.add_argument("--simulations", type=int, default=2000)
    bf_stop.add_argument("--seed", type=int, default=20261007)
    bf_stop.add_argument("--prior-alpha", type=float, default=1.0)
    bf_stop.add_argument("--prior-beta", type=float, default=1.0)

    p_srm = sub.add_parser(
        "srm",
        help="sample ratio mismatch check (chi-square + Dirichlet Bayes factor)",
    )
    p_srm.add_argument(
        "--arm",
        action="append",
        required=True,
        metavar="NAME:UNITS",
        help="arm as name:units assigned (repeat at least twice)",
    )
    p_srm.add_argument(
        "--expected",
        default=None,
        help="comma-separated planned shares in --arm order (default: even split)",
    )
    p_srm.add_argument("--alpha", type=float, default=DEFAULT_SRM_ALPHA)
    p_srm.add_argument("--prior-concentration", type=float, default=1.0)


    p_pps = sub.add_parser(
        "pps",
        help="posterior predictive probability of eventual B>A success",
    )
    p_pps.add_argument("--conversions-a", type=int, required=True)
    p_pps.add_argument("--trials-a", type=int, required=True)
    p_pps.add_argument("--conversions-b", type=int, required=True)
    p_pps.add_argument("--trials-b", type=int, required=True)
    p_pps.add_argument(
        "--remaining",
        type=int,
        required=True,
        help="planned additional trials per arm (use --remaining-b to override B)",
    )
    p_pps.add_argument("--remaining-b", type=int, default=None)
    p_pps.add_argument("--threshold", type=float, default=0.95)
    p_pps.add_argument("--futility-threshold", type=float, default=0.10)
    p_pps.add_argument("--prior-alpha", type=float, default=1.0)
    p_pps.add_argument("--prior-beta", type=float, default=1.0)
    p_pps.add_argument("--sims", type=int, default=4000)
    p_pps.add_argument("--decision-samples", type=int, default=4000)
    p_pps.add_argument("--seed", type=int, default=20261014)

    p_rates = sub.add_parser(
        "rates",
        help="Gamma-Poisson comparison of event rates (events per exposure)",
    )
    p_rates.add_argument("--events-a", type=int, required=True)
    p_rates.add_argument("--exposure-a", type=float, required=True)
    p_rates.add_argument("--events-b", type=int, required=True)
    p_rates.add_argument("--exposure-b", type=float, required=True)
    p_rates.add_argument(
        "--lower-is-better",
        action="store_true",
        help="treat fewer events as better (errors, crashes, churn)",
    )
    p_rates.add_argument("--prior-shape", type=float, default=0.5)
    p_rates.add_argument("--prior-rate", type=float, default=1e-6)
    p_rates.add_argument("--ci", type=float, default=0.95)
    p_rates.add_argument("--loss-threshold", type=float, default=None)
    p_rates.add_argument("--samples", type=int, default=50_000)
    p_rates.add_argument("--seed", type=int, default=20261016)

    p_msprt = sub.add_parser(
        "msprt",
        help="always-valid mSPRT p-values and confidence sequence over cumulative looks",
    )
    p_msprt.add_argument(
        "--look",
        action="append",
        default=[],
        metavar="CA:NA:CB:NB",
        help="cumulative conversions/trials for A and B at one look (repeat, in order)",
    )
    p_msprt.add_argument("--alpha", type=float, default=0.05)
    p_msprt.add_argument("--tau", type=float, default=0.02, help="mixing sd on the absolute lift")
    p_msprt.add_argument("--theta0", type=float, default=0.0, help="null value of p_B - p_A")
    p_msprt.add_argument(
        "--simulate",
        action="store_true",
        help="instead, simulate A/A false positives of mSPRT vs naive peeking",
    )
    p_msprt.add_argument("--rate", type=float, default=0.10)
    p_msprt.add_argument("--per-look", type=int, default=500)
    p_msprt.add_argument("--looks", type=int, default=20)
    p_msprt.add_argument("--sims", type=int, default=2000)
    p_msprt.add_argument("--seed", type=int, default=20261018)

    p_ratio = sub.add_parser(
        "ratio",
        help="ratio metric (e.g. clicks/views per user) with delta-method variance",
    )
    p_ratio.add_argument(
        "--data",
        required=True,
        help="CSV with one row per randomization unit: arm,numerator,denominator",
    )
    p_ratio.add_argument("--control", default="A", help="control arm label (default A)")
    p_ratio.add_argument("--treatment", default=None, help="treatment arm label (default: the other arm)")
    p_ratio.add_argument("--ci", type=float, default=0.95)
    p_ratio.add_argument("--lower-is-better", action="store_true")

    p_boot = sub.add_parser(
        "bootstrap",
        help="Bayesian bootstrap A/B comparison of a median, quantile, trimmed mean, std or mean",
    )
    p_boot.add_argument("--data", required=True, help="CSV with one row per unit: arm,value")
    p_boot.add_argument("--control", default="A", help="control arm label (default A)")
    p_boot.add_argument("--treatment", default=None, help="treatment arm label (default: the other arm)")
    p_boot.add_argument("--statistic", choices=["mean", "median", "quantile", "trimmed_mean", "std"],
                        default="median")
    p_boot.add_argument("--q", type=float, default=0.9, help="quantile level for --statistic quantile (default 0.9)")
    p_boot.add_argument("--trim", type=float, default=0.1, help="tail fraction for trimmed_mean (default 0.1)")
    p_boot.add_argument("--draws", type=int, default=4000)
    p_boot.add_argument("--ci", type=float, default=0.95)
    p_boot.add_argument("--seed", type=int, default=20261022)
    p_boot.add_argument("--lower-is-better", action="store_true")

    p_cat = sub.add_parser(
        "categorical",
        help="Dirichlet-multinomial comparison of categorical / ordinal outcomes (e.g. star ratings)",
    )
    p_cat.add_argument("--counts-a", required=True, help="comma-separated category counts for A")
    p_cat.add_argument("--counts-b", required=True, help="comma-separated category counts for B")
    p_cat.add_argument("--scores", default=None, help="comma-separated category scores (default 1..K)")
    p_cat.add_argument("--prior", type=float, default=1.0, help="symmetric Dirichlet concentration (default 1)")
    p_cat.add_argument("--lower-is-better", action="store_true")
    p_cat.add_argument("--ci", type=float, default=0.95)
    p_cat.add_argument("--samples", type=int, default=50_000)
    p_cat.add_argument("--seed", type=int, default=20261025)

    p_meta = sub.add_parser(
        "meta",
        help="Bayesian random-effects meta-analysis of experiment lifts (partial pooling, prediction)",
    )
    p_meta.add_argument(
        "--experiment",
        action="append",
        required=True,
        metavar="NAME:ESTIMATE:SE",
        help="one experiment's effect estimate and standard error (repeat, >= 2)",
    )
    p_meta.add_argument("--tau-prior", choices=["uniform", "half-normal", "half-cauchy"], default="half-cauchy")
    p_meta.add_argument("--tau-scale", type=float, default=None, help="prior scale for tau (default median SE)")
    p_meta.add_argument("--threshold", type=float, default=0.0, help="report P(effect > threshold)")
    p_meta.add_argument("--ci", type=float, default=0.95)

    p_robust = sub.add_parser(
        "robust",
        help="robust Student-t (BEST-style) comparison of means for heavy-tailed metrics",
    )
    p_robust.add_argument("--data", required=True, help="CSV with one row per unit: arm,value")
    p_robust.add_argument("--control", default="A", help="control arm label (default A)")
    p_robust.add_argument("--treatment", default=None, help="treatment arm label (default: the other arm)")
    p_robust.add_argument("--nu", type=float, default=None, help="fix the t degrees of freedom (default: learn)")
    p_robust.add_argument("--draws", type=int, default=4000)
    p_robust.add_argument("--burn-in", type=int, default=1000)
    p_robust.add_argument("--ci", type=float, default=0.95)
    p_robust.add_argument("--seed", type=int, default=20261030)
    p_robust.add_argument("--lower-is-better", action="store_true")

    return parser


def _cmd_convert(args: argparse.Namespace) -> None:
    posterior_a = BetaBinomialPosterior.from_counts(
        args.conversions_a, args.trials_a,
        alpha0=args.prior_alpha, beta0=args.prior_beta,
    )
    posterior_b = BetaBinomialPosterior.from_counts(
        args.conversions_b, args.trials_b,
        alpha0=args.prior_alpha, beta0=args.prior_beta,
    )
    result = superiority_decision(
        posterior_a, posterior_b, ci=args.ci, n_samples=args.samples, seed=args.seed
    )
    uplift = variant_vs_control(
        posterior_a, posterior_b, ci=args.ci, n_samples=args.samples, seed=args.seed
    )

    def line(name: str, post: BetaBinomialPosterior) -> VariantLine:
        lo, hi = post.credible_interval(args.ci)
        return VariantLine(name, post.successes, post.trials, post.mean(), lo, hi)

    report = ConversionReport(
        title=f"Conversion test: A vs B at {args.ci:.0%} credibility",
        alpha0=posterior_a.alpha0,
        beta0=posterior_a.beta0,
        variants=[line("A", posterior_a), line("B", posterior_b)],
        comparison=ComparisonBlock(
            prob_b_beats_a=result.prob_b_beats_a,
            diff_lower=result.diff_ci_lower,
            diff_upper=result.diff_ci_upper,
            decision=result.decision.value,
            notes=[f"Monte Carlo samples per variant: {args.samples}"],
            expected_uplift=uplift.expected_uplift,
            expected_relative_uplift=uplift.expected_relative_uplift,
            relative_diff_lower=uplift.relative_uplift_ci_lower,
            relative_diff_upper=uplift.relative_uplift_ci_upper,
        ),
    )
    markdown = render_conversion_report(report)
    if args.out:
        write_report(markdown, Path(args.out))
        print(f"report written to {args.out}")
    else:
        print(markdown)


def _cmd_power(args: argparse.Namespace) -> None:
    plan = required_sample_size(
        baseline_rate=args.baseline,
        expected_rate=args.expected,
        alpha=args.alpha,
        power=args.target_power,
        alternative=args.alternative,
    )
    print(f"visitors per arm : {plan.n_per_arm}")
    print(f"baseline rate    : {plan.baseline_rate:.4f}")
    print(f"expected rate    : {plan.expected_rate:.4f}")
    print(f"alpha            : {plan.alpha:g} ({plan.alternative})")
    print(f"target power     : {plan.power:.2f}")


def _cmd_rope(args: argparse.Namespace) -> None:
    posterior_a = BetaBinomialPosterior.from_counts(
        args.conversions_a, args.trials_a,
        alpha0=args.prior_alpha, beta0=args.prior_beta,
    )
    posterior_b = BetaBinomialPosterior.from_counts(
        args.conversions_b, args.trials_b,
        alpha0=args.prior_alpha, beta0=args.prior_beta,
    )
    rope = rope_decision(
        posterior_a,
        posterior_b,
        rope=args.rope,
        ci=args.ci,
        n_samples=args.samples,
        seed=args.seed,
    )
    loss = expected_loss_stop(
        posterior_a,
        posterior_b,
        threshold=args.loss_threshold,
        n_samples=args.samples,
        seed=args.seed,
    )
    print(f"ROPE interval              : [{rope.rope_lower:+.4f}, {rope.rope_upper:+.4f}]")
    print(f"P(diff in ROPE)            : {rope.prob_in_rope:.4f}")
    print(f"P(B practically better)    : {rope.prob_above_rope:.4f}")
    print(f"P(A practically better)    : {rope.prob_below_rope:.4f}")
    print(
        f"{rope.ci:.0%} CI for B - A          : "
        f"[{rope.diff_ci_lower:+.4f}, {rope.diff_ci_upper:+.4f}]"
    )
    print(f"ROPE decision              : {rope.decision.value}")
    print(f"leader                     : {loss.leader}")
    print(f"expected loss of leader    : {loss.expected_loss:.5f}")
    print(f"loss threshold             : {loss.threshold:g}")
    print(f"stop for expected loss     : {'yes' if loss.should_stop else 'no'}")
    print(f"expected-loss decision     : {loss.decision.value}")


def _parse_arm_spec(spec: str) -> tuple[str, int, int]:
    parts = spec.rsplit(":", 2)
    if len(parts) != 3:
        raise ValueError("arm must be NAME:CONVERSIONS:TRIALS (for example A:95:1000)")
    name, conversions_s, trials_s = parts
    name = name.strip()
    if not name:
        raise ValueError("arm name must be non-empty")
    try:
        conversions = int(conversions_s)
        trials = int(trials_s)
    except ValueError as exc:
        raise ValueError("conversions and trials must be integers") from exc
    return name, conversions, trials


def _cmd_best(args: argparse.Namespace) -> None:
    if len(args.arm) < 3:
        raise ValueError("best requires at least three --arm flags")
    arms: dict[str, BetaBinomialPosterior] = {}
    for spec in args.arm:
        name, conversions, trials = _parse_arm_spec(spec)
        if name in arms:
            raise ValueError(f"duplicate arm name: {name}")
        arms[name] = BetaBinomialPosterior.from_counts(
            conversions,
            trials,
            alpha0=args.prior_alpha,
            beta0=args.prior_beta,
        )
    result = probability_of_being_best(
        arms, n_samples=args.samples, seed=args.seed
    )
    table = markdown_table(
        ["arm", "conversions", "visitors", "mean", "P(best)"],
        [
            [
                arm.name,
                str(arm.conversions),
                str(arm.trials),
                f"{arm.posterior_mean:.4f}",
                f"{arm.prob_best:.4f}",
            ]
            for arm in result.arms
        ],
    )
    print(table)
    print(f"leader                     : {result.leader}")
    print(f"P(leader is best)          : {result.probabilities[result.leader]:.4f}")
    print(f"Monte Carlo samples        : {result.n_samples}")


def _posterior_from_spec(
    spec: str, prior_alpha: float, prior_beta: float
) -> tuple[str, BetaBinomialPosterior]:
    name, conversions, trials = _parse_arm_spec(spec)
    return name, BetaBinomialPosterior.from_counts(
        conversions,
        trials,
        alpha0=prior_alpha,
        beta0=prior_beta,
    )


def _cmd_uplift(args: argparse.Namespace) -> None:
    control_name, control_post = _posterior_from_spec(
        args.control, args.prior_alpha, args.prior_beta
    )
    arms: dict[str, BetaBinomialPosterior] = {control_name: control_post}
    for spec in args.arm:
        name, posterior = _posterior_from_spec(spec, args.prior_alpha, args.prior_beta)
        if name in arms:
            raise ValueError(f"duplicate arm name: {name}")
        arms[name] = posterior
    result = variants_vs_control(
        arms,
        control=control_name,
        ci=args.ci,
        n_samples=args.samples,
        seed=args.seed,
    )

    def rel_cell(value: float) -> str:
        return "inf" if not math.isfinite(value) else f"{value:+.4f}"

    table = markdown_table(
        [
            "arm",
            "conversions",
            "visitors",
            "mean",
            "P(> control)",
            "E[uplift]",
            "E[rel. uplift]",
            f"{result.ci:.0%} CI abs",
            f"{result.ci:.0%} CI rel",
        ],
        [
            [
                arm.name,
                str(arm.conversions),
                str(arm.trials),
                f"{arm.posterior_mean:.4f}",
                f"{arm.uplift.prob_beats_control:.4f}",
                f"{arm.uplift.expected_uplift:+.4f}",
                rel_cell(arm.uplift.expected_relative_uplift),
                f"[{arm.uplift.uplift_ci_lower:+.4f}, {arm.uplift.uplift_ci_upper:+.4f}]",
                (
                    f"[{arm.uplift.relative_uplift_ci_lower:+.4f}, "
                    f"{arm.uplift.relative_uplift_ci_upper:+.4f}]"
                ),
            ]
            for arm in result.variants
        ],
    )
    print(table)
    print(f"control                    : {result.control_name}")
    print(f"control mean               : {result.control_mean:.4f}")
    print(f"Monte Carlo samples        : {result.n_samples}")


def _cmd_shrink(args: argparse.Namespace) -> None:
    if len(args.arm) < 2:
        raise ValueError("shrink requires at least two --arm flags")
    arms: dict[str, tuple[int, int]] = {}
    for spec in args.arm:
        name, conversions, trials = _parse_arm_spec(spec)
        if name in arms:
            raise ValueError(f"duplicate arm name: {name}")
        arms[name] = (conversions, trials)
    result = hierarchical_beta_shrinkage(arms, ci=args.ci)
    table = markdown_table(
        [
            "arm",
            "conversions",
            "visitors",
            "raw rate",
            "posterior mean",
            "prior weight",
            f"{result.ci:.0%} CI",
        ],
        [
            [
                arm.name,
                str(arm.conversions),
                str(arm.trials),
                f"{arm.raw_rate:.4f}",
                f"{arm.posterior_mean:.4f}",
                f"{arm.prior_weight:.4f}",
                f"[{arm.ci_lower:.4f}, {arm.ci_upper:.4f}]",
            ]
            for arm in result.arms
        ],
    )
    print(table)
    print(f"shared prior alpha         : {result.prior_alpha:.4f}")
    print(f"shared prior beta          : {result.prior_beta:.4f}")
    print(f"prior mean (grand mean)    : {result.prior_mean:.4f}")
    print(f"prior strength             : {result.prior_strength:.4f}")


def _cmd_peek(args: argparse.Namespace) -> None:
    result = simulate_null_peeking(
        p_true=args.rate,
        per_look_n=args.per_look,
        n_looks=args.looks,
        n_sims=args.sims,
        alpha=args.alpha,
        seed=args.seed,
    )
    print(f"A/A simulations            : {result.n_sims}")
    print(f"looks                      : {result.n_looks} x {result.per_look_n} visitors")
    print(f"nominal one-sided alpha    : {result.alpha:g}")
    print(f"false stops                : {result.false_stops}")
    print(f"empirical false-stop rate  : {result.false_stop_rate:.3f}")



def _cmd_bayes_factor(args: argparse.Namespace) -> None:
    posterior_a = BetaBinomialPosterior.from_counts(
        args.conversions_a, args.trials_a,
        alpha0=args.prior_alpha, beta0=args.prior_beta,
    )
    posterior_b = BetaBinomialPosterior.from_counts(
        args.conversions_b, args.trials_b,
        alpha0=args.prior_alpha, beta0=args.prior_beta,
    )
    result = bayes_factor_decision(posterior_a, posterior_b)
    print(f"BF10                       : {result.bf10:.6g}")
    print(f"log BF10                   : {result.log_bf10:.4f}")
    print(f"Kass-Raftery               : {result.label}")
    print(f"decision                   : {result.decision.value}")


def _cmd_bf_stop(args: argparse.Namespace) -> None:
    import math
    threshold = args.log_bf_threshold if args.log_bf_threshold is not None else math.log(10.0)
    result = simulate_bayes_factor_stopping(
        true_rate_a=args.rate_a,
        true_rate_b=args.rate_b,
        per_look=args.per_look,
        looks=args.looks,
        log_bf_threshold=threshold,
        alpha0=args.prior_alpha,
        beta0=args.prior_beta,
        n_simulations=args.simulations,
        seed=args.seed,
    )
    print(f"stop rate                  : {result.stop_rate:.4f}")
    print(f"mean looks at stop         : {result.mean_looks_at_stop:.2f}")
    print(f"mean log BF at stop        : {result.mean_log_bf_at_stop:.3f}")
    print(f"stop for H1 rate           : {result.decisive_for_h1_rate:.4f}")
    print(f"stop for H0 rate           : {result.decisive_for_h0_rate:.4f}")


def _cmd_pps(args: argparse.Namespace) -> None:
    result = predictive_probability_of_success(
        args.conversions_a,
        args.trials_a,
        args.conversions_b,
        args.trials_b,
        remaining_a=args.remaining,
        remaining_b=args.remaining_b,
        threshold=args.threshold,
        futility_threshold=args.futility_threshold,
        alpha0=args.prior_alpha,
        beta0=args.prior_beta,
        n_simulations=args.sims,
        decision_samples=args.decision_samples,
        seed=args.seed,
    )
    print(f"{'predictive P(success)':<27}: {result.predictive_probability:.4f}")
    print(f"{'mean final P(B>A)':<27}: {result.mean_final_prob_b_beats_a:.4f}")
    print(f"{'already success':<27}: {'yes' if result.prob_already_success else 'no'}")
    print(f"{'futility probability':<27}: {result.futility_probability:.4f}")
    print(f"{'continue recommended':<27}: {'yes' if result.continue_recommended else 'no'}")
    print(f"{'remaining A / B':<27}: {result.remaining_a} / {result.remaining_b}")
    print(f"{'threshold':<27}: {result.threshold:g}")


def _cmd_rates(args: argparse.Namespace) -> None:
    result = compare_rates(
        args.events_a,
        args.exposure_a,
        args.events_b,
        args.exposure_b,
        shape0=args.prior_shape,
        rate0=args.prior_rate,
        higher_is_better=not args.lower_is_better,
        ci=args.ci,
        loss_threshold=args.loss_threshold,
        n_samples=args.samples,
        seed=args.seed,
    )
    pct = f"{result.ci:.0%}"
    rows = [
        [
            "A",
            str(result.events_a),
            f"{result.exposure_a:g}",
            f"{result.rate_a_mean:.6g}",
            f"[{result.rate_a_interval[0]:.6g}, {result.rate_a_interval[1]:.6g}]",
        ],
        [
            "B",
            str(result.events_b),
            f"{result.exposure_b:g}",
            f"{result.rate_b_mean:.6g}",
            f"[{result.rate_b_interval[0]:.6g}, {result.rate_b_interval[1]:.6g}]",
        ],
    ]
    print(markdown_table(["arm", "events", "exposure", "rate mean", f"{pct} CrI"], rows))
    print()
    direction = "higher" if result.higher_is_better else "lower"
    print(f"{'P(B better)':<27}: {result.prob_b_better:.4f}  ({direction} rate is better)")
    lo, hi = result.rate_ratio_interval
    print(f"{'rate ratio B/A (median)':<27}: {result.rate_ratio_median:.4f}")
    print(f"rate ratio {pct} CrI".ljust(27) + f": [{lo:.4f}, {hi:.4f}]")
    print(f"{'expected loss choose A':<27}: {result.expected_loss_choose_a:.6g}")
    print(f"{'expected loss choose B':<27}: {result.expected_loss_choose_b:.6g}")
    if result.loss_threshold is not None:
        print(f"{'winner (loss <= threshold)':<27}: {result.winner or 'none yet'}")


def _cmd_srm(args: argparse.Namespace) -> None:
    names: list[str] = []
    counts: list[int] = []
    for spec in args.arm:
        parts = spec.rsplit(":", 1)
        if len(parts) != 2 or not parts[0].strip():
            raise ValueError("arm must be NAME:UNITS (for example A:5000)")
        try:
            units = int(parts[1])
        except ValueError as exc:
            raise ValueError("units must be an integer") from exc
        names.append(parts[0].strip())
        counts.append(units)
    shares = None
    if args.expected:
        try:
            shares = [float(x) for x in args.expected.split(",")]
        except ValueError as exc:
            raise ValueError("--expected must be comma-separated numbers") from exc
    result = srm_check(
        counts,
        shares,
        arms=names,
        alpha=args.alpha,
        prior_concentration=args.prior_concentration,
    )
    rows = [
        [arm, str(n), f"{exp:.4f}", f"{obs:.4f}"]
        for arm, n, exp, obs in zip(
            result.arms, result.counts, result.expected_shares, result.observed_shares
        )
    ]
    print(markdown_table(["arm", "units", "expected share", "observed share"], rows))
    print()
    print(f"chi-square ({result.dof} dof)".ljust(27) + f": {result.chi_square:.4f}")
    print(f"{'p-value':<27}: {result.p_value:.4g}")
    print(f"{'log BF10 (Dirichlet/plan)':<27}: {result.log_bf10:.4f}")
    print(f"SRM (p < {result.alpha:g})".ljust(27) + f": {'yes' if result.mismatch else 'no'}")
    print(f"{'SRM (BF10 >= 1/alpha)':<27}: {'yes' if result.bayesian_mismatch else 'no'}")


def _cmd_msprt(args: argparse.Namespace) -> None:
    if args.simulate:
        sim = simulate_msprt_false_positives(
            rate=args.rate,
            per_look=args.per_look,
            looks=args.looks,
            alpha=args.alpha,
            tau=args.tau,
            simulations=args.sims,
            seed=args.seed,
        )
        print(
            f"A/A at rate {sim.rate:g}, {sim.looks} looks of {sim.per_look} per arm, "
            f"{sim.simulations} simulations"
        )
        print(f"{'naive z-test peeking FPR':<27}: {sim.naive_false_positive_rate:.4f}")
        print(f"{'mSPRT always-valid FPR':<27}: {sim.msprt_false_positive_rate:.4f}  (alpha {sim.alpha:g})")
        return
    if not args.look:
        raise ValueError("pass at least one --look CA:NA:CB:NB (or --simulate)")
    rows_in = []
    for spec in args.look:
        parts = spec.split(":")
        if len(parts) != 4:
            raise ValueError("look must be CA:NA:CB:NB (for example 48:500:62:500)")
        try:
            rows_in.append(tuple(int(x) for x in parts))
        except ValueError as exc:
            raise ValueError("look counts must be integers") from exc
    result = msprt_proportions(rows_in, alpha=args.alpha, tau=args.tau, theta0=args.theta0)
    pct = f"{(1 - result.alpha) * 100:g}%"
    rows = [
        [
            str(look.look + 1),
            f"{look.delta:+.4f}",
            f"{look.std_error:.4f}",
            f"{look.p_value:.4g}",
            f"[{look.ci_low:+.4f}, {look.ci_high:+.4f}]",
        ]
        for look in result.looks
    ]
    print(markdown_table(["look", "p_B - p_A", "SE", "always-valid p", f"{pct} conf. sequence"], rows))
    print()
    print(f"{'mixing tau':<27}: {result.tau:g}")
    if result.stopped_at is None:
        print(f"{'decision':<27}: keep collecting (p > {result.alpha:g})")
    else:
        print(f"{'decision':<27}: reject p_B - p_A = {result.theta0:g} at look {result.stopped_at + 1}")


def _read_ratio_csv(path: str) -> dict[str, tuple[list[float], list[float]]]:
    try:
        with open(path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            missing = {"arm", "numerator", "denominator"} - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"{path} is missing column(s): {', '.join(sorted(missing))}")
            arms: dict[str, tuple[list[float], list[float]]] = {}
            for line, row in enumerate(reader, start=2):
                try:
                    x, y = float(row["numerator"]), float(row["denominator"])
                except (TypeError, ValueError):
                    raise ValueError(f"{path} line {line}: numerator/denominator must be numbers") from None
                xs, ys = arms.setdefault(row["arm"], ([], []))
                xs.append(x)
                ys.append(y)
    except OSError as exc:
        raise ValueError(f"could not read {path}: {exc}") from None
    return arms


def _cmd_ratio(args: argparse.Namespace) -> None:
    arms = _read_ratio_csv(args.data)
    if args.control not in arms:
        raise ValueError(f"control arm {args.control!r} not found; arms: {sorted(arms)}")
    others = [k for k in arms if k != args.control]
    treatment = args.treatment
    if treatment is None:
        if len(others) != 1:
            raise ValueError(f"pass --treatment; arms other than control: {sorted(others)}")
        treatment = others[0]
    if treatment not in arms or treatment == args.control:
        raise ValueError(f"treatment arm {treatment!r} not found or equals control")
    a = ratio_summary(*arms[args.control])
    b = ratio_summary(*arms[treatment])
    res = compare_ratio_metrics(a, b, ci=args.ci, higher_is_better=not args.lower_is_better)
    pct = f"{res.ci:.0%}"
    rows = []
    for label, s in ((args.control, a), (treatment, b)):
        deff = "n/a" if s.design_effect is None else f"{s.design_effect:.2f}"
        rows.append([label, str(s.n), f"{s.sum_numerator:g}", f"{s.sum_denominator:g}",
                     f"{s.ratio:.6g}", f"{s.se:.4g}", deff])
    print(markdown_table(["arm", "units", "numerator", "denominator", "ratio", "delta SE", "design effect"], rows))
    print()
    lo, hi = res.difference_interval
    print(f"{'difference B - A':<27}: {res.difference:+.6g}  {pct} CI [{lo:+.6g}, {hi:+.6g}]")
    llo, lhi = res.relative_lift_interval
    print(f"{'relative lift':<27}: {res.relative_lift:+.2%}  {pct} CI [{llo:+.2%}, {lhi:+.2%}]")
    print(f"{'p-value (delta method)':<27}: {res.p_value:.4g}")
    if res.naive_p_value is not None:
        print(f"{'p-value (naive pooled)':<27}: {res.naive_p_value:.4g}  (ignores within-unit correlation)")
    direction = "higher" if res.higher_is_better else "lower"
    print(f"{'P(B better)':<27}: {res.prob_b_better:.4f}  ({direction} is better)")
    print(f"{'expected loss choose A':<27}: {res.expected_loss_choose_a:.6g}")
    print(f"{'expected loss choose B':<27}: {res.expected_loss_choose_b:.6g}")


def _read_value_csv(path: str) -> dict[str, list[float]]:
    try:
        with open(path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            missing = {"arm", "value"} - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"{path} is missing column(s): {', '.join(sorted(missing))}")
            arms: dict[str, list[float]] = {}
            for line, row in enumerate(reader, start=2):
                try:
                    v = float(row["value"])
                except (TypeError, ValueError):
                    raise ValueError(f"{path} line {line}: value must be a number") from None
                arms.setdefault(row["arm"], []).append(v)
    except OSError as exc:
        raise ValueError(f"could not read {path}: {exc}") from None
    return arms


def _cmd_bootstrap(args: argparse.Namespace) -> None:
    from .bootstrap import compare_bayesian_bootstrap

    arms = _read_value_csv(args.data)
    if args.control not in arms:
        raise ValueError(f"control arm {args.control!r} not found; arms: {sorted(arms)}")
    others = [k for k in arms if k != args.control]
    treatment = args.treatment
    if treatment is None:
        if len(others) != 1:
            raise ValueError(f"pass --treatment; arms other than control: {sorted(others)}")
        treatment = others[0]
    if treatment not in arms or treatment == args.control:
        raise ValueError(f"treatment arm {treatment!r} not found or equals control")
    res = compare_bayesian_bootstrap(
        arms[args.control], arms[treatment], args.statistic, q=args.q, trim=args.trim,
        n_draws=args.draws, ci=args.ci, seed=args.seed, higher_is_better=not args.lower_is_better,
    )
    pct = f"{res.ci:.0%}"
    rows = []
    for label, post in ((args.control, res.a), (treatment, res.b)):
        lo, hi = post.interval
        rows.append([label, str(post.n), f"{post.point_estimate:.6g}", f"{post.sd:.4g}", f"[{lo:.6g}, {hi:.6g}]"])
    print(f"Bayesian bootstrap: {res.statistic} ({args.draws} Dirichlet draws)")
    print(markdown_table(["arm", "units", "estimate", "posterior SD", f"{pct} CrI"], rows))
    print()
    lo, hi = res.difference_interval
    print(f"{'difference B - A':<24}: {res.difference:+.6g}  {pct} CrI [{lo:+.6g}, {hi:+.6g}]")
    if res.relative_lift is not None:
        llo, lhi = res.relative_lift_interval
        print(f"{'relative lift':<24}: {res.relative_lift:+.2%}  {pct} CrI [{llo:+.2%}, {lhi:+.2%}]")
    direction = "higher" if res.higher_is_better else "lower"
    print(f"{'P(B better)':<24}: {res.prob_b_better:.4f}  ({direction} is better)")
    print(f"{'expected loss choose A':<24}: {res.expected_loss_choose_a:.6g}")
    print(f"{'expected loss choose B':<24}: {res.expected_loss_choose_b:.6g}")


def _cmd_robust(args: argparse.Namespace) -> None:
    from .robust import compare_robust_means

    arms = _read_value_csv(args.data)
    if args.control not in arms:
        raise ValueError(f"control arm {args.control!r} not found; arms: {sorted(arms)}")
    others = [k for k in arms if k != args.control]
    treatment = args.treatment
    if treatment is None:
        if len(others) != 1:
            raise ValueError(f"pass --treatment; arms other than control: {sorted(others)}")
        treatment = others[0]
    if treatment not in arms or treatment == args.control:
        raise ValueError(f"treatment arm {treatment!r} not found or equals control")
    res = compare_robust_means(
        arms[args.control], arms[treatment], nu=args.nu, n_draws=args.draws, burn_in=args.burn_in,
        ci=args.ci, seed=args.seed, higher_is_better=not args.lower_is_better,
    )
    pct = f"{res.ci:.0%}"
    rows = []
    for label, post in ((args.control, res.a), (treatment, res.b)):
        lo, hi = post.mu_interval
        raw = float(sum(arms[label]) / len(arms[label]))
        rows.append([label, str(post.n), f"{raw:.6g}", f"{post.mu:.6g}", f"[{lo:.6g}, {hi:.6g}]",
                     f"{post.sigma:.4g}", f"{post.nu_median:.3g}"])
    print(f"Robust Student-t comparison ({args.draws} Gibbs draws, nu {'fixed' if args.nu else 'learned'})")
    print(markdown_table(["arm", "units", "raw mean", "robust mu", f"{pct} CrI", "scale", "nu (median)"], rows))
    print()
    lo, hi = res.difference_interval
    print(f"{'difference B - A':<24}: {res.difference:+.6g}  {pct} CrI [{lo:+.6g}, {hi:+.6g}]")
    if res.relative_lift is not None:
        llo, lhi = res.relative_lift_interval
        print(f"{'relative lift':<24}: {res.relative_lift:+.2%}  {pct} CrI [{llo:+.2%}, {lhi:+.2%}]")
    elo, ehi = res.effect_size_interval
    print(f"{'effect size':<24}: {res.effect_size:+.4f}  {pct} CrI [{elo:+.4f}, {ehi:+.4f}]")
    direction = "higher" if res.higher_is_better else "lower"
    print(f"{'P(B better)':<24}: {res.prob_b_better:.4f}  ({direction} is better)")
    print(f"{'expected loss choose A':<24}: {res.expected_loss_choose_a:.6g}")
    print(f"{'expected loss choose B':<24}: {res.expected_loss_choose_b:.6g}")


def _parse_list(text: str, flag: str, cast=float) -> list:
    try:
        return [cast(v) for v in text.split(",") if v.strip() != ""]
    except ValueError as exc:
        raise ValueError(f"{flag} must be a comma-separated list of numbers") from exc


def _cmd_meta(args: argparse.Namespace) -> None:
    from .meta import meta_analysis

    names, ys, ses = [], [], []
    for spec in args.experiment:
        parts = spec.rsplit(":", 2)
        if len(parts) != 3:
            raise ValueError(f"--experiment expects NAME:ESTIMATE:SE, got {spec!r}")
        try:
            ys.append(float(parts[1]))
            ses.append(float(parts[2]))
        except ValueError as exc:
            raise ValueError(f"--experiment {spec!r}: estimate and SE must be numbers") from exc
        names.append(parts[0])
    res = meta_analysis(ys, ses, names=names, tau_prior=args.tau_prior, tau_scale=args.tau_scale,
                        ci=args.ci, threshold=args.threshold)
    pct = f"{res.ci:.0%}"
    thr = f"{res.threshold:g}"
    print(f"Random-effects meta-analysis ({len(names)} experiments, tau prior: {res.tau_prior})")
    print(f"{'experiment':<14} {'estimate':>9} {'se':>8} {'pooled':>9} {'post sd':>8} "
          f"{pct + ' CrI':>22} {'pooling':>8} {'P(>' + thr + ')':>8}")
    for i, name in enumerate(res.names):
        lo, hi = res.theta_intervals[i]
        print(f"{name:<14} {res.estimates[i]:>9.4g} {res.std_errors[i]:>8.4g} {res.theta_means[i]:>9.4g} "
              f"{res.theta_sds[i]:>8.4g} {f'[{lo:.4g}, {hi:.4g}]':>22} {res.pooling[i]:>8.2f} "
              f"{res.prob_theta_above[i]:>8.4f}")
    lo, hi = res.mu_interval
    print(f"{'overall effect mu':<26}: {res.mu_mean:.4g} (sd {res.mu_sd:.4g})  {pct} CrI [{lo:.4g}, {hi:.4g}]")
    print(f"{'P(mu > ' + thr + ')':<26}: {res.prob_mu_above:.4f}")
    lo, hi = res.tau_interval
    print(f"{'heterogeneity tau':<26}: median {res.tau_median:.4g}  {pct} CrI [{lo:.4g}, {hi:.4g}]")
    lo, hi = res.predictive_interval
    print(f"{'new experiment effect':<26}: {res.predictive_mean:.4g}  {pct} PI [{lo:.4g}, {hi:.4g}]")
    print(f"{'P(new effect > ' + thr + ')':<26}: {res.prob_new_above:.4f}")


def _cmd_categorical(args: argparse.Namespace) -> None:
    from .categorical import compare_categorical

    res = compare_categorical(
        _parse_list(args.counts_a, "--counts-a", int),
        _parse_list(args.counts_b, "--counts-b", int),
        scores=None if args.scores is None else _parse_list(args.scores, "--scores"),
        prior=args.prior,
        higher_is_better=not args.lower_is_better,
        ci=args.ci,
        n_samples=args.samples,
        seed=args.seed,
    )
    pct = f"{res.ci:.0%}"
    print(f"Dirichlet-multinomial comparison ({len(res.scores)} categories, {args.samples} draws)")
    print(f"{'category':>8} {'score':>7} {'n_A':>6} {'n_B':>6} {'share diff':>11} {'P(B share >)':>13}")
    for i, sc in enumerate(res.scores):
        print(f"{i + 1:>8} {sc:>7g} {res.counts_a[i]:>6} {res.counts_b[i]:>6} "
              f"{res.share_diff_means[i]:>+11.4f} {res.prob_share_b_higher[i]:>13.4f}")
    lo, hi = res.mean_score_a_interval
    print(f"{'mean score A':<24}: {res.mean_score_a:.4f}  {pct} CrI [{lo:.4f}, {hi:.4f}]")
    lo, hi = res.mean_score_b_interval
    print(f"{'mean score B':<24}: {res.mean_score_b:.4f}  {pct} CrI [{lo:.4f}, {hi:.4f}]")
    lo, hi = res.mean_diff_interval
    print(f"{'mean diff (B - A)':<24}: {res.mean_diff:+.4f}  {pct} CrI [{lo:+.4f}, {hi:+.4f}]")
    direction = "higher" if res.higher_is_better else "lower"
    print(f"{'P(B better)':<24}: {res.prob_b_better:.4f}  ({direction} is better)")
    lo, hi = res.superiority_interval
    print(f"{'P(superiority)':<24}: {res.superiority:.4f}  {pct} CrI [{lo:.4f}, {hi:.4f}]")
    print(f"{'expected loss choose A':<24}: {res.expected_loss_choose_a:.6g}")
    print(f"{'expected loss choose B':<24}: {res.expected_loss_choose_b:.6g}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handlers = {
        "convert": _cmd_convert,
        "power": _cmd_power,
        "peek": _cmd_peek,
        "rope": _cmd_rope,
        "best": _cmd_best,
        "uplift": _cmd_uplift,
        "shrink": _cmd_shrink,
        "bayes-factor": _cmd_bayes_factor,
        "bf-stop": _cmd_bf_stop,
        "srm": _cmd_srm,
        "pps": _cmd_pps,
        "rates": _cmd_rates,
        "msprt": _cmd_msprt,
        "ratio": _cmd_ratio,
        "bootstrap": _cmd_bootstrap,
        "categorical": _cmd_categorical,
        "meta": _cmd_meta,
        "robust": _cmd_robust,
    }
    try:
        handlers[args.command](args)
    except ValueError as exc:
        parser.exit(2, f"bayes-ab: error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())