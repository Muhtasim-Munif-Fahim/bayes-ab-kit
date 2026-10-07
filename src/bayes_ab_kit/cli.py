"""Command-line interface for bayes-ab-kit."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from . import __version__
from .decisions import superiority_decision
from .hierarchical import hierarchical_beta_shrinkage
from .multiarms import probability_of_being_best
from .power import required_sample_size
from .posteriors import BetaBinomialPosterior
from .reporting import ComparisonBlock, ConversionReport, VariantLine, markdown_table, render_conversion_report, write_report
from .bayes_factor import bayes_factor_decision, simulate_bayes_factor_stopping
from .rope import expected_loss_stop, rope_decision
from .sequential import simulate_null_peeking
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
    }
    try:
        handlers[args.command](args)
    except ValueError as exc:
        parser.exit(2, f"bayes-ab: error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())