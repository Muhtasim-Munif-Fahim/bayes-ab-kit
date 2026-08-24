"""Command-line interface for bayes-ab-kit."""

from __future__ import annotations

import argparse
from pathlib import Path

from . import __version__
from .decisions import superiority_decision
from .power import required_sample_size
from .posteriors import BetaBinomialPosterior
from .reporting import ComparisonBlock, ConversionReport, VariantLine, render_conversion_report, write_report
from .sequential import simulate_null_peeking


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


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handlers = {"convert": _cmd_convert, "power": _cmd_power, "peek": _cmd_peek}
    try:
        handlers[args.command](args)
    except ValueError as exc:
        parser.exit(2, f"bayes-ab: error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())