# bayes-ab-kit

Bayesian A/B testing and experiment analysis toolkit: conversion posteriors,
hierarchical Beta-Binomial shrinkage, control-referenced uplift, revenue
models, decision rules, sequential guardrails, power planning, synthetic
data, Markdown reports, and a command-line interface.

## Install

Requires Python 3.10+.

```bash
pip install -r requirements.txt
# or, from a checkout:
pip install -e .
```

## Command line

```bash
# Compare two conversion arms and print a Markdown report
bayes-ab convert --conversions-a 95 --trials-a 1000 \
                 --conversions-b 130 --trials-b 1000

# Same analysis written to a file
bayes-ab convert --conversions-a 95 --trials-a 1000 \
                 --conversions-b 130 --trials-b 1000 --out report.md

# Visitors per arm needed to move 10.0% -> 12.5% at 80% power
bayes-ab power --baseline 0.10 --expected 0.125

# False-stop risk of checking significance after every 500 visitors
bayes-ab peek --rate 0.10 --per-look 500 --looks 5

# Exact BF10 for independent vs shared conversion rates
bayes-ab bayes-factor --conversions-a 95 --trials-a 1000 \
                    --conversions-b 130 --trials-b 1000

# Sequential early-stopping operating characteristics on |log BF10|
bayes-ab bf-stop --rate-a 0.10 --rate-b 0.12 --per-look 500 --looks 5

# ROPE win / loss / practical equivalence, plus expected-loss stopping
bayes-ab rope --conversions-a 95 --trials-a 1000 \
              --conversions-b 130 --trials-b 1000 \
              --rope 0.01 --loss-threshold 0.0025

# P(each arm is best) for three or more conversion variants
bayes-ab best --arm A:95:1000 --arm B:130:1000 --arm C:110:1000

# P(variant > control), expected uplift, and credible intervals
bayes-ab uplift --control A:95:1000 --arm B:130:1000 --arm C:110:1000

# Shrink noisy conversion arms toward a shared empirical-Bayes Beta prior
bayes-ab shrink --arm noisy:8:20 --arm control:500:5000 --arm winner:2000:5000
```

The console script is provided by `pip install -e .`; alternatively run
`PYTHONPATH=src python -m bayes_ab_kit.cli ...` from a checkout.

## Python API

```python
import numpy as np
from bayes_ab_kit import (
    BetaBinomialPosterior,
    superiority_decision,
    expected_loss,
    expected_loss_stop,
    probability_of_being_best,
    rope_decision,
    evaluate_variants,
    VariantData,
    hierarchical_beta_shrinkage,
    variant_vs_control,
    variants_vs_control,
)

a = BetaBinomialPosterior.from_counts(95, trials=1000)
b = BetaBinomialPosterior.from_counts(130, trials=1000)
print(superiority_decision(a, b).decision)          # ship_b / ship_a / keep_running
print(rope_decision(a, b, rope=0.01).decision)      # ship_* / practical_equivalence / keep_running
print(expected_loss(b, a))                          # E[max(rate_A - rate_B, 0)]
print(expected_loss_stop(a, b, threshold=0.0025).should_stop)
uplift = variant_vs_control(a, b)
print(uplift.prob_beats_control)                    # P(B > A) by quadrature
print(uplift.expected_uplift, uplift.expected_relative_uplift)
print(uplift.uplift_ci_lower, uplift.uplift_ci_upper)
c = BetaBinomialPosterior.from_counts(110, trials=1000)
print(probability_of_being_best({"A": a, "B": b, "C": c}).probabilities)
print(variants_vs_control({"A": a, "B": b, "C": c}, control="A").variants)
shrunk = hierarchical_beta_shrinkage(
    {"noisy": (8, 20), "control": (500, 5000), "winner": (2000, 5000)}
)
print(shrunk.prior_mean, shrunk.arms[0].posterior_mean, shrunk.arms[0].ci_lower)

orders_a = np.random.default_rng(1).lognormal(size=120)
orders_b = np.random.default_rng(2).lognormal(size=150)
result = evaluate_variants(
    VariantData("A", 95, 1000, orders_a),
    VariantData("B", 130, 1000, orders_b),
)
print(result.summary_b.mean, result.decision)       # revenue-per-visitor comparison
```

Every stochastic function takes an explicit `seed` (defaulting to one fixed
value), so reports are reproducible.

## Modules

| Module | Purpose |
| --- | --- |
| `posteriors` | Beta-Binomial conjugate posteriors for conversion rates |
| `hierarchical` | Empirical-Bayes hierarchical Beta shrinkage across many variants |
| `decisions` | Credible-interval superiority rules, P(B beats A) by quadrature or Monte Carlo |
| `multiarms` | Monte Carlo P(each arm is best) for three or more conversion variants |
| `uplift` | P(variant > control), conjugate E[absolute/relative uplift], and CIs |
| `rope` | ROPE win / loss / practical-equivalence decisions and expected-loss stopping |
| `risk` | Expected-loss stopping rule with a tolerance threshold |
| `sampling` | Seeded generators and draw summaries |
| `revenue` | Normal and Lognormal order-value models with analytic expected value |
| `bayes_factor` | Exact Beta-Binomial BF10 vs a shared-rate null, Kass–Raftery labels, and sequential `|log BF|` early stopping |
| `evaluation` | Joint conversion x revenue (ARPU) comparison |
| `sequential` | A/A peeking simulations quantifying false-stop inflation |
| `stopping` | Bayesian stopping plans: detection rate and expected sample size |
| `power` | Analytic sample-size planning plus simulated power curves |
| `synthdata` | Synthetic experiments, including A/A and uplift presets |
| `reporting` | GitHub-flavoured Markdown report rendering |
| `cli` | `bayes-ab` argparse interface |

## Example

`examples/run_demo.py` simulates a two-arm experiment end to end — planning,
conversion and revenue analysis, sequential guardrails — and writes
`examples/demo_report.md`:

```bash
python examples/run_demo.py [output_path]
```

## Tests

```bash
python -m pytest tests -q
```

## Limitations

- Conversion decisions use equal-tailed credible intervals on Monte Carlo
  estimates of the rate difference; results vary slightly across sample
  counts even with a fixed seed.
- ROPE decisions compare that same equal-tailed interval with
  ``[-rope, rope]`` (or an explicit interval) on ``rate_B - rate_A``;
  expected-loss stopping ships the posterior-mean leader only when its
  expected loss is below the chosen threshold.
- Multi-arm P(best) is a seeded Monte Carlo estimate of which arm has the
  highest conversion rate; exact ties (rare for continuous Beta draws)
  are split equally so the shares sum to one. It is a ranking diagnostic,
  not a ROPE or expected-loss stopping rule.
- Hierarchical Beta shrinkage (`hierarchical_beta_shrinkage`) fits one
  shared `Beta(α, β)` by maximising the beta-binomial marginal likelihood,
  then updates every arm with that prior. Posterior means are
  `(1 - w) * raw_rate + w * prior_mean` with prior weight
  `w = (α + β) / (trials + α + β)`, so noisy arms move farther toward the
  grand mean than precise ones. Pass `ci=None` to skip equal-tailed
  intervals. The fitted prior is a plug-in: intervals ignore uncertainty
  in `(α, β)`, and when extra-binomial variation is absent the prior
  strength sits on a numerical cap and pools arms tightly. The same
  bound is used when every arm is all successes or all failures; the
  equal-tailed interval can then sit on 0 or 1 while the posterior mean
  stays slightly inside the unit interval, because that Beta is extremely
  skewed.
  `result.posteriors()` rebuilds `BetaBinomialPosterior` objects from the
  shared prior for ROPE, P(best), and uplift; those helpers still treat
  the arms as independent given the fixed prior.
- Control-referenced uplift (`variant_vs_control`) reports P(variant >
  control) by quadrature, `E[rate_v - rate_c]` from Beta means, and
  `E[(rate_v - rate_c) / rate_c] = E[rate_v] E[1 / rate_c] - 1` when the
  control posterior shape `alpha` is greater than 1. That relative
  expectation is not `(μ_v - μ_c) / μ_c`; it diverges when `alpha <= 1`
  (for example a uniform prior and zero control conversions). Equal-tailed
  intervals for both uplift scales use the same seeded Monte Carlo stream
  as the two-arm difference interval, so they line up with ROPE decisions
  at a matching seed and sample count. `variants_vs_control` scores each
  non-control arm this way and is not a P(best) ranking.
- The lognormal revenue model treats the log-scale dispersion as fixed;
  uncertainty is modelled only over the mean.
- Sequential guardrails rely on Normal approximations for speed and are
  meant for planning intuition, not as a replacement for group-sequential
  designs.

## License

MIT — see [LICENSE](LICENSE).
## CUPED covariate adjustment

`cuped_adjust` implements Deng et al. (2013) Controlled-experiment Using
Pre-Experiment Data. Given an outcome `Y` and a pre-period covariate `X`,
it returns `Y - θ (X - mean(X))` with `θ = Cov(Y, X) / Var(X)`, preserving
`E[Y]` while cutting variance by roughly `Corr(Y, X)²`.

```python
from bayes_ab_kit import cuped_adjust, cuped_variance_reduction

result = cuped_adjust(y_outcome, x_pre_period)
print(result.fit.theta, result.fit.variance_reduction)
print(result.adjusted.mean(), result.var_adjusted)
```

