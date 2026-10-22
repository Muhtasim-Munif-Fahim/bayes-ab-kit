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

# Sample ratio mismatch: is a 50.8/49.2 split on a 50/50 test a bug?
bayes-ab srm --arm control:101600 --arm treatment:98400 --expected 0.5,0.5

# Shrink noisy conversion arms toward a shared empirical-Bayes Beta prior
bayes-ab shrink --arm noisy:8:20 --arm control:500:5000 --arm winner:2000:5000

# Will collecting 2k more visitors per arm likely declare B > A?
bayes-ab pps --conversions-a 95 --trials-a 1000 \
             --conversions-b 130 --trials-b 1000 --remaining 2000

# Count/rate metrics (events per exposure): orders per user-day, crashes per session
bayes-ab rates --events-a 40 --exposure-a 1000 --events-b 70 --exposure-b 1000 \
               --loss-threshold 0.001
bayes-ab rates --events-a 12 --exposure-a 5000 --events-b 21 --exposure-b 5000 \
               --lower-is-better   # fewer errors is better

# Always-valid sequential test: peek after every batch without inflating alpha
bayes-ab msprt --look 48:500:62:500 --look 98:1000:131:1000 --look 150:1500:205:1500
bayes-ab msprt --simulate --rate 0.10 --per-look 500 --looks 20

# Ratio metric (clicks/views, revenue/session) per randomized user, delta-method SE
bayes-ab ratio --data users.csv --control A   # columns: arm,numerator,denominator
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
    predictive_probability_of_success,
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
| `srm` | Sample ratio mismatch checks: chi-square, Dirichlet-multinomial BF10, and an anytime-valid sequential monitor |
| `evaluation` | Joint conversion x revenue (ARPU) comparison |
| `ratio` | Ratio metrics (CTR, revenue per session) with delta-method variance, design effect vs naive pooling, and A/B comparison |
| `bootstrap` | Bayesian bootstrap (Dirichlet-weighted) posteriors and A/B comparison for medians, quantiles, trimmed means, std, ratios or any weighted statistic |
| `msprt` | Mixture SPRT: always-valid p-values, confidence sequences, and A/A peeking simulations |
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

## Sample ratio mismatch (SRM)

Check traffic allocation before you read any metric. A split that differs
from the plan usually means an assignment or logging bug.
`srm_check(counts, expected_shares)` runs:

- a Pearson chi-square goodness-of-fit test (`p_value`, `mismatch` when
  `p < alpha`, with `alpha = 0.001` by default), and
- a Dirichlet-multinomial Bayes factor, `H1: p ~ Dirichlet(a)` against
  `H0: p = planned shares`. It is exact and in closed form, and the result
  also reports `posterior_mean_shares`.

Under `H0`, BF10 evaluated on the growing counts is a test martingale.
`sequential_srm_monitor(cumulative_counts)` uses this to raise an alarm the
first time `BF10 >= 1/alpha`. Because of Ville's inequality, the false-alarm
rate stays at or below `alpha` however often you look (Lindon & Malek, 2020).

```python
from bayes_ab_kit import srm_check, sequential_srm_monitor

res = srm_check([101_600, 98_400], [0.5, 0.5], arms=["control", "treatment"])
print(res.p_value, res.log_bf10, res.mismatch, res.bayesian_mismatch)

daily = [[5_100, 4_900], [10_350, 9_650], [15_800, 14_200]]   # cumulative
mon = sequential_srm_monitor(daily, alpha=0.001)
print(mon.stopped_at, mon.anytime_p_values)
```

## Always-valid sequential testing (mSPRT)

`sequential.simulate_null_peeking` shows how far a fixed-horizon p-value's
false-positive rate climbs when you check it after every batch. The mixture
sequential probability ratio test (Robbins 1970; Johari, Koomen, Pekelis &
Walsh, *Always Valid Inference*, 2022) is the fix behind most commercial
"peek-any-time" dashboards. With an approximately normal lift estimate
`delta_hat ~ N(theta, V)` and a normal mixing prior `theta ~ N(theta0, tau^2)`,
the mixture likelihood ratio has a closed form and is a martingale under the
null. Ville's inequality then bounds the chance it *ever* reaches `1/alpha` by
`alpha`.

- `msprt_monitor(deltas, variances, tau=...)` is the generic version for any
  metric with a normal estimate (means, CUPED-adjusted means, ...).
- `msprt_proportions(cumulative)` takes running
  `(conversions_a, trials_a, conversions_b, trials_b)` totals.
- Each look reports the always-valid p-value
  `min(1, 1 / max_s Lambda_s)` and a confidence-sequence interval. The
  intervals are intersected over looks and cover the true lift at every look
  simultaneously with probability `>= 1 - alpha`.
- `simulate_msprt_false_positives` compares mSPRT with naive repeated
  z-tests on A/A data.

```python
from bayes_ab_kit import msprt_proportions, simulate_msprt_false_positives

looks = [(48, 500, 62, 500), (98, 1000, 131, 1000), (150, 1500, 205, 1500), (199, 2000, 281, 2000)]
res = msprt_proportions(looks, alpha=0.05, tau=0.02)
print(res.stopped_at, res.p_value, res.confidence_interval)

sim = simulate_msprt_false_positives(rate=0.10, per_look=500, looks=20)
print(sim.naive_false_positive_rate, sim.msprt_false_positive_rate)  # about 0.25 vs about 0.02
```

`tau` is the size of lift the test is tuned to find quickly, not an
assumption validity depends on: any `tau > 0` controls `alpha`. The price of
peeking freely is a wider interval than a single fixed-horizon look would give.

## Count and rate metrics (Gamma-Poisson)

`compare_rates` handles metrics that are event counts over an exposure
(purchases per user-day, crashes per 1,000 sessions). Each arm gets a
conjugate `Gamma(shape0 + events, rate0 + exposure)` posterior
(`GammaPoissonPosterior`, default Jeffreys-like `shape0=0.5` and a vague
`rate0=1e-6`).

* `P(rate_B > rate_A)` is exact: `G_b / (G_a + G_b)` is Beta-distributed, so
  the probability is a regularized incomplete beta (`prob_rate_b_exceeds_a`).
* The rate ratio `rate_B / rate_A` is a scaled beta-prime variable; its
  median and credible interval come from Beta quantiles (`rate_ratio_interval`).
* Expected loss of choosing each arm (in rate units) is Monte Carlo from
  seeded draws; set `loss_threshold` to get a `winner` once one arm's loss is
  small enough.
* `higher_is_better=False` (CLI `--lower-is-better`) flips the orientation for
  "bad" counts such as errors or churn events.
* `GammaPoissonPosterior.predictive_events(exposure)` returns the negative
  binomial posterior predictive for future counts.

```python
from bayes_ab_kit import compare_rates

res = compare_rates(40, 1000.0, 70, 1000.0, loss_threshold=1e-3)
res.prob_b_better, res.rate_ratio_interval, res.winner
```

## Ratio metrics (delta method)

Click-through rate (clicks / page views) and revenue per session are
**ratio metrics**: the denominator is not the randomization unit. If you
pool every page view as an independent Bernoulli trial, you ignore the
correlation between views from the same user. That standard error is too
small, so A/A tests come out "significant" far more often than `alpha`.
`ratio_summary` uses per-user numerators and denominators and the
delta-method variance (Deng, Knoblich & Lu, 2018):

    Var(r) ~ (1/n) [ s_xx/mu_y^2 - 2 mu_x s_xy/mu_y^3 + mu_x^2 s_yy/mu_y^4 ]

* `ratio_summary(x, y)` takes raw per-unit arrays. `ratio_summary_from_moments(n, Σx, Σy, Σx², Σy², Σxy)` takes aggregates from SQL.
* `design_effect` reports how many times the naive pooled variance understates the truth.
* `compare_ratio_metrics(a, b)` reports:
  * the difference with its CI and p-value;
  * the relative lift (delta method on `log r`) with its CI;
  * the naive p-value, for contrast;
  * the flat-prior posterior `P(B better)` and closed-form expected losses.

```python
from bayes_ab_kit import compare_ratio_metrics, ratio_summary

a = ratio_summary(clicks_a, views_a)   # one entry per user
b = ratio_summary(clicks_b, views_b)
res = compare_ratio_metrics(a, b)
a.design_effect, res.p_value, res.naive_p_value, res.relative_lift_interval
```

## Bayesian bootstrap (medians, quantiles, any metric)

Conjugate models cover conversions, counts and revenue. Decision metrics such as **median order value**, **p90 latency** or a **trimmed mean** have no convenient likelihood. The Bayesian bootstrap (Rubin, 1981) puts a flat Dirichlet prior on the probabilities of the observed values. Each posterior draw reweights the sample with `w ~ Dirichlet(1, …, 1)` and evaluates the weighted statistic. No parametric assumption is needed.

* `bayesian_bootstrap(x, statistic)` supports:
  * `"mean"` (exact: posterior variance `Σ(x − x̄)² / (n(n+1))`);
  * `"median"`, `"quantile"` (with `q=`) and `"trimmed_mean"` (with `trim=`), which use the weighted inverse-CDF definitions; uniform weights reproduce `numpy.quantile(method="inverted_cdf")` and `scipy.stats.trim_mean`;
  * `"std"`;
  * any callable `f(values, weights)`.
  
  Draws are vectorised and generated in chunks.
* `bayesian_bootstrap_ratio(x, y)` gives the posterior of `Σx / Σy` with the same unit weights on numerator and denominator, for example clicks / views per user.
* `compare_bayesian_bootstrap(a, b, statistic)` reports the `B − A` difference, the relative lift with credible intervals, `P(B better)`, and the expected loss of each choice. Use `higher_is_better=False` for latency-type metrics.

```python
from bayes_ab_kit import bayesian_bootstrap, compare_bayesian_bootstrap

post = bayesian_bootstrap(order_values, "median", n_draws=4000, seed=1)
post.point_estimate, post.interval
res = compare_bayesian_bootstrap(latency_a, latency_b, "quantile", q=0.9, higher_is_better=False, seed=7)
res.prob_b_better, res.difference_interval, res.expected_loss_choose_b
```

```bash
bayes-ab bootstrap --data values.csv --statistic quantile --q 0.9 --lower-is-better   # CSV: arm,value
```
