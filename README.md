# bayes-ab-kit

Bayesian A/B testing and experiment analysis toolkit: conversion posteriors,
revenue models, decision rules, sequential guardrails, power planning,
synthetic data, Markdown reports, and a command-line interface.

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
    evaluate_variants,
    VariantData,
)

a = BetaBinomialPosterior.from_counts(95, trials=1000)
b = BetaBinomialPosterior.from_counts(130, trials=1000)
print(superiority_decision(a, b).decision)          # ship_b / ship_a / keep_running
print(expected_loss(b, a))                          # E[max(rate_A - rate_B, 0)]

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
| `decisions` | Credible-interval superiority rules, P(B beats A) by quadrature or Monte Carlo |
| `risk` | Expected-loss stopping rule with a tolerance threshold |
| `sampling` | Seeded generators and draw summaries |
| `revenue` | Normal and Lognormal order-value models with analytic expected value |
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
- The lognormal revenue model treats the log-scale dispersion as fixed;
  uncertainty is modelled only over the mean.
- Sequential guardrails rely on Normal approximations for speed and are
  meant for planning intuition, not as a replacement for group-sequential
  designs.

## License

MIT — see [LICENSE](LICENSE).