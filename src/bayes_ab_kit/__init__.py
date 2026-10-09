"""bayes-ab-kit: Bayesian A/B testing and experiment analysis toolkit."""

__version__ = "0.1.0"

from .bayes_factor import (
    BayesFactorResult,
    SequentialBayesFactorResult,
    bayes_factor_10,
    bayes_factor_decision,
    kass_raftery_label,
    log_bayes_factor_10,
    simulate_bayes_factor_stopping,
)
from .decisions import ComparisonResult, Decision, superiority_decision
from .evaluation import ArpuEvaluation, VariantData, evaluate_variants
from .hierarchical import HierarchicalBetaResult, ShrunkArm, hierarchical_beta_shrinkage
from .multiarms import ArmBestShare, MultiArmBestResult, probability_of_being_best
from .posteriors import BetaBinomialPosterior
from .power import SampleSizePlan, required_sample_size
from .reporting import ConversionReport, render_conversion_report, write_report
from .risk import RiskResult, expected_loss, risk_decision
from .rope import ExpectedLossStopResult, RopeResult, expected_loss_stop, rope_decision
from .sampling import make_rng, summarize_draws
from .sequential import PeekSimulationResult, simulate_null_peeking
from .stopping import StoppingPlanResult, simulate_stopping_plan
from .uplift import (
    ControlUpliftResult,
    UpliftResult,
    VariantVsControl,
    expected_absolute_uplift,
    expected_relative_uplift,
    variant_vs_control,
    variants_vs_control,
)


from .cuped import (
    CupedFit,
    CupedResult,
    cuped_adjust,
    cuped_variance_reduction,
    estimate_cuped_theta,
)

__all__ = [
    "ArmBestShare",
    "ArpuEvaluation",
    "simulate_bayes_factor_stopping",
    "log_bayes_factor_10",
    "kass_raftery_label",
    "bayes_factor_decision",
    "bayes_factor_10",
    "SequentialBayesFactorResult",
    "BayesFactorResult",
    "BetaBinomialPosterior",
    "ComparisonResult",
    "ControlUpliftResult",
    "ConversionReport",
    "Decision",
    "ExpectedLossStopResult",
    "HierarchicalBetaResult",
    "MultiArmBestResult",
    "PeekSimulationResult",
    "RiskResult",
    "RopeResult",
    "SampleSizePlan",
    "ShrunkArm",
    "StoppingPlanResult",
    "UpliftResult",
    "VariantData",
    "VariantVsControl",
    "CupedFit",
    "CupedResult",
    "cuped_adjust",
    "cuped_variance_reduction",
    "estimate_cuped_theta",
    "__version__",
    "evaluate_variants",
    "expected_absolute_uplift",
    "expected_loss",
    "expected_loss_stop",
    "expected_relative_uplift",
    "hierarchical_beta_shrinkage",
    "make_rng",
    "probability_of_being_best",
    "render_conversion_report",
    "required_sample_size",
    "risk_decision",
    "rope_decision",
    "simulate_null_peeking",
    "simulate_stopping_plan",
    "summarize_draws",
    "superiority_decision",
    "variant_vs_control",
    "variants_vs_control",
    "write_report",
]