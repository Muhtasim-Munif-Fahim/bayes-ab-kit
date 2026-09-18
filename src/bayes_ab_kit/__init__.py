"""bayes-ab-kit: Bayesian A/B testing and experiment analysis toolkit."""

__version__ = "0.1.0"

from .decisions import ComparisonResult, Decision, superiority_decision
from .evaluation import ArpuEvaluation, VariantData, evaluate_variants
from .multiarms import ArmBestShare, MultiArmBestResult, probability_of_being_best
from .posteriors import BetaBinomialPosterior
from .power import SampleSizePlan, required_sample_size
from .reporting import ConversionReport, render_conversion_report, write_report
from .risk import RiskResult, expected_loss, risk_decision
from .rope import ExpectedLossStopResult, RopeResult, expected_loss_stop, rope_decision
from .sampling import make_rng, summarize_draws
from .sequential import PeekSimulationResult, simulate_null_peeking
from .stopping import StoppingPlanResult, simulate_stopping_plan

__all__ = [
    "ArmBestShare",
    "ArpuEvaluation",
    "BetaBinomialPosterior",
    "ComparisonResult",
    "ConversionReport",
    "Decision",
    "ExpectedLossStopResult",
    "MultiArmBestResult",
    "PeekSimulationResult",
    "RiskResult",
    "RopeResult",
    "SampleSizePlan",
    "StoppingPlanResult",
    "VariantData",
    "__version__",
    "evaluate_variants",
    "expected_loss",
    "expected_loss_stop",
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
    "write_report",
]