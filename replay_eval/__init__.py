"""Replay-based evaluation for early-exit methods.

Implements the protocol described in ``agent-v2/proposal.md``. Each method is
a triple ``(S, pi, G)``:

* ``S`` (checkpoint) decides whether the current prefix should be checked.
* ``pi`` (verifier) estimates whether the current state is sufficient.
* ``G`` (stop rule) turns the verifier output into a stop decision.

Compose a new method by writing any of the three pieces and passing them to
``EarlyExitMethod``. The replay agent runs a set of methods against one shared
reasoning trajectory.
"""

from replay_eval.method import (
    CheckpointFn,
    EarlyExitMethod,
    MethodResult,
    StopRule,
    Verifier,
    VerifierContext,
)
from replay_eval.checkpoints import (
    AlwaysCheckpoint,
    AnyCheckpoint,
    EveryNTokens,
    IntervalSignalCheckpoint,
    SemanticStepCheckpoint,
    StopMarkerCheckpoint,
)
from replay_eval.stop_rules import (
    AlwaysStop,
    CodeStopRule,
    PUMAStopRule,
    ThresholdGE,
    ThresholdGT,
    ThresholdLE,
)
from replay_eval.verifiers import (
    AnswerCoverageVerifier,
    CodeStopVerifier,
    DynasorVerifier,
    EATVerifier,
    EntrocutVerifier,
    KeywordCountVerifier,
    LLMSufficiencyVerifier,
    PUMAVerifier,
    ProbeLogprobVerifier,
    StatisticalUncertaintyVerifier,
)
from replay_eval.methods import (
    build_answer_coverage_method,
    build_codestop_method,
    build_deer_method,
    build_dtsr_method,
    build_dynasor_method,
    build_eat_method,
    build_entrocut_method,
    build_puma_method,
    build_simple_method,
    build_statistical_method,
)
from replay_eval.replay import (
    ReplayEvaluatorAgent,
    ReplayEvaluatorAgentFullGeneration,
)
__all__ = [
    "CheckpointFn",
    "EarlyExitMethod",
    "MethodResult",
    "StopRule",
    "Verifier",
    "VerifierContext",
    "AlwaysCheckpoint",
    "AnyCheckpoint",
    "EveryNTokens",
    "IntervalSignalCheckpoint",
    "SemanticStepCheckpoint",
    "StopMarkerCheckpoint",
    "AlwaysStop",
    "CodeStopRule",
    "PUMAStopRule",
    "ThresholdGE",
    "ThresholdGT",
    "ThresholdLE",
    "AnswerCoverageVerifier",
    "CodeStopVerifier",
    "DynasorVerifier",
    "EATVerifier",
    "EntrocutVerifier",
    "KeywordCountVerifier",
    "LLMSufficiencyVerifier",
    "PUMAVerifier",
    "ProbeLogprobVerifier",
    "StatisticalUncertaintyVerifier",
    "build_answer_coverage_method",
    "build_codestop_method",
    "build_deer_method",
    "build_dtsr_method",
    "build_dynasor_method",
    "build_eat_method",
    "build_entrocut_method",
    "build_puma_method",
    "build_simple_method",
    "build_statistical_method",
    "ReplayEvaluatorAgent",
    "ReplayEvaluatorAgentFullGeneration",
    "HiddenStateLogProbs",
    "TokenLogProbOutput",
]


def __getattr__(name):
    if name in {"HiddenStateLogProbs", "TokenLogProbOutput"}:
        from replay_eval.hidden_logprobs import (
            HiddenStateLogProbs,
            TokenLogProbOutput,
        )

        return {
            "HiddenStateLogProbs": HiddenStateLogProbs,
            "TokenLogProbOutput": TokenLogProbOutput,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
