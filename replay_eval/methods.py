"""Convenience factories that compose the built-in methods.

Adding a new method is a one-liner: pick or write (S, pi, G) and call
``EarlyExitMethod("my_method", S, pi, G)``.
"""

from __future__ import annotations

from typing import Any
from typing import Callable
from typing import Sequence

from replay_eval.checkpoints import (
    AlwaysCheckpoint,
    EveryNTokens,
    IntervalSignalCheckpoint,
    SemanticStepCheckpoint,
    StopMarkerCheckpoint,
)
from replay_eval.method import EarlyExitMethod
from replay_eval.stop_rules import CodeStopRule
from replay_eval.stop_rules import PUMAStopRule
from replay_eval.stop_rules import ThresholdGE
from replay_eval.stop_rules import ThresholdGT
from replay_eval.stop_rules import ThresholdLE
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


def build_deer_method(
    name: str = "deer",
    threshold: float = 0.95,
    probe_tokens: int = 20,
    probe_logprobs: int = 1,
    markers: Sequence[str] | None = None,
    reflection_signals: Sequence[str] | None = None,
) -> EarlyExitMethod:
    """DEER: check markers and score a shared temporary-answer probe."""
    checkpoint_markers = markers if markers is not None else reflection_signals
    return EarlyExitMethod(
        name=name,
        checkpoint=StopMarkerCheckpoint(checkpoint_markers),
        verifier=ProbeLogprobVerifier(
            probe_tokens=probe_tokens,
            probe_logprobs=probe_logprobs,
        ),
        stop_rule=ThresholdGE(threshold, scale_for_code=True),
    )


def build_entrocut_method(
    name: str = "entrocut",
    threshold: float = 0.2,
    tail_mass_threshold: float = 0.05,
    probe_tokens: int = 20,
    probe_logprobs: int = 20,
    markers: Sequence[str] | None = None,
    reflection_signals: Sequence[str] | None = None,
) -> EarlyExitMethod:
    """ENTROCUT: DEER checkpoint/probe path with top-logprob entropy scoring."""
    checkpoint_markers = markers if markers is not None else reflection_signals
    return EarlyExitMethod(
        name=name,
        checkpoint=StopMarkerCheckpoint(checkpoint_markers),
        verifier=EntrocutVerifier(
            threshold=threshold,
            tail_mass_threshold=tail_mass_threshold,
            probe_tokens=probe_tokens,
            probe_logprobs=probe_logprobs,
        ),
        stop_rule=ThresholdLE(threshold, scale_for_code=True),
    )


def build_dtsr_method(
    name: str = "dtsr",
    threshold: float = 80.0,
    min_interval: int = 32,
    reflection_signals: Sequence[str] | None = None,
    prompt_template: str | None = None,
    system_prompt: str | None = None,
    llm: str = "same",
    llm_name: str | None = None,
    chat_template: dict[str, Any] | None = None,
    completion_kwargs: dict[str, Any] | None = None,
    llm_kwargs: dict[str, Any] | None = None,
    close_thinking: bool = True,
    probe_tokens: int = 20,
) -> EarlyExitMethod:
    """DTSR: reflection-signal + min-interval checkpoint, LLM sufficiency."""
    return EarlyExitMethod(
        name=name,
        checkpoint=IntervalSignalCheckpoint(
            reflection_signals, min_interval=min_interval
        ),
        verifier=LLMSufficiencyVerifier(
            prompt_template=prompt_template,
            system_prompt=system_prompt,
            llm=llm,
            llm_name=llm_name,
            chat_template=chat_template,
            completion_kwargs=completion_kwargs,
            llm_kwargs=llm_kwargs,
            close_thinking=close_thinking,
            probe_tokens=probe_tokens,
        ),
        stop_rule=ThresholdGE(threshold, scale_for_code=True),
    )


def build_flash_think_method(
    name: str = "dtsr",
    threshold: float = 80.0,
    min_interval: int = 32,
    reflection_signals: Sequence[str] | None = None,
    prompt_template: str | None = None,
    system_prompt: str | None = None,
    llm: str = "same",
    llm_name: str | None = None,
    chat_template: dict[str, Any] | None = None,
    completion_kwargs: dict[str, Any] | None = None,
    llm_kwargs: dict[str, Any] | None = None,
    close_thinking: bool = True,
    probe_tokens: int = 20,
) -> EarlyExitMethod:
    """DTSR: reflection-signal + min-interval checkpoint, LLM sufficiency."""
    return EarlyExitMethod(
        name=name,
        checkpoint=IntervalSignalCheckpoint(
            reflection_signals, min_interval=min_interval
        ),
        verifier=LLMSufficiencyVerifier(
            prompt_template=prompt_template,
            system_prompt=system_prompt,
            llm=llm,
            llm_name=llm_name,
            chat_template=chat_template,
            completion_kwargs=completion_kwargs,
            llm_kwargs=llm_kwargs,
            close_thinking=close_thinking,
            probe_tokens=probe_tokens,
        ),
        stop_rule=ThresholdGE(threshold, scale_for_code=True),
    )


def build_simple_method(
    name: str = "simple_keyword",
    keywords: Sequence[str] = ("wait", "but"),
    max_count: int = 10,
    check_every: int = 32,
) -> EarlyExitMethod:
    """SimpleAgent-style: stop when combined keyword count exceeds ``max_count``."""
    return EarlyExitMethod(
        name=name,
        checkpoint=EveryNTokens(check_every),
        verifier=KeywordCountVerifier(keywords),
        stop_rule=ThresholdGE(max_count),
    )


def build_statistical_method(
    name: str = "statistical_early_stop",
    B: int = 100,
    alpha: float = 0.05,
    uncertainty_keywords: Sequence[str | Sequence[str]] | None = None,
    delta: int = 5,
) -> EarlyExitMethod:
    """Statistical uncertainty density: stop when density > alpha every B tokens."""
    return EarlyExitMethod(
        name=name,
        checkpoint=EveryNTokens(B),
        verifier=StatisticalUncertaintyVerifier(
            uncertainty_keywords=uncertainty_keywords,
            delta=delta,
        ),
        stop_rule=ThresholdGT(alpha),
    )


def build_codestop_method(
    name: str = "codestop",
    conf_threshold: float = 0.95,
    degen_threshold: float = 1.0,
    stop_word: str = "Wait",
    markers: Sequence[str] | None = None,
    reflection_signals: Sequence[str] | None = None,
    probe_tokens: int = 20,
    probe_logprobs: int = 1,
    use_log: bool = True,
    ewt: bool = True,
    ramp_steps: int | None = None,
    ramp_min: float | None = None,
    trial_answer_prefix: str = "\n**Final Answer**\n\nThe final answer is \\boxed",
) -> EarlyExitMethod:
    """CodeStop: probe answer confidence at stop-word checkpoints."""
    checkpoint_markers = markers if markers is not None else reflection_signals
    if checkpoint_markers is None:
        checkpoint_markers = (stop_word,)
    return EarlyExitMethod(
        name=name,
        checkpoint=StopMarkerCheckpoint(checkpoint_markers),
        verifier=CodeStopVerifier(
            conf_threshold=conf_threshold,
            degen_threshold=degen_threshold,
            probe_tokens=probe_tokens,
            probe_logprobs=probe_logprobs,
            use_log=use_log,
            ewt=ewt,
            ramp_steps=ramp_steps,
            ramp_min=ramp_min,
            trial_answer_prefix=trial_answer_prefix,
        ),
        stop_rule=CodeStopRule(),
    )


def build_answer_coverage_method(
    name: str = "answer-coverage",
    check_every: int = 128,
    window_size: int = 10,
    llm: str = "same",
    llm_name: str | None = None,
    chat_template: dict[str, Any] | None = None,
    completion_kwargs: dict[str, Any] | None = None,
    llm_kwargs: dict[str, Any] | None = None,
    close_thinking: bool = True,
    collect_answer: str | Callable[[str], Any] | None = None,
    collect_coding_answer: str | Callable[[str, Any], Any] | None = None,
) -> EarlyExitMethod:
    """Answer coverage: stop after ``window_size`` matching probes."""
    return EarlyExitMethod(
        name=name,
        checkpoint=EveryNTokens(check_every),
        verifier=AnswerCoverageVerifier(
            window_size=window_size,
            llm=llm,
            llm_name=llm_name,
            chat_template=chat_template,
            completion_kwargs=completion_kwargs,
            llm_kwargs=llm_kwargs,
            close_thinking=close_thinking,
            collect_answer=collect_answer,
            collect_coding_answer=collect_coding_answer,
        ),
        stop_rule=ThresholdGE(1.0),
    )


def build_dynasor_method(
    name: str = "dynasor",
    check_every: int = 128,
    window_size: int = 5,
    consistency_threshold: float = 0.8,
    uncertainty_signals: Sequence[str] = ("wait", "hmm"),
    llm: str = "same",
    llm_name: str | None = None,
    chat_template: dict[str, Any] | None = None,
    completion_kwargs: dict[str, Any] | None = None,
    llm_kwargs: dict[str, Any] | None = None,
    close_thinking: bool = True,
    collect_answer: str | Callable[[str], Any] | None = None,
    collect_coding_answer: str | Callable[[str, Any], Any] | None = None,
) -> EarlyExitMethod:
    """Dynasor: stop when the current answer agrees with its probe window."""
    return EarlyExitMethod(
        name=name,
        checkpoint=EveryNTokens(check_every),
        verifier=DynasorVerifier(
            window_size=window_size,
            consistency_threshold=consistency_threshold,
            uncertainty_signals=uncertainty_signals,
            llm=llm,
            llm_name=llm_name,
            chat_template=chat_template,
            completion_kwargs=completion_kwargs,
            llm_kwargs=llm_kwargs,
            close_thinking=close_thinking,
            collect_answer=collect_answer,
            collect_coding_answer=collect_coding_answer,
        ),
        stop_rule=ThresholdGE(consistency_threshold),
    )


def build_puma_method(
    name: str = "puma",
    similarity_threshold: float = 0.35,
    confidence_threshold: float = 0.98,
    epsilon: float = 0.03,
    consecutive: int = 2,
    probe_tokens: int = 30,
    probe_logprobs: int = 1,
    window_size: int = 1,
    consecutive_redundancy_stop: int = 50,
    min_loop_breaker_step: int = 50,
    forced_stop_min_confidence: float = 0.8,
    always_check_first_n: int = 1,
    trigger_mode: str = "previous",
    loop_breaker_similarity_threshold: float | None = None,
    task_type: str | None = None,
    embedding_model: str | None = "ZhishanQ/qwen3-embedding-redundancy-detector-0.6B",
    embedding_device: str | None = None,
    embedding_batch_size: int = 32,
    embedding_max_tokens: int | None = 2048,
    code_similarity_threshold: float = 0.8,
    collect_answer: str | Callable[[str], Any] | None = None,
    collect_coding_answer: str | Callable[[str, Any], Any] | None = None,
    min_step_chars: int = 200,
    max_step_chars: int = 1000,
) -> EarlyExitMethod:
    """PUMA-v2: check semantic reasoning steps, verify with trial-answer stability."""
    return EarlyExitMethod(
        name=name,
        checkpoint=SemanticStepCheckpoint(
            min_step_chars=min_step_chars,
            max_step_chars=max_step_chars,
        ),
        verifier=PUMAVerifier(
            similarity_threshold=similarity_threshold,
            confidence_threshold=confidence_threshold,
            epsilon=epsilon,
            consecutive=consecutive,
            probe_tokens=probe_tokens,
            probe_logprobs=probe_logprobs,
            window_size=window_size,
            consecutive_redundancy_stop=consecutive_redundancy_stop,
            min_loop_breaker_step=min_loop_breaker_step,
            forced_stop_min_confidence=forced_stop_min_confidence,
            always_check_first_n=always_check_first_n,
            trigger_mode=trigger_mode,
            loop_breaker_similarity_threshold=loop_breaker_similarity_threshold,
            task_type=task_type,
            embedding_model=embedding_model,
            embedding_device=embedding_device,
            embedding_batch_size=embedding_batch_size,
            embedding_max_tokens=embedding_max_tokens,
            code_similarity_threshold=code_similarity_threshold,
            collect_answer=collect_answer,
            collect_coding_answer=collect_coding_answer,
        ),
        stop_rule=PUMAStopRule(),
    )


def build_eat_method(
    name: str = "eat",
    threshold: float = 2.0,
    warmup_steps: int = 50,
    entropy_decoding_steps: int = 20,
    entropy_rollouts: int = 10,
    timescale: float = 10.0,
    probe_logprobs: int = 20,
    min_step_chars: int = 200,
    max_step_chars: int = 120,
) -> EarlyExitMethod:
    """EAT: stop when EMA variance of forced-answer entropy stabilizes."""
    return EarlyExitMethod(
        name=name,
        checkpoint=SemanticStepCheckpoint(
            min_step_chars=min_step_chars,
            max_step_chars=max_step_chars,
        ),
        verifier=EATVerifier(
            threshold=threshold,
            warmup_steps=warmup_steps,
            entropy_decoding_steps=entropy_decoding_steps,
            entropy_rollouts=entropy_rollouts,
            timescale=timescale,
            probe_logprobs=probe_logprobs,
        ),
        stop_rule=ThresholdLE(threshold, scale_for_code=True),
    )


def build_baseline_no_thinking(
    name: str = "no_thinking",
) -> EarlyExitMethod:
    """Baseline: answer directly without enabling model reasoning."""
    return EarlyExitMethod(
        name=name,
        checkpoint=AlwaysCheckpoint(),
        verifier=None,
        stop_rule=None,
        no_thinking=True,
    )


def build_baseline_500_method(
    name: str = "baseline_500",
) -> EarlyExitMethod:
    """Baseline: stop every ``check_every`` tokens, no verifier or stop rule."""
    return EarlyExitMethod(
        name=name,
        checkpoint=EveryNTokens(500),
        verifier=None,
        stop_rule=None,
    )


def build_baseline_1000_method(
    name: str = "baseline_1K",
) -> EarlyExitMethod:
    """Baseline: stop every ``check_every`` tokens, no verifier or stop rule."""
    return EarlyExitMethod(
        name=name,
        checkpoint=EveryNTokens(1024),
        verifier=None,
        stop_rule=None,
    )


def build_baseline_1500_method(
    name: str = "baseline_1.5K",
) -> EarlyExitMethod:
    """Baseline: stop every ``check_every`` tokens, no verifier or stop rule."""
    return EarlyExitMethod(
        name=name,
        checkpoint=EveryNTokens(1536),
        verifier=None,
        stop_rule=None,
    )


def build_baseline_2000_method(
    name: str = "baseline_2K",
) -> EarlyExitMethod:
    """Baseline: stop every ``check_every`` tokens, no verifier or stop rule."""
    return EarlyExitMethod(
        name=name,
        checkpoint=EveryNTokens(2048),
        verifier=None,
        stop_rule=None,
    )


def build_baseline_3000_method(
    name: str = "baseline_3K",
) -> EarlyExitMethod:
    """Baseline: stop every ``check_every`` tokens, no verifier or stop rule."""
    return EarlyExitMethod(
        name=name,
        checkpoint=EveryNTokens(3072),
        verifier=None,
        stop_rule=None,
    )


def build_baseline_4000_method(
    name: str = "baseline_4K",
) -> EarlyExitMethod:
    """Baseline: stop every ``check_every`` tokens, no verifier or stop rule."""
    return EarlyExitMethod(
        name=name,
        checkpoint=EveryNTokens(4096),
        verifier=None,
        stop_rule=None,
    )


def build_baseline_6000_method(
    name: str = "baseline_6K",
) -> EarlyExitMethod:
    """Baseline: stop every ``check_every`` tokens, no verifier or stop rule."""
    return EarlyExitMethod(
        name=name,
        checkpoint=EveryNTokens(6144),
        verifier=None,
        stop_rule=None,
    )


def build_baseline_8000_method(
    name: str = "baseline_8K",
) -> EarlyExitMethod:
    """Baseline: stop every ``check_every`` tokens, no verifier or stop rule."""
    return EarlyExitMethod(
        name=name,
        checkpoint=EveryNTokens(8192),
        verifier=None,
        stop_rule=None,
    )


def build_baseline_12000_method(
    name: str = "baseline_12K",
) -> EarlyExitMethod:
    """Baseline: stop every ``check_every`` tokens, no verifier or stop rule."""
    return EarlyExitMethod(
        name=name,
        checkpoint=EveryNTokens(12288),
        verifier=None,
        stop_rule=None,
    )


def build_baseline_16000_method(
    name: str = "baseline_16K",
) -> EarlyExitMethod:
    """Baseline: stop every ``check_every`` tokens, no verifier or stop rule."""
    return EarlyExitMethod(
        name=name,
        checkpoint=EveryNTokens(16384),
        verifier=None,
        stop_rule=None,
    )


def build_baseline_24000_method(
    name: str = "baseline_24K",
) -> EarlyExitMethod:
    """Baseline: stop every ``check_every`` tokens, no verifier or stop rule."""
    return EarlyExitMethod(
        name=name,
        checkpoint=EveryNTokens(24576),
        verifier=None,
        stop_rule=None,
    )
