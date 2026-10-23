"""Verifier functions (pi_i)."""

from __future__ import annotations

import logging
import math
import re
from importlib import import_module
from typing import Any, Callable, Sequence

from methods.chat_template import ChatTemplateController
from methods.response import Response
from replay_eval.method import Verifier, VerifierContext
from replay_eval.utils.codestop import (
    compute_degeneration_score,
    compute_ramping_threshold,
)
from replay_eval.utils.deer import best_token_logprobs
from replay_eval.utils.deer import mean_token_probability
from replay_eval.utils.deer import residual_bucket_entropy
from replay_eval.utils.deer import slice_temp_answer_probe
from replay_eval.utils.puma import (
    DEFAULT_PUMA_EMBEDDING_MAX_TOKENS,
    DEFAULT_PUMA_EMBEDDING_MODEL,
    best_trial_confidence,
    extract_trial_answer,
    get_confident_ending,
    get_puma_similarity_backend,
    loop_breaker_fires,
    should_exit_verified,
)
from replay_eval.utils.steps import separate_step_spans
from replay_eval.utils.statistical import (
    normalize_uncertainty_keywords,
    uncertainty_density_score,
)
from llm import get_llm_wrapper


logger = logging.getLogger(__name__)


def _resolve_collector(
    collector: str | Callable[..., Any] | None,
    *,
    name: str,
) -> Callable[..., Any] | None:
    if collector is None:
        return None
    if callable(collector):
        return collector
    module_name, object_name = collector.rsplit(".", 1)
    resolved = getattr(import_module(module_name), object_name)
    if not callable(resolved):
        raise TypeError(f"{name} must resolve to a callable: {collector}")
    return resolved


class KeywordCountVerifier(Verifier):
    """Count how many times any keyword appears in the reasoning prefix.

    Mirrors the heuristic in ``methods/simple.py``. Combine with ``ThresholdGE``
    to reproduce SimpleAgent's behavior.
    """

    def __init__(self, keywords: Sequence[str], case_sensitive: bool = False):
        self.keywords = tuple(keywords)
        self.case_sensitive = case_sensitive

    def __call__(self, ctx: VerifierContext) -> float:
        text = ctx.prefix.content
        if not self.case_sensitive:
            text = text.lower()
            kws = [k.lower() for k in self.keywords]
        else:
            kws = list(self.keywords)
        return float(sum(text.count(k) for k in kws))


class StatisticalUncertaintyVerifier(Verifier):
    """Uncertainty keyword density over the current reasoning prefix."""

    def __init__(
        self,
        uncertainty_keywords: Sequence[str | Sequence[str]] | None = None,
        delta: int = 5,
    ):
        self.uncertainty_keywords = normalize_uncertainty_keywords(uncertainty_keywords)
        self.delta = int(delta)

    def __call__(self, ctx: VerifierContext) -> float:
        score = uncertainty_density_score(
            ctx.prefix.content,
            self.uncertainty_keywords,
            length=ctx.position,
            delta=self.delta,
        )
        ctx.logs.setdefault("statistical_uncertainty_scores", []).append(
            {
                "position": ctx.position,
                "density": score.density,
                # "count": score.count,
                # "arrivals": list(score.arrivals),
            }
        )
        return score.density


class ProbeLogprobVerifier(Verifier):
    """DEER-style probe over a forced temporary answer.

    The verifier asks the replay agent for the cached temporary answer at the
    current position, then scores up to the first final-answer terminator, capped
    by ``probe_tokens``.
    """

    def __init__(
        self,
        probe_tokens: int = 20,
        probe_logprobs: int = 1,
    ):
        self.probe_tokens = int(probe_tokens)
        self.probe_logprobs = int(probe_logprobs)

    @property
    def temp_answer_tokens(self) -> int:
        return self.probe_tokens

    @property
    def temp_answer_logprobs(self) -> int:
        return self.probe_logprobs

    def __call__(self, ctx: VerifierContext) -> float:
        probe = self._probe_from_temp_answer(ctx)
        score = mean_token_probability(probe)
        if probe.content or probe.logprobs:
            ctx.logs.setdefault("probe_scores", []).append([ctx.position, score])
        logger.info(
            "Probe score method=%s position=%s score=%s probe_tokens=%s",
            ctx.method_name or "deer",
            ctx.position,
            score,
            len(probe.logprobs),
        )
        return score

    def _probe_from_temp_answer(self, ctx: VerifierContext) -> Response:
        probe_tokens = ctx.agent.scale_probe_tokens(self.probe_tokens, ctx.kwargs)
        response = ctx.agent.generate_temp_answer(
            ctx.prefix,
            ctx.position,
            ctx.kwargs,
            logprobs=self.probe_logprobs,
        )
        probe = slice_temp_answer_probe(
            response,
            probe_tokens=probe_tokens,
            token_counter=ctx.agent.chat_template.count_tokens,
            is_code=bool(getattr(ctx.agent, "_is_code_task")(ctx.kwargs)),
            think_end_token=ctx.agent.think_end_token(),
        )
        state = ctx.state
        state["_probe_prompt_tokens"] = state.get("_probe_prompt_tokens", 0) + (
            response.prompt_tokens or 0
        )
        state["_probe_completion_tokens"] = state.get("_probe_completion_tokens", 0) + (
            probe.completion_tokens or 0
        )
        ctx.logs.setdefault("temp_answer_positions", []).append(ctx.position)
        ctx.logs.setdefault("probe_attempts", []).append(
            {
                "position": ctx.position,
                # "temp_answer_position": ctx.position,
                "prompt_tokens": response.prompt_tokens or 0,
                "completion_tokens": probe.completion_tokens or 0,
                # "output_tokens": probe.completion_tokens or 0,
                "content": probe.content,
            }
        )
        return probe


class CodeStopVerifier(ProbeLogprobVerifier):
    """CodeStop: DEER temp-answer score plus degeneration and threshold ramping."""

    def __init__(
        self,
        conf_threshold: float = 0.95,
        degen_threshold: float = 1.0,
        probe_tokens: int = 20,
        probe_logprobs: int = 1,
        use_log: bool = True,
        ewt: bool = True,
        ramp_steps: int | None = None,
        ramp_min: float | None = None,
        trial_answer_prefix: str = "\n**Final Answer**\n\nThe final answer is \\boxed",
    ):
        super().__init__(
            probe_tokens=probe_tokens,
            probe_logprobs=probe_logprobs,
        )
        self.conf_threshold = float(conf_threshold)
        self.degen_threshold = float(degen_threshold)
        self.use_log = bool(use_log)
        self.ewt = bool(ewt)
        self.ramp_steps = ramp_steps
        self.ramp_min = ramp_min
        self.trial_answer_prefix = trial_answer_prefix

    def reset(self, method) -> None:
        method.state["_codestop_step_index"] = 0
        method.state["_codestop_should_stop"] = False

    def __call__(self, ctx: VerifierContext) -> float:
        state = ctx.state
        step_index = int(state.get("_codestop_step_index", 0))
        probe = self._probe_from_temp_answer(ctx)
        score = mean_token_probability(probe)
        prob_checks = ctx.logs.setdefault("prob_checks", [])
        positions = [item["stop_token_idx"] for item in prob_checks] + [ctx.position]
        probabilities = [item["total_prob_max"] for item in prob_checks] + [score]
        degen_score = compute_degeneration_score(
            positions,
            probabilities,
            use_log=self.use_log,
        )
        effective_threshold = compute_ramping_threshold(
            step_index,
            self.ramp_steps,
            self.ramp_min,
            ctx.agent.scale_threshold(self.conf_threshold, ctx.kwargs),
        )
        confidence_stop = score > effective_threshold
        degeneration_stop = degen_score > self.degen_threshold
        state["_codestop_should_stop"] = bool(confidence_stop or degeneration_stop)
        state["_codestop_step_index"] = step_index + 1

        prob_checks.append(
            {
                "position": ctx.position,
                "stop_token_idx": ctx.position,
                "total_prob_max": score,
                "predicted_answer": probe.content,
                "token_ids": [],
                "token_probs": [math.exp(lp) for lp in best_token_logprobs(probe)],
                "degen_score": degen_score,
                "effective_deer_threshold": effective_threshold,
                "confidence_stop": confidence_stop,
                "degeneration_stop": degeneration_stop,
                "prompt_tokens": probe.prompt_tokens or 0,
                "completion_tokens": probe.completion_tokens or 0,
                "output_tokens": probe.completion_tokens or 0,
            }
        )
        logger.info(
            "CodeStop score method=%s position=%s confidence=%s degen=%s stop=%s",
            ctx.method_name or "codestop",
            ctx.position,
            score,
            degen_score,
            state["_codestop_should_stop"],
        )
        return score


class EntrocutVerifier(ProbeLogprobVerifier):
    """ENTROCUT: DEER-style probe scored by top-logprob entropy."""

    def __init__(
        self,
        threshold: float = 0.2,
        tail_mass_threshold: float = 0.05,
        probe_tokens: int = 20,
        probe_logprobs: int = 20,
    ):
        super().__init__(
            probe_tokens=probe_tokens,
            probe_logprobs=probe_logprobs,
        )
        self.threshold = float(threshold)
        self.tail_mass_threshold = float(tail_mass_threshold)

    def __call__(self, ctx: VerifierContext) -> float:
        probe = self._probe_from_temp_answer(ctx)
        entropy, tail_mass, token_count = self._probe_entropy(probe)
        score = entropy if tail_mass <= self.tail_mass_threshold else math.inf
        ctx.logs.setdefault("entrocut_checks", []).append(
            {
                "position": ctx.position,
                "entropy": entropy,
                "tail_mass": tail_mass,
                "score": score,
                "threshold": self.threshold,
                "tail_mass_threshold": self.tail_mass_threshold,
                "probe_tokens": token_count,
            }
        )
        if token_count:
            ctx.logs.setdefault("probe_scores", []).append([ctx.position, score])
        logger.info(
            "ENTROCUT check method=%s position=%s entropy=%s tail_mass=%s score=%s",
            ctx.method_name or "entrocut",
            ctx.position,
            entropy,
            tail_mass,
            score,
        )
        return score

    def _probe_entropy(self, probe: Response) -> tuple[float, float, int]:
        entropies: list[float] = []
        tail_masses: list[float] = []
        for item in probe.logprobs:
            logprobs = [
                float(top.logprob)
                for top in item.top_logprobs[: self.probe_logprobs]
                if top.logprob is not None
            ]
            if not logprobs:
                continue
            token_entropy = residual_bucket_entropy(logprobs)
            entropies.append(token_entropy.entropy)
            tail_masses.append(token_entropy.tail_mass)
        if not entropies:
            return math.inf, math.inf, 0
        return (
            sum(entropies) / len(entropies),
            sum(tail_masses) / len(tail_masses),
            len(entropies),
        )


class EATVerifier(Verifier):
    """Entropy After Think verifier using fresh forced temporary-answer rollouts."""

    def __init__(
        self,
        *,
        threshold: float = 2.0,
        warmup_steps: int = 50,
        entropy_decoding_steps: int = 20,
        entropy_rollouts: int = 10,
        timescale: float = 10.0,
        probe_logprobs: int = 20,
    ):
        self.threshold = float(threshold)
        self.warmup_steps = int(warmup_steps)
        self.entropy_decoding_steps = int(entropy_decoding_steps)
        self.entropy_rollouts = int(entropy_rollouts)
        self.timescale = float(timescale)
        self.probe_logprobs = int(probe_logprobs)

    @property
    def temp_answer_tokens(self) -> int:
        return self.entropy_decoding_steps

    @property
    def temp_answer_logprobs(self) -> int:
        return self.probe_logprobs

    def reset(self, method) -> None:
        method.state["_eat_entropy_history"] = []
        method.state["_eat_ema"] = None
        method.state["_eat_ema_variance"] = None
        method.state["_eat_step_index"] = 0

    def __call__(self, ctx: VerifierContext) -> float:
        entropy, rollout_logs = self._estimate_entropy(ctx)
        state = ctx.state
        logs = ctx.logs
        history = state.setdefault("_eat_entropy_history", [])
        step_index = int(state.get("_eat_step_index", 0)) + 1
        state["_eat_step_index"] = step_index

        if math.isinf(entropy):
            ema_var = math.inf
        else:
            history.append(entropy)
            ema_var = self._update_ema_variance(state, entropy)

        score = math.inf if step_index < self.warmup_steps else ema_var
        logs.setdefault("eat_checks", []).append(
            {
                "step": step_index,
                "position": ctx.position,
                "entropy": entropy,
                "ema_variance": ema_var,
                "score": score,
                "threshold": self.threshold,
                "warmup_steps": self.warmup_steps,
                "rollouts": rollout_logs,
            }
        )
        logger.info(
            "EAT check method=%s step=%s position=%s entropy=%s ema_var=%s score=%s",
            ctx.method_name or "eat",
            step_index,
            ctx.position,
            entropy,
            ema_var,
            score,
        )
        return score

    def _estimate_entropy(
        self,
        ctx: VerifierContext,
    ) -> tuple[float, list[dict[str, Any]]]:
        rollout_entropies: list[list[float]] = []
        rollout_logs: list[dict[str, Any]] = []
        missing_logprob_rollouts = 0
        state = ctx.state
        probe_tokens = ctx.agent.scale_probe_tokens(
            self.entropy_decoding_steps, ctx.kwargs
        )

        for rollout_idx in range(self.entropy_rollouts):
            response = ctx.agent.generate_temp_answer(
                ctx.prefix,
                ctx.position,
                ctx.kwargs,
                logprobs=self.probe_logprobs,
                force_new=True,
            )
            state["_eat_prompt_tokens"] = state.get("_eat_prompt_tokens", 0) + (
                response.prompt_tokens or 0
            )
            state["_eat_completion_tokens"] = state.get("_eat_completion_tokens", 0) + (
                response.completion_tokens or 0
            )

            token_entropies = self._token_entropies(response, probe_tokens)
            effective_logprobs = [
                len(item.top_logprobs) for item in response.logprobs[:probe_tokens]
            ]
            if not token_entropies:
                missing_logprob_rollouts += 1
                logger.info(
                    "EAT rollout missing top logprobs method=%s position=%s rollout=%s",
                    ctx.method_name or "eat",
                    ctx.position,
                    rollout_idx,
                )
            else:
                rollout_entropies.append(token_entropies)

            rollout_logs.append(
                {
                    "rollout": rollout_idx,
                    "position": ctx.position,
                    "prompt_tokens": response.prompt_tokens or 0,
                    "completion_tokens": response.completion_tokens or 0,
                    "output_tokens": response.completion_tokens or 0,
                    "effective_logprobs": effective_logprobs,
                    "entropy": sum(token_entropies) if token_entropies else math.inf,
                    "content": response.content,
                }
            )

        if missing_logprob_rollouts:
            ctx.logs.setdefault("eat_provider_events", []).append(
                {
                    "position": ctx.position,
                    "event": "missing_logprobs",
                    "rollouts": missing_logprob_rollouts,
                }
            )

        if not rollout_entropies:
            return math.inf, rollout_logs

        cumulative_totals = [
            sum(token_entropies) for token_entropies in rollout_entropies
        ]
        return sum(cumulative_totals) / len(cumulative_totals), rollout_logs

    def _token_entropies(self, response: Response, probe_tokens: int) -> list[float]:
        entropies: list[float] = []
        for item in response.logprobs[:probe_tokens]:
            logprobs = [
                float(top.logprob)
                for top in item.top_logprobs
                if top.logprob is not None
            ]
            if not logprobs:
                continue
            entropies.append(self._entropy_from_logprobs(logprobs))
        return entropies

    @staticmethod
    def _entropy_from_logprobs(logprobs: Sequence[float]) -> float:
        if not logprobs:
            return math.inf
        return residual_bucket_entropy(logprobs).entropy

    def _update_ema_variance(self, state: dict[str, Any], entropy: float) -> float:
        alpha = 1.0 / self.timescale
        ema = state.get("_eat_ema")
        ema_var = state.get("_eat_ema_variance")
        if ema is None:
            state["_eat_ema"] = entropy
            state["_eat_ema_variance"] = 0.0
            return 0.0

        previous_ema = float(ema)
        previous_var = float(ema_var or 0.0)
        diff = entropy - previous_ema
        next_var = (1.0 - alpha) * previous_var + alpha * diff * diff
        state["_eat_ema"] = alpha * entropy + (1.0 - alpha) * previous_ema
        state["_eat_ema_variance"] = next_var
        return next_var


class PUMAVerifier(Verifier):
    """PUMA verifier over semantic reasoning-step checkpoints."""

    def __init__(
        self,
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
        embedding_model: str | None = DEFAULT_PUMA_EMBEDDING_MODEL,
        embedding_device: str | None = None,
        embedding_batch_size: int = 32,
        embedding_max_tokens: int | None = DEFAULT_PUMA_EMBEDDING_MAX_TOKENS,
        code_similarity_threshold: float = 0.8,
        collect_answer: str | Callable[[str], Any] | None = None,
        collect_coding_answer: str | Callable[[str, Any], Any] | None = None,
        similarity_backend: Any | None = None,
    ):
        self.similarity_threshold = float(similarity_threshold)
        self.confidence_threshold = float(confidence_threshold)
        self.epsilon = float(epsilon)
        self.consecutive = int(consecutive)
        self.probe_tokens = int(probe_tokens)
        self.probe_logprobs = int(probe_logprobs)
        self.window_size = int(window_size)
        self.consecutive_redundancy_stop = int(consecutive_redundancy_stop)
        self.min_loop_breaker_step = int(min_loop_breaker_step)
        self.forced_stop_min_confidence = float(forced_stop_min_confidence)
        self.always_check_first_n = int(always_check_first_n)
        if trigger_mode not in {"current", "previous"}:
            raise ValueError(f"Unknown PUMA trigger_mode: {trigger_mode!r}")
        self.trigger_mode = trigger_mode
        self.loop_breaker_similarity_threshold = (
            self.similarity_threshold
            if loop_breaker_similarity_threshold is None
            else float(loop_breaker_similarity_threshold)
        )
        self.task_type = task_type
        self.embedding_model = embedding_model
        self.embedding_device = embedding_device
        self.embedding_batch_size = int(embedding_batch_size)
        self.embedding_max_tokens = (
            int(embedding_max_tokens) if embedding_max_tokens is not None else None
        )
        self.code_similarity_threshold = float(code_similarity_threshold)
        self.answer_collector = _resolve_collector(
            collect_answer, name="collect_answer"
        )
        self.coding_answer_collector = _resolve_collector(
            collect_coding_answer,
            name="collect_coding_answer",
        )
        self._similarity = similarity_backend
        self._step_texts: list[str] = []
        self._full_path_filter_cache: dict[str, list[Any]] | None = None

    @property
    def temp_answer_tokens(self) -> None:
        return None

    @property
    def temp_answer_logprobs(self) -> int:
        return self.probe_logprobs

    def preload(self) -> None:
        self._similarity_backend()

    def reset(self, method) -> None:
        method.state["_puma_should_stop"] = False
        method.state["_puma_consecutive_redundant"] = 0
        method.state["_puma_last_prefix_chars"] = 0
        method.state["_puma_trial_results"] = []
        self._step_texts = []
        self._full_path_filter_cache = None
        if self._similarity is not None:
            self._similarity.reset()

    def __call__(self, ctx: VerifierContext) -> float:
        state = ctx.state
        logs = ctx.logs
        state["_puma_should_stop"] = False

        task_type = self._effective_task_type(ctx)
        step_index = int(state.get("_semantic_step_index") or 0)
        if step_index <= 0:
            step_index = len(self._step_texts) + 1
        step_text = self._latest_step_text(ctx)
        self._step_texts.append(step_text)

        full_path_filter = self._full_path_filter(ctx)
        step_offset = step_index - 1
        is_redundant = bool(
            step_offset < len(full_path_filter["redundant"])
            and full_path_filter["redundant"][step_offset]
        )
        loop_redundant = bool(
            step_offset < len(full_path_filter["loop_redundant"])
            and full_path_filter["loop_redundant"][step_offset]
        )
        consecutive_redundant = int(state.get("_puma_consecutive_redundant", 0))
        consecutive_redundant = consecutive_redundant + 1 if loop_redundant else 0
        state["_puma_consecutive_redundant"] = consecutive_redundant

        trial_results = state.setdefault("_puma_trial_results", [])
        if loop_breaker_fires(
            step_index=step_index,
            consecutive_redundant=consecutive_redundant,
            trial_results=trial_results,
            consecutive_redundancy_stop=self.consecutive_redundancy_stop,
            min_loop_breaker_step=self.min_loop_breaker_step,
            forced_stop_min_confidence=ctx.agent.scale_threshold(
                self.forced_stop_min_confidence, ctx.kwargs
            ),
        ):
            state["_puma_should_stop"] = True
            logs.setdefault("puma_events", []).append(
                {
                    "step": step_index,
                    "position": ctx.position,
                    "reason": "loop_breaker",
                    "consecutive_redundant": consecutive_redundant,
                    "best_trial_confidence": best_trial_confidence(trial_results),
                }
            )
            return best_trial_confidence(trial_results)

        should_trial = bool(
            step_offset < len(full_path_filter["should_generate_trial"])
            and full_path_filter["should_generate_trial"][step_offset]
        )
        similarity = (
            full_path_filter["step_similarities"][step_offset]
            if step_offset < len(full_path_filter["step_similarities"])
            else None
        )

        logs.setdefault("puma_step_checks", []).append(
            {
                "step": step_index,
                "position": ctx.position,
                "chars": len(step_text),
                "similarity": similarity,
                "redundant": is_redundant,
                "loop_redundant": loop_redundant,
                "consecutive_redundant": consecutive_redundant,
                "should_trial": should_trial,
            }
        )

        if not should_trial:
            logger.info(
                "PUMA skipped trial method=%s step=%s position=%s similarity=%s",
                ctx.method_name or "puma",
                step_index,
                ctx.position,
                similarity,
            )
            return 0.0

        logger.info(
            "PUMA generating trial method=%s step=%s position=%s similarity=%s",
            ctx.method_name or "puma",
            step_index,
            ctx.position,
            similarity,
        )
        trial_resp = self._generate_trial_answer(ctx, task_type)
        probe_resp = self._slice_trial_probe(trial_resp, ctx)
        confidence = mean_token_probability(probe_resp)
        answer_source = (
            trial_resp.content
            if task_type == "code"
            else get_confident_ending(task_type) + trial_resp.content
        )
        answer = extract_trial_answer(answer_source, task_type)
        collected_answer = self.collect_answer(answer_source, answer, task_type, ctx)
        trial = {
            "position": ctx.position,
            "temp_answer_position": getattr(
                trial_resp, "_temp_answer_position", ctx.position
            ),
            "answer": answer,
            "collected_answer": collected_answer,
            "confidence": confidence,
            "confidence_tokens": probe_resp.completion_tokens or 0,
            "step": step_index,
            "prompt_tokens": trial_resp.prompt_tokens or 0,
            "completion_tokens": trial_resp.completion_tokens or 0,
            "from_cache": bool(getattr(trial_resp, "_from_temp_answer_cache", False)),
        }
        trial_results.append(trial)
        logs.setdefault("puma_trials", []).append(trial)
        logs.setdefault("probe_scores", []).append([ctx.position, confidence])
        logger.info(
            "PUMA trial complete method=%s step=%s position=%s confidence=%s answer=%r from_cache=%s",
            ctx.method_name or "puma",
            step_index,
            ctx.position,
            confidence,
            answer,
            trial["from_cache"],
        )

        state["_puma_probe_prompt_tokens"] = state.get(
            "_puma_probe_prompt_tokens", 0
        ) + (trial_resp.prompt_tokens or 0)
        state["_puma_probe_completion_tokens"] = state.get(
            "_puma_probe_completion_tokens", 0
        ) + (trial_resp.completion_tokens or 0)

        if should_exit_verified(
            trial_results,
            task_type=task_type,
            consecutive=self.consecutive,
            confidence_threshold=ctx.agent.scale_threshold(
                self.confidence_threshold, ctx.kwargs
            ),
            epsilon=self.epsilon,
            code_similarity_threshold=ctx.agent.scale_threshold(
                self.code_similarity_threshold, ctx.kwargs
            ),
        ):
            state["_puma_should_stop"] = True
            logs.setdefault("puma_events", []).append(
                {
                    "step": step_index,
                    "position": ctx.position,
                    "reason": "verified_early_exit",
                    "confidence": confidence,
                    "answer": answer,
                }
            )
        return confidence

    def collect_answer(
        self,
        answer_source: str,
        extracted_answer: str,
        task_type: str,
        ctx: VerifierContext,
    ) -> Any:
        if task_type == "code" and self.coding_answer_collector is not None:
            reference = ctx.kwargs.get("answer")
            return bool(
                reference is not None
                and self.coding_answer_collector(answer_source, reference)
            )
        if task_type != "code" and self.answer_collector is not None:
            return str(self.answer_collector(answer_source)).strip()
        return extracted_answer

    def _latest_step_text(self, ctx: VerifierContext) -> str:
        previous_end = int(ctx.state.get("_puma_last_prefix_chars", 0))
        current_end = len(ctx.prefix.content)
        ctx.state["_puma_last_prefix_chars"] = current_end
        return ctx.prefix.content[previous_end:current_end].strip()

    def _similarity_backend(self) -> Any:
        if self._similarity is None:
            model_name = self.embedding_model or DEFAULT_PUMA_EMBEDDING_MODEL
            self._similarity = get_puma_similarity_backend(
                model_name,
                device=self.embedding_device,
                batch_size=self.embedding_batch_size,
                max_tokens=self.embedding_max_tokens,
            )
        return self._similarity

    def _full_path_filter(self, ctx: VerifierContext) -> dict[str, list[Any]]:
        if self._full_path_filter_cache is not None:
            return self._full_path_filter_cache

        if ctx.trajectory is None:
            step_texts = list(self._step_texts)
        else:
            thinking_end = ctx.agent._thinking_end_char(ctx.trajectory)
            reasoning = ctx.trajectory.content[:thinking_end]
            boundaries = ctx.state.get("_semantic_step_boundaries")
            if boundaries is None:
                spans = separate_step_spans(reasoning)
                boundaries = [span.end for span in spans if span.end > 0]
                if not boundaries and reasoning:
                    boundaries = [len(reasoning)]
            step_texts = []
            previous = 0
            for boundary in boundaries:
                step_texts.append(reasoning[previous:boundary].strip())
                previous = boundary

        step_similarities = self._step_similarities(step_texts)
        similarity_threshold = ctx.agent.scale_threshold(
            self.similarity_threshold, ctx.kwargs
        )
        loop_breaker_similarity_threshold = ctx.agent.scale_threshold(
            self.loop_breaker_similarity_threshold, ctx.kwargs
        )
        redundant = [
            sim is not None and sim >= similarity_threshold for sim in step_similarities
        ]
        loop_redundant = [
            sim is not None and sim >= loop_breaker_similarity_threshold
            for sim in step_similarities
        ]
        should_generate_trial = [False] * len(step_texts)
        for index in range(min(self.always_check_first_n, len(step_texts))):
            should_generate_trial[index] = True
        for index in range(1, len(step_texts)):
            if not redundant[index]:
                continue
            if self.trigger_mode == "current":
                should_generate_trial[index] = True
            else:
                should_generate_trial[index - 1] = True

        full_path_filter = {
            "step_similarities": step_similarities,
            "redundant": redundant,
            "loop_redundant": loop_redundant,
            "should_generate_trial": should_generate_trial,
        }
        self._full_path_filter_cache = full_path_filter
        return full_path_filter

    def _step_similarities(self, step_texts: list[str]) -> list[float | None]:
        if not step_texts:
            return []
        backend = self._similarity_backend()
        batch_fn = getattr(backend, "max_similarities", None)
        if callable(batch_fn):
            return list(batch_fn(step_texts, self.window_size))

        similarities: list[float | None] = [None]
        for index in range(1, len(step_texts)):
            start = 0 if self.window_size <= 0 else max(0, index - self.window_size)
            window = step_texts[start:index]
            similarities.append(backend.max_similarity(step_texts[index], window))
        return similarities

    def _generate_trial_answer(self, ctx: VerifierContext, task_type: str) -> Response:
        prefix, position = self._prefix_with_verify_batch_buffer(ctx)
        return ctx.agent.generate_temp_answer(
            prefix,
            position,
            ctx.kwargs,
            logprobs=self.probe_logprobs,
        )

    def _slice_trial_probe(self, response: Response, ctx: VerifierContext) -> Response:
        probe_tokens = ctx.agent.scale_probe_tokens(self.probe_tokens, ctx.kwargs)
        return slice_temp_answer_probe(
            response,
            probe_tokens=probe_tokens,
            token_counter=ctx.agent.chat_template.count_tokens,
            is_code=self._effective_task_type(ctx) == "code",
            think_end_token=ctx.agent.think_end_token(),
        )

    def _prefix_with_verify_batch_buffer(
        self, ctx: VerifierContext
    ) -> tuple[Response, int]:
        batch = max(1, int(getattr(ctx.agent, "verify_batch_token", 1)))
        if batch <= 1 or ctx.trajectory is None:
            return ctx.prefix, ctx.position

        target_position = ((ctx.position + batch - 1) // batch) * batch
        if target_position <= ctx.position:
            return ctx.prefix, ctx.position

        thinking_end_char = ctx.agent._thinking_end_char(ctx.trajectory)
        if ctx.trajectory.logprobs and ctx.prefix.logprobs:
            token_texts = ctx.trajectory.token_texts()
            char_end = sum(len(token or "") for token in token_texts[:target_position])
            char_end = min(char_end, thinking_end_char)
            if char_end <= len(ctx.prefix.content):
                return ctx.prefix, ctx.position
            buffered = ctx.trajectory.char_slice_from_tokens(0, char_end)
            ctx.logs.setdefault("puma_trial_prompt_buffers", []).append(
                {
                    "position": ctx.position,
                    "prompt_position": len(buffered.logprobs),
                    "chars": char_end,
                }
            )
            prefix = Response(
                input=ctx.trajectory.input,
                content=buffered.content,
                logprobs=buffered.logprobs,
                completion_tokens=len(buffered.logprobs),
            )
            return prefix, len(buffered.logprobs)

        char_end = min(target_position, thinking_end_char, len(ctx.trajectory.content))
        if char_end <= len(ctx.prefix.content):
            return ctx.prefix, ctx.position
        ctx.logs.setdefault("puma_trial_prompt_buffers", []).append(
            {
                "position": ctx.position,
                "prompt_position": char_end,
                "chars": char_end,
            }
        )
        prefix = Response(
            input=ctx.trajectory.input,
            content=ctx.trajectory.content[:char_end],
            completion_tokens=None,
        )
        return prefix, char_end

    def _effective_task_type(self, ctx: VerifierContext) -> str:
        if self.task_type:
            return self.task_type
        if ctx.agent._is_code_task(ctx.kwargs):
            return "code"
        dataset = str(
            ctx.kwargs.get("dataset") or ctx.kwargs.get("dataloader") or ""
        ).lower()
        if "gpqa" in dataset:
            return "gpqa"
        return "math"


class TextCompletionScoringHandler:
    """Render verifier prompts and call /completions, not chat completions."""

    INTERNAL_RUNTIME_KEYS = {
        "answer",
        "evaluator",
        "evaluator_name",
        "dataloader",
        "dataset",
        "is_code",
        "reuse_generation_content",
        "reuse_generation_input",
        "reuse_generation_response",
        "verify_batch_token",
        "final_answer_batch_size",
    }

    def __init__(
        self,
        llm_name: str | None = None,
        llm: Any | None = None,
        chat_template: dict[str, Any] | None = None,
        chat_template_controller: ChatTemplateController | None = None,
        completion_kwargs: dict[str, Any] | None = None,
        llm_kwargs: dict[str, Any] | None = None,
        close_thinking: bool = True,
    ):
        if llm is None and not llm_name:
            raise ValueError("TextCompletionScoringHandler requires llm or llm_name.")
        self.llm = (
            llm if llm is not None else get_llm_wrapper(llm_name, **(llm_kwargs or {}))
        )
        self.chat_template = chat_template_controller or ChatTemplateController(
            chat_template
        )
        self.completion_kwargs = dict(completion_kwargs or {})
        self.close_thinking = bool(close_thinking)

    def score(
        self, messages: list[dict[str, str]], runtime_kwargs: dict[str, Any]
    ) -> Response:
        prompt = self.render_prompt(messages)
        kwargs = {
            key: value
            for key, value in runtime_kwargs.items()
            if key not in self.INTERNAL_RUNTIME_KEYS
        }
        kwargs.update(self.completion_kwargs)
        kwargs.pop("max_thinking_completion_token", None)
        kwargs.pop("max_temp_answer_completion_token", None)
        kwargs.pop("max_final_answer_completion_token", None)
        kwargs.pop("max_competion_token", None)
        kwargs.setdefault("max_completion_token", 1024)
        kwargs["detail"] = True
        response = Response.from_completion(
            self.llm.completion(prompt, **kwargs),
            token_counter=self.chat_template.count_tokens,
        )
        if response.prompt_tokens is None:
            response.prompt_tokens = self.chat_template.count_tokens(prompt)
        return response

    def generate_temp_answer(
        self,
        input_text: str,
        runtime_kwargs: dict[str, Any],
    ) -> Response:
        """Close an open thinking block, then continue with text completion."""
        prompt = self.close_thinking_if_needed(input_text)
        kwargs = {
            key: value
            for key, value in runtime_kwargs.items()
            if key not in self.INTERNAL_RUNTIME_KEYS
        }
        kwargs.update(self.completion_kwargs)
        kwargs.pop("max_thinking_completion_token", None)
        kwargs.pop("max_competion_token", None)
        temp_answer_tokens = kwargs.pop("max_temp_answer_completion_token", None)
        final_answer_tokens = kwargs.pop("max_final_answer_completion_token", None)
        if temp_answer_tokens is not None:
            kwargs["max_completion_token"] = temp_answer_tokens
        elif final_answer_tokens is not None and "max_completion_token" not in kwargs:
            kwargs["max_completion_token"] = final_answer_tokens
        kwargs["detail"] = True
        response = Response.from_completion(
            self.llm.completion(prompt, **kwargs),
            token_counter=self.chat_template.count_tokens,
        )
        if response.prompt_tokens is None:
            response.prompt_tokens = self.chat_template.count_tokens(prompt)
        return response

    def render_prompt(self, messages: list[dict[str, str]]) -> str:
        prompt = self.chat_template.render(messages)
        if not self.close_thinking:
            return prompt
        return self._append_closed_thinking(prompt)

    def close_thinking_if_needed(self, prompt: str) -> str:
        if not self.close_thinking:
            return prompt
        end = self.chat_template.think_end_token or "</think>"
        if end in prompt:
            return prompt
        return (
            prompt.rstrip()
            + f"You are out of thinking budget {end}\n\nThe final answer is"
        )

    def _append_closed_thinking(self, prompt: str) -> str:
        start = self.chat_template.think_start_token or "<think>"
        end = self.chat_template.think_end_token or "</think>"
        stripped = prompt.rstrip()
        if stripped.endswith(end):
            return prompt if prompt.endswith("\n") else prompt + "\n"
        if stripped.endswith(start):
            return prompt + f"\n\n{end}\n"
        return prompt + f"{start}\n\n{end}\n"


DEFAULT_SUFFICIENCY_TEMPLATE = (
    "Assess the confidence that the provided thought is sufficient to solve the question.\n"
    "Your entire response must be exactly: Confidence: \\boxed{{<number>}}, where <number> "
    "is a real value "
    "between 0 and 100. Here, 100 means the thought is fully sufficient to reach the "
    "final answer directly without additional reasoning, and 0 means the thought is "
    "completely insufficient to do so.\n\n"
    "Do not include analysis, explanation, or any other text. Return the boxed score "
    "immediately.\n\n"
    "### Question\n{question}\n\n"
    "### Thought\n{reasoning_trace}\n"
)


class LLMSufficiencyVerifier(Verifier):
    """DTSR-style sufficiency scorer.

    Prompts ``ctx.agent``'s LLM with a scoring template. Extracts a number in
    ``[0, 100]`` from ``\\boxed{...}``. Returns ``None`` if the model output
    could not be parsed.
    """

    def __init__(
        self,
        prompt_template: str | None = None,
        system_prompt: str | None = None,
        llm: str = "same",
        llm_name: str | None = None,
        chat_template: dict[str, Any] | None = None,
        completion_kwargs: dict[str, Any] | None = None,
        llm_kwargs: dict[str, Any] | None = None,
        close_thinking: bool = True,
        probe_tokens: int = 20,
    ):
        self.prompt_template = prompt_template or DEFAULT_SUFFICIENCY_TEMPLATE
        self.system_prompt = system_prompt
        if llm not in {"same", "custom"}:
            raise ValueError(
                f"LLMSufficiencyVerifier llm must be 'same' or 'custom', got {llm!r}"
            )
        if llm == "same" and llm_name:
            llm = "custom"
        if llm == "custom" and not llm_name:
            raise ValueError(
                "LLMSufficiencyVerifier with llm='custom' requires llm_name."
            )
        self.llm = llm
        self.custom_scoring_handler = (
            TextCompletionScoringHandler(
                llm_name=llm_name,
                chat_template=chat_template,
                completion_kwargs=completion_kwargs,
                llm_kwargs=llm_kwargs,
                close_thinking=close_thinking,
            )
            if llm == "custom"
            else None
        )
        self.same_chat_template = chat_template
        self.same_completion_kwargs = completion_kwargs
        self.close_thinking = close_thinking
        if probe_tokens <= 0:
            raise ValueError("LLMSufficiencyVerifier probe_tokens must be positive.")
        self.probe_tokens = int(probe_tokens)

    def __call__(self, ctx: VerifierContext) -> float | None:
        messages = self._build_messages(ctx.question, ctx.prefix.content)
        scoring_handler = self._scoring_handler(ctx)
        score_response = scoring_handler.score(messages, runtime_kwargs=ctx.kwargs)
        state = ctx.state
        state["_sufficiency_prompt_tokens"] = state.get(
            "_sufficiency_prompt_tokens", 0
        ) + (score_response.prompt_tokens or 0)
        state["_sufficiency_completion_tokens"] = state.get(
            "_sufficiency_completion_tokens", 0
        ) + (score_response.completion_tokens or 0)
        score = self._parse_score(score_response.content)
        ctx.logs.setdefault("sufficiency_attempts", []).append(
            {
                "position": ctx.position,
                "prompt_tokens": score_response.prompt_tokens or 0,
                "completion_tokens": score_response.completion_tokens or 0,
                "output_tokens": score_response.completion_tokens or 0,
                "content": score_response.content,
                "score": score,
            }
        )
        return score

    def _scoring_handler(self, ctx: VerifierContext) -> TextCompletionScoringHandler:
        if self.custom_scoring_handler is not None:
            return self.custom_scoring_handler
        completion_kwargs = self.same_completion_kwargs
        if completion_kwargs is None:
            completion_kwargs = ctx.agent._completion_kwargs_for_phase(
                "temp_answer", ctx.kwargs
            )
        return TextCompletionScoringHandler(
            llm=ctx.agent.llm,
            chat_template=self.same_chat_template,
            chat_template_controller=None
            if self.same_chat_template
            else ctx.agent.chat_template,
            completion_kwargs=completion_kwargs,
            close_thinking=self.close_thinking,
        )

    def _build_messages(
        self, question: str, reasoning_trace: str
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append(
            {
                "role": "user",
                "content": self.prompt_template.format(
                    question=question,
                    reasoning_trace=reasoning_trace,
                ),
            }
        )
        return messages

    @staticmethod
    def _parse_score(content: str) -> float | None:
        if not content:
            return None
        match = re.search(r"\\?boxed\{\s*(-?\d+(?:\.\d+)?)\s*\}", content)
        if not match:
            match = re.search(
                r"(?i)\bconfidence\s*:\s*(?:\\?boxed\{\s*)?(-?\d+(?:\.\d+)?)",
                content,
            )
        if not match:
            match = re.fullmatch(r"\s*(-?\d+(?:\.\d+)?)\s*", content)
        if not match:
            return None
        try:
            score = float(match.group(1))
        except (TypeError, ValueError):
            return None
        return max(0.0, min(100.0, score))


class AnswerCoverageVerifier(Verifier):
    """Generate temporary answers and require consecutive answer stability."""

    def __init__(
        self,
        window_size: int = 3,
        llm: str = "same",
        llm_name: str | None = None,
        chat_template: dict[str, Any] | None = None,
        completion_kwargs: dict[str, Any] | None = None,
        llm_kwargs: dict[str, Any] | None = None,
        close_thinking: bool = True,
        collect_answer: str | Callable[[str], Any] | None = None,
        collect_coding_answer: str | Callable[[str, Any], Any] | None = None,
    ):
        if window_size <= 0:
            raise ValueError("window_size must be >= 1")
        if llm not in {"same", "custom"}:
            raise ValueError(
                f"AnswerCoverageVerifier llm must be 'same' or 'custom', got {llm!r}"
            )
        if llm == "same" and llm_name:
            llm = "custom"
        if llm == "custom" and not llm_name:
            raise ValueError(
                "AnswerCoverageVerifier with llm='custom' requires llm_name."
            )
        self.window_size = int(window_size)
        self.llm = llm
        self.custom_scoring_handler = (
            TextCompletionScoringHandler(
                llm_name=llm_name,
                chat_template=chat_template,
                completion_kwargs=completion_kwargs,
                llm_kwargs=llm_kwargs,
                close_thinking=close_thinking,
            )
            if llm == "custom"
            else None
        )
        self.same_chat_template = chat_template
        self.same_completion_kwargs = completion_kwargs
        self.close_thinking = close_thinking
        self.answer_collector = _resolve_collector(
            collect_answer, name="collect_answer"
        )
        self.coding_answer_collector = _resolve_collector(
            collect_coding_answer,
            name="collect_coding_answer",
        )

    @property
    def temp_answer_tokens(self) -> int | None:
        if (
            self.custom_scoring_handler is not None
            or self.same_chat_template is not None
        ):
            raise AttributeError(
                "custom answer-coverage temp-answer generation is not shared"
            )
        return None

    @property
    def temp_answer_logprobs(self) -> int:
        return 1

    def __call__(self, ctx: VerifierContext) -> float:
        response = self._generate_temp_answer(ctx)
        state = ctx.state
        state["_answer_coverage_prompt_tokens"] = state.get(
            "_answer_coverage_prompt_tokens", 0
        ) + (response.prompt_tokens or 0)
        state["_answer_coverage_completion_tokens"] = state.get(
            "_answer_coverage_completion_tokens", 0
        ) + (response.completion_tokens or 0)

        answer = response.content.strip()
        collected_answer = self.collect_answer(
            self._answer_extraction_content(response, ctx), ctx
        )
        normalized_answer = self._normalize_answer(collected_answer)
        previous_answer = state.get("_answer_coverage_previous_answer")
        if normalized_answer and normalized_answer == previous_answer:
            consecutive_matches = (
                int(state.get("_answer_coverage_consecutive_matches", 1)) + 1
            )
        else:
            consecutive_matches = 1
        state["_answer_coverage_previous_answer"] = normalized_answer
        state["_answer_coverage_consecutive_matches"] = consecutive_matches

        attempts = ctx.logs.setdefault("answer_coverage_attempts", [])
        attempts.append(
            {
                "position": ctx.position,
                "answer": answer,
                "collected_answer": collected_answer,
                "consecutive_matches": consecutive_matches,
                # "temp_answer_position": ctx.position,
                "prompt_tokens": response.prompt_tokens or 0,
                "completion_tokens": response.completion_tokens or 0,
                # "output_tokens": response.completion_tokens or 0,
            }
        )

        score = float(consecutive_matches >= self.window_size)
        logger.info(
            "Answer coverage check method=%s position=%s answer=%r consecutive=%s/%s score=%s from_cache=%s",
            ctx.method_name or "answer-coverage",
            ctx.position,
            collected_answer,
            consecutive_matches,
            self.window_size,
            score,
            bool(getattr(response, "_from_temp_answer_cache", False)),
        )
        return score

    def _generate_temp_answer(self, ctx: VerifierContext) -> Response:
        if self.custom_scoring_handler is None and self.same_chat_template is None:
            return ctx.agent.generate_temp_answer(
                ctx.prefix,
                ctx.position,
                ctx.kwargs,
                logprobs=1,
            )

        handler = self._scoring_handler(ctx)
        return handler.generate_temp_answer(
            ctx.prefix.get_generation_content(),
            runtime_kwargs=ctx.kwargs,
        )

    def collect_answer(self, answer: str, ctx: VerifierContext) -> Any:
        if self.coding_answer_collector is not None and ctx.agent._is_code_task(
            ctx.kwargs
        ):
            reference = ctx.kwargs.get("answer")
            return bool(
                reference is not None
                and self.coding_answer_collector(answer, reference)
            )
        if self.answer_collector is None:
            return answer
        return str(self.answer_collector(answer)).strip()

    def _answer_extraction_content(
        self,
        response: Response,
        ctx: VerifierContext,
    ) -> str:
        """Use the same forced ending that preceded a shared temp answer."""
        is_code = ctx.agent._is_code_task(ctx.kwargs)
        if (is_code and self.coding_answer_collector is None) or (
            not is_code and self.answer_collector is None
        ):
            return response.content
        if (
            self.custom_scoring_handler is not None
            or self.same_chat_template is not None
        ):
            return response.content
        return ctx.agent.temp_answer_extraction_content(response, ctx.kwargs)

    def _scoring_handler(self, ctx: VerifierContext) -> TextCompletionScoringHandler:
        if self.custom_scoring_handler is not None:
            return self.custom_scoring_handler
        return TextCompletionScoringHandler(
            llm=ctx.agent.llm,
            chat_template=self.same_chat_template,
            chat_template_controller=None
            if self.same_chat_template
            else ctx.agent.chat_template,
            completion_kwargs=self.same_completion_kwargs,
            close_thinking=self.close_thinking,
        )

    @staticmethod
    def _normalize_answer(answer: Any) -> str:
        if isinstance(answer, bool):
            return "true" if answer else ""
        return re.sub(r"\s+", " ", answer).strip().lower()


class DynasorVerifier(AnswerCoverageVerifier):
    """Assess certainty from answer consistency over a sliding probe window."""

    def __init__(
        self,
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
    ):
        if window_size <= 0:
            raise ValueError("window_size must be >= 1")
        if not 0.0 < consistency_threshold <= 1.0:
            raise ValueError("consistency_threshold must be in (0, 1]")
        super().__init__(
            window_size=1,
            llm=llm,
            llm_name=llm_name,
            chat_template=chat_template,
            completion_kwargs=completion_kwargs,
            llm_kwargs=llm_kwargs,
            close_thinking=close_thinking,
            collect_answer=collect_answer,
            collect_coding_answer=collect_coding_answer,
        )
        self.window_size = int(window_size)
        self.consistency_threshold = float(consistency_threshold)
        self.uncertainty_signals = tuple(
            signal.strip().lower() for signal in uncertainty_signals if signal.strip()
        )

    def __call__(self, ctx: VerifierContext) -> float:
        response = self._generate_temp_answer(ctx)
        state = ctx.state
        state["_dynasor_prompt_tokens"] = state.get("_dynasor_prompt_tokens", 0) + (
            response.prompt_tokens or 0
        )
        state["_dynasor_completion_tokens"] = state.get(
            "_dynasor_completion_tokens", 0
        ) + (response.completion_tokens or 0)

        answer = response.content.strip()
        uncertainty_signal = self._find_uncertainty_signal(answer)
        collected_answer: Any = None
        normalized_answer = ""
        consistency = 0.0
        valid_answers = state.setdefault("_dynasor_valid_answers", [])
        window: list[str] = valid_answers[-self.window_size :]

        if uncertainty_signal is None:
            collected_answer = self.collect_answer(
                self._answer_extraction_content(response, ctx), ctx
            )
            normalized_answer = self._normalize_answer(collected_answer)
            if normalized_answer:
                valid_answers.append(normalized_answer)
                window = valid_answers[-self.window_size :]
                if len(window) == self.window_size:
                    consistency = (
                        sum(candidate == normalized_answer for candidate in window)
                        / self.window_size
                    )

        state["_dynasor_consistency"] = consistency
        state["_dynasor_window"] = list(window)

        attempts = ctx.logs.setdefault("dynasor_attempts", [])
        attempts.append(
            {
                "position": ctx.position,
                "answer": answer,
                "collected_answer": collected_answer,
                "consistency": consistency,
                "window_count": len(window),
                "omitted": uncertainty_signal is not None or not normalized_answer,
                "uncertainty_signal": uncertainty_signal,
                # "temp_answer_position": ctx.position,
                "prompt_tokens": response.prompt_tokens or 0,
                "completion_tokens": response.completion_tokens or 0,
                # "output_tokens": response.completion_tokens or 0,
            }
        )
        logger.info(
            "Dynasor check method=%s position=%s answer=%r consistency=%s threshold=%s window=%s/%s omitted=%s signal=%r from_cache=%s",
            ctx.method_name or "dynasor",
            ctx.position,
            collected_answer,
            consistency,
            self.consistency_threshold,
            len(window),
            self.window_size,
            uncertainty_signal is not None or not normalized_answer,
            uncertainty_signal,
            bool(getattr(response, "_from_temp_answer_cache", False)),
        )
        return consistency

    def _find_uncertainty_signal(self, answer: str) -> str | None:
        lowered = answer.lower()
        for signal in self.uncertainty_signals:
            if re.search(rf"(?<!\w){re.escape(signal)}(?!\w)", lowered):
                return signal
        return None


class ProbeInTheMiddleVerifier(Verifier):
    """Check if any of the probe signals appear in the reasoning prefix."""

    def __init__(self, signals: Sequence[str]):
        self.signals = tuple(signals)

    def __call__(self, ctx: VerifierContext) -> float:
        text = ctx.prefix.content
        return float(any(sig in text for sig in self.signals))
