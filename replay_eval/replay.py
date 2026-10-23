"""Algorithm 1: replay-based evaluation for early-exit methods.

See ``agent-v2/proposal.md``.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from methods.base import (
    BaseAgent,
    BaseAgentFullGeneration,
    BaseAgentReuseGeneration,
    MAX_FINAL_COMPLETION_REQUEST_TOKENS,
    Response,
)
from replay_eval.method import EarlyExitMethod, MethodResult, VerifierContext
from replay_eval.utils import (
    convert_chat_full_generation_to_replay_response,
    full_generation_kwargs,
)
from scripts.config_loader import import_from_path


logger = logging.getLogger(__name__)


class _ReplayEvaluatorMixin:
    """Core replay logic shared by the reuse- and full-generation variants."""

    exit: bool = True
    methods: list[EarlyExitMethod]
    verify_batch_token: int
    final_answer_batch_size: int
    code_probe_token_multiplier: int = 1
    code_threshold_multiplier: float = 1.0

    def _init_replay_config(
        self,
        methods: Iterable[EarlyExitMethod] | None,
        verify_batch_token: int,
        final_answer_batch_size: int,
        code_probe_token_multiplier: int,
        code_threshold_multiplier: float,
        exit: bool = True,
    ) -> None:
        if not isinstance(exit, bool):
            raise ValueError("replay_eval.exit must be a boolean")
        self.exit = exit
        if code_probe_token_multiplier < 1:
            raise ValueError("code_probe_token_multiplier must be >= 1")
        if not 0 < code_threshold_multiplier <= 1:
            raise ValueError("code_threshold_multiplier must be in (0, 1]")
        self.verify_batch_token = int(verify_batch_token)
        self.final_answer_batch_size = int(final_answer_batch_size)
        self.code_probe_token_multiplier = int(code_probe_token_multiplier)
        self.code_threshold_multiplier = float(code_threshold_multiplier)
        self._init_methods(methods)

    def _init_methods(
        self,
        methods: Iterable[EarlyExitMethod] | Iterable[dict[str, Any]] | None,
    ) -> None:
        self.methods = _resolve_methods(methods)
        self._temp_answer_cache: dict[int, Response] = {}
        self._temp_answer_logs: list[dict[str, Any]] = []
        self._temp_answer_token_needs: dict[int, int] = {}
        self._temp_answer_logprob_needs: dict[int, int] = {}

    # ------------------------------------------------------------------ #
    # Public helpers exposed to verifiers via VerifierContext.agent.     #
    # ------------------------------------------------------------------ #

    def configured_stop_tokens(self, kwargs: dict[str, Any]) -> list[str]:
        return self._configured_stop_tokens(kwargs)  # type: ignore[attr-defined]

    def think_end_token(self) -> str:
        return self.chat_template.think_end_token or "</think>"  # type: ignore[attr-defined]

    def think_start_token(self) -> str:
        return self.chat_template.think_start_token or "<think>"  # type: ignore[attr-defined]

    def temp_answer_extraction_content(
        self,
        response: Response,
        kwargs: dict[str, Any],
    ) -> str:
        """Restore the forced ending needed to interpret a temp completion.

        ``generate_temp_answer`` completes after ``\\boxed`` for non-code
        tasks. Thus a returned completion beginning with ``{A}`` represents
        ``\\boxed{A}`` when passed to an answer extractor. Completions that
        provide their own prose or boxed form remain self-contained.
        """
        is_code_task = self._is_code_task(kwargs)  # type: ignore[attr-defined]
        if is_code_task or not response.content.startswith("{"):
            return response.content
        return self._temp_answer_suffix(kwargs) + response.content

    def generate_temp_answer(
        self,
        prefix: Response,
        position: int,
        kwargs: dict[str, Any],
        *,
        logprobs: int = 1,
        force_new: bool = False,
    ) -> Response:
        """Generate or reuse the forced immediate answer for a replay position."""
        cached = self._temp_answer_cache.get(position)
        if cached is not None and not force_new:
            response = cached.copy(deep=True)
            object.__setattr__(response, "_from_temp_answer_cache", True)
            object.__setattr__(response, "_temp_answer_position", position)
            return response

        prompt = prefix.get_generation_content() + self._temp_answer_suffix(kwargs)
        completion_kwargs = self._completion_kwargs_for_phase("temp_answer", kwargs)  # type: ignore[attr-defined]
        temp_answer_limit = self._effective_temp_answer_token_limit(kwargs)
        if temp_answer_limit is not None:
            completion_kwargs["max_completion_token"] = temp_answer_limit
        token_need = self._temp_answer_token_needs.get(position)
        if token_need is not None:
            completion_kwargs["max_completion_token"] = int(token_need)
        logprob_need = self._temp_answer_logprob_needs.get(position, logprobs)
        completion_kwargs["logprobs"] = max(1, int(logprobs), int(logprob_need))
        completion_kwargs["detail"] = True
        response = Response.from_completion(
            self.llm.completion(prompt, **completion_kwargs),  # type: ignore[attr-defined]
            token_counter=self._count_completion_text_tokens,  # type: ignore[attr-defined]
        )
        self._fill_missing_completion_tokens(response)  # type: ignore[attr-defined]
        if not force_new:
            self._temp_answer_cache[position] = response.copy(deep=True)
        self._temp_answer_logs.append(
            {
                "position": position,
                "answer": response.content,
                "prompt_tokens": response.prompt_tokens or 0,
                "completion_tokens": response.completion_tokens or 0,
                **({"force_new": True} if force_new else {}),
            }
        )
        object.__setattr__(response, "_from_temp_answer_cache", False)
        object.__setattr__(response, "_temp_answer_position", position)
        return response

    # ------------------------------------------------------------------ #
    # Replay loop (Algorithm 1).                                         #
    # ------------------------------------------------------------------ #

    def solve(self, question: str, **kwargs) -> Response:
        self._temp_answer_cache = {}
        self._temp_answer_logs = []
        self._temp_answer_token_needs = {}
        self._temp_answer_logprob_needs = {}
        self._fallback_final_response: Response | None = None
        self._fallback_final_truncated = False
        self._preload_verifiers()
        trajectory = self._obtain_trajectory(question, kwargs)
        unfinished_thinking = self.think_end_token() not in trajectory.content
        thinking_limit_reached = (
            self._reached_thinking_limit(trajectory, kwargs)
            if unfinished_thinking
            else False
        )
        results = self._replay(question, trajectory, kwargs)
        if unfinished_thinking and (
            thinking_limit_reached
            or not self.methods
            or self._fallback_final_response is not None
        ):
            trajectory = self._force_finalize_trajectory(
                trajectory,
                kwargs,
                thinking_limit_reached=thinking_limit_reached,
                final_response=self._fallback_final_response,
                final_answer_truncated=self._fallback_final_truncated,
            )
            self.full_generation_response = trajectory  # type: ignore[attr-defined]
        object.__setattr__(trajectory, "method_results", results)
        object.__setattr__(trajectory, "method_results_dicts", [r.to_dict() for r in results])
        if self._temp_answer_logs:
            object.__setattr__(trajectory, "temp_answers", list(self._temp_answer_logs))
        return trajectory

    def _force_finalize_trajectory(
        self,
        trajectory: Response,
        kwargs: dict[str, Any],
        *,
        thinking_limit_reached: bool,
        final_response: Response | None = None,
        final_answer_truncated: bool = False,
    ) -> Response:
        """Force a normal final-answer rollout after unfinished thinking.

        ``max_completion_token`` is the trajectory's thinking budget. The
        follow-up completion gets its own final-answer budget and is retained
        even when that second request itself reaches its length limit.
        """
        thinking_tokens = trajectory.token_count()
        final_kwargs = self._with_default_final_answer_budget(kwargs)

        if final_response is None:
            final_response = self._generate_early_answer(trajectory, final_kwargs)
        final_answer_truncated = final_answer_truncated or bool(
            getattr(final_response, "final_answer_truncated", False)
        )

        answer_tokens = final_response.token_count()
        reasoning = trajectory.content
        trajectory.content = (
            f"{reasoning}{self.think_end_token()}{final_response.content}"
        )
        trajectory.finish_reason = final_response.finish_reason
        trajectory.completion_tokens = thinking_tokens + answer_tokens
        object.__setattr__(trajectory, "thinking_tokens", thinking_tokens)
        object.__setattr__(trajectory, "answer_tokens", answer_tokens)
        object.__setattr__(trajectory, "forced_final_answer", True)
        object.__setattr__(
            trajectory,
            "thinking_limit_reached",
            thinking_limit_reached,
        )
        object.__setattr__(
            trajectory,
            "final_answer_prompt_tokens",
            final_response.prompt_tokens or 0,
        )
        object.__setattr__(
            trajectory,
            "final_answer_truncated",
            final_answer_truncated,
        )
        logical_total_tokens = (
            (trajectory.prompt_tokens or 0) + thinking_tokens + answer_tokens
        )
        object.__setattr__(
            trajectory,
            "logical_total_tokens",
            logical_total_tokens,
        )
        logger.info(
            "Forced final answer thinking_tokens=%s answer_tokens=%s "
            "thinking_limit_reached=%s final_answer_truncated=%s "
            "logical_total_tokens=%s",
            thinking_tokens,
            answer_tokens,
            thinking_limit_reached,
            final_answer_truncated,
            logical_total_tokens,
        )
        return trajectory

    def _preload_verifiers(self) -> None:
        for method in self.methods:
            preload = getattr(method.verifier, "preload", None)
            if callable(preload):
                logger.info(
                    "preloading method=%s verifier=%s",
                    method.name,
                    method.verifier.__class__.__name__,
                )
                try:
                    preload()
                except RuntimeError as exc:
                    if not _is_already_borrowed_error(exc):
                        raise
                    logger.warning(
                        "Skipping verifier preload after tokenizer contention "
                        "method=%s error=%s",
                        method.name,
                        exc,
                    )

    def _obtain_trajectory(self, question: str, kwargs: dict[str, Any]) -> Response:
        """Fetch a single full trajectory without running the chunked loop.

        Uses whichever ``_get_full_generation_response`` is mixed in
        (fresh call in the ``FullGeneration`` flavour, or supplied content
        in the ``Reuse`` flavour).
        """
        generation_messages = BaseAgent.build_generation_messages(self, question)  # type: ignore[arg-type]
        rendered_prompt = self.chat_template.render(generation_messages)  # type: ignore[attr-defined]
        response = self._get_full_generation_response(  # type: ignore[attr-defined]
            question,
            generation_messages,
            rendered_prompt,
            kwargs,
        )
        if not response.input:
            response.input = rendered_prompt
        if response.prompt_tokens is None:
            prompt_tokens = self._count_completion_text_tokens(response.input)  # type: ignore[attr-defined]
            if prompt_tokens is not None:
                response.prompt_tokens = prompt_tokens
        self.full_generation_response = response  # type: ignore[attr-defined]
        return response

    def _replay(
        self,
        question: str,
        trajectory: Response,
        kwargs: dict[str, Any],
    ) -> list[MethodResult]:
        for method in self.methods:
            method.reset()

        active: list[EarlyExitMethod] = list(self.methods)
        if not active:
            return []
        results: list[MethodResult] = []
        self._exit_counts: dict[str, int] = {}

        thinking_end_char = self._thinking_end_char(trajectory)

        if trajectory.logprobs:
            self._replay_over_tokens(
                question, trajectory, thinking_end_char, active, results, kwargs
            )
        else:
            self._replay_over_content(
                question, trajectory, thinking_end_char, active, results, kwargs
            )

        self._generate_stopped_answers(question, results, kwargs)

        # Methods that never stopped fall back on y_full.
        if active:
            full_prefix, full_answer, full_finish = self._extract_full_answer(
                trajectory, kwargs, thinking_end_char
            )
            trajectory_position = int(
                getattr(trajectory, "thinking_tokens", trajectory.token_count())
            )
            fallback_final_response = self._fallback_final_response
            answer_tokens = int(
                getattr(
                    trajectory,
                    "answer_tokens",
                    fallback_final_response.token_count()
                    if fallback_final_response is not None
                    else 0,
                )
            )
            final_prompt_tokens = getattr(
                trajectory,
                "final_answer_prompt_tokens",
                fallback_final_response.prompt_tokens
                if fallback_final_response is not None
                else None,
            )
            fallback_logs = (
                {"final_answer_truncated": True}
                if self._fallback_final_truncated
                else {}
            )
            for method in active:
                results.append(
                    MethodResult(
                        method_name=method.name,
                        position=trajectory_position,
                        stopped_early=False,
                        verifier_score=None,
                        reasoning_prefix=full_prefix,
                        final_answer=full_answer,
                        finish_reason=full_finish,
                        prompt_tokens=final_prompt_tokens,
                        thinking_tokens=trajectory_position,
                        answer_tokens=answer_tokens,
                        completion_tokens=trajectory_position + answer_tokens,
                        verifier_prompt_tokens=self._verifier_prompt_tokens(method),
                        verifier_completion_tokens=self._verifier_completion_tokens(method),
                        state=dict(method.state),
                        logs={**dict(method.logs), **fallback_logs},
                    )
                )

        if not self.exit:
            for result in results:
                if not result.stopped_early:
                    result.mark_continued(None)
        return results

    # ------------------------------------------------------------------ #
    # Iteration strategies.                                              #
    # ------------------------------------------------------------------ #

    def _replay_over_tokens(
        self,
        question: str,
        trajectory: Response,
        thinking_end_char: int,
        active: list[EarlyExitMethod],
        results: list[MethodResult],
        kwargs: dict[str, Any],
    ) -> None:
        token_texts = trajectory.token_texts()
        prefix_chars = 0
        for t, token_lp in enumerate(trajectory.logprobs):
            latest_token = token_lp.token or ""
            prefix_chars += len(latest_token)
            if prefix_chars > thinking_end_char:
                break
            prefix = self._prefix_response(trajectory, t + 1, prefix_chars, token_texts)
            self._evaluate_active_methods(
                question, prefix, trajectory, latest_token, t + 1, active, results, kwargs
            )
            if not active:
                return

    def _replay_over_content(
        self,
        question: str,
        trajectory: Response,
        thinking_end_char: int,
        active: list[EarlyExitMethod],
        results: list[MethodResult],
        kwargs: dict[str, Any],
    ) -> None:
        """Fallback when the provider gave us no logprobs.

        We walk the content one character at a time. Callers that need
        token-precise positions should use a provider that returns logprobs.
        """
        content = trajectory.content[:thinking_end_char]
        for i, ch in enumerate(content, start=1):
            prefix = Response(
                input=trajectory.input,
                content=content[:i],
                completion_tokens=None,
            )
            self._evaluate_active_methods(
                question, prefix, trajectory, ch, i, active, results, kwargs
            )
            if not active:
                return

    def _prefix_response(
        self,
        trajectory: Response,
        token_end: int,
        char_end: int,
        token_texts: list[str],
    ) -> Response:
        return Response(
            input=trajectory.input,
            content=trajectory.content[:char_end],
            logprobs=list(trajectory.logprobs[:token_end]),
            completion_tokens=token_end,
        )

    def _evaluate_active_methods(
        self,
        question: str,
        prefix: Response,
        trajectory: Response,
        latest_token: str,
        position: int,
        active: list[EarlyExitMethod],
        results: list[MethodResult],
        kwargs: dict[str, Any],
    ) -> None:
        fired: list[tuple[EarlyExitMethod, VerifierContext]] = []
        for method in list(active):
            ctx = VerifierContext(
                question=question,
                prefix=prefix,
                latest_token=latest_token,
                position=position,
                agent=self,  # type: ignore[arg-type]
                kwargs=kwargs,
                state=method.state,
                logs=method.logs,
                trajectory=trajectory,
                method_name=method.name,
            )
            if not method.should_check(ctx):
                continue
            fired.append((method, ctx))

        self._register_temp_answer_need(position, [method for method, _ctx in fired], kwargs)

        for method, ctx in fired:
            if method not in active:
                continue
            logger.info(
                "checking method=%s position=%s latest_token=%r",
                method.name,
                position,
                latest_token,
            )
            try:
                verifier_output = method.verify(ctx)
            except RuntimeError as exc:
                if not _is_already_borrowed_error(exc):
                    raise
                logger.warning(
                    "Skipping verifier check after tokenizer contention "
                    "method=%s position=%s error=%s",
                    method.name,
                    position,
                    exc,
                )
                method.logs.setdefault("runtime_errors", []).append(
                    {
                        "position": position,
                        "type": exc.__class__.__name__,
                        "message": str(exc),
                        "action": "skipped_check",
                    }
                )
                continue
            if not method.should_stop(verifier_output, ctx):
                continue

            logger.info(
                "method=%s stopped position=%s score=%s",
                method.name,
                position,
                verifier_output,
            )
            results.append(
                MethodResult(
                    method_name=method.name,
                    position=0 if method.no_thinking else position,
                    stopped_early=True,
                    verifier_score=_as_float(verifier_output),
                    reasoning_prefix="" if method.no_thinking else prefix.content,
                    final_answer="",
                    finish_reason=None,
                    prompt_tokens=None,
                    thinking_tokens=0 if method.no_thinking else position,
                    answer_tokens=0,
                    completion_tokens=0 if method.no_thinking else position,
                    verifier_prompt_tokens=self._verifier_prompt_tokens(method),
                    verifier_completion_tokens=self._verifier_completion_tokens(method),
                    state=dict(method.state),
                    logs=dict(method.logs),
                )
            )
            if self.exit:
                active.remove(method)
            else:
                count = self._exit_counts.get(method.name, 0) + 1
                self._exit_counts[method.name] = count
                results[-1].mark_continued(count)
            if not active:
                return

    def _register_temp_answer_need(
        self,
        position: int,
        methods: list[EarlyExitMethod],
        kwargs: dict[str, Any],
    ) -> None:
        token_needs = [
            getattr(method.verifier, "temp_answer_tokens")
            for method in methods
            if hasattr(method.verifier, "temp_answer_tokens")
        ]
        logprob_needs = [
            getattr(method.verifier, "temp_answer_logprobs")
            for method in methods
            if hasattr(method.verifier, "temp_answer_logprobs")
        ]
        if logprob_needs:
            self._temp_answer_logprob_needs[position] = max(
                int(need) for need in logprob_needs
            )
        if not token_needs:
            return
        if any(need is None for need in token_needs):
            full_budget = self._configured_temp_answer_token_limit(kwargs)
            if full_budget is not None:
                self._temp_answer_token_needs[position] = self.scale_probe_tokens(
                    full_budget, kwargs
                )
            return
        self._temp_answer_token_needs[position] = self.scale_probe_tokens(
            max(int(need) for need in token_needs), kwargs
        )

    def scale_probe_tokens(self, probe_tokens: int, kwargs: dict[str, Any]) -> int:
        if self._is_code_task(kwargs):
            scaled_tokens = int(probe_tokens) * self.code_probe_token_multiplier
            max_tokens = self._effective_temp_answer_token_limit(kwargs)
            return (
                min(scaled_tokens, max_tokens)
                if max_tokens is not None
                else scaled_tokens
            )
        return int(probe_tokens)

    def _effective_temp_answer_token_limit(
        self, kwargs: dict[str, Any]
    ) -> int | None:
        merged_kwargs = {**self.completion_kwargs, **kwargs}  # type: ignore[attr-defined]
        max_tokens = self._int_or_none(
            merged_kwargs.get("max_temp_answer_completion_token")
        )
        if max_tokens is not None and self._is_code_task(kwargs):
            max_tokens *= self.code_probe_token_multiplier
        return max_tokens

    def scale_threshold(self, threshold: float, kwargs: dict[str, Any]) -> float:
        if self._is_code_task(kwargs):
            return float(threshold) * self.code_threshold_multiplier
        return float(threshold)

    def _configured_temp_answer_token_limit(self, kwargs: dict[str, Any]) -> int | None:
        merged_kwargs = {**self.completion_kwargs, **kwargs}  # type: ignore[attr-defined]
        for key in (
            "max_temp_answer_completion_token",
            "max_final_answer_completion_token",
            "max_completion_token",
            "max_competion_token",
            "max_tokens",
        ):
            value = self._int_or_none(merged_kwargs.get(key))  # type: ignore[attr-defined]
            if value is not None:
                return value
        return None

    # ------------------------------------------------------------------ #
    # Answer generation & bookkeeping helpers.                           #
    # ------------------------------------------------------------------ #

    def _generate_early_answer(
        self,
        prefix: Response,
        kwargs: dict[str, Any],
    ) -> Response:
        """Fresh y_i after the replay pass for a stopped prefix."""
        input_text = (
            prefix.get_generation_content()
            + f"You are out of thinking budget {self.think_end_token()}"
        )
        prefix_with_tau = Response(
            input=prefix.input,
            content=input_text[len(prefix.input):],
            logprobs=list(prefix.logprobs),
            completion_tokens=prefix.completion_tokens,
        )
        merged_kwargs = {**self.completion_kwargs, **kwargs}  # type: ignore[attr-defined]
        finalize_kwargs = self._finalize_kwargs_with_remaining_budget(  # type: ignore[attr-defined]
            prefix_with_tau,
            merged_kwargs,
        )
        # Use BaseAgent's finalize path (real completion call), NOT the
        # replay path inherited from BaseAgentFullGeneration / *Reuse*.
        response = BaseAgent._generate_final_answer(
            self,  # type: ignore[arg-type]
            input_text,
            **finalize_kwargs,
        )
        final_request_kwargs = self._completion_kwargs_for_phase(  # type: ignore[attr-defined]
            "final",
            finalize_kwargs,
        )
        requested_final_tokens = self._int_or_none(
            final_request_kwargs.get("max_completion_token")
        )
        if (
            requested_final_tokens is not None
            and requested_final_tokens > MAX_FINAL_COMPLETION_REQUEST_TOKENS
        ):
            final_request_kwargs["max_completion_token"] = (
                MAX_FINAL_COMPLETION_REQUEST_TOKENS
            )
        object.__setattr__(
            response,
            "final_answer_truncated",
            self._response_reached_token_limit(response, final_request_kwargs),
        )
        return response

    def _generate_stopped_answers(
        self,
        question: str,
        results: list[MethodResult],
        kwargs: dict[str, Any],
    ) -> None:
        pending = [
            result
            for result in results
            if result.stopped_early and result.final_answer == ""
        ]
        answer_cache: dict[tuple[int | None, str, bool], Response] = {}
        batch_size = max(1, int(getattr(self, "final_answer_batch_size", 16)))
        for start in range(0, len(pending), batch_size):
            for result in pending[start : start + batch_size]:
                method = next(
                    (
                        item for item in self.methods
                        if item.name == (result.base_method or result.method_name)
                    ),
                    None,
                )
                no_thinking = bool(
                    getattr(self.chat_template, "is_gemma4", False)
                    and method is not None
                    and method.no_thinking
                )
                cache_key = (result.position, result.reasoning_prefix, no_thinking)
                outcome = answer_cache.get(cache_key)
                if outcome is None:
                    if no_thinking:
                        messages = BaseAgent.build_generation_messages(self, question)  # type: ignore[arg-type]
                        input_text = self.chat_template.render(  # type: ignore[attr-defined]
                            messages,
                            enable_thinking=False,
                        )
                        outcome = BaseAgent._generate_final_answer(
                            self,  # type: ignore[arg-type]
                            input_text,
                            **self._with_default_final_answer_budget(kwargs),
                        )
                    else:
                        prefix = Response(
                            input=self.full_generation_response.input,  # type: ignore[attr-defined]
                            content=result.reasoning_prefix,
                            completion_tokens=result.thinking_tokens,
                        )
                        outcome = self._generate_early_answer(prefix, kwargs)
                    answer_cache[cache_key] = outcome

                answer_response = outcome
                if getattr(answer_response, "final_answer_truncated", False):
                    result.logs = {
                        **result.logs,
                        "final_answer_truncated": True,
                    }
                result.final_answer = answer_response.content
                result.finish_reason = answer_response.finish_reason
                result.prompt_tokens = answer_response.prompt_tokens
                result.answer_tokens = answer_response.completion_tokens or 0
                result.completion_tokens = (result.thinking_tokens or 0) + (
                    answer_response.completion_tokens or 0
                )

    def _extract_full_answer(
        self,
        trajectory: Response,
        kwargs: dict[str, Any],
        thinking_end_char: int,
    ) -> tuple[str, str, str | None]:
        tau = self.think_end_token()
        content = trajectory.content
        if thinking_end_char < len(content):
            reasoning = content[:thinking_end_char]
            y_full = content[thinking_end_char:]
            if y_full.startswith(tau):
                y_full = y_full[len(tau):]
            return reasoning, y_full, trajectory.finish_reason

        # No natural end-of-thinking in the trajectory: run a fresh finalize
        # so the "no early exit" methods still get a comparable y_full.
        finalized = self._generate_early_answer(
            trajectory,
            self._with_default_final_answer_budget(kwargs),
        )
        self._fallback_final_truncated = bool(
            getattr(finalized, "final_answer_truncated", False)
        )
        self._fallback_final_response = finalized
        reasoning = trajectory.content
        y_full = finalized.content
        return reasoning, y_full, finalized.finish_reason

    def _with_default_final_answer_budget(
        self,
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        final_kwargs = dict(kwargs)
        merged_kwargs = {**self.completion_kwargs, **final_kwargs}  # type: ignore[attr-defined]
        if self._int_or_none(
            merged_kwargs.get("max_final_answer_completion_token")
        ) is None:
            final_kwargs["max_final_answer_completion_token"] = (
                MAX_FINAL_COMPLETION_REQUEST_TOKENS
            )
        return final_kwargs

    def _thinking_end_char(self, trajectory: Response) -> int:
        tau = self.think_end_token()
        idx = trajectory.content.find(tau)
        return idx if idx != -1 else len(trajectory.content)

    def _temp_answer_suffix(self, kwargs: dict[str, Any]) -> str:
        tau = self.think_end_token()
        if self._is_code_task(kwargs):
            return f"You are out of thinking budget {tau} the final code snippet is \n\n"
        return f"You are out of thinking budget {tau} the final answer is \\boxed"

    def _reached_thinking_limit(
        self,
        trajectory: Response,
        kwargs: dict[str, Any],
    ) -> bool:
        max_completion_token = self._configured_generation_token_limit(kwargs)
        return (
            max_completion_token is not None
            and trajectory.completion_tokens is not None
            and trajectory.completion_tokens >= max_completion_token
        )

    def _configured_generation_token_limit(self, kwargs: dict[str, Any]) -> int | None:
        merged_kwargs = {**self.completion_kwargs, **kwargs}  # type: ignore[attr-defined]
        for key in ("max_completion_token", "max_competion_token", "max_tokens"):
            value = self._int_or_none(merged_kwargs.get(key))  # type: ignore[attr-defined]
            if value is not None:
                return value
        return None

    def _response_reached_token_limit(
        self,
        response: Response,
        completion_kwargs: dict[str, Any],
    ) -> bool:
        token_limit = self._completion_token_limit_from_kwargs(completion_kwargs)
        return (
            token_limit is not None
            and response.completion_tokens is not None
            and response.completion_tokens == token_limit
        )

    def _completion_token_limit_from_kwargs(self, completion_kwargs: dict[str, Any]) -> int | None:
        for key in ("max_completion_token", "max_competion_token", "max_tokens"):
            value = self._int_or_none(completion_kwargs.get(key))  # type: ignore[attr-defined]
            if value is not None:
                return value
        return None

    @staticmethod
    def _is_code_task(kwargs: dict[str, Any]) -> bool:
        if kwargs.get("is_code") is not None:
            return bool(kwargs.get("is_code"))
        dataset = str(kwargs.get("dataset") or kwargs.get("dataloader") or "").lower()
        evaluator_name = str(kwargs.get("evaluator_name") or "").lower()
        evaluator = kwargs.get("evaluator")
        evaluator_module = str(getattr(evaluator, "__module__", "")).lower()
        code_names = ("code", "humaneval", "livecodebench")
        return any(name in dataset for name in code_names) or any(
            name in evaluator_name or name in evaluator_module for name in code_names
        )

    @staticmethod
    def _verifier_prompt_tokens(method: EarlyExitMethod) -> int:
        state = method.state
        return (
            int(state.get("_probe_prompt_tokens", 0))
            + int(state.get("_sufficiency_prompt_tokens", 0))
            + int(state.get("_dynasor_prompt_tokens", 0))
            + int(state.get("_answer_coverage_prompt_tokens", 0))
            + int(state.get("_codestop_prompt_tokens", 0))
            + int(state.get("_puma_probe_prompt_tokens", 0))
            + int(state.get("_eat_prompt_tokens", 0))
        )

    @staticmethod
    def _verifier_completion_tokens(method: EarlyExitMethod) -> int:
        state = method.state
        return (
            int(state.get("_probe_completion_tokens", 0))
            + int(state.get("_sufficiency_completion_tokens", 0))
            + int(state.get("_dynasor_completion_tokens", 0))
            + int(state.get("_answer_coverage_completion_tokens", 0))
            + int(state.get("_codestop_completion_tokens", 0))
            + int(state.get("_puma_probe_completion_tokens", 0))
            + int(state.get("_eat_completion_tokens", 0))
        )


def _resolve_methods(
    methods: Iterable[EarlyExitMethod] | Iterable[dict[str, Any]] | None,
) -> list[EarlyExitMethod]:
    """Accept either prebuilt methods or spec dicts (for YAML configs).

    Spec format::

        {
            "factory": "replay_eval.methods.build_deer_method",
            "name": "deer",
            "kwargs": {"threshold": 0.95, "probe_tokens": 20},
        }

    ``factory`` may also point at any callable returning an ``EarlyExitMethod``.
    """
    if not methods:
        return []
    resolved: list[EarlyExitMethod] = []
    for item in methods:
        if isinstance(item, EarlyExitMethod):
            resolved.append(item)
            continue
        if isinstance(item, dict):
            factory_path = item.get("factory")
            if not factory_path:
                raise ValueError(
                    "Method spec is missing required 'factory' dotted path."
                )
            factory = import_from_path(factory_path)
            kwargs = dict(item.get("kwargs") or {})
            if "name" in item and "name" not in kwargs:
                kwargs["name"] = item["name"]
            method = factory(**kwargs)
            if not isinstance(method, EarlyExitMethod):
                raise TypeError(
                    f"Factory {factory_path!r} did not return an EarlyExitMethod."
                )
            resolved.append(method)
            continue
        raise TypeError(
            f"Unsupported method entry: {item!r}. "
            "Expected an EarlyExitMethod or a dict spec."
        )
    return resolved


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_already_borrowed_error(exc: BaseException) -> bool:
    """Identify the transient fast-tokenizer re-entrancy failure."""
    return "already borrowed" in str(exc).casefold()


class ReplayEvaluatorAgent(_ReplayEvaluatorMixin, BaseAgentReuseGeneration):
    """Reuse-mode replay evaluator.

    Requires ``reuse_generation_content`` (or ``reuse_generation_response``)
    to be passed via ``solve()`` kwargs, exactly like ``DTSRAgentReuseGeneration``.
    """

    def __init__(
        self,
        llm_name: str,
        methods: Iterable[EarlyExitMethod] | None = None,
        verify_batch_token: int = 16,
        final_answer_batch_size: int = 16,
        code_probe_token_multiplier: int = 1,
        code_threshold_multiplier: float = 1.0,
        exit: bool = True,
        **kwargs: Any,
    ):
        super().__init__(llm_name=llm_name, **kwargs)
        self._init_replay_config(
            methods,
            verify_batch_token,
            final_answer_batch_size,
            code_probe_token_multiplier,
            code_threshold_multiplier,
            exit,
        )


class ReplayEvaluatorAgentFullGeneration(_ReplayEvaluatorMixin, BaseAgentFullGeneration):
    """Full-generation replay evaluator: rolls the trajectory itself."""

    def __init__(
        self,
        llm_name: str,
        methods: Iterable[EarlyExitMethod] | None = None,
        verify_batch_token: int = 16,
        final_answer_batch_size: int = 16,
        code_probe_token_multiplier: int = 1,
        code_threshold_multiplier: float = 1.0,
        exit: bool = True,
        **kwargs: Any,
    ):
        super().__init__(llm_name=llm_name, **kwargs)
        self._init_replay_config(
            methods,
            verify_batch_token,
            final_answer_batch_size,
            code_probe_token_multiplier,
            code_threshold_multiplier,
            exit,
        )

    def _get_full_generation_response(
        self,
        question: str,
        messages: list[dict[str, str]],
        rendered_prompt: str,
        kwargs: dict[str, Any],
    ) -> Response:
        response = self.full_generation(
            messages,
            **full_generation_kwargs(self.completion_kwargs, kwargs),
        )
        response.input = rendered_prompt
        return convert_chat_full_generation_to_replay_response(
            response,
            template=getattr(self.chat_template, "template", "hermes"),
            think_start_token=self.chat_template.think_start_token or "<think>",
            think_end_token=self.chat_template.think_end_token or "</think>",
        )
