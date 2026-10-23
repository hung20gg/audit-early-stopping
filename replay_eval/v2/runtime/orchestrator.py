from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import Iterable
from typing import Any

from methods.response import LogProb, Response, TopLogProb
from replay_eval.method import EarlyExitMethod, MethodResult, VerifierContext
from replay_eval.replay import _resolve_methods
from replay_eval.verifiers import (
    EATVerifier,
    EntrocutVerifier,
    LLMSufficiencyVerifier,
    PUMAVerifier,
)
from replay_eval.v2.engine.cache import RequestCache
from replay_eval.v2.engine.access import EngineAccessGate
from replay_eval.v2.engine.extraction import ExtractionConnector, ExtractionResult
from replay_eval.v2.engine.types import RequestKind, RequestResult, RequestSpec
from replay_eval.v2.runtime.adapter import ReplayCompatibilityAdapter
from replay_eval.v2.runtime.metrics import RunMetrics


logger = logging.getLogger(__name__)


class AsyncReplayOrchestrator:
    """Stateful per-example replay over a shared asynchronous engine."""

    def __init__(
        self,
        *,
        engine: Any,
        renderer: Any,
        methods: Iterable[EarlyExitMethod] | Iterable[dict[str, Any]],
        extraction_connector: ExtractionConnector | None = None,
        completion_kwargs: dict[str, Any] | None = None,
        stop_tokens: list[str] | None = None,
        code_probe_token_multiplier: int = 1,
        code_threshold_multiplier: float = 1.0,
        verify_batch_token: int = 16,
        min_interval: int = 0,
        exit: bool = True,
        metrics: RunMetrics | None = None,
        engine_access_gate: EngineAccessGate | None = None,
    ) -> None:
        if not isinstance(exit, bool):
            raise ValueError("replay_eval.exit must be a boolean")
        self.exit = exit
        self.engine = engine
        self.renderer = renderer
        self.methods = _resolve_methods(methods)
        for method in self.methods:
            if isinstance(method.verifier, EntrocutVerifier):
                method.verifier.probe_logprobs = 50
            if getattr(method.verifier, "custom_scoring_handler", None) is not None:
                raise ValueError(
                    f"V2 method {method.name!r} requests a separate custom LLM; "
                    "native v2 execution requires llm='same'"
                )
            if getattr(method.verifier, "same_chat_template", None) is not None:
                raise ValueError(
                    f"V2 method {method.name!r} overrides the canonical chat template"
                )
        self.extraction_connector = extraction_connector
        self.completion_kwargs = dict(completion_kwargs or {})
        self.cache = RequestCache()
        self.metrics = metrics or RunMetrics()
        self.engine_access_gate = engine_access_gate or EngineAccessGate()
        self.llm_calls = 0
        self.llm_call_seconds = 0.0
        self.cache_salt = f"replay-v2-{uuid.uuid4().hex}"
        if min_interval < 0:
            raise ValueError("min_interval must be >= 0")
        self.min_interval = int(min_interval)
        self._last_checkpoint_positions: dict[int, int] = {}
        self.agent = ReplayCompatibilityAdapter(
            renderer.tokenizer,
            renderer,
            completion_kwargs=self.completion_kwargs,
            stop_tokens=stop_tokens,
            code_probe_token_multiplier=code_probe_token_multiplier,
            code_threshold_multiplier=code_threshold_multiplier,
            verify_batch_token=verify_batch_token,
        )

    async def evaluate(
        self,
        *,
        example_id: str,
        question: str,
        messages: list[dict[str, Any]],
        trajectory: Response | None = None,
        kwargs: dict[str, Any] | None = None,
    ) -> tuple[Response, list[MethodResult], dict[str, Any]]:
        kwargs = dict(kwargs or {})
        self.llm_calls = 0
        self.llm_call_seconds = 0.0
        self.agent.reset()
        self._last_checkpoint_positions.clear()
        for method in self.methods:
            method.reset()
        started = time.perf_counter()
        extraction: ExtractionResult | None = None
        hidden_pair: tuple[Any, Any] | None = None
        timing = {
            "rendering_alignment_seconds": 0.0,
            "hidden_extraction_seconds": 0.0,
            "hidden_loading_seconds": 0.0,
            "hidden_processing_seconds": 0.0,
            "terminator_seconds": 0.0,
            "thought_calibration_seconds": 0.0,
            "replay_seconds": 0.0,
            "final_prompt_build_seconds": 0.0,
        }
        try:
            rendering_started = time.perf_counter()
            source_ids = self.renderer.source(messages)
            source_text = self.renderer.source_text(messages)
            if trajectory is None:
                logger.info(
                    "Full-trace generation started example=%s reason=missing_trajectory",
                    example_id,
                )
                full_spec = RequestSpec.create(
                    source_ids,
                    {**self.completion_kwargs, "logprobs": 50},
                    0,
                    RequestKind.FULL_TRACE,
                    example_id=example_id,
                )
                full_result = await self._request(full_spec)
                trajectory = full_result.response
                trajectory.input = source_text
                object.__setattr__(
                    trajectory, "output_token_ids", full_result.output_token_ids
                )
                logger.info(
                    "Full-trace generation finished example=%s output_tokens=%s",
                    example_id,
                    len(full_result.output_token_ids),
                )
            elif not trajectory.input:
                trajectory.input = source_text
            output_ids = self._trajectory_token_ids(trajectory)
            trace = self.renderer.full_trace(
                messages,
                trajectory.content,
                stored_input=trajectory.input,
                output_token_ids=output_ids or None,
            )
            timing["rendering_alignment_seconds"] = (
                time.perf_counter() - rendering_started
            )
            self.metrics.rendering_alignment_seconds += timing[
                "rendering_alignment_seconds"
            ]
            if output_ids and list(trace.output_token_ids) != output_ids:
                raise ValueError(
                    "Stored output token IDs differ from canonical HF assistant rendering"
                )
            if not output_ids:
                output_ids = list(trace.output_token_ids)

            if self.extraction_connector is not None:
                extraction_started = time.perf_counter()
                logger.info(
                    "Hidden-state extraction waiting example=%s tokens=%s",
                    example_id,
                    len(output_ids),
                )
                async with self.engine_access_gate.exclusive():
                    logger.info(
                        "Hidden-state extraction started example=%s tokens=%s wait=%.2fs",
                        example_id,
                        len(output_ids),
                        time.perf_counter() - extraction_started,
                    )
                    extraction = await self.extraction_connector.extract(
                        engine=self.engine,
                        full_token_ids=list(trace.full_token_ids),
                        output_start=len(trace.source_token_ids),
                        output_token_ids=output_ids,
                        example_id=example_id,
                        cache_salt=self.cache_salt,
                    )
                logger.info(
                    "Hidden-state extraction finished example=%s elapsed=%.2fs",
                    example_id,
                    time.perf_counter() - extraction_started,
                )
                timing["hidden_extraction_seconds"] = (
                    time.perf_counter() - extraction_started
                )
                self.metrics.hidden_extraction_seconds += extraction.extraction_seconds
                self.metrics.hidden_normalization_seconds += (
                    extraction.normalization_seconds
                )
                loading_started = time.perf_counter()
                hidden_trace = extraction.load_trace()
                timing["hidden_loading_seconds"] = time.perf_counter() - loading_started
                hidden_pair = (hidden_trace.raw, hidden_trace.final_normalized)
                self.metrics.hidden_loading_seconds += extraction.loading_seconds
                if any(
                    getattr(value, "shape", [0])[0] != len(output_ids)
                    for value in hidden_pair
                ):
                    raise ValueError(
                        "Hidden tensors and canonical output positions do not align"
                    )
                self.agent.hidden_states = hidden_pair
                self.agent.hidden_trace = hidden_trace
                self._apply_observed_logprobs(
                    trajectory, output_ids, extraction.observed_logprobs
                )

                processing_started = time.perf_counter()
                for method in self.methods:
                    prepare = getattr(method.verifier, "prepare_hidden_trace", None)
                    if not callable(prepare):
                        continue
                    phase_started = time.perf_counter()
                    prepare(hidden_trace)
                    elapsed = time.perf_counter() - phase_started
                    phase = self._hidden_method_phase(method)
                    timing[phase] += elapsed
                    setattr(self.metrics, phase, getattr(self.metrics, phase) + elapsed)
                hidden_verifiers = [
                    method.verifier
                    for method in self.methods
                    if getattr(method.verifier, "uses_hidden_trace", False)
                ]
                if hidden_verifiers and all(
                    getattr(verifier, "permits_token_state_release", False)
                    for verifier in hidden_verifiers
                ):
                    hidden_trace.release_token_states()
                    self.agent.hidden_states = None
                    hidden_pair = None
                timing["hidden_processing_seconds"] = (
                    time.perf_counter() - processing_started
                )
                self.metrics.hidden_processing_seconds += timing[
                    "hidden_processing_seconds"
                ]
                cleanup_started = time.perf_counter()
                extraction.cleanup()
                extraction = None
                self.metrics.cleanup_seconds += time.perf_counter() - cleanup_started

            self._validate_observed_logprobs(trajectory, output_ids)
            replay_started = time.perf_counter()
            results = await self._replay(
                example_id,
                question,
                messages,
                trajectory,
                list(trace.source_token_ids),
                output_ids,
                kwargs,
                timing,
            )
            timing["replay_seconds"] = time.perf_counter() - replay_started
            self.metrics.replay_seconds += timing["replay_seconds"]
            object.__setattr__(trajectory, "method_results", results)
            object.__setattr__(
                trajectory, "method_results_dicts", [item.to_dict() for item in results]
            )
            self.metrics.scan_tokens += len(output_ids)
            timing["llm_calls"] = self.llm_calls
            timing["llm_call_seconds"] = self.llm_call_seconds
            timing["avg_llm_call_seconds"] = (
                self.llm_call_seconds / self.llm_calls if self.llm_calls else 0.0
            )
            return (
                trajectory,
                results,
                {"example_seconds": time.perf_counter() - started, **timing},
            )
        finally:
            self.agent.hidden_states = None
            self.agent.hidden_trace = None
            hidden_pair = None
            if extraction is not None:
                cleanup_started = time.perf_counter()
                extraction.cleanup()
                self.metrics.cleanup_seconds += time.perf_counter() - cleanup_started

    async def rerun_final_answers(
        self,
        *,
        example_id: str,
        messages: list[dict[str, Any]],
        trajectory: Response,
        results: list[dict[str, Any]],
        kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Refill removed method answers without replaying any stop method."""
        kwargs = dict(kwargs or {})
        started = time.perf_counter()
        calls_before = self.llm_calls
        call_seconds_before = self.llm_call_seconds
        source_text = self.renderer.source_text(messages)
        if not trajectory.input:
            trajectory.input = source_text
        output_ids = self._trajectory_token_ids(trajectory)
        trace = self.renderer.full_trace(
            messages,
            trajectory.content,
            stored_input=trajectory.input,
            output_token_ids=output_ids or None,
        )
        if not output_ids:
            output_ids = list(trace.output_token_ids)

        pending: list[MethodResult] = []
        for item in results:
            position = int(item.get("position", 0))
            reasoning_prefix = self.renderer.tokenizer.decode(
                output_ids[:position],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            pending.append(
                MethodResult(
                    method_name=str(item.get("method") or item.get("method_name")),
                    base_method=item.get("base_method"),
                    exit=item.get("exit"),
                    exit_index=item.get("exit_index"),
                    position=position,
                    stopped_early=bool(item.get("stopped_early", True)),
                    verifier_score=item.get("verifier_score"),
                    reasoning_prefix=reasoning_prefix,
                    final_answer="",
                    thinking_tokens=item.get("thinking_tokens", position),
                    verifier_prompt_tokens=int(item.get("verifier_prompt_tokens", 0) or 0),
                    verifier_completion_tokens=int(
                        item.get("verifier_completion_tokens", 0) or 0
                    ),
                    state=dict(item.get("state") or {}),
                    logs=dict(item.get("logs") or {}),
                )
            )

        timing = {"final_prompt_build_seconds": 0.0}
        await self._fill_answers(
            example_id,
            messages,
            list(trace.source_token_ids),
            output_ids,
            pending,
            kwargs,
            timing,
        )
        for item, refilled in zip(results, pending):
            item.update(
                {
                    "final_answer": refilled.final_answer,
                    "finish_reason": refilled.finish_reason,
                    "prompt_tokens": refilled.prompt_tokens,
                    "answer_tokens": refilled.answer_tokens,
                    "completion_tokens": refilled.completion_tokens,
                }
            )
            logs = dict(item.get("logs") or {})
            logs.pop("final_answer_truncated", None)
            if logs:
                item["logs"] = logs
            else:
                item.pop("logs", None)
        answer_calls = self.llm_calls - calls_before
        answer_call_seconds = self.llm_call_seconds - call_seconds_before
        return {
            "final_answer_only": True,
            "final_answers_rerun": len(pending),
            "final_prompt_build_seconds": timing["final_prompt_build_seconds"],
            "example_seconds": time.perf_counter() - started,
            "llm_calls": answer_calls,
            "llm_call_seconds": answer_call_seconds,
            "avg_llm_call_seconds": (
                answer_call_seconds / answer_calls if answer_calls else 0.0
            ),
        }

    async def _replay(
        self,
        example_id: str,
        question: str,
        messages: list[dict[str, Any]],
        trajectory: Response,
        source_token_ids: list[int],
        output_ids: list[int],
        kwargs: dict[str, Any],
        timing: dict[str, float],
    ) -> list[MethodResult]:
        active = list(self.methods)
        results: list[MethodResult] = []
        exit_counts: dict[str, int] = {}
        thinking_end = self.agent._thinking_end_char(trajectory)
        prefix_chars = 0
        prefix_logprobs: list[LogProb] = []
        indexed_prefix_placeholder = Response()
        for index, logprob in enumerate(trajectory.logprobs):
            latest = logprob.token or self.renderer.decode_token(output_ids[index])
            prefix_chars += len(latest)
            if prefix_chars > thinking_end:
                break
            prefix_logprobs.append(logprob)
            prefix: Response | None = None

            def materialize_prefix() -> Response:
                nonlocal prefix
                if prefix is None:
                    prefix = Response(
                        input=trajectory.input,
                        content=trajectory.content[:prefix_chars],
                        logprobs=list(prefix_logprobs),
                        completion_tokens=index + 1,
                    )
                return prefix

            fired: list[tuple[EarlyExitMethod, VerifierContext]] = []
            for method in list(active):
                ctx = VerifierContext(
                    question=question,
                    # Built-in checkpoints use the indexed replay path below,
                    # so avoid copying the growing content/logprob prefix until
                    # a checkpoint actually fires. Custom checkpoints receive
                    # the fully materialized prefix through the fallback.
                    prefix=indexed_prefix_placeholder,
                    latest_token=latest,
                    position=index + 1,
                    agent=self.agent,
                    kwargs=kwargs,
                    state=method.state,
                    logs=method.logs,
                    trajectory=trajectory,
                    method_name=method.name,
                )
                should_check = method.checkpoint.replay_check(
                    ctx,
                    full_content=trajectory.content,
                    prefix_chars=prefix_chars,
                )
                if should_check is None:
                    ctx.prefix = materialize_prefix()
                    should_check = method.should_check(ctx)
                if should_check:
                    if not self._checkpoint_interval_allows(method, ctx.position):
                        continue
                    ctx.prefix = materialize_prefix()
                    fired.append((method, ctx))
            if not fired:
                continue
            await self._schedule_checkpoint(example_id, messages, fired, kwargs)
            for method, ctx in fired:
                verifier_started = time.perf_counter()
                value = method.verify(ctx)
                elapsed = time.perf_counter() - verifier_started
                self.metrics.verifier_seconds += elapsed
                phase = self._hidden_method_phase(method, required=False)
                if phase is not None:
                    timing[phase] += elapsed
                    setattr(self.metrics, phase, getattr(self.metrics, phase) + elapsed)
                if not method.should_stop(value, ctx):
                    continue
                results.append(self._stopped_result(method, ctx, value))
                if self.exit:
                    active.remove(method)
                else:
                    count = exit_counts.get(method.name, 0) + 1
                    exit_counts[method.name] = count
                    results[-1].mark_continued(count)
            if not active:
                break

        if active:
            reasoning, answer = self._full_answer(trajectory)
            position = self._thinking_token_count(trajectory, thinking_end)
            if position == 0:
                position = min(len(output_ids), len(trajectory.logprobs))
            for method in active:
                results.append(
                    MethodResult(
                        method_name=method.name,
                        position=position,
                        stopped_early=False,
                        verifier_score=None,
                        reasoning_prefix=reasoning,
                        final_answer=answer,
                        finish_reason=trajectory.finish_reason,
                        thinking_tokens=position,
                        answer_tokens=max(0, len(output_ids) - position),
                        completion_tokens=len(output_ids),
                        verifier_prompt_tokens=self._verifier_prompt_tokens(method),
                        verifier_completion_tokens=self._verifier_completion_tokens(
                            method
                        ),
                        state=dict(method.state),
                        logs=dict(method.logs),
                    )
                )
        if not self.exit:
            for result in results:
                if not result.stopped_early:
                    result.mark_continued(None)
        await self._fill_answers(
            example_id,
            messages,
            source_token_ids,
            output_ids,
            [item for item in results if not item.final_answer],
            kwargs,
            timing,
        )
        return results

    def _checkpoint_interval_allows(
        self, method: EarlyExitMethod, position: int
    ) -> bool:
        """Accept the first checkpoint, then require the configured token gap."""
        if self.min_interval <= 0:
            return True
        key = id(method)
        last_position = self._last_checkpoint_positions.get(key)
        if (
            last_position is not None
            and position < last_position + self.min_interval
        ):
            return False
        self._last_checkpoint_positions[key] = position
        return True

    @staticmethod
    def _hidden_method_phase(
        method: EarlyExitMethod, *, required: bool = True
    ) -> str | None:
        name = method.verifier.__class__.__name__
        if name == "TerminatorHiddenStateVerifier":
            return "terminator_seconds"
        if name == "ThoughtCalibrationVerifier":
            return "thought_calibration_seconds"
        if required:
            raise RuntimeError(f"No detailed timing phase for {name}")
        return None

    async def _request(self, spec: RequestSpec) -> RequestResult:
        result = await self.cache.get_or_submit(spec, self._submit_engine)
        self._record_request(spec.kind, result)
        return result

    def _record_request(self, kind: RequestKind, result: RequestResult) -> None:
        self.metrics.add_request(kind, result)
        if not result.cached:
            self.llm_calls += 1
            self.llm_call_seconds += float(result.timing.end_to_end)

    async def _submit_engine(self, spec: RequestSpec) -> RequestResult:
        async with self.engine_access_gate.shared():
            return await self.engine.submit(spec)

    async def _schedule_checkpoint(
        self,
        example_id: str,
        messages: list[dict[str, Any]],
        fired: list[tuple[EarlyExitMethod, VerifierContext]],
        kwargs: dict[str, Any],
    ) -> None:
        normal = [
            pair
            for pair in fired
            if not isinstance(pair[0].verifier, LLMSufficiencyVerifier)
            and self._uses_temp_answer(pair[0])
        ]
        tasks: list[Any] = []
        specs: list[tuple[str, Any, RequestSpec]] = []
        grouped: dict[
            tuple[tuple[int, ...], int],
            tuple[list[tuple[EarlyExitMethod, VerifierContext]], Response],
        ] = {}
        for method, ctx in normal:
            prefix, position = self._temp_answer_target(method, ctx)
            prompt_ids = self._normal_prompt_ids(messages, prefix.content, kwargs)
            key = (tuple(prompt_ids), position)
            if key not in grouped:
                grouped[key] = ([], prefix)
            grouped[key][0].append((method, ctx))
        for (prompt_key, position), (consumers, prefix) in grouped.items():
            token_need, logprob_need, rollout_count, force_start = (
                self._normal_requirements(consumers, kwargs)
            )
            prompt_ids = list(prompt_key)
            self.agent.set_force_start(position, force_start, prefix=prefix)
            total_slots = force_start + rollout_count
            for slot in range(total_slots):
                spec = RequestSpec.create(
                    prompt_ids,
                    self._probe_sampling(token_need, logprob_need, kwargs, slot),
                    position,
                    RequestKind.NORMAL_PROBE,
                    example_id=example_id,
                    cache_slot=slot,
                    cache_salt=self.cache_salt,
                )
                specs.append(("normal", (position, slot, prefix), spec))
                tasks.append(self.cache.get_or_submit(spec, self._submit_engine))
        for method, ctx in fired:
            if not isinstance(method.verifier, LLMSufficiencyVerifier):
                continue
            _prompt_text, prompt_ids = self._dtsr_prompt(method.verifier, ctx)
            spec = RequestSpec.create(
                prompt_ids,
                self._dtsr_sampling(method.verifier, kwargs),
                ctx.position,
                RequestKind.DTSR,
                example_id=example_id,
                cache_salt=self.cache_salt,
            )
            specs.append(("dtsr", prompt_ids, spec))
            tasks.append(self.cache.get_or_submit(spec, self._submit_engine))
        if not tasks:
            return
        outcomes = await asyncio.gather(*tasks)
        for (kind, target, spec), outcome in zip(specs, outcomes):
            self._record_request(spec.kind, outcome)
            if kind == "normal":
                position, slot, prefix = target
                self.agent.put_temp_answer(
                    position, outcome.response, int(slot), prefix=prefix
                )
            else:
                self.agent.put_completion(list(target), outcome.response)

    def _normal_requirements(
        self,
        methods: list[tuple[EarlyExitMethod, VerifierContext]],
        kwargs: dict[str, Any],
    ) -> tuple[int, int, int, int]:
        tokens: list[int] = []
        logprobs: list[int] = []
        rollouts = 1
        has_shared_consumer = False
        for method, _ctx in methods:
            verifier = method.verifier
            need = getattr(verifier, "temp_answer_tokens", None)
            if need is None:
                need = kwargs.get(
                    "max_temp_answer_completion_token",
                    self.completion_kwargs.get("max_temp_answer_completion_token", 256),
                )
            tokens.append(self.agent.scale_probe_tokens(int(need), kwargs))
            logprobs.append(int(getattr(verifier, "temp_answer_logprobs", 1)))
            if isinstance(verifier, EATVerifier):
                rollouts = max(rollouts, int(verifier.entropy_rollouts))
            else:
                has_shared_consumer = True
        force_start = (
            1
            if has_shared_consumer
            and any(
                isinstance(method.verifier, EATVerifier) for method, _ctx in methods
            )
            else 0
        )
        return max(tokens), max(logprobs), rollouts, force_start

    @staticmethod
    def _temp_answer_target(
        method: EarlyExitMethod, ctx: VerifierContext
    ) -> tuple[Response, int]:
        if isinstance(method.verifier, PUMAVerifier):
            return method.verifier._prefix_with_verify_batch_buffer(ctx)
        return ctx.prefix, ctx.position

    @staticmethod
    def _uses_temp_answer(method: EarlyExitMethod) -> bool:
        try:
            getattr(method.verifier, "temp_answer_tokens")
            getattr(method.verifier, "temp_answer_logprobs")
            return True
        except AttributeError:
            return False

    def _normal_prompt_ids(
        self, messages: list[dict[str, Any]], thinking: str, kwargs: dict[str, Any]
    ) -> list[int]:
        open_ids = self.renderer.checkpoint(messages, thinking)
        suffix_ids = self.renderer.tokenizer.encode(
            self.agent.temp_answer_suffix(kwargs), add_special_tokens=False
        )
        return [*open_ids, *suffix_ids]

    def _probe_sampling(
        self, tokens: int, logprobs: int, kwargs: dict[str, Any], slot: int
    ) -> dict[str, Any]:
        values = self.agent._completion_kwargs_for_phase("temp_answer", kwargs)
        values.update({"max_tokens": tokens, "logprobs": logprobs})
        values.pop("max_completion_token", None)
        if slot:
            values["seed"] = int(values.get("seed", 0)) + slot
        return values

    def _dtsr_prompt(
        self, verifier: LLMSufficiencyVerifier, ctx: VerifierContext
    ) -> tuple[str, list[int]]:
        if (
            verifier.custom_scoring_handler is not None
            or verifier.same_chat_template is not None
        ):
            raise ValueError(
                "V2 currently requires DTSR llm='same' with the canonical chat template"
            )
        messages = verifier._build_messages(ctx.question, ctx.prefix.content)
        prompt = self.agent.chat_template.render(messages)
        if verifier.close_thinking:
            start = self.agent.think_start_token()
            end = self.agent.think_end_token()
            stripped = prompt.rstrip()
            if stripped.endswith(end):
                prompt = prompt if prompt.endswith("\n") else prompt + "\n"
            elif stripped.endswith(start):
                prompt += f"\n\n{end}\n"
            else:
                prompt += f"{start}\n\n{end}\n"
        return prompt, list(
            self.renderer.tokenizer.encode(prompt, add_special_tokens=False)
        )

    def _dtsr_sampling(
        self, verifier: LLMSufficiencyVerifier, kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        values = self.agent._completion_kwargs_for_phase("temp_answer", kwargs)
        values.pop("max_completion_token", None)
        values["max_tokens"] = verifier.probe_tokens
        values["logprobs"] = max(1, int(values.get("logprobs", 1)))
        return values

    async def _fill_answers(
        self,
        example_id: str,
        messages: list[dict[str, Any]],
        source_token_ids: list[int],
        output_token_ids: list[int],
        results: list[MethodResult],
        kwargs: dict[str, Any],
        timing: dict[str, Any],
    ) -> None:
        prompt_build_started = time.perf_counter()
        suffix_ids = self.renderer.tokenizer.encode(
            f"You are out of thinking budget {self.agent.think_end_token()}",
            add_special_tokens=False,
        )
        unique: dict[tuple[int, str], RequestSpec] = {}
        for result in results:
            if result.position < 0 or result.position > len(output_token_ids):
                raise ValueError(
                    f"Final-answer position {result.position} is outside the canonical "
                    f"output token range 0..{len(output_token_ids)}"
                )
            method = next(
                (
                        item for item in self.methods
                        if item.name == (result.base_method or result.method_name)
                    ),
                None,
            )
            if (
                self.renderer.is_gemma4
                and method is not None
                and method.no_thinking
                and result.stopped_early
            ):
                # Gemma 4's true no-thinking baseline is a fresh assistant
                # turn: no <|think|> system switch and no `reasoning` field.
                prompt_ids = self.renderer.source(messages, enable_thinking=False)
            elif self.renderer.is_gemma4:
                prompt_ids = [
                    *self.renderer.checkpoint(messages, result.reasoning_prefix),
                    *suffix_ids,
                ]
            else:
                prompt_ids = [
                    *source_token_ids,
                    *output_token_ids[: result.position],
                    *suffix_ids,
                ]
            sampling = self.agent._completion_kwargs_for_phase("final", kwargs)
            sampling["max_tokens"] = sampling.pop("max_completion_token", 1024)
            unique[(result.position, result.reasoning_prefix)] = RequestSpec.create(
                prompt_ids,
                sampling,
                result.position,
                RequestKind.FINAL,
                example_id=example_id,
                cache_salt=self.cache_salt,
            )
        prompt_build_seconds = time.perf_counter() - prompt_build_started
        timing["final_prompt_build_seconds"] += prompt_build_seconds
        self.metrics.final_prompt_build_seconds += prompt_build_seconds
        logger.info(
            "Final-answer prompts prepared example=%s requests=%s elapsed=%.3fs",
            example_id,
            len(unique),
            prompt_build_seconds,
        )
        outcomes = (
            await self._await_final_answers(example_id, list(unique.values()))
            if unique
            else []
        )
        for outcome in outcomes:
            self._record_request(RequestKind.FINAL, outcome)
        mapped = dict(zip(unique, outcomes))
        for result in results:
            outcome = mapped[(result.position, result.reasoning_prefix)]
            result.final_answer = outcome.response.content
            result.finish_reason = outcome.response.finish_reason
            result.prompt_tokens = outcome.response.prompt_tokens
            result.answer_tokens = outcome.response.completion_tokens or 0
            result.completion_tokens = result.position + result.answer_tokens

    async def _await_final_answers(
        self,
        example_id: str,
        specs: list[RequestSpec],
    ) -> list[RequestResult]:
        started = time.perf_counter()
        max_tokens = max(
            (int(spec.sampling_dict().get("max_tokens", 0)) for spec in specs),
            default=0,
        )
        logger.info(
            "Final-answer generation started example=%s requests=%s max_tokens=%s",
            example_id,
            len(specs),
            max_tokens,
        )
        tasks = [
            asyncio.create_task(self.cache.get_or_submit(spec, self._submit_engine))
            for spec in specs
        ]
        try:
            pending = set(tasks)
            completed = 0
            while pending:
                done, pending = await asyncio.wait(pending, timeout=30.0)
                completed += len(done)
                logger.info(
                    "Final-answer generation progress example=%s completed=%s/%s elapsed=%.1fs",
                    example_id,
                    completed,
                    len(tasks),
                    time.perf_counter() - started,
                )
            outcomes = [task.result() for task in tasks]
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        logger.info(
            "Final-answer generation finished example=%s requests=%s elapsed=%.2fs output_tokens=%s",
            example_id,
            len(tasks),
            time.perf_counter() - started,
            sum(len(outcome.output_token_ids) for outcome in outcomes),
        )
        return outcomes

    @staticmethod
    def _stopped_result(
        method: EarlyExitMethod, ctx: VerifierContext, value: Any
    ) -> MethodResult:
        try:
            score = float(value) if value is not None else None
        except (TypeError, ValueError):
            score = None
        return MethodResult(
            method_name=method.name,
            position=0 if method.no_thinking else ctx.position,
            stopped_early=True,
            verifier_score=score,
            reasoning_prefix="" if method.no_thinking else ctx.prefix.content,
            final_answer="",
            thinking_tokens=0 if method.no_thinking else ctx.position,
            completion_tokens=0 if method.no_thinking else ctx.position,
            verifier_prompt_tokens=AsyncReplayOrchestrator._verifier_prompt_tokens(
                method
            ),
            verifier_completion_tokens=AsyncReplayOrchestrator._verifier_completion_tokens(
                method
            ),
            state=dict(method.state),
            logs=dict(method.logs),
        )

    def _trajectory_token_ids(self, trajectory: Response) -> list[int]:
        explicit = getattr(trajectory, "output_token_ids", None)
        if explicit is not None:
            return [int(item) for item in explicit]
        if trajectory.logprobs:
            # Per-token decode is not always invertible, so encode the complete
            # stored content and validate against the canonical render instead.
            return list(
                self.renderer.tokenizer.encode(
                    trajectory.content, add_special_tokens=False
                )
            )
        return []

    def _apply_observed_logprobs(
        self, trajectory: Response, output_ids: list[int], values: list[float]
    ) -> None:
        if len(output_ids) != len(values):
            raise ValueError(
                "Extracted logprobs and canonical output IDs have different lengths"
            )
        if len(trajectory.logprobs) != len(values):
            trajectory.logprobs = [
                LogProb(
                    token=self.renderer.decode_token(token_id),
                    logprob=value,
                    top_logprobs=[
                        TopLogProb(
                            token=self.renderer.decode_token(token_id), logprob=value
                        )
                    ],
                )
                for token_id, value in zip(output_ids, values)
            ]
            trajectory.completion_tokens = len(values)
            return
        for row, value in zip(trajectory.logprobs, values):
            row.logprob = float(value)

            if not row.top_logprobs:
                row.top_logprobs = [TopLogProb(token=row.token, logprob=float(value))]

    @staticmethod
    def _validate_observed_logprobs(
        trajectory: Response, output_ids: list[int]
    ) -> None:
        if len(trajectory.logprobs) != len(output_ids):
            raise ValueError(
                f"Expected one logprob row per output token: {len(trajectory.logprobs)} != {len(output_ids)}"
            )
        missing = [
            index
            for index, row in enumerate(trajectory.logprobs)
            if row.logprob is None
        ]
        if missing:
            raise ValueError(
                f"Missing observed-token logprobs at output positions {missing[:8]}"
            )

    def _full_answer(self, trajectory: Response) -> tuple[str, str]:
        marker = self.agent.think_end_token()
        index = trajectory.content.find(marker)
        if index < 0:
            return trajectory.content, ""
        return trajectory.content[:index], trajectory.content[index + len(marker) :]

    @staticmethod
    def _thinking_token_count(trajectory: Response, thinking_end: int) -> int:
        cursor = 0
        count = 0
        for row in trajectory.logprobs:
            cursor += len(row.token or "")
            if cursor > thinking_end:
                break
            count += 1
        return count

    @staticmethod
    def _verifier_prompt_tokens(method: EarlyExitMethod) -> int:
        return sum(
            int(method.state.get(key, 0))
            for key in (
                "_probe_prompt_tokens",
                "_sufficiency_prompt_tokens",
                "_dynasor_prompt_tokens",
                "_answer_coverage_prompt_tokens",
                "_codestop_prompt_tokens",
                "_puma_probe_prompt_tokens",
                "_eat_prompt_tokens",
            )
        )

    @staticmethod
    def _verifier_completion_tokens(method: EarlyExitMethod) -> int:
        return sum(
            int(method.state.get(key, 0))
            for key in (
                "_probe_completion_tokens",
                "_sufficiency_completion_tokens",
                "_dynasor_completion_tokens",
                "_answer_coverage_completion_tokens",
                "_codestop_completion_tokens",
                "_puma_probe_completion_tokens",
                "_eat_completion_tokens",
            )
        )
