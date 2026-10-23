from __future__ import annotations

import asyncio
import json
import logging
import time
from importlib import import_module
from pathlib import Path
from typing import Any

from methods.response import Response
from replay_eval.v2.config.schema import ReplayV2Config
from replay_eval.v2.engine.extraction import (
    FileExtractionConnector,
    NativeVLLMExtractionConnector,
)
from replay_eval.v2.engine.access import EngineAccessGate
from replay_eval.v2.engine.vllm_engine import AsyncVLLMEngine
from replay_eval.v2.rendering.hf import CanonicalRenderer
from replay_eval.v2.runtime.metrics import RunMetrics
from replay_eval.v2.runtime.orchestrator import AsyncReplayOrchestrator
from replay_eval.replay import _resolve_methods
from replay_eval.verifiers import PUMAVerifier


LOGGER = logging.getLogger(__name__)


class ReplayV2Runner:
    def __init__(
        self,
        config: ReplayV2Config,
        evaluator: Any | None = None,
        evaluator_name: str | None = None,
        *,
        engine: Any | None = None,
        extraction_connector: Any | None = None,
    ) -> None:
        self.config = config
        self.engine = engine or AsyncVLLMEngine(
            config.model.id,
            max_concurrency=config.runtime.max_concurrency,
            max_logprobs=config.model.max_logprobs,
            engine_kwargs=config.engine_kwargs(),
            environment=config.model.environment,
        )
        self.extraction_connector = (
            self._build_extraction_connector()
            if extraction_connector is None
            else extraction_connector
        )
        self.evaluator = evaluator
        self.evaluator_name = evaluator_name
        self.metrics = RunMetrics()
        self.engine_access_gate = EngineAccessGate()
        self.renderer: CanonicalRenderer | None = None

    def _build_extraction_connector(self) -> Any:
        extraction = self.config.extraction
        if extraction.mode == "native":
            return NativeVLLMExtractionConnector(
                extraction.temp_dir,
                dola_layers=extraction.dola_layers,
                dola_chunk_size=extraction.dola_chunk_size,
            )
        assert extraction.callback is not None
        module_name, object_name = extraction.callback.rsplit(".", 1)
        callback = getattr(import_module(module_name), object_name)
        if not callable(callback):
            raise TypeError(
                f"Extraction callback is not callable: {extraction.callback}"
            )
        return FileExtractionConnector(
            callback,
            dola_layers=extraction.dola_layers,
            dola_chunk_size=extraction.dola_chunk_size,
        )

    def _preload_puma_backends(self) -> None:
        for spec in self.config.methods:
            if not str(spec.get("factory", "")).endswith("build_puma_method"):
                continue
            for method in _resolve_methods([spec]):
                if isinstance(method.verifier, PUMAVerifier):
                    method.verifier.preload()

    async def run(
        self,
        input_file: str | Path,
        output_file: str | Path,
        *,
        resume_from: str | Path | None = None,
        recovery_dataloader: Any | None = None,
        recovery_split: str = "test",
        dataloader_name: str | None = None,
    ) -> dict[str, Any]:
        records = list(_read_jsonl(input_file))
        if self.config.random_exit.enabled:
            return await self._run_random_exit(
                records,
                Path(input_file),
                Path(output_file),
                resume_from=Path(resume_from) if resume_from is not None else None,
            )
        if recovery_dataloader is not None:
            dataset_records = list(
                recovery_dataloader(split=recovery_split, limit=None)
            )
            missing_numbers = _missing_dataset_record_numbers(
                records, len(dataset_records)
            )
            if missing_numbers:
                recovered = _recover_missing_records(
                    records,
                    dataset_records,
                    missing_numbers=missing_numbers,
                    dataloader_name=dataloader_name,
                )
                records.extend(recovered)
                LOGGER.info(
                    "Recovered %s missing records from dataloader positions=%s; "
                    "all configured methods will run for those records",
                    len(missing_numbers),
                    missing_numbers,
                )
        else:
            missing_numbers = _missing_record_numbers(records)
            if missing_numbers:
                LOGGER.warning(
                    "Input is missing record numbers %s, but no recovery dataloader "
                    "was supplied",
                    missing_numbers,
                )
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        completed_rows: list[dict[str, Any]] = []
        completed_keys: set[tuple[str, str]] = set()
        checkpoint_path = Path(resume_from) if resume_from is not None else None
        if checkpoint_path is not None:
            if not checkpoint_path.is_file():
                raise FileNotFoundError(
                    f"Replay v2 checkpoint does not exist: {checkpoint_path}"
                )
            for checkpoint_index, row in enumerate(
                _read_jsonl(checkpoint_path), start=1
            ):
                if not _is_completed_output_record(row):
                    continue
                key = _record_identity(row, checkpoint_index)
                if key in completed_keys:
                    continue
                completed_keys.add(key)
                completed_rows.append(row)

        indexed_records = list(enumerate(records, start=1))
        pending_records = [
            (index, record)
            for index, record in indexed_records
            if _record_identity(record, index) not in completed_keys
        ]
        self.metrics.resumed_examples = len(indexed_records) - len(pending_records)
        if self.metrics.resumed_examples:
            LOGGER.info(
                "Replay v2 resume loaded=%s pending=%s checkpoint=%s",
                self.metrics.resumed_examples,
                len(pending_records),
                checkpoint_path,
            )

        # Preload PUMA before vLLM sizes its KV cache; the one native engine
        # then stays alive and retains its allocated KV blocks for the run.
        if any(
            self._record_needs_method_replay(record)
            for _index, record in pending_records
        ):
            self._preload_puma_backends()
        if not pending_records:
            if checkpoint_path != output_path:
                with output_path.open("w", encoding="utf-8") as handle:
                    for row in completed_rows:
                        handle.write(
                            json.dumps(row, ensure_ascii=False, default=_json_default)
                            + "\n"
                        )
            return self.metrics.summary()
        await self.engine.start()
        self.metrics.model_load_seconds = float(
            getattr(self.engine, "load_seconds", 0.0)
        )
        self.renderer = CanonicalRenderer(
            self.engine.tokenizer,
            model_id=self.config.model.id,
            template_kwargs=self.config.rendering.template_kwargs,
            think_start_token=self.config.rendering.think_start_token,
            think_end_token=self.config.rendering.think_end_token,
        )
        if self.config.runtime.warmup:
            warmup_messages = self._messages("Warm up.")
            await self.engine.warmup(
                self.renderer.source(warmup_messages), self.config.runtime.sampling
            )
            self.metrics.warmup_seconds = float(
                getattr(self.engine, "warmup_seconds", 0.0)
            )

        semaphore = asyncio.Semaphore(self.config.runtime.example_concurrency)
        output_lock = asyncio.Lock()
        measured_started = time.perf_counter()

        async def bounded(index: int, record: dict[str, Any], handle: Any) -> None:
            async with semaphore:
                item_started = time.perf_counter()
                try:
                    result = await asyncio.wait_for(
                        self._evaluate_record(index, record),
                        timeout=self.config.runtime.example_timeout_seconds,
                    )
                except asyncio.CancelledError:
                    raise
                except asyncio.TimeoutError:
                    self.metrics.failed_examples += 1
                    self.metrics.timed_out_examples += 1
                    LOGGER.warning(
                        "Replay v2 skipped record after exceeding %.1f-second "
                        "timeout index=%s id=%r",
                        self.config.runtime.example_timeout_seconds,
                        index,
                        _record_identity(record, index)[1],
                    )
                    return
                except Exception as exc:
                    if _is_fatal_engine_error(exc):
                        raise
                    self.metrics.failed_examples += 1
                    LOGGER.exception(
                        "Replay v2 skipped failed record index=%s id=%r",
                        index,
                        _record_identity(record, index)[1],
                    )
                    return
                self.metrics.latencies.append(time.perf_counter() - item_started)
                self.metrics.examples += 1
            async with output_lock:
                self.metrics.cache_hits += result.pop("_cache_hits", 0)
                self.metrics.cache_misses += result.pop("_cache_misses", 0)
                handle.write(
                    json.dumps(result, ensure_ascii=False, default=_json_default) + "\n"
                )
                handle.flush()

        try:
            append_checkpoint = checkpoint_path == output_path
            mode = "a" if append_checkpoint else "w"
            with output_path.open(mode, encoding="utf-8") as handle:
                if not append_checkpoint:
                    for row in completed_rows:
                        handle.write(
                            json.dumps(row, ensure_ascii=False, default=_json_default)
                            + "\n"
                        )
                    handle.flush()
                await asyncio.gather(
                    *(
                        bounded(index, record, handle)
                        for index, record in pending_records
                    )
                )
            self.metrics.measured_seconds = time.perf_counter() - measured_started
            self.metrics.peak_gpu_memory_bytes = self.engine.peak_gpu_memory_bytes()
            return self.metrics.summary()
        finally:
            await self.engine.shutdown()

    async def _run_random_exit(
        self,
        records: list[dict[str, Any]],
        input_file: Path,
        output_file: Path,
        *,
        resume_from: Path | None,
    ) -> dict[str, Any]:
        from replay_eval.v2.runtime.random_exit import (
            RandomExitReplayRunner,
            _input_manifest_identity,
            resolved_settings,
            validate_resume_metadata,
        )

        self.config.validate()
        validate_resume_metadata(
            self.config,
            input_file=input_file,
            records=records,
            resume_from=resume_from,
        )
        # Snapshot serializable settings before vLLM initializes. Some vLLM
        # versions normalize nested engine dictionaries in place.
        random_exit_settings = resolved_settings(
            self.config,
            manifest_identity=_input_manifest_identity(input_file, records),
        )
        LOGGER.info(
            "Random-exit replay validated model=%s records=%d input=%s output=%s "
            "resume_from=%s",
            self.config.model.key or self.config.model.id,
            len(records),
            input_file,
            output_file,
            resume_from,
        )
        engine_started = time.perf_counter()
        LOGGER.info("Random-exit loading generation engine")
        await self.engine.start()
        LOGGER.info(
            "Random-exit generation engine ready elapsed=%.1fs",
            time.perf_counter() - engine_started,
        )
        self.renderer = CanonicalRenderer(
            self.engine.tokenizer,
            model_id=self.config.model.id,
            template_kwargs=self.config.rendering.template_kwargs,
            think_start_token=self.config.rendering.think_start_token,
            think_end_token=self.config.rendering.think_end_token,
        )
        try:
            if self.config.runtime.warmup:
                LOGGER.info("Random-exit engine warmup started")
                warmup_started = time.perf_counter()
                warmup_messages = self._messages("Warm up.")
                await self.engine.warmup(
                    self.renderer.source(warmup_messages),
                    self.config.runtime.sampling,
                )
                LOGGER.info(
                    "Random-exit engine warmup finished elapsed=%.1fs",
                    time.perf_counter() - warmup_started,
                )
            return await RandomExitReplayRunner(
                self.config,
                engine=self.engine,
                renderer=self.renderer,
                settings=random_exit_settings,
            ).run(
                records,
                input_file,
                output_file,
                resume_from=resume_from,
            )
        finally:
            LOGGER.info("Random-exit shutting down generation engine")
            await self.engine.shutdown()
            LOGGER.info("Random-exit generation engine stopped")

    def _record_needs_method_replay(self, record: dict[str, Any]) -> bool:
        processed = _completed_method_names(
            _existing_method_results(record), exit=self.config.runtime.exit
        )
        return bool(
            _select_method_specs(
                self.config.methods,
                processed_methods=processed,
                rerun_all=self.config.runtime.rerun_all,
            )
        )

    async def _evaluate_record(
        self, index: int, record: dict[str, Any]
    ) -> dict[str, Any]:
        assert self.renderer is not None
        nested_input = record.get("input")
        nested_input = nested_input if isinstance(nested_input, dict) else {}
        question = str(
            record.get("problem")
            or record.get("question")
            or record.get("prompt")
            or nested_input.get("problem")
            or nested_input.get("question")
            or nested_input.get("prompt")
            or (record.get("input") if isinstance(record.get("input"), str) else None)
            or ""
        )
        if not question:
            raise ValueError(
                f"Input record {index} has no problem/question/prompt/input field"
            )
        trajectory_payload = (
            record.get("response")
            or record.get("trajectory")
            or record.get("generation_response")
        )
        if isinstance(trajectory_payload, dict):
            trajectory = Response(**trajectory_payload)
        elif isinstance(record.get("output"), str):
            trajectory = Response(
                content=record["output"],
                logprobs=Response.normalize_logprobs(record.get("logprobs") or []),
                finish_reason=record.get("finish_reason"),
                prompt_tokens=record.get("prompt_tokens"),
                completion_tokens=record.get("completion_tokens")
                or record.get("num_token"),
            )
        else:
            trajectory = None
        if trajectory is not None and record.get("output_token_ids") is not None:
            object.__setattr__(
                trajectory, "output_token_ids", record["output_token_ids"]
            )
        record_id = (
            record.get("id")
            or nested_input.get("id")
            or record.get("line_number")
            or index
        )
        example_id = str(record_id)
        existing_results = _existing_method_results(record)
        final_answer_results = (
            [item for item in existing_results if "final_answer" not in item]
            if self.config.runtime.rerun_final_answer
            and not self.config.runtime.rerun_all
            else []
        )
        processed_methods = _completed_method_names(
            existing_results, exit=self.config.runtime.exit
        )
        method_specs = _select_method_specs(
            self.config.methods,
            processed_methods=processed_methods,
            rerun_all=self.config.runtime.rerun_all,
        )
        if not method_specs and not final_answer_results:
            row = {
                **_discard_previous_replay_logs(record),
                "id": record_id,
                "model": self.config.model.key or self.config.model.id,
                "method_results": existing_results,
                "v2_timing": {
                    "example_seconds": 0.0,
                    "evaluator_seconds": 0.0,
                    "skipped_methods": True,
                },
                "_cache_hits": 0,
                "_cache_misses": 0,
            }
            return _strip_serialized_logprobs(row)

        answer = record.get("answer", nested_input.get("answer"))
        runtime_kwargs = {
            "answer": answer,
            "dataset": record.get("dataset", record.get("dataloader")),
            "is_code": record.get("is_code"),
            "evaluator": self.evaluator,
            "evaluator_name": self.evaluator_name,
            "max_temp_answer_completion_token": self.config.runtime.max_temp_answer_completion_token,
            "max_final_answer_completion_token": self.config.runtime.max_final_answer_completion_token,
        }
        orchestrator = AsyncReplayOrchestrator(
            engine=self.engine,
            renderer=self.renderer,
            methods=method_specs,
            extraction_connector=self.extraction_connector if method_specs else None,
            completion_kwargs=self.config.runtime.sampling,
            code_probe_token_multiplier=self.config.runtime.code_probe_token_multiplier,
            code_threshold_multiplier=self.config.runtime.code_threshold_multiplier,
            verify_batch_token=self.config.runtime.verify_batch_token,
            min_interval=self.config.runtime.min_interval,
            exit=self.config.runtime.exit,
            metrics=self.metrics,
            engine_access_gate=self.engine_access_gate,
        )
        messages = self._messages(question)
        if method_specs:
            response, results, timing = await orchestrator.evaluate(
                example_id=example_id,
                question=question,
                messages=messages,
                trajectory=trajectory,
                kwargs=runtime_kwargs,
            )
        else:
            if trajectory is None:
                raise ValueError(
                    f"Input record {index} cannot rerun final answers without a stored trajectory"
                )
            response = trajectory
            results = []
            timing = {}
        if final_answer_results:
            refill_timing = await orchestrator.rerun_final_answers(
                example_id=example_id,
                messages=messages,
                trajectory=response,
                results=final_answer_results,
                kwargs=runtime_kwargs,
            )
            timing = _merge_timing(timing, refill_timing)
        timing["evaluator_seconds"] = 0.0
        evaluator_started = time.perf_counter()
        new_results = [result.to_dict() for result in results]
        serialized_results = (
            new_results
            if self.config.runtime.rerun_all
            else _merge_method_results(existing_results, new_results)
        )
        if self.evaluator is not None and answer is not None:
            for result in serialized_results:
                result["correct"] = bool(
                    self.evaluator(result.get("final_answer", ""), answer)
                )
        row = {
            **_discard_previous_replay_logs(record),
            "id": record_id,
            "model": self.config.model.key or self.config.model.id,
            "output": response.content,
            "num_token": response.token_count(),
            "completion_tokens": response.completion_tokens,
            "response": _strip_serialized_logprobs(_model_dict(response)),
            "method_results": serialized_results,
            "v2_timing": timing,
            "_cache_hits": orchestrator.cache.hits,
            "_cache_misses": orchestrator.cache.misses,
        }
        if self.evaluator_name:
            row["evaluator"] = self.evaluator_name
        if self.evaluator is not None and answer is not None:
            final_output = _answer_channel(
                response.content, self.renderer.think_end_token
            )
            row["correct"] = bool(self.evaluator(final_output, answer))
        timing["evaluator_seconds"] = time.perf_counter() - evaluator_started
        self.metrics.evaluator_seconds += timing["evaluator_seconds"]
        row["v2_timing"] = timing
        return _strip_serialized_logprobs(row)

    def _messages(self, question: str) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if self.config.rendering.system_prompt:
            messages.append(
                {"role": "system", "content": self.config.rendering.system_prompt}
            )
        messages.append({"role": "user", "content": question})
        return messages


def _strip_serialized_logprobs(value: Any) -> Any:
    """Remove trajectory log-prob payloads from every serialized output path."""
    if isinstance(value, dict):
        return {
            key: _strip_serialized_logprobs(item)
            for key, item in value.items()
            if key not in {"logprobs", "top_logprobs"}
        }
    if isinstance(value, list):
        return [_strip_serialized_logprobs(item) for item in value]
    return value


def _method_result_name(result: dict[str, Any]) -> str | None:
    value = result.get("method") or result.get("method_name")
    return str(value) if value else None


def _discard_previous_replay_logs(record: dict[str, Any]) -> dict[str, Any]:
    """Keep source data while removing replay results and timing from older runs."""
    discarded = {"additional_logs", "method_results", "v2_timing"}
    cleaned = {key: value for key, value in record.items() if key not in discarded}
    response = cleaned.get("response")
    if isinstance(response, dict):
        cleaned["response"] = {
            key: value
            for key, value in response.items()
            if key
            not in {
                "additional_logs",
                "method_results",
                "method_results_dicts",
                "v2_timing",
            }
        }
    return cleaned


def _existing_method_results(record: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = [record.get("method_results")]
    additional_logs = record.get("additional_logs")
    if isinstance(additional_logs, dict):
        candidates.append(additional_logs.get("method_results"))
    response = record.get("response")
    if isinstance(response, dict):
        candidates.extend(
            [response.get("method_results_dicts"), response.get("method_results")]
        )
    for candidate in candidates:
        if isinstance(candidate, list):
            return [dict(item) for item in candidate if isinstance(item, dict)]
    return []


def _select_method_specs(
    method_specs: list[dict[str, Any]],
    *,
    processed_methods: set[str],
    rerun_all: bool,
) -> list[dict[str, Any]]:
    if rerun_all or not processed_methods:
        return list(method_specs)
    selected = []
    for spec in method_specs:
        kwargs = spec.get("kwargs") if isinstance(spec, dict) else None
        configured_name = (
            kwargs.get("name") if isinstance(kwargs, dict) else None
        ) or spec.get("name")
        if configured_name is None or str(configured_name) not in processed_methods:
            selected.append(spec)
    return selected


def _completed_method_names(results: list[dict[str, Any]], *, exit: bool) -> set[str]:
    completed: set[str] = set()
    for item in results:
        name = _method_result_name(item)
        if exit:
            if item.get("exit") is not False and name:
                completed.add(name)
        elif (
            item.get("exit") is False
            and item.get("base_method")
            and name == f"{item['base_method']}:full"
        ):
            completed.add(str(item["base_method"]))
    return completed


def _merge_method_results(
    existing: list[dict[str, Any]],
    new: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    replaced_names = {
        name for item in new if (name := _method_result_name(item)) is not None
    }
    replaced_groups = {
        item["base_method"] for item in new
        if item.get("exit") is False and item.get("base_method")
    }
    preserved = [
        item for item in existing
        if _method_result_name(item) not in replaced_names
        and not (item.get("exit") is False and item.get("base_method") in replaced_groups)
    ]
    return [*preserved, *new]


def _answer_channel(content: str, think_end_token: str) -> str:
    if think_end_token and think_end_token in content:
        return content.rsplit(think_end_token, 1)[-1].strip()
    return content


def _merge_timing(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    merged = dict(left)
    for key, value in right.items():
        if isinstance(value, (int, float)) and isinstance(merged.get(key), (int, float)):
            merged[key] += value
        else:
            merged[key] = value
    calls = merged.get("llm_calls")
    seconds = merged.get("llm_call_seconds")
    if isinstance(calls, (int, float)) and isinstance(seconds, (int, float)):
        merged["avg_llm_call_seconds"] = seconds / calls if calls else 0.0
    return merged


def _read_jsonl(path: str | Path):
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"JSONL line {line_number} is not an object")
            yield value


def _record_identity(
    record: dict[str, Any], fallback_index: int
) -> tuple[str, str]:
    nested = record.get("input")
    nested = nested if isinstance(nested, dict) else {}
    value = (
        record.get("id")
        or nested.get("id")
        or record.get("line_number")
        or fallback_index
    )
    return type(value).__name__, str(value)


def _is_completed_output_record(record: dict[str, Any]) -> bool:
    return (
        "v2_error" not in record
        and isinstance(record.get("method_results"), list)
        and isinstance(record.get("v2_timing"), dict)
    )


def _is_fatal_engine_error(exc: BaseException) -> bool:
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        name = type(current).__name__
        if name in {"EngineDeadError", "AsyncEngineDeadError", "OutOfMemoryError"}:
            return True
        current = current.__cause__ or current.__context__
    return False


def _record_number(record: dict[str, Any]) -> int | None:
    """Return the stable, one-based dataset position carried by an output row."""
    for key in ("line_number", "id"):
        value = record.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return None


def _missing_record_numbers(records: list[dict[str, Any]]) -> list[int]:
    """Find internal holes without guessing omitted leading or trailing ranges."""
    present = {
        number for record in records if (number := _record_number(record)) is not None
    }
    if not present:
        return []
    return sorted(set(range(min(present), max(present) + 1)) - present)


def _missing_dataset_record_numbers(
    records: list[dict[str, Any]], dataset_size: int
) -> list[int]:
    """Find every absent one-based dataset position, including both tails."""
    if dataset_size < 0:
        raise ValueError("dataset_size must be non-negative")
    present = {
        number
        for record in records
        if (number := _record_number(record)) is not None
        and number <= dataset_size
    }
    if not present and records:
        # Inputs without explicit positions can only be interpreted as the
        # leading contiguous dataset slice.
        present = set(range(1, min(len(records), dataset_size) + 1))
    return sorted(set(range(1, dataset_size + 1)) - present)


def _recover_missing_records(
    records: list[dict[str, Any]],
    dataset_records: Any,
    *,
    missing_numbers: list[int] | None = None,
    dataloader_name: str | None = None,
) -> list[dict[str, Any]]:
    """Rebuild absent output rows from their one-based dataloader positions."""
    missing = (
        _missing_record_numbers(records)
        if missing_numbers is None
        else list(missing_numbers)
    )
    if not missing:
        return []
    dataset = list(dataset_records)
    if max(missing) > len(dataset):
        raise ValueError(
            f"Cannot recover record {max(missing)}: dataloader returned only "
            f"{len(dataset)} records"
        )
    recovered = []
    for number in missing:
        source = dataset[number - 1]
        if not isinstance(source, dict):
            raise TypeError(f"Dataloader record {number} is not an object")
        record = {**source, "line_number": number, "id": number}
        if dataloader_name:
            record.setdefault("dataloader", dataloader_name)
            record.setdefault("dataset", dataloader_name)
        recovered.append(record)
    return recovered


def _json_default(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def _model_dict(value: Any) -> dict[str, Any]:
    dump = getattr(value, "model_dump", None)
    return dump() if callable(dump) else value.dict()
