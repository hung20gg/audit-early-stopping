from __future__ import annotations

import asyncio
import copy
import inspect
import os
import time
import uuid
from pathlib import Path
import tempfile
from collections.abc import AsyncIterator
from typing import Any

from methods.response import LogProb, Response, TopLogProb
from replay_eval.v2.engine.types import RequestResult, RequestSpec, Timing


class AsyncVLLMEngine:
    """Small version-tolerant wrapper around one long-lived vLLM engine."""

    def __init__(
        self,
        model: str,
        *,
        max_concurrency: int = 32,
        max_logprobs: int = 50,
        engine_kwargs: dict[str, Any] | None = None,
        environment: dict[str, str] | None = None,
        engine: Any = None,
        tokenizer: Any = None,
    ) -> None:
        if max_logprobs < 50:
            raise ValueError("max_logprobs must be >= 50 for ENTROCUT")
        self.model = model
        self.max_logprobs = int(max_logprobs)
        self.max_concurrency = int(max_concurrency)
        # AsyncEngineArgs normalizes nested dictionaries such as
        # speculative_config in place. Keep the typed replay configuration
        # immutable so it remains JSON-serializable for checkpoint metadata.
        self.engine_kwargs = copy.deepcopy(engine_kwargs or {})
        self.environment = {
            str(key): str(value) for key, value in (environment or {}).items()
        }
        self._engine = engine
        self._tokenizer = tokenizer
        self._semaphore = asyncio.Semaphore(self.max_concurrency)
        self.load_seconds = 0.0
        self.warmup_seconds = 0.0
        self.started = engine is not None

    @property
    def tokenizer(self) -> Any:
        if self._tokenizer is None:
            raise RuntimeError("Engine has not been started; tokenizer is unavailable")
        return self._tokenizer

    async def start(self) -> None:
        if self.started:
            return
        # vLLM reads several kernel choices at import time. Apply configured
        # values before importing it so spawned EngineCore workers inherit them.
        os.environ.update(self.environment)
        started = time.perf_counter()
        try:
            from vllm.engine.arg_utils import AsyncEngineArgs
            try:
                from vllm.v1.engine.async_llm import AsyncLLM
            except ImportError:
                from vllm.engine.async_llm_engine import AsyncLLMEngine as AsyncLLM
        except ImportError as exc:
            raise RuntimeError(
                "Replay Eval v2 requires vLLM in the execution environment. "
                "Install the repository requirements on a supported CUDA host."
            ) from exc
        args = {
            "model": self.model,
            "max_logprobs": self.max_logprobs,
            # Prefix caching is the safe fallback for standalone v2 configs.
            # Consolidated configs supply the complete serving policy through
            # llm.vllm.serving and can override this value.
            "enable_prefix_caching": True,
            **self.engine_kwargs,
        }
        self._engine = AsyncLLM.from_engine_args(AsyncEngineArgs(**args))
        get_tokenizer = getattr(self._engine, "get_tokenizer", None)
        if get_tokenizer is None:
            raise RuntimeError("This vLLM AsyncLLM version does not expose get_tokenizer()")
        tokenizer = get_tokenizer()
        self._tokenizer = await tokenizer if inspect.isawaitable(tokenizer) else tokenizer
        self.load_seconds = time.perf_counter() - started
        self.started = True

    async def submit(self, spec: RequestSpec) -> RequestResult:
        if not self.started:
            await self.start()
        async with self._semaphore:
            return await self._generate(spec)

    async def _generate(self, spec: RequestSpec) -> RequestResult:
        try:
            from vllm import SamplingParams
        except ImportError as exc:
            raise RuntimeError("vLLM is required to submit generation requests") from exc
        sampling = spec.sampling_dict()
        max_completion = sampling.pop("max_completion_token", None)
        if max_completion is not None and "max_tokens" not in sampling:
            sampling["max_tokens"] = max_completion
        sampling.pop("detail", None)
        request_id = f"replay-v2-{spec.kind}-{uuid.uuid4().hex}"
        request_started = time.perf_counter()
        prompt = {"prompt_token_ids": list(spec.prompt_token_ids)}
        if spec.cache_salt is not None:
            prompt["cache_salt"] = spec.cache_salt
        generator = self._engine.generate(
            prompt,
            SamplingParams(**sampling),
            request_id=request_id,
        )
        final = await _consume_final(generator)
        elapsed = time.perf_counter() - request_started
        return self._convert(final, spec, elapsed)

    def _convert(self, output: Any, spec: RequestSpec, elapsed: float) -> RequestResult:
        candidate = output.outputs[0] if getattr(output, "outputs", None) else None
        output_ids = list(getattr(candidate, "token_ids", []) or [])
        content = str(getattr(candidate, "text", "") or "")
        response = Response(
            content=content,
            logprobs=self._output_logprobs(candidate, output_ids),
            finish_reason=getattr(candidate, "finish_reason", None),
            prompt_tokens=len(spec.prompt_token_ids),
            completion_tokens=len(output_ids),
        )
        timing = _timing_from_metrics(getattr(output, "metrics", None), elapsed)
        return RequestResult(
            response=response,
            prompt_token_ids=list(spec.prompt_token_ids),
            output_token_ids=output_ids,
            timing=timing,
            cached_prompt_tokens=int(getattr(output, "num_cached_tokens", 0) or 0),
            raw=output,
        )

    def _output_logprobs(self, candidate: Any, token_ids: list[int]) -> list[LogProb]:
        rows = list(getattr(candidate, "logprobs", None) or [])
        result: list[LogProb] = []
        for index, token_id in enumerate(token_ids):
            row = rows[index] if index < len(rows) else {}
            observed = row.get(token_id) if hasattr(row, "get") else None
            tops: list[TopLogProb] = []
            items = row.items() if hasattr(row, "items") else []
            for top_id, item in items:
                tops.append(
                    TopLogProb(
                        token=getattr(item, "decoded_token", None)
                        or self.tokenizer.decode([int(top_id)]),
                        logprob=_float_attr(item, "logprob"),
                    )
                )
            # Providers may insert the observed token before their top-k
            # candidates. EntroCut slices this list, so preserve probability
            # rank rather than relying on mapping insertion order.
            tops.sort(
                key=lambda item: float("-inf") if item.logprob is None else item.logprob,
                reverse=True,
            )
            result.append(
                LogProb(
                    token=getattr(observed, "decoded_token", None)
                    or self.tokenizer.decode([token_id]),
                    logprob=_float_attr(observed, "logprob"),
                    top_logprobs=tops,
                )
            )
        return result

    async def warmup(self, prompt_token_ids: list[int], sampling: dict[str, Any]) -> None:
        from replay_eval.v2.engine.types import RequestKind

        started = time.perf_counter()
        await self.submit(
            RequestSpec.create(
                prompt_token_ids,
                {**sampling, "max_tokens": 1},
                -1,
                RequestKind.WARMUP,
                example_id="warmup",
            )
        )
        self.warmup_seconds = time.perf_counter() - started

    async def extract_trace(
        self,
        *,
        full_token_ids: list[int],
        output_start: int,
        output_token_ids: list[int],
        example_id: str,
        temp_dir: Path | None = None,
        cache_salt: str | None = None,
        dola_layers: list[int] | None = None,
        dola_chunk_size: int = 64,
    ) -> Any:
        """Extract prompt logprobs and token hidden states from the live engine.

        Token embeddings use vLLM's native ``token_embed`` pooling task.  The
        same canonical token list is used for prompt scoring and extraction.
        """
        from replay_eval.v2.engine.extraction import ExtractionResult
        del dola_chunk_size
        if dola_layers:
            raise RuntimeError(
                "DoLa layer extraction requires callback mode with "
                "replay_eval.v2.engine.vllm_hidden_states.extract_trace"
            )
        try:
            import torch
            from vllm import PoolingParams, SamplingParams
        except ImportError as exc:
            raise RuntimeError("Native extraction requires vLLM and torch") from exc
        started = time.perf_counter()
        request_id = f"replay-v2-extract-logprobs-{uuid.uuid4().hex}"
        prompt = {"prompt_token_ids": full_token_ids}
        if cache_salt is not None:
            prompt["cache_salt"] = cache_salt
        generator = self._engine.generate(
            prompt,
            SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=50),
            request_id=request_id,
        )
        scored = await _consume_final(generator)
        prompt_rows = list(getattr(scored, "prompt_logprobs", None) or [])
        observed: list[float] = []
        for index, token_id in enumerate(full_token_ids[output_start:], start=output_start):
            row = prompt_rows[index] if index < len(prompt_rows) else None
            item = row.get(token_id) if hasattr(row, "get") else None
            value = getattr(item, "logprob", None)
            if value is None:
                raise RuntimeError(f"vLLM omitted observed prompt logprob at canonical token {index}")
            observed.append(float(value))

        encode = getattr(self._engine, "encode", None)
        if not callable(encode):
            raise RuntimeError(
                "This vLLM build lacks AsyncLLM.encode(); token_embed extraction is unavailable"
            )
        pooling = PoolingParams(task="token_embed", use_activation=False)
        encoded = await _consume_final(
            encode(
                prompt,
                pooling,
                request_id=f"replay-v2-extract-hidden-{uuid.uuid4().hex}",
            )
        )
        data = getattr(getattr(encoded, "outputs", None), "data", None)
        if data is None:
            raise RuntimeError("vLLM token_embed request returned no token embeddings")
        hidden = data if isinstance(data, torch.Tensor) else torch.as_tensor(data)
        hidden = hidden[output_start : output_start + len(output_token_ids)].cpu()
        if hidden.shape[0] != len(output_token_ids):
            raise RuntimeError("vLLM hidden-state length does not match canonical output tokens")
        normalization_started = time.perf_counter()
        normalized = torch.nn.functional.normalize(hidden, dim=-1)
        normalization_seconds = time.perf_counter() - normalization_started
        directory = temp_dir or Path(tempfile.gettempdir())
        directory.mkdir(parents=True, exist_ok=True)
        stem = f"replay-v2-{example_id}-{uuid.uuid4().hex}"
        raw_path = directory / f"{stem}-raw.pt"
        normalized_path = directory / f"{stem}-normalized.pt"
        try:
            torch.save(hidden, raw_path)
            torch.save(normalized, normalized_path)
        except BaseException:
            raw_path.unlink(missing_ok=True)
            normalized_path.unlink(missing_ok=True)
            raise
        result = ExtractionResult(
            token_ids=list(output_token_ids),
            observed_logprobs=observed,
            raw_hidden_path=raw_path,
            normalized_hidden_path=normalized_path,
            extraction_seconds=time.perf_counter() - started,
            normalization_seconds=normalization_seconds,
            hidden_state_contract="vllm_token_embed",
            prompt_tokens=output_start,
            model_id=self.model,
        )
        result.validate(output_token_ids)
        return result

    async def shutdown(self) -> None:
        if self._engine is None:
            return
        shutdown = getattr(self._engine, "shutdown", None)
        if callable(shutdown):
            outcome = shutdown()
            if inspect.isawaitable(outcome):
                await outcome

    def peak_gpu_memory_bytes(self) -> int | None:
        try:
            import torch

            return int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
        except ImportError:
            return None


async def _consume_final(generator: AsyncIterator[Any]) -> Any:
    final = None
    async for output in generator:
        final = output
    if final is None:
        raise RuntimeError("vLLM request completed without a RequestOutput")
    return final


def _float_attr(item: Any, name: str) -> float | None:
    value = getattr(item, name, None)
    return float(value) if value is not None else None


def _timing_from_metrics(metrics: Any, elapsed: float) -> Timing:
    if metrics is None:
        return Timing(end_to_end=elapsed)
    arrival = getattr(metrics, "arrival_time", None)
    scheduled = getattr(metrics, "first_scheduled_time", None)
    first_token = getattr(metrics, "first_token_time", None)
    finished = getattr(metrics, "finished_time", None)
    queue = getattr(metrics, "time_in_queue", None)
    if queue is None and arrival is not None and scheduled is not None:
        queue = scheduled - arrival
    prefill = first_token - scheduled if scheduled is not None and first_token is not None else 0.0
    decode = finished - first_token if finished is not None and first_token is not None else 0.0
    e2e = finished - arrival if finished is not None and arrival is not None else elapsed
    return Timing(float(queue or 0), float(prefill), float(decode), float(e2e))
