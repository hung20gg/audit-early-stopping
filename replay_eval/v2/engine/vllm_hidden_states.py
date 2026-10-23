"""vLLM worker callback for exact pre-final-RMSNorm trace extraction.

The worker extension captures Qwen's final norm input during an ordinary
prompt-logprob request. This keeps generation and extraction on the same
long-lived vLLM engine while preserving the hidden-state contract used to
train Terminator and Thought Calibration.
"""

from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path
from typing import Any

from replay_eval.v2.dola.worker import DoLaWorkerCapture, find_final_norm


LOGGER = logging.getLogger(__name__)


class HiddenStateWorkerExtension:
    """Methods injected into a vLLM worker for collective-RPC extraction."""

    def replay_v2_begin_hidden_capture(
        self, dola_layers: list[int] | None = None
    ) -> dict[str, Any]:
        import torch

        if getattr(self, "_replay_v2_capture_handles", None):
            raise RuntimeError("A replay-v2 hidden-state capture is already active")
        model = self.model_runner.get_model()
        norm = find_final_norm(model)
        raw_chunks: list[torch.Tensor] = []
        normalized_chunks: list[torch.Tensor] = []
        copy_events: list[torch.cuda.Event] = []

        def copy_to_host(value: torch.Tensor) -> torch.Tensor:
            source = value.detach()
            if source.device.type != "cuda":
                return source.to(device="cpu", dtype=torch.float16)
            try:
                host = torch.empty(
                    source.shape,
                    device="cpu",
                    dtype=torch.float16,
                    pin_memory=True,
                )
            except RuntimeError:
                # Preserve compatibility on hosts whose pinned-memory quota is
                # too small for long traces.
                return source.to(device="cpu", dtype=torch.float16)
            host.copy_(source, non_blocking=True)
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(source.device))
            copy_events.append(event)
            return host

        def capture_raw(_module: Any, args: tuple[Any, ...]) -> None:
            hidden = args[0]
            residual = args[1] if len(args) > 1 else None
            raw = hidden if residual is None else hidden + residual
            raw_chunks.append(copy_to_host(raw))

        def capture_normalized(
            _module: Any, _args: tuple[Any, ...], output: Any
        ) -> None:
            normalized = output[0] if isinstance(output, tuple) else output
            normalized_chunks.append(copy_to_host(normalized))

        self._replay_v2_raw_chunks = raw_chunks
        self._replay_v2_normalized_chunks = normalized_chunks
        self._replay_v2_copy_events = copy_events
        dola_capture = DoLaWorkerCapture.start(
            model, list(dola_layers or []), copy_to_host
        )
        try:
            self._replay_v2_capture_handles = (
                norm.register_forward_pre_hook(capture_raw),
                norm.register_forward_hook(capture_normalized),
            )
        except BaseException:
            dola_capture.close()
            raise
        self._replay_v2_dola_capture = dola_capture
        return {
            "norm": type(norm).__name__,
            "dola_layers": self._replay_v2_dola_capture.layers,
        }

    def replay_v2_finish_hidden_capture(
        self,
        raw_path: str,
        normalized_path: str,
        expected_tokens: int,
        output_start: int = 0,
        output_tokens: int | None = None,
        dola_chunk_size: int = 64,
    ) -> dict[str, Any]:
        import torch

        handles = getattr(self, "_replay_v2_capture_handles", ())
        try:
            for event in getattr(self, "_replay_v2_copy_events", ()):
                event.synchronize()
            raw_chunks = list(getattr(self, "_replay_v2_raw_chunks", ()))
            normalized_chunks = list(
                getattr(self, "_replay_v2_normalized_chunks", ())
            )
            if not raw_chunks or len(raw_chunks) != len(normalized_chunks):
                raise RuntimeError(
                    "vLLM final-norm hooks did not capture matching hidden-state chunks"
                )
            raw = torch.cat(raw_chunks, dim=0)[:expected_tokens].contiguous()
            normalized = (
                torch.cat(normalized_chunks, dim=0)[:expected_tokens].contiguous()
            )
            if raw.shape[0] != expected_tokens or normalized.shape != raw.shape:
                raise RuntimeError(
                    "vLLM hidden-state capture length does not match the scored token stream: "
                    f"raw={tuple(raw.shape)}, normalized={tuple(normalized.shape)}, "
                    f"expected_tokens={expected_tokens}"
                )
            if output_tokens is None:
                output_tokens = expected_tokens - output_start
            output_end = output_start + output_tokens
            raw = raw[output_start:output_end].contiguous()
            normalized = normalized[output_start:output_end].contiguous()
            if raw.shape[0] != output_tokens or normalized.shape != raw.shape:
                raise RuntimeError(
                    "vLLM hidden-state assistant slice has the wrong shape: "
                    f"raw={tuple(raw.shape)}, normalized={tuple(normalized.shape)}, "
                    f"output_start={output_start}, output_tokens={output_tokens}"
                )
            torch.save(raw, raw_path)
            torch.save(normalized, normalized_path)
            dola_capture = getattr(self, "_replay_v2_dola_capture", None)
            model_outputs = (
                dola_capture.project_jsd(
                    model=self.model_runner.get_model(),
                    final_normalized=normalized,
                    expected_tokens=expected_tokens,
                    output_start=output_start,
                    output_end=output_end,
                    chunk_size=int(dola_chunk_size),
                )
                if dola_capture is not None
                else {}
            )
            return {
                "shape": list(raw.shape),
                "chunks": len(raw_chunks),
                "model_outputs": model_outputs,
            }
        finally:
            for handle in handles:
                handle.remove()
            self._replay_v2_capture_handles = ()
            self._replay_v2_raw_chunks = []
            self._replay_v2_normalized_chunks = []
            self._replay_v2_copy_events = []
            dola_capture = getattr(self, "_replay_v2_dola_capture", None)
            if dola_capture is not None:
                dola_capture.close()
            self._replay_v2_dola_capture = None

    def replay_v2_cancel_hidden_capture(self) -> None:
        for handle in getattr(self, "_replay_v2_capture_handles", ()):
            handle.remove()
        self._replay_v2_capture_handles = ()
        self._replay_v2_raw_chunks = []
        self._replay_v2_normalized_chunks = []
        self._replay_v2_copy_events = []
        dola_capture = getattr(self, "_replay_v2_dola_capture", None)
        if dola_capture is not None:
            dola_capture.close()
        self._replay_v2_dola_capture = None


async def extract_trace(
    *,
    engine: Any,
    full_token_ids: list[int],
    output_start: int,
    output_token_ids: list[int],
    example_id: str,
    cache_salt: str | None = None,
    dola_layers: list[int] | None = None,
    dola_chunk_size: int = 64,
) -> dict[str, Any]:
    """Score one canonical trace and capture its vLLM pre-final norm states."""
    from vllm import SamplingParams

    from replay_eval.v2.engine.vllm_engine import _consume_final

    live_engine = getattr(engine, "_engine", None)
    if live_engine is None or not callable(getattr(live_engine, "collective_rpc", None)):
        raise RuntimeError("The configured vLLM engine does not support collective RPC")

    temp_dir = Path("/tmp/replay-v2-extraction")
    temp_dir.mkdir(parents=True, exist_ok=True)
    stem = f"replay-v2-{example_id}-{uuid.uuid4().hex}"
    raw_path = temp_dir / f"{stem}-raw.pt"
    normalized_path = temp_dir / f"{stem}-normalized.pt"
    capture_active = False
    try:
        begin_started = time.perf_counter()
        await live_engine.collective_rpc(
            "replay_v2_begin_hidden_capture", args=(list(dola_layers or []),)
        )
        begin_seconds = time.perf_counter() - begin_started
        capture_active = True
        request_id = f"replay-v2-extract-{uuid.uuid4().hex}"
        prompt = {"prompt_token_ids": list(full_token_ids)}
        if cache_salt is not None:
            prompt["cache_salt"] = cache_salt
        generation_started = time.perf_counter()
        scored = await _consume_final(
            live_engine.generate(
                prompt,
                SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=50),
                request_id=request_id,
            )
        )
        generation_seconds = time.perf_counter() - generation_started
        logprobs_started = time.perf_counter()
        prompt_rows = list(getattr(scored, "prompt_logprobs", None) or [])
        observed: list[float] = []
        for index, token_id in enumerate(
            full_token_ids[output_start:], start=output_start
        ):
            row = prompt_rows[index] if index < len(prompt_rows) else None
            item = row.get(token_id) if hasattr(row, "get") else None
            value = getattr(item, "logprob", None)
            if value is None:
                raise RuntimeError(
                    f"vLLM omitted observed prompt logprob at canonical token {index}"
                )
            observed.append(float(value))
        logprobs_seconds = time.perf_counter() - logprobs_started

        finish_started = time.perf_counter()
        finish_results = await live_engine.collective_rpc(
            "replay_v2_finish_hidden_capture",
            args=(
                str(raw_path),
                str(normalized_path),
                len(full_token_ids),
                output_start,
                len(output_token_ids),
                int(dola_chunk_size),
            ),
        )
        finish_seconds = time.perf_counter() - finish_started
        capture_active = False
        LOGGER.info(
            "Hidden-state callback phases example=%s begin=%.2fs prefill=%.2fs "
            "logprobs=%.2fs finalize=%.2fs",
            example_id,
            begin_seconds,
            generation_seconds,
            logprobs_seconds,
            finish_seconds,
        )
        worker_payloads = [
            value for value in (finish_results or []) if isinstance(value, dict)
        ]
        if dola_layers and len(worker_payloads) != 1:
            raise RuntimeError(
                "DoLa layer projection currently requires tensor_parallel_size=1"
            )
        model_outputs = (
            dict(worker_payloads[0].get("model_outputs") or {})
            if worker_payloads
            else {}
        )
        return {
            "token_ids": list(output_token_ids),
            "observed_logprobs": observed,
            "raw_hidden_path": raw_path,
            "normalized_hidden_path": normalized_path,
            "hidden_state_contract": "pre_final_rmsnorm",
            "prompt_tokens": output_start,
            "model_id": engine.model,
            "model_outputs": model_outputs,
        }
    except BaseException:
        if capture_active:
            try:
                await live_engine.collective_rpc("replay_v2_cancel_hidden_capture")
            except BaseException:
                # EngineCore failures (especially CUDA OOM) make utility RPC
                # unavailable. Preserve the original generation exception;
                # the dead worker is about to shut down anyway.
                LOGGER.warning(
                    "Could not cancel hidden capture after extraction failure",
                    exc_info=True,
                )
        raw_path.unlink(missing_ok=True)
        normalized_path.unlink(missing_ok=True)
        raise
