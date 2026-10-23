from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass
class ExtractionResult:
    token_ids: list[int]
    observed_logprobs: list[float]
    raw_hidden_path: Path
    normalized_hidden_path: Path
    extraction_seconds: float = 0.0
    loading_seconds: float = 0.0
    normalization_seconds: float = 0.0
    hidden_state_contract: str = "unknown"
    prompt_tokens: int | None = None
    model_id: str | None = None
    model_outputs: dict[str, list[float]] = field(default_factory=dict)

    def validate(self, expected_output_ids: list[int]) -> None:
        if self.token_ids != expected_output_ids:
            raise ValueError("Hidden-state token IDs do not align with the canonical output IDs")
        if len(self.observed_logprobs) != len(expected_output_ids):
            raise ValueError("Every stored output token must have one observed logprob")
        for name, values in self.model_outputs.items():
            if len(values) != len(expected_output_ids):
                raise ValueError(
                    f"Model output {name!r} must have one score per stored output token"
                )

    def load_trace(self) -> "HiddenStateTrace":
        raw, normalized = self.load()
        return HiddenStateTrace(
            raw=raw,
            final_normalized=normalized,
            token_ids=list(self.token_ids),
            prompt_tokens=self.prompt_tokens,
            hidden_state_contract=self.hidden_state_contract,
            model_id=self.model_id,
            model_outputs={key: list(value) for key, value in self.model_outputs.items()},
        )

    def load(self) -> tuple[Any, Any]:
        started = time.perf_counter()
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("Loading extracted hidden states requires torch") from exc
        raw = torch.load(self.raw_hidden_path, map_location="cpu", weights_only=True)
        normalized = torch.load(
            self.normalized_hidden_path, map_location="cpu", weights_only=True
        )
        self.loading_seconds += time.perf_counter() - started
        return raw, normalized

    def cleanup(self) -> None:
        for path in (self.raw_hidden_path, self.normalized_hidden_path):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass


@dataclass
class HiddenStateTrace:
    """Example-scoped canonical assistant states and optional worker scores."""

    raw: Any
    final_normalized: Any
    token_ids: list[int]
    prompt_tokens: int | None
    hidden_state_contract: str
    model_id: str | None = None
    model_outputs: dict[str, list[float]] = field(default_factory=dict)
    pooled_hidden_states: dict[int, Any] = field(default_factory=dict, repr=False)
    reported_precomputed_score_keys: set[str] = field(default_factory=set, repr=False)

    def release_token_states(self) -> None:
        """Release the large token-by-hidden tensors after probe preparation."""
        self.raw = None
        self.final_normalized = None
    precomputed_score_keys: set[str] = field(default_factory=set, repr=False)

    def require_pre_final_rmsnorm(self) -> None:
        if self.hidden_state_contract != "pre_final_rmsnorm":
            raise RuntimeError(
                "This hidden-state model requires contract='pre_final_rmsnorm', "
                f"but extraction reported {self.hidden_state_contract!r}. Use a vLLM "
                "worker callback that captures the input to the base model's final norm."
            )


class ExtractionConnector(Protocol):
    async def extract(
        self,
        *,
        engine: Any,
        full_token_ids: list[int],
        output_start: int,
        output_token_ids: list[int],
        example_id: str,
        cache_salt: str | None = None,
    ) -> ExtractionResult: ...


class FileExtractionConnector:
    """Connector for the native vLLM token-embedding extraction worker.

    The generation engine remains the sole long-lived engine. A deployment
    supplies ``extract_callback`` which captures the raw layer
    ``num_hidden_layers`` representation and applies the model's final norm on
    that engine, returning file paths plus prompt logprobs. Keeping this narrow
    boundary avoids retaining GPU tensors in Python orchestration.
    """

    def __init__(
        self,
        extract_callback: Any,
        *,
        dola_layers: list[int] | None = None,
        dola_chunk_size: int = 64,
    ) -> None:
        self.extract_callback = extract_callback
        self.dola_layers = list(dola_layers or [])
        self.dola_chunk_size = int(dola_chunk_size)

    async def extract(self, **kwargs: Any) -> ExtractionResult:
        started = time.perf_counter()
        callback_kwargs = dict(kwargs)
        if self.dola_layers:
            callback_kwargs.update(
                dola_layers=self.dola_layers,
                dola_chunk_size=self.dola_chunk_size,
            )
        payload = await self.extract_callback(**callback_kwargs)
        result = payload if isinstance(payload, ExtractionResult) else ExtractionResult(**payload)
        if result.prompt_tokens is None:
            result.prompt_tokens = int(kwargs["output_start"])
        if result.model_id is None:
            result.model_id = getattr(kwargs.get("engine"), "model", None)
        result.extraction_seconds += time.perf_counter() - started
        try:
            result.validate(list(kwargs["output_token_ids"]))
        except BaseException:
            result.cleanup()
            raise
        return result


class PrecomputedExtractionConnector:
    """Test/offline connector backed by precomputed extraction results."""

    def __init__(self, results: dict[str, ExtractionResult]) -> None:
        self.results = results

    async def extract(self, **kwargs: Any) -> ExtractionResult:
        result = self.results[str(kwargs["example_id"])]
        try:
            result.validate(list(kwargs["output_token_ids"]))
        except BaseException:
            result.cleanup()
            raise
        return result


class NativeVLLMExtractionConnector:
    """Use the live AsyncLLM's prompt scoring and token-embedding hook."""

    def __init__(
        self,
        temp_dir: str | Path | None = None,
        *,
        dola_layers: list[int] | None = None,
        dola_chunk_size: int = 64,
    ) -> None:
        self.temp_dir = Path(temp_dir) if temp_dir is not None else None
        self.dola_layers = list(dola_layers or [])
        self.dola_chunk_size = int(dola_chunk_size)

    async def extract(self, **kwargs: Any) -> ExtractionResult:
        engine = kwargs["engine"]
        extract = getattr(engine, "extract_trace", None)
        if not callable(extract):
            raise RuntimeError("Configured engine does not support native trace extraction")
        return await extract(
            full_token_ids=list(kwargs["full_token_ids"]),
            output_start=int(kwargs["output_start"]),
            output_token_ids=list(kwargs["output_token_ids"]),
            example_id=str(kwargs["example_id"]),
            temp_dir=self.temp_dir,
            cache_salt=kwargs.get("cache_salt"),
            dola_layers=self.dola_layers,
            dola_chunk_size=self.dola_chunk_size,
        )


def normalize_hidden_file(raw_path: Path, normalized_path: Path) -> float:
    """Apply the model's final RMS/LayerNorm in normal model dtype.

    This helper intentionally does not cast to FP32.  The caller passes a raw
    tensor already captured at layer ``num_hidden_layers`` and should normally
    perform model-specific final normalization in the extraction worker.  This
    fallback performs L2 normalization for fixtures and generic connectors.
    """
    started = time.perf_counter()
    import torch

    hidden = torch.load(raw_path, map_location="cpu", weights_only=True)
    normalized = torch.nn.functional.normalize(hidden, dim=-1)
    torch.save(normalized, normalized_path)
    return time.perf_counter() - started
