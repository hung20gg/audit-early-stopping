"""vLLM worker-side intermediate-layer capture and JSD projection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from replay_eval.v2.dola.math import js_divergence_from_logits


def find_final_norm(model: Any) -> Any:
    for path in (
        "language_model.model.norm",
        "model.norm",
        "model.model.norm",
    ):
        value = model
        for part in path.split("."):
            value = getattr(value, part, None)
            if value is None:
                break
        if callable(value):
            return value
    raise RuntimeError(
        f"Could not locate the base model's final norm on {type(model).__name__}"
    )


def _find_transformer_layers(model: Any) -> list[Any]:
    for path in (
        "language_model.model.layers",
        "model.layers",
        "model.model.layers",
        "transformer.h",
    ):
        value = model
        for part in path.split("."):
            value = getattr(value, part, None)
            if value is None:
                break
        if value is not None and hasattr(value, "__len__"):
            return list(value)
    raise RuntimeError(f"Could not locate transformer layers on {type(model).__name__}")


def _effective_layer_output(output: Any) -> Any:
    import torch

    hidden = output[0] if isinstance(output, tuple) else output
    residual = (
        output[1]
        if isinstance(output, tuple)
        and len(output) > 1
        and isinstance(output[1], torch.Tensor)
        and output[1].shape == hidden.shape
        else None
    )
    return hidden if residual is None else hidden + residual


@dataclass
class DoLaWorkerCapture:
    """Own all intermediate-layer hooks and projection state for one request."""

    layers: list[int]
    chunks: dict[int, list[Any]]
    handles: tuple[Any, ...]

    @classmethod
    def start(
        cls,
        model: Any,
        requested_layers: list[int],
        copy_to_host: Callable[[Any], Any],
    ) -> "DoLaWorkerCapture":
        requested = [int(value) for value in requested_layers]
        if not requested:
            return cls([], {}, ())
        if len(requested) != len(set(requested)):
            raise ValueError("DoLa layers must not contain duplicates")

        from vllm.distributed import get_tensor_model_parallel_world_size

        tensor_parallel_size = get_tensor_model_parallel_world_size()
        if tensor_parallel_size != 1:
            raise RuntimeError(
                "DoLa layer projection currently requires tensor_parallel_size=1; "
                f"got {tensor_parallel_size}"
            )
        model_layers = _find_transformer_layers(model)
        invalid = [
            layer for layer in requested if layer < 0 or layer >= len(model_layers)
        ]
        if invalid:
            raise ValueError(
                f"Invalid DoLa layers {invalid}; model has {len(model_layers)} layers"
            )

        chunks: dict[int, list[Any]] = {layer: [] for layer in requested}

        def capture(layer: int):
            def hook(_module: Any, _args: tuple[Any, ...], output: Any) -> None:
                chunks[layer].append(copy_to_host(_effective_layer_output(output)))

            return hook

        handles = tuple(
            model_layers[layer].register_forward_hook(capture(layer))
            for layer in requested
        )
        return cls(requested, chunks, handles)

    def project_jsd(
        self,
        *,
        model: Any,
        final_normalized: Any,
        expected_tokens: int,
        output_start: int,
        output_end: int,
        chunk_size: int,
    ) -> dict[str, list[float]]:
        import torch

        if not self.layers:
            return {}
        if chunk_size < 1:
            raise ValueError("DoLa projection chunk_size must be positive")
        norm = find_final_norm(model)
        compute_logits = getattr(model, "compute_logits", None)
        if not callable(compute_logits):
            raise RuntimeError("DoLa requires the vLLM model's compute_logits()")
        complete = {}
        for layer in self.layers:
            chunks = self.chunks[layer]
            if not chunks:
                raise RuntimeError(f"DoLa hook for layer {layer} captured no states")
            states = torch.cat(chunks, dim=0)[:expected_tokens]
            complete[layer] = states[output_start:output_end].contiguous()

        try:
            projection_parameter = next(norm.parameters())
        except StopIteration:
            projection_parameter = None
        if projection_parameter is None:
            raise RuntimeError("DoLa could not determine projection device/dtype")

        score_chunks: dict[int, list[Any]] = {layer: [] for layer in self.layers}
        for start in range(0, final_normalized.shape[0], chunk_size):
            end = start + chunk_size
            mature = final_normalized[start:end].to(
                device=projection_parameter.device,
                dtype=projection_parameter.dtype,
            )
            final_logits = compute_logits(mature)
            if final_logits is None:
                raise RuntimeError(
                    "DoLa projection is only available on the final pipeline rank"
                )
            for layer in self.layers:
                hidden = complete[layer][start:end].to(
                    device=mature.device, dtype=mature.dtype
                )
                normalized = norm(hidden)
                if isinstance(normalized, tuple):
                    normalized = normalized[0]
                premature_logits = compute_logits(normalized)
                if premature_logits is None:
                    raise RuntimeError(
                        "DoLa projection is only available on the final pipeline rank"
                    )
                score_chunks[layer].append(
                    js_divergence_from_logits(final_logits, premature_logits).cpu()
                )

        outputs: dict[str, list[float]] = {}
        layer_scores = []
        for layer in self.layers:
            score = torch.cat(score_chunks[layer], dim=0)
            outputs[f"dola_jsd.layer_{layer}"] = [
                float(value) for value in score.tolist()
            ]
            layer_scores.append(score)
        stacked = torch.stack(layer_scores, dim=-1)
        outputs["dola_jsd.max"] = [
            float(value) for value in stacked.amax(dim=-1).tolist()
        ]
        outputs["dola_jsd.selected_layer"] = [
            float(self.layers[index]) for index in stacked.argmax(dim=-1).tolist()
        ]
        return outputs

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = ()
        self.chunks.clear()


__all__ = ["DoLaWorkerCapture", "find_final_norm"]
