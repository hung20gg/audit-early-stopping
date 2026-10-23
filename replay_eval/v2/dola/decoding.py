"""Hugging Face DoLa contrastive decoder."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import time
from typing import Any, Iterable

from replay_eval.v2.dola.math import (
    contrast_logits,
    layer_js_divergences,
)


LOGGER = logging.getLogger(__name__)


def _final_norm(model: Any) -> Any:
    for path in (
        "model.norm",
        "model.model.norm",
        "language_model.model.norm",
        "model.language_model.model.norm",
        "transformer.ln_f",
    ):
        value = model
        for part in path.split("."):
            value = getattr(value, part, None)
            if value is None:
                break
        if callable(value):
            return value
    raise RuntimeError("Could not locate the model's final normalization layer")


def _project_candidates(model: Any, hidden: Any) -> Any:
    """Project all ``[batch, layers, hidden]`` candidates in one LM-head call."""
    head = model.get_output_embeddings()
    if head is None:
        raise RuntimeError("The model does not expose output embeddings")
    batch, layers, width = hidden.shape
    normalized = _final_norm(model)(hidden.reshape(batch * layers, width))
    if isinstance(normalized, tuple):
        normalized = normalized[0]
    return head(normalized).reshape(batch, layers, -1)


def _validate_layers(hidden_states: tuple[Any, ...], layers: Iterable[int]) -> list[int]:
    # Transformers returns embedding output at index 0, block outputs after
    # that, and the mature final state at -1. Candidate indices use this
    # public hidden_states tuple convention.
    values = [int(layer) for layer in layers]
    if not values:
        raise ValueError("dola_layers must contain at least one layer")
    invalid = [
        layer for layer in values if layer < 0 or layer >= len(hidden_states) - 1
    ]
    if invalid:
        raise ValueError(
            f"Invalid premature layers {invalid}; expected values in "
            f"[0, {len(hidden_states) - 2}]"
        )
    if len(values) != len(set(values)):
        raise ValueError("dola_layers must not contain duplicates")
    return values


@dataclass
class DoLaGeneration:
    token_ids: list[int]
    text: str
    selected_layers: list[int]
    layer_jsd: list[dict[int, float]]
    finish_reason: str
    total_seconds: float
    first_token_seconds: float | None
    token_logprobs: list[dict[int, float]] = field(default_factory=list)

    @property
    def tokens_per_second(self) -> float:
        return len(self.token_ids) / self.total_seconds if self.total_seconds else 0.0


class DoLaDecoder:
    """Single-example DoLa decoder for a Hugging Face causal language model."""

    def __init__(self, model: Any, tokenizer: Any) -> None:
        self.model = model
        self.tokenizer = tokenizer

    def generate(
        self,
        prompt_token_ids: list[int],
        *,
        dola_layers: Iterable[int],
        max_new_tokens: int,
        alpha: float = 0.1,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 0,
        seed: int | None = None,
        eos_token_ids: Iterable[int] | None = None,
        log_every: int = 0,
        num_logprobs: int = 0,
    ) -> DoLaGeneration:
        import torch

        if not prompt_token_ids:
            raise ValueError("prompt_token_ids must not be empty")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if temperature < 0.0 or not 0.0 < top_p <= 1.0 or top_k < 0:
            raise ValueError("Invalid temperature/top_p/top_k sampling values")
        if num_logprobs < 0:
            raise ValueError("num_logprobs must be nonnegative")
        if log_every < 0:
            raise ValueError("log_every must be >= 0")

        device = next(self.model.parameters()).device
        input_ids = torch.tensor([prompt_token_ids], device=device, dtype=torch.long)
        model_kwargs: dict[str, Any] = {
            "attention_mask": torch.ones_like(input_ids),
            "use_cache": True,
        }
        generated: list[int] = []
        token_logprobs: list[dict[int, float]] = []
        selected_layers: list[int] = []
        layer_jsd: list[dict[int, float]] = []
        eos = set(int(value) for value in (eos_token_ids or []))
        if not eos:
            configured_eos = getattr(
                getattr(self.model, "generation_config", None), "eos_token_id", None
            )
            configured_eos = (
                configured_eos
                if isinstance(configured_eos, (list, tuple, set))
                else [configured_eos]
            )
            eos.update(int(value) for value in configured_eos if value is not None)
            tokenizer_eos = getattr(self.tokenizer, "eos_token_id", None)
            if tokenizer_eos is not None:
                eos.add(int(tokenizer_eos))
        generator = None
        if seed is not None:
            generator = torch.Generator(device=device).manual_seed(int(seed))
        candidate_layers: list[int] | None = None
        finish_reason = "length"
        started = time.perf_counter()
        first_token_seconds: float | None = None

        with torch.inference_mode():
            for _ in range(max_new_tokens):
                model_inputs = self.model.prepare_inputs_for_generation(
                    input_ids, **model_kwargs
                )
                outputs = self.model(
                    **model_inputs,
                    output_hidden_states=True,
                    return_dict=True,
                )
                hidden_states = tuple(outputs.hidden_states)
                if candidate_layers is None:
                    candidate_layers = _validate_layers(hidden_states, dola_layers)
                final_logits = outputs.logits[:, -1, :]
                candidate_hidden = torch.stack(
                    [hidden_states[layer][:, -1, :] for layer in candidate_layers],
                    dim=1,
                )
                candidate_logits = _project_candidates(self.model, candidate_hidden)
                divergences = layer_js_divergences(final_logits, candidate_logits)
                selected_index = divergences.argmax(dim=-1)
                gather_index = selected_index[:, None, None].expand(
                    -1, 1, candidate_logits.shape[-1]
                )
                premature = candidate_logits.gather(1, gather_index).squeeze(1)
                layer_tensor = torch.as_tensor(
                    candidate_layers, device=selected_index.device
                )
                layer_id = layer_tensor[selected_index]
                scores = contrast_logits(final_logits, premature, alpha=alpha)
                token = self._sample(
                    scores,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    generator=generator,
                )
                token_id = int(token.item())
                if num_logprobs:
                    # Retain the untempered contrast distribution, like vLLM
                    # raw logprobs, independently of decoder sampling filters.
                    logp = torch.log_softmax(scores.float(), dim=-1)[0]
                    values, indices = logp.topk(min(num_logprobs, logp.numel()))
                    row = dict(zip(indices.cpu().tolist(), values.cpu().tolist()))
                    row[token_id] = float(logp[token_id].item())
                    token_logprobs.append(row)
                generated.append(token_id)
                selected_layers.append(int(layer_id.item()))
                layer_jsd.append(
                    {
                        layer: float(divergences[0, index].item())
                        for index, layer in enumerate(candidate_layers)
                    }
                )
                elapsed = time.perf_counter() - started
                if first_token_seconds is None:
                    first_token_seconds = elapsed
                if log_every and (
                    len(generated) == 1 or len(generated) % log_every == 0
                ):
                    LOGGER.info(
                        "DoLa decode progress tokens=%d elapsed=%.2fs tokens_per_second=%.2f "
                        "selected_layer=%d max_jsd=%.6f",
                        len(generated),
                        elapsed,
                        len(generated) / elapsed if elapsed else 0.0,
                        selected_layers[-1],
                        float(divergences.amax().item()),
                    )
                input_ids = torch.cat((input_ids, token.reshape(1, 1)), dim=-1)
                model_kwargs = self.model._update_model_kwargs_for_generation(
                    outputs,
                    model_kwargs,
                    is_encoder_decoder=False,
                )
                if token_id in eos:
                    finish_reason = "stop"
                    break

        total_seconds = time.perf_counter() - started
        if finish_reason == "length":
            LOGGER.warning(
                "DoLa generation reached max_new_tokens=%d without an EOS token",
                max_new_tokens,
            )
        LOGGER.info(
            "DoLa decode finished tokens=%d finish_reason=%s total=%.2fs "
            "first_token=%.2fs tokens_per_second=%.2f",
            len(generated),
            finish_reason,
            total_seconds,
            first_token_seconds or 0.0,
            len(generated) / total_seconds if total_seconds else 0.0,
        )
        return DoLaGeneration(
            token_ids=generated,
            text=self.tokenizer.decode(generated, skip_special_tokens=False),
            selected_layers=selected_layers,
            layer_jsd=layer_jsd,
            finish_reason=finish_reason,
            total_seconds=total_seconds,
            first_token_seconds=first_token_seconds,
            token_logprobs=token_logprobs,
        )

    @staticmethod
    def _sample(
        scores: Any,
        *,
        temperature: float,
        top_p: float,
        top_k: int,
        generator: Any,
    ) -> Any:
        import torch

        if temperature == 0.0:
            return scores.argmax(dim=-1)
        logits = scores / temperature
        if top_k:
            threshold = logits.topk(
                min(top_k, logits.shape[-1]), dim=-1
            ).values[:, -1:]
            logits = logits.masked_fill(logits < threshold, float("-inf"))
        if top_p < 1.0:
            ordered, indices = logits.sort(dim=-1, descending=True)
            cumulative = ordered.softmax(dim=-1).cumsum(dim=-1)
            remove = cumulative > top_p
            remove[:, 1:] = remove[:, :-1].clone()
            remove[:, 0] = False
            ordered = ordered.masked_fill(remove, float("-inf"))
            logits = torch.full_like(logits, float("-inf")).scatter(
                1, indices, ordered
            )
        return torch.multinomial(
            logits.softmax(dim=-1), 1, generator=generator
        ).squeeze(1)


__all__ = ["DoLaDecoder", "DoLaGeneration"]
