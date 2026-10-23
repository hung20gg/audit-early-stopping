"""Recover next-token log probabilities from extracted hidden states.

``replay_eval.hidden_states`` stores the decoder states immediately before the
model's final normalization.  This module applies that normalization and the
causal-LM vocabulary head without running the transformer again.

Notebook example::

    from replay_eval.hidden_logprobs import HiddenStateLogProbs

    scorer = HiddenStateLogProbs.from_artifact(
        "gpt-oss-20b-output-head.pt", device="cuda"
    )
    result = scorer.score_tokens(hidden_states, assistant_token_ids)
    print(result.token_logprobs, result.perplexity)

Directory/CLI example::

    python -m replay_eval.hidden_logprobs \
      --input embeddings/ --weights gpt-oss-20b-output-head.pt \
      --output token_logprobs.jsonl --device cuda

The state at token ``i`` predicts token ``i + 1``.  Consequently, scoring an
extraction with ``T`` assistant states produces ``T - 1`` log probabilities.
The first assistant token cannot be recovered because the extraction does not
store the final prompt state.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import torch
from torch import nn

from replay_eval.extract_output_weights import ARTIFACT_FORMAT


LOGGER = logging.getLogger("replay_eval.hidden_logprobs")


def _get_text_backbone(model: Any) -> Any:
    """Resolve the decoder module from text-only and multimodal wrappers."""
    backbone = getattr(model, "model", model)
    language_model = getattr(backbone, "language_model", None)
    if language_model is None:
        language_model = getattr(model, "language_model", None)
    return language_model if language_model is not None else backbone


def _get_output_head(model: Any) -> Any:
    getter = getattr(model, "get_output_embeddings", None)
    if callable(getter):
        head = getter()
        if head is not None:
            return head
    for owner in (model, getattr(model, "language_model", None)):
        head = getattr(owner, "lm_head", None) if owner is not None else None
        if head is not None:
            return head
    raise ValueError(f"Could not locate the output head on {type(model).__name__}")


@dataclass(frozen=True)
class TokenLogProbOutput:
    """Observed next-token scores aligned to their target token IDs."""

    token_ids: Any
    token_logprobs: Any
    source_indices: Any

    @property
    def mean_logprob(self) -> float:
        if self.token_logprobs.numel() == 0:
            return float("nan")
        return float(self.token_logprobs.float().mean().item())

    @property
    def perplexity(self) -> float:
        mean = self.mean_logprob
        return math.exp(-mean) if math.isfinite(mean) else float("nan")


class _StoredRMSNorm(nn.Module):
    def __init__(
        self,
        weight: Any,
        *,
        eps: float,
        weight_offset: float,
        operation: str,
    ) -> None:
        super().__init__()
        if operation not in {"rsqrt", "pow"}:
            raise ValueError(f"Unsupported RMSNorm operation {operation!r}")
        self.weight = nn.Parameter(torch.as_tensor(weight), requires_grad=False)
        self.eps = eps
        self.weight_offset = weight_offset
        self.operation = operation

    def forward(self, hidden_states: Any) -> Any:
        input_dtype = hidden_states.dtype
        value = hidden_states.float()
        variance = value.pow(2).mean(-1, keepdim=True) + self.eps
        inverse_rms = (
            torch.pow(variance, -0.5)
            if self.operation == "pow"
            else torch.rsqrt(variance)
        )
        value = value * inverse_rms
        value = value * (self.weight.float() + self.weight_offset)
        return value.to(input_dtype)


class _StoredOutputHead(nn.Module):
    def __init__(self, weight: Any, bias: Any | None) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.as_tensor(weight), requires_grad=False)
        self.bias = (
            nn.Parameter(torch.as_tensor(bias), requires_grad=False)
            if bias is not None
            else None
        )

    def forward(self, hidden_states: Any) -> Any:
        return torch.nn.functional.linear(hidden_states, self.weight, self.bias)


class HiddenStateLogProbs:
    """Final normalization and vocabulary head for pre-norm hidden states.

    Use :meth:`from_artifact` for output weights produced by
    ``replay_eval.extract_output_weights``.  This path never constructs or
    loads the transformer.  :meth:`from_model` is available when a causal LM
    already exists in notebook memory.
    """

    def __init__(
        self,
        final_norm: Any,
        output_head: Any,
        *,
        model_id: str | None = None,
        device: str = "auto",
        dtype: str = "auto",
        final_logit_softcapping: float | None = None,
    ) -> None:
        self.final_norm = final_norm.eval()
        self.output_head = output_head.eval()
        self.model_id = model_id
        self.final_logit_softcapping = final_logit_softcapping
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if dtype == "auto":
            self.final_norm.to(self.device)
            self.output_head.to(self.device)
        else:
            try:
                torch_dtype = getattr(torch, dtype)
            except AttributeError as exc:
                raise ValueError(f"Unknown torch dtype {dtype!r}") from exc
            self.final_norm.to(device=self.device, dtype=torch_dtype)
            self.output_head.to(device=self.device, dtype=torch_dtype)

    @classmethod
    def from_model(
        cls,
        model: Any,
        *,
        model_id: str | None = None,
        device: str = "auto",
    ) -> "HiddenStateLogProbs":
        """Build from a loaded Hugging Face causal LM without copying weights."""
        backbone = _get_text_backbone(model)
        final_norm = getattr(backbone, "norm", None)
        if final_norm is None:
            raise ValueError(
                f"Could not locate the final norm on {type(backbone).__name__}"
            )
        if model_id is None:
            model_id = getattr(getattr(model, "config", None), "name_or_path", None)
        config = getattr(model, "config", None)
        text_config = getattr(config, "text_config", None) or config
        return cls(
            final_norm,
            _get_output_head(model),
            model_id=model_id,
            device=device,
            final_logit_softcapping=getattr(
                text_config, "final_logit_softcapping", None
            ),
        )

    @classmethod
    def from_artifact(
        cls,
        path: str | Path,
        *,
        device: str = "auto",
        dtype: str = "auto",
    ) -> "HiddenStateLogProbs":
        """Load only a saved final norm and vocabulary head artifact."""
        artifact = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(artifact, dict) or artifact.get("format") != ARTIFACT_FORMAT:
            raise ValueError(
                f"{path} is not a supported output-weight artifact "
                f"({ARTIFACT_FORMAT})"
            )
        norm = artifact.get("norm") or {}
        output_head = artifact.get("output_head") or {}
        if "weight" not in norm or "weight" not in output_head:
            raise ValueError(f"Output-weight artifact {path} is incomplete")
        return cls(
            _StoredRMSNorm(
                norm["weight"],
                eps=float(norm["eps"]),
                weight_offset=float(norm.get("weight_offset", 0.0)),
                operation=str(norm.get("operation", "rsqrt")),
            ),
            _StoredOutputHead(output_head["weight"], output_head.get("bias")),
            model_id=artifact.get("model_id"),
            device=device,
            dtype=dtype,
            final_logit_softcapping=artifact.get("final_logit_softcapping"),
        )

    def _prepare_hidden(self, hidden_states: Any) -> Any:
        hidden = torch.as_tensor(hidden_states, device=self.device)
        if hidden.ndim < 2:
            raise ValueError(
                f"hidden_states must have shape (..., T, H), got {tuple(hidden.shape)}"
            )
        weight = getattr(self.output_head, "weight", None)
        if weight is not None and hidden.dtype != weight.dtype:
            hidden = hidden.to(dtype=weight.dtype)
        return hidden

    def logits(self, hidden_states: Any) -> Any:
        """Return vocabulary logits with shape ``(..., T, vocab_size)``."""
        hidden = self._prepare_hidden(hidden_states)
        with torch.inference_mode():
            return self._apply_output_head(self.final_norm(hidden))

    def _apply_output_head(self, normalized_hidden: Any) -> Any:
        """Apply the vocabulary projection and model-configured logit softcap."""
        logits = self.output_head(normalized_hidden)
        if self.final_logit_softcapping is not None:
            cap = self.final_logit_softcapping
            logits = torch.tanh(logits / cap) * cap
        return logits

    def log_probs(self, hidden_states: Any) -> Any:
        """Return full-vocabulary float32 log probabilities.

        This can be very large.  Prefer :meth:`score_tokens` when only the
        observed/generated token probabilities are needed.
        """
        return torch.log_softmax(self.logits(hidden_states).float(), dim=-1)

    def score_tokens(
        self,
        hidden_states: Any,
        token_ids: Any,
        *,
        shift: bool = True,
        chunk_size: int = 128,
    ) -> TokenLogProbOutput:
        """Score target tokens without materializing all sequence logits.

        With the default ``shift=True``, hidden state ``[..., i, :]`` is scored
        against ``token_ids[..., i + 1]``.  Set ``shift=False`` only when the
        supplied target IDs are already aligned one-for-one with the states.
        """
        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
        hidden = self._prepare_hidden(hidden_states)
        targets = torch.as_tensor(token_ids, dtype=torch.long, device=self.device)
        if hidden.shape[:-1] != targets.shape:
            raise ValueError(
                "hidden/token shape mismatch: "
                f"hidden={tuple(hidden.shape)}, token_ids={tuple(targets.shape)}"
            )
        if shift:
            hidden = hidden[..., :-1, :]
            targets = targets[..., 1:]

        leading_shape = targets.shape
        flat_hidden = hidden.reshape(-1, hidden.shape[-1])
        flat_targets = targets.reshape(-1)
        chunks = []
        with torch.inference_mode():
            for start in range(0, flat_targets.numel(), chunk_size):
                stop = min(start + chunk_size, flat_targets.numel())
                logits = self._apply_output_head(
                    self.final_norm(flat_hidden[start:stop])
                ).float()
                selected = logits.gather(
                    1, flat_targets[start:stop, None]
                ).squeeze(1)
                chunks.append(selected - torch.logsumexp(logits, dim=-1))
        if chunks:
            scores = torch.cat(chunks).reshape(leading_shape)
        else:
            scores = torch.empty(
                leading_shape, dtype=torch.float32, device=self.device
            )
        source_length = leading_shape[-1] if leading_shape else 0
        source_indices = torch.arange(source_length, device=self.device)
        if leading_shape[:-1]:
            source_indices = source_indices.expand(*leading_shape[:-1], source_length)
        return TokenLogProbOutput(targets, scores, source_indices)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_extractions(path: Path) -> Iterator[tuple[dict[str, Any], Any]]:
    """Yield ``(metadata, hidden_states)`` from a NumPy or PT extraction."""
    import numpy as np
    import torch

    if path.is_file():
        if path.suffix == ".npy":
            metadata_path = path.with_suffix(".json")
            if not metadata_path.exists():
                raise FileNotFoundError(f"Missing metadata file {metadata_path}")
            yield _load_json(metadata_path), torch.from_numpy(np.load(path))
            return
        if path.suffix in {".pt", ".pth"}:
            payload = torch.load(path, map_location="cpu", weights_only=True)
            items = payload if isinstance(payload, list) else [payload]
            for item in items:
                if not isinstance(item, dict) or "hidden_states" not in item:
                    raise ValueError(f"Unsupported PT payload in {path}")
                yield item, item["hidden_states"]
            return
        raise ValueError(f"Unsupported input file {path}")

    index_path = path / "index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing extraction index {index_path}")
    index = _load_json(index_path)
    shard_cache: dict[str, Any] = {}
    for entry in index:
        if "embedding_file" in entry:
            metadata = _load_json(path / entry["metadata_file"])
            hidden = torch.from_numpy(np.load(path / entry["embedding_file"]))
            yield metadata, hidden
            continue
        shard_name = entry.get("shard_file")
        if not shard_name:
            raise ValueError(f"Unrecognized extraction index entry: {entry!r}")
        if shard_name not in shard_cache:
            shard_cache = {
                shard_name: torch.load(
                    path / shard_name, map_location="cpu", weights_only=True
                )
            }
        item = shard_cache[shard_name][int(entry["position"])]
        yield item, item["hidden_states"]


def _infer_model(input_path: Path) -> str | None:
    manifest_path = input_path / "manifest.json" if input_path.is_dir() else None
    if manifest_path is None or not manifest_path.exists():
        return None
    return _load_json(manifest_path).get("model")


def run_cli(args: argparse.Namespace) -> None:
    inferred_model = _infer_model(args.input)
    scorer = HiddenStateLogProbs.from_artifact(
        args.weights,
        device=args.device,
        dtype=args.dtype,
    )
    model_id = scorer.model_id
    if inferred_model and model_id and model_id != inferred_model:
        raise ValueError(
            f"Input manifest uses {inferred_model!r}, but --weights contains "
            f"{model_id!r}"
        )
    model_id = model_id or inferred_model
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.output.open("w", encoding="utf-8") as handle:
        for metadata, hidden in iter_extractions(args.input):
            representation = metadata.get("hidden_representation")
            if representation not in (None, "pre_final_rmsnorm"):
                raise ValueError(
                    f"Expected pre_final_rmsnorm states, got {representation!r}"
                )
            token_ids = metadata.get("assistant_token_ids")
            if token_ids is None:
                raise ValueError("Extraction metadata has no assistant_token_ids")
            embedding_model = metadata.get("embedding_model")
            if embedding_model and model_id and embedding_model != model_id:
                raise ValueError(
                    f"Hidden states use {embedding_model!r}, but the output head is "
                    f"from {model_id!r}"
                )
            result = scorer.score_tokens(
                hidden, token_ids, shift=True, chunk_size=args.chunk_size
            )
            mean_logprob = result.mean_logprob
            perplexity = result.perplexity
            row = {
                "problem_id": metadata.get("problem_id"),
                "line_number": metadata.get("line_number"),
                "model": model_id,
                "alignment": "hidden[i] predicts assistant_token_ids[i+1]",
                "first_token_unscored": True,
                "token_ids": result.token_ids.detach().cpu().tolist(),
                "token_logprobs": result.token_logprobs.detach().cpu().tolist(),
                "mean_logprob": (
                    mean_logprob if math.isfinite(mean_logprob) else None
                ),
                "perplexity": perplexity if math.isfinite(perplexity) else None,
            }
            handle.write(json.dumps(row, allow_nan=False) + "\n")
            count += 1
    LOGGER.info("Wrote %d scored trajectories to %s", count, args.output)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Extraction directory, .npy record, or .pt shard.",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Output JSONL file."
    )
    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="Artifact created by replay_eval.extract_output_weights.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=("auto", "float16", "bfloat16", "float32"),
    )
    parser.add_argument("--chunk-size", type=int, default=128)
    return parser.parse_args(argv)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    run_cli(parse_args())


if __name__ == "__main__":
    main()
