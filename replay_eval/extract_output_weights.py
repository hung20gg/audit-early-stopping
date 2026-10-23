"""Extract only a model's final norm and vocabulary-head weights.

The extractor reads the Hugging Face safetensor index and opens only the
shards containing the final normalization and LM-head (or tied embedding)
weights.  It never constructs the transformer model.

Example::

    python -m replay_eval.extract_output_weights \
      --model openai/gpt-oss-20b --output gpt-oss-20b-output-head.pt
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger("replay_eval.extract_output_weights")
ARTIFACT_FORMAT = "replay_eval.output_weights.v1"


def _download_or_local(
    model: str | Path,
    filename: str,
    *,
    revision: str,
    token: str | None,
    cache_dir: Path | None,
    local_files_only: bool,
) -> Path:
    local = Path(model)
    if local.is_dir():
        path = local / filename
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            str(model),
            filename,
            revision=revision,
            token=token,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
    )


def _checkpoint_index(
    model: str | Path,
    *,
    revision: str,
    token: str | None,
    cache_dir: Path | None,
    local_files_only: bool,
) -> tuple[dict[str, str], dict[str, Path]]:
    """Return the tensor-to-shard map and any already-resolved shard paths."""
    from huggingface_hub.errors import EntryNotFoundError

    local = Path(model)
    index_name = "model.safetensors.index.json"
    try:
        index_path = _download_or_local(
            model,
            index_name,
            revision=revision,
            token=token,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
    except (FileNotFoundError, EntryNotFoundError):
        single = _download_or_local(
            model,
            "model.safetensors",
            revision=revision,
            token=token,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        from safetensors import safe_open

        with safe_open(single, framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
        return {key: single.name for key in keys}, {single.name: single}

    payload = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError(f"Invalid safetensor index {index_path}")
    resolved = {}
    if local.is_dir():
        resolved = {name: local / name for name in set(weight_map.values())}
    return {str(key): str(value) for key, value in weight_map.items()}, resolved


def _select_key(
    keys: set[str], candidates: tuple[str, ...], *, kind: str
) -> str:
    for candidate in candidates:
        if candidate in keys:
            return candidate
    if kind == "final norm":
        fallback = sorted(
            key
            for key in keys
            if key.endswith(".norm.weight") and ".layers." not in key
        )
    else:
        fallback = sorted(key for key in keys if key.endswith(".lm_head.weight"))
    if len(fallback) == 1:
        return fallback[0]
    raise ValueError(
        f"Could not uniquely locate {kind}; candidates present={fallback!r}"
    )


def _load_selected_tensors(
    model: str | Path,
    tensor_to_shard: dict[str, str],
    selected_keys: set[str],
    resolved_shards: dict[str, Path],
    *,
    revision: str,
    token: str | None,
    cache_dir: Path | None,
    local_files_only: bool,
) -> dict[str, Any]:
    from safetensors import safe_open

    by_shard: dict[str, list[str]] = {}
    for key in selected_keys:
        by_shard.setdefault(tensor_to_shard[key], []).append(key)
    tensors = {}
    for shard_name, shard_keys in by_shard.items():
        shard_path = resolved_shards.get(shard_name)
        if shard_path is None:
            shard_path = _download_or_local(
                model,
                shard_name,
                revision=revision,
                token=token,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            )
        LOGGER.info("Reading %s from %s", ", ".join(shard_keys), shard_name)
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            for key in shard_keys:
                tensors[key] = handle.get_tensor(key)
    return tensors


def extract_output_weights(
    model: str | Path,
    output: Path,
    *,
    revision: str = "main",
    token: str | None = None,
    cache_dir: Path | None = None,
    local_files_only: bool = False,
    dtype: str = "preserve",
) -> dict[str, Any]:
    """Extract and save a compact output-weight artifact."""
    import torch

    config_path = _download_or_local(
        model,
        "config.json",
        revision=revision,
        token=token,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    text_config = config.get("text_config") or config
    model_type = str(text_config.get("model_type") or config.get("model_type") or "")

    weight_map, resolved = _checkpoint_index(
        model,
        revision=revision,
        token=token,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    keys = set(weight_map)
    norm_key = _select_key(
        keys,
        (
            "model.language_model.norm.weight",
            "model.norm.weight",
            "language_model.model.norm.weight",
            "transformer.norm.weight",
        ),
        kind="final norm",
    )
    head_key = _select_key(
        keys,
        (
            "lm_head.weight",
            "model.language_model.embed_tokens.weight",
            "model.embed_tokens.weight",
            "language_model.model.embed_tokens.weight",
            "transformer.wte.weight",
        ),
        kind="output head",
    )
    bias_key = "lm_head.bias" if "lm_head.bias" in keys else None
    selected = {norm_key, head_key}
    if bias_key:
        selected.add(bias_key)
    tensors = _load_selected_tensors(
        model,
        weight_map,
        selected,
        resolved,
        revision=revision,
        token=token,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )

    if dtype == "preserve":
        torch_dtype = None
    else:
        try:
            torch_dtype = getattr(torch, dtype)
        except AttributeError as exc:
            raise ValueError(f"Unknown torch dtype {dtype!r}") from exc
    norm_weight = tensors[norm_key]
    head_weight = tensors[head_key]
    head_bias = tensors[bias_key] if bias_key else None
    if torch_dtype is not None:
        norm_weight = norm_weight.to(torch_dtype)
        head_weight = head_weight.to(torch_dtype)
        if head_bias is not None:
            head_bias = head_bias.to(torch_dtype)

    # Qwen3.5 stores RMSNorm's learned delta initialized at zero; its forward
    # uses (1 + weight). GPT-OSS and Gemma4 store the multiplicative weight.
    normalized_model_type = model_type.replace("_", ".")
    norm_weight_offset = 1.0 if normalized_model_type.startswith("qwen3.5") else 0.0
    norm_operation = "pow" if normalized_model_type.startswith("gemma4") else "rsqrt"
    configured_model_id = config.get("_name_or_path")
    if not configured_model_id and not Path(model).is_dir():
        configured_model_id = str(model)
    artifact = {
        "format": ARTIFACT_FORMAT,
        "model_id": configured_model_id,
        "source": str(model),
        "revision": revision,
        "model_type": model_type,
        "hidden_size": int(text_config.get("hidden_size", head_weight.shape[1])),
        "vocab_size": int(text_config.get("vocab_size", head_weight.shape[0])),
        "norm": {
            "weight": norm_weight.contiguous(),
            "eps": float(text_config.get("rms_norm_eps", 1e-6)),
            "weight_offset": norm_weight_offset,
            "operation": norm_operation,
            "source_key": norm_key,
        },
        "output_head": {
            "weight": head_weight.contiguous(),
            "bias": head_bias.contiguous() if head_bias is not None else None,
            "source_key": head_key,
        },
        "final_logit_softcapping": text_config.get("final_logit_softcapping"),
    }
    if tuple(head_weight.shape) != (artifact["vocab_size"], artifact["hidden_size"]):
        raise ValueError(
            "Output-head shape disagrees with config: "
            f"weight={tuple(head_weight.shape)}, expected="
            f"({artifact['vocab_size']}, {artifact['hidden_size']})"
        )
    if norm_weight.numel() != artifact["hidden_size"]:
        raise ValueError(
            "Final-norm shape disagrees with config: "
            f"weight={tuple(norm_weight.shape)}, hidden_size={artifact['hidden_size']}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, output)
    LOGGER.info(
        "Saved %s output weights (%s, %s) to %s",
        model,
        tuple(head_weight.shape),
        head_weight.dtype,
        output,
    )
    return artifact


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="HF model ID or local model directory.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--token", default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--dtype",
        default="preserve",
        choices=("preserve", "float16", "bfloat16", "float32"),
    )
    return parser.parse_args(argv)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    extract_output_weights(
        args.model,
        args.output,
        revision=args.revision,
        token=args.token,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
        dtype=args.dtype,
    )


if __name__ == "__main__":
    main()
