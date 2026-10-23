"""Extract reusable, label-free TERMINATOR embeddings from replay traces.

Each input row is processed once.  The resulting ``hidden_states`` tensor is
stored on CPU as ``torch.float16`` with shape ``(T, H)``: ``T`` contiguous
assistant reasoning tokens and ``H`` the base model hidden size (2560 for
Qwen/Qwen3.5-4B).  These are the pre-final-RMSNorm backbone states consumed by
the TERMINATOR extra decoder layer/FFN; they are not sentence embeddings.

The replay file may contain many curation methods for one trajectory.  They
share the one extracted tensor.  Labels are intentionally not written here.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

LOGGER = logging.getLogger("replay_eval.hidden_states")


def iter_runner_curation(path: Path) -> Iterator[dict[str, Any]]:
    """Yield replay rows, including traces whose records omit prefixes."""
    with path.open("r", encoding="utf-8") as handle:
        for source_row, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            logs = row.get("additional_logs") or {}
            trace = logs.get("curation_trace") or {}
            content = trace.get("content") or row.get("output")
            prompt = trace.get("input")
            records = logs.get("curation_records") or []
            if not isinstance(content, str) or not isinstance(records, list):
                continue
            # Prefer the trajectory paired with the stored rendered prompt.
            # Keep reasoning only; exclude </think> and the final answer.
            if "</think>" in content:
                content = content.split("</think>", 1)[0]
            method_results = logs.get("method_results") or []
            allow_over_16k = any(
                isinstance(result, dict)
                and str(result.get("method", "")).lower() in {"baseline_16k", "baseline_24k"}
                and bool(result.get("stopped_early"))
                for result in method_results
            )
            input_data = row.get("input") or {}
            yield {
                "source_row": source_row,
                "line_number": row.get("line_number", source_row),
                "problem_id": input_data.get("id") or input_data.get("problem_id"),
                "question": input_data.get("question", ""),
                "dataloader": row.get("dataloader"),
                "source_model": row.get("model"),
                "prompt": prompt if isinstance(prompt, str) else None,
                "content": content,
                "records": records,
                "allow_over_16k": allow_over_16k,
            }


def load_system_prompt(config_path: Path, dataloader: str) -> str:
    import yaml

    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    path = ((config.get("system_prompt") or {}).get("dataset") or {}).get(dataloader)
    if not path:
        raise ValueError(f"No dataset system prompt configured for {dataloader!r}")
    prompt_path = Path(path)
    if not prompt_path.is_absolute():
        prompt_path = PROJECT_ROOT / prompt_path
    return prompt_path.read_text(encoding="utf-8").strip()


@dataclass
class TextBackboneRuntime:
    model: Any
    tokenizer: Any
    backbone: Any
    device: str
    dtype: Any
    completed_forwards: int = 0


def _get_text_config(config: Any) -> Any:
    """Resolve the decoder config from text-only or multimodal configs."""
    text_config = getattr(config, "text_config", None)
    if text_config is not None:
        return text_config

    get_text_config = getattr(config, "get_text_config", None)
    if callable(get_text_config):
        try:
            resolved = get_text_config()
        except TypeError:
            resolved = get_text_config(decoder=True)
        if resolved is not None:
            return resolved
    return config


def _get_text_backbone(model: Any) -> Any:
    """Resolve the decoder module from text-only or multimodal model wrappers."""
    backbone = getattr(model, "model", model)
    language_model = getattr(backbone, "language_model", None)
    if language_model is None:
        language_model = getattr(model, "language_model", None)
    return language_model if language_model is not None else backbone


def _select_attention_backend(config: Any) -> str:
    """Choose an attention backend compatible with every decoder head size."""
    text_config = _get_text_config(config)
    head_dims = []
    for name in ("head_dim", "global_head_dim"):
        value = getattr(text_config, name, None)
        if isinstance(value, int):
            head_dims.append(value)

    per_layer_config = getattr(text_config, "per_layer_config", None)
    if hasattr(per_layer_config, "values"):
        for layer_config in per_layer_config.values():
            value = (
                layer_config.get("head_dim")
                if isinstance(layer_config, dict)
                else getattr(layer_config, "head_dim", None)
            )
            if isinstance(value, int):
                head_dims.append(value)

    # flash-attn 2 rejects head dimensions above 256. Gemma4E full-attention
    # layers use 512, so let PyTorch SDPA choose a compatible CUDA kernel.
    return "sdpa" if head_dims and max(head_dims) > 256 else "flash_attention_2"


def compute_last_hidden(loaded, text: str):
    """Backward-compatible helper for callers of the pre-curation API."""
    import torch

    input_ids = loaded.tokenizer(
        text, return_tensors="pt", add_special_tokens=False
    ).input_ids.to(loaded.device)
    if input_ids.shape[1] == 0:
        raise ValueError("Cannot extract a hidden state from empty text")
    with torch.no_grad():
        outputs = loaded.model(input_ids=input_ids, output_hidden_states=True)
    return outputs.hidden_states[-1][0, -1].detach().to(
        dtype=torch.float16, device="cpu"
    )


def build_sequences(loaded, rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compatibility path for the original sparse, labeled shard format.

    New extraction uses :func:`extract_sequence` and never calls this helper.
    It remains available so older downstream consumers can migrate without an
    import break.
    """
    import torch

    sequences: list[dict[str, Any]] = []
    for row in rows:
        by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in row.get("records", []):
            if not isinstance(record, dict) or not isinstance(record.get("method"), str):
                continue
            if not isinstance(record.get("reasoning_prefix"), str):
                continue
            if not isinstance(record.get("position"), int) or record["position"] < 1:
                continue
            by_method[record["method"]].append(record)
        cache: dict[tuple[int, str], Any] = {}
        for method, records in by_method.items():
            ordered = sorted(records, key=lambda item: item["position"])
            states = []
            for record in ordered:
                key = (record["position"], record["reasoning_prefix"])
                if key not in cache:
                    cache[key] = compute_last_hidden(
                        loaded, row.get("prompt", "") + record["reasoning_prefix"]
                    )
                states.append(cache[key])
            if states:
                sequences.append(
                    {
                        "problem_id": row.get("line_number"),
                        "method": method,
                        "positions": torch.tensor(
                            [item["position"] for item in ordered], dtype=torch.long
                        ),
                        "hidden_states": torch.stack(states),
                        "labels": torch.tensor(
                            [int(bool(item.get("triggered"))) for item in ordered],
                            dtype=torch.long,
                        ),
                    }
                )
    return sequences


def load_text_backbone(model_id: str, device: str = "auto", dtype: str = "bfloat16"):
    """Load a text decoder backbone without running the vocabulary head."""
    import torch
    from transformers import AutoConfig, AutoTokenizer

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = getattr(torch, dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    config = AutoConfig.from_pretrained(model_id)
    attention_backend = _select_attention_backend(config)
    LOGGER.info(
        "Using attention backend=%s for model=%s",
        attention_backend,
        model_id,
    )

    # Multimodal models expose their text decoder below ``language_model``.
    # Keeping the other modality towers on CPU avoids involving them in
    # text-only extraction.
    if "Qwen3.5" in model_id or "Qwen3_5" in model_id:
        from transformers.models.qwen3_5.modeling_qwen3_5 import (
            Qwen3_5ForConditionalGeneration,
        )

        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            attn_implementation=attention_backend,
        )
    else:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            attn_implementation=attention_backend,
        )
    backbone = _get_text_backbone(model)
    backbone.to(device).eval()
    return TextBackboneRuntime(model, tokenizer, backbone, device, torch_dtype)


def _render_chat(runtime: TextBackboneRuntime, system: str, question: str) -> str:
    tokenizer = runtime.tokenizer
    base = [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]
    config = runtime.model.config
    text_config = _get_text_config(config)
    if str(getattr(config, "model_type", "")).startswith("gemma4") or str(
        getattr(text_config, "model_type", "")
    ).startswith("gemma4"):
        # Gemma 4 controls thinking through a token in the system message;
        # this must match the prompt used by the generation runner.
        from methods.chat_template import prepare_gemma4_messages

        base = prepare_gemma4_messages(base, enable_thinking=True)
    return tokenizer.apply_chat_template(
        base, tokenize=False, add_generation_prompt=True, enable_thinking=True
    )


def _resolve_prompt_text(
    runtime: TextBackboneRuntime,
    row: dict[str, Any],
    system_prompt: str,
) -> str:
    """Use the generation-time rendered prompt when the replay stores it."""
    prompt = row.get("prompt")
    if isinstance(prompt, str) and prompt:
        return prompt
    return _render_chat(runtime, system_prompt, row["question"])


def extract_sequence(
    runtime: TextBackboneRuntime,
    row: dict[str, Any],
    system_prompt: str,
    *,
    position_unit: str = "char",
    max_embedding_tokens: int = 16384,
    cleanup_every: int = 64,
) -> dict[str, Any] | None:
    result = extract_batch(
        runtime, [row], system_prompt, position_unit=position_unit,
        max_embedding_tokens=max_embedding_tokens,
        cleanup_every=cleanup_every,
    )
    return result[0] if result else None


def extract_batch(
    runtime: TextBackboneRuntime,
    rows: list[dict[str, Any]],
    system_prompt: str,
    *,
    position_unit: str = "char",
    max_embedding_tokens: int = 16384,
    cleanup_every: int = 64,
) -> list[dict[str, Any]]:
    """Extract a batch of variable-length trajectories in one backbone call.

    Padding is used only for the batched forward; each returned item contains
    only its real assistant-token states.  The returned state tensor remains
    CPU ``float16`` with shape ``(T, H)``.
    """
    import torch

    tokenizer = runtime.tokenizer
    prepared: list[dict[str, Any]] = []
    for row in rows:
        positions = sorted(
            {
                int(record["position"])
                for record in row["records"]
                if isinstance(record, dict)
                and isinstance(record.get("position"), int)
                and record["position"] > 0
            }
        )
        if position_unit != "char":
            raise ValueError("Only character positions are supported for traces with empty logprobs")
        # Curation positions in these logprob-free traces are character
        # boundaries. First crop to the furthest grid checkpoint, then apply
        # the model-token cap below: effective range = min(max_grid, 16K).
        max_grid_position = min(max(positions), len(row["content"])) if positions else None
        assistant_content = (
            row["content"][:max_grid_position]
            if max_grid_position is not None
            else row["content"]
        )
        # Keep prompt and continuation tokenization separate: Qwen's BPE can
        # merge the prompt's final ``<think>\n`` with the first answer token.
        encoded = tokenizer(
            assistant_content,
            add_special_tokens=False,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        assistant_ids = encoded.input_ids[0]
        offsets = encoded.offset_mapping[0]
        original_token_count = int(assistant_ids.numel())
        capped = (
            not row.get("allow_over_16k", False)
            and max_embedding_tokens > 0
            and original_token_count > max_embedding_tokens
        )
        if capped:
            assistant_ids = assistant_ids[:max_embedding_tokens]
            offsets = offsets[:max_embedding_tokens]
            assistant_content = assistant_content[: int(offsets[-1, 1])]
        prompt_text = _resolve_prompt_text(runtime, row, system_prompt)
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
        max_position = len(assistant_content)
        prepared.append(
            {
                "row": row,
                "positions": positions,
                "max_position": max_position,
                "prompt_len": len(prompt_ids),
                "assistant_ids": assistant_ids,
                "offsets": offsets.tolist(),
                "original_token_count": original_token_count,
                "capped": capped,
                "max_grid_position": max_grid_position,
                "input_ids": torch.cat(
                    [torch.tensor(prompt_ids, dtype=assistant_ids.dtype), assistant_ids]
                ),
            }
        )
    if not prepared:
        return []

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        raise ValueError("Tokenizer needs pad_token_id or eos_token_id for batched extraction")
    max_len = max(item["input_ids"].numel() for item in prepared)
    input_ids = torch.full((len(prepared), max_len), pad_id, dtype=prepared[0]["input_ids"].dtype)
    attention_mask = torch.zeros((len(prepared), max_len), dtype=torch.long)
    for index, item in enumerate(prepared):
        length = item["input_ids"].numel()
        input_ids[index, :length] = item["input_ids"]
        attention_mask[index, :length] = 1

    captured: dict[str, torch.Tensor] = {}

    def capture_norm_input(_module, args):
        captured["states"] = args[0].detach()

    handle = runtime.backbone.norm.register_forward_pre_hook(capture_norm_input)
    device_input_ids = input_ids.to(runtime.device)
    device_attention_mask = attention_mask.to(runtime.device)
    try:
        with torch.inference_mode():
            runtime.backbone(
                input_ids=device_input_ids,
                attention_mask=device_attention_mask,
                use_cache=False,
            )
    finally:
        handle.remove()
    all_states = captured["states"]
    output = []
    for index, item in enumerate(prepared):
        prompt_len = item["prompt_len"]
        token_count = item["assistant_ids"].numel()
        states = all_states[index, prompt_len : prompt_len + token_count]
        states = states.detach().to(device="cpu", dtype=torch.float16)
        if states.shape[0] != token_count:
            raise RuntimeError(
                "Batched hidden-state slicing produced an invalid length: "
                f"expected {token_count}, got {states.shape[0]}"
            )
        char_ends = [int(end) for _start, end in item["offsets"]]
        checkpoint_tokens = {
            str(position): bisect.bisect_right(
                char_ends, min(position, item["max_position"])
            )
            for position in item["positions"]
            if position <= item["max_position"]
        }
        row = item["row"]
        output.append(
            {
                "problem_id": row["problem_id"] or row["line_number"],
                "line_number": row["line_number"],
                "source_row": row["source_row"],
                "source_model": row.get("source_model"),
                "embedding_model": runtime.model.config.name_or_path,
                "hidden_states": states,
                "assistant_token_ids": item["assistant_ids"].to(dtype=torch.long, device="cpu"),
                "position_ids": torch.arange(prompt_len, prompt_len + token_count, dtype=torch.long),
                "assistant_char_ends": torch.tensor(char_ends, dtype=torch.int32),
                "prompt_tokens": prompt_len,
                "checkpoint_token_indices": checkpoint_tokens,
                "original_assistant_tokens": item["original_token_count"],
                "embedding_was_capped": item["capped"],
                "max_curation_grid_position": item["max_grid_position"],
                "allow_over_16k": bool(row.get("allow_over_16k")),
                "tensor_dtype": "torch.float16",
                "tensor_shape": [int(states.shape[0]), int(states.shape[1])],
                "padding_tokens_discarded": int(input_ids.shape[1] - (prompt_len + token_count)),
            }
        )
    # Do not keep the full padded activation batch alive between iterations.
    # The returned tensors above are CPU float16 copies; everything below is
    # temporary GPU/input state.  empty_cache releases reserved (not active)
    # allocator blocks, preventing long runs from accumulating VRAM reserve.
    del all_states, captured, device_input_ids, device_attention_mask
    runtime.completed_forwards += 1
    should_cleanup = cleanup_every > 0 and runtime.completed_forwards % cleanup_every == 0
    if runtime.device.startswith("cuda") and should_cleanup:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        LOGGER.info(
            "CUDA cleanup after %d forwards: allocated=%.2f GB reserved=%.2f GB",
            runtime.completed_forwards,
            torch.cuda.memory_allocated() / 2**30,
            torch.cuda.memory_reserved() / 2**30,
        )
    return output


def write_shards(
    sequences: Iterable[dict[str, Any]], output_dir: Path, shard_size: int = 8
) -> dict[str, int]:
    import torch

    output_dir.mkdir(parents=True, exist_ok=True)
    index: list[dict[str, Any]] = []
    shard: list[dict[str, Any]] = []
    shard_index = 0
    states = 0
    for sequence in sequences:
        shard.append(sequence)
        states += int(sequence["hidden_states"].shape[0])
        if len(shard) >= shard_size:
            name = f"shard_{shard_index:05d}.pt"
            torch.save(shard, output_dir / name)
            index.extend(
                {"problem_id": item["problem_id"], "shard_file": name, "position": i}
                for i, item in enumerate(shard)
            )
            shard = []
            shard_index += 1
    if shard:
        name = f"shard_{shard_index:05d}.pt"
        torch.save(shard, output_dir / name)
        index.extend(
            {"problem_id": item["problem_id"], "shard_file": name, "position": i}
            for i, item in enumerate(shard)
        )
        shard_index += 1
    (output_dir / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    return {"sequences": len(index), "states": states, "shards": shard_index}


def write_numpy_records(
    sequences: Iterable[dict[str, Any]],
    output_dir: Path,
    *,
    flush_every: int = 8,
    total: int | None = None,
    save_workers: int = 2,
) -> dict[str, int]:
    """Write mmap-friendly per-trajectory NumPy files and a small JSON index."""
    import numpy as np
    import torch

    if flush_every < 1:
        raise ValueError("flush_every must be >= 1")
    if save_workers < 1:
        raise ValueError("save_workers must be >= 1")
    output_dir.mkdir(parents=True, exist_ok=True)
    index = []
    states = 0
    pending = []

    def flush_index() -> None:
        (output_dir / "index.json").write_text(
            json.dumps(index, indent=2), encoding="utf-8"
        )

    try:
        from tqdm import tqdm
        iterator = tqdm(
            sequences,
            total=total,
            desc="Embedding trajectories",
            unit="row",
            dynamic_ncols=True,
        )
    except ImportError:
        iterator = sequences
    def save_one(item_index: int, sequence: dict[str, Any]) -> dict[str, Any]:
        import torch

        stem = f"embedding_{item_index:06d}"
        hidden_path = output_dir / f"{stem}.npy"
        metadata_path = output_dir / f"{stem}.json"
        # The tensor is already CPU float16. NumPy writes release the GIL, so
        # multiple workers can serialize while the GPU computes the next batch.
        np.save(hidden_path, sequence["hidden_states"].numpy())
        metadata = {
            key: (value.tolist() if isinstance(value, torch.Tensor) else value)
            for key, value in sequence.items()
            if key != "hidden_states"
        }
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        return {
            "problem_id": sequence["problem_id"],
            "line_number": sequence["line_number"],
            "embedding_key": str(sequence["line_number"]),
            "embedding_file": hidden_path.name,
            "metadata_file": metadata_path.name,
            "_saved_name": hidden_path.name,
        }

    executor = ThreadPoolExecutor(max_workers=save_workers)
    try:
        def harvest_completed(*, wait_for_one: bool = False) -> None:
            nonlocal pending
            if wait_for_one and pending:
                pending[0].result()
            done = [future for future in pending if future.done()]
            if not done:
                return
            pending = [future for future in pending if not future.done()]
            completed = [future.result() for future in done]
            index.extend(
                {key: value for key, value in item.items() if key != "_saved_name"}
                for item in completed
            )
            flush_index()
            # LOGGER.info(
            #     "Saved %d file(s): indexed=%d pending=%d latest=%s",
            #     len(completed), len(index), len(pending), completed[-1]["_saved_name"],
            # )

        for item_index, sequence in enumerate(iterator):
            # Do not wait for the whole save batch before publishing progress.
            harvest_completed()
            pending.append(executor.submit(save_one, item_index, sequence))
            states += int(sequence["hidden_states"].shape[0])
            if len(pending) >= flush_every:
                # Bound queued CPU memory, but only wait for the oldest save.
                harvest_completed(wait_for_one=True)
    finally:
        if pending:
            completed = [future.result() for future in pending]
            index.extend({key: value for key, value in item.items() if key != "_saved_name"} for item in completed)
            flush_index()
            LOGGER.info("Saved final batch: trajectories=%d states=%d", len(index), states)
        executor.shutdown(wait=True)
    return {"sequences": len(index), "states": states, "shards": len(index)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "scripts/config.yml")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="bfloat16", choices=("float16", "bfloat16", "float32"))
    parser.add_argument("--position-unit", default="char", choices=("char",))
    parser.add_argument("--shard-size", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--max-embedding-tokens", type=int, default=16384,
        help="Reasoning-token cap; ignored when baseline_16K/24K exits.",
    )
    parser.add_argument(
        "--cleanup-every", type=int, default=64,
        help="Run CUDA synchronize/empty_cache every N model forwards; 0 disables it.",
    )
    parser.add_argument(
        "--save-batch-size", type=int, default=8,
        help="Flush index/progress after this many saved trajectories.",
    )
    parser.add_argument(
        "--save-workers", type=int, default=2,
        help="Background NumPy serialization workers.",
    )
    parser.add_argument(
        "--storage", choices=("numpy", "pt"), default="numpy",
        help="NumPy mmap files are the fast, low-overhead default.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import yaml
    import torch

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
        force=True,
    )
    started = time.monotonic()
    print(
        f"[hidden_states] starting model={args.model} batch_size={args.batch_size} "
        f"storage={args.storage}",
        flush=True,
    )
    LOGGER.info(
        "Starting hidden-state extraction: input=%s model=%s batch_size=%d storage=%s",
        args.input, args.model, args.batch_size, args.storage,
    )
    total_rows = sum(1 for line in args.input.open("r", encoding="utf-8") if line.strip())
    LOGGER.info("Input contains %d non-empty JSONL rows", total_rows)
    LOGGER.info("Loading tokenizer and model (this may take several minutes)")
    try:
        runtime = load_text_backbone(args.model, args.device, args.dtype)
    except Exception:
        LOGGER.exception("Model loading failed")
        raise
    text_config = _get_text_config(runtime.model.config)
    LOGGER.info(
        "Model loaded on %s; hidden size=%s",
        runtime.device,
        text_config.hidden_size,
    )
    if runtime.device.startswith("cuda"):
        LOGGER.info(
            "CUDA after model load: allocated=%.2f GB reserved=%.2f GB",
            torch.cuda.memory_allocated() / 2**30,
            torch.cuda.memory_reserved() / 2**30,
        )
    config = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    prompt_paths = (config.get("system_prompt") or {}).get("dataset") or {}
    cache: dict[str, str] = {}

    def rows():
        for row in iter_runner_curation(args.input):
            dataloader = row.get("dataloader") or "math"
            if dataloader not in cache:
                path = prompt_paths.get(dataloader)
                if not path:
                    raise ValueError(f"No dataset system prompt configured for {dataloader!r}")
                prompt_path = Path(path)
                if not prompt_path.is_absolute():
                    prompt_path = PROJECT_ROOT / prompt_path
                cache[dataloader] = prompt_path.read_text(encoding="utf-8").strip()
            row["_system_prompt"] = cache[dataloader]
            yield row

    def sequences():
        batch = []
        batch_index = 0
        for row in rows():
            if batch and row["_system_prompt"] != batch[0]["_system_prompt"]:
                yield from extract_batch(
                    runtime, batch, batch[0]["_system_prompt"],
                    position_unit=args.position_unit,
                    max_embedding_tokens=args.max_embedding_tokens,
                    cleanup_every=args.cleanup_every,
                )
                batch = []
            batch.append(row)
            if len(batch) >= args.batch_size:
                batch_index += 1
                LOGGER.info("Running batch %d with %d trajectory(ies)", batch_index, len(batch))
                prompt = batch[0]["_system_prompt"]
                yield from extract_batch(
                    runtime, batch, prompt, position_unit=args.position_unit,
                    max_embedding_tokens=args.max_embedding_tokens,
                    cleanup_every=args.cleanup_every,
                )
                batch = []
        if batch:
            batch_index += 1
            # LOGGER.info("Running final batch %d with %d trajectory(ies)", batch_index, len(batch))
            yield from extract_batch(
                runtime, batch, batch[0]["_system_prompt"],
                position_unit=args.position_unit,
                max_embedding_tokens=args.max_embedding_tokens,
                cleanup_every=args.cleanup_every,
            )

    try:
        if args.storage == "numpy":
            stats = write_numpy_records(
                sequences(), args.output_dir, flush_every=args.save_batch_size,
                total=total_rows, save_workers=args.save_workers,
            )
        else:
            stats = write_shards(sequences(), args.output_dir, args.shard_size)
    except Exception:
        LOGGER.exception("Extraction failed")
        raise
    manifest = {
        "format_version": 2,
        "model": args.model,
        "source": str(args.input),
        "batch_size": args.batch_size,
        "storage": args.storage,
        "tensor": {"dtype": "torch.float16", "shape": "(T, H)", "hidden_representation": "pre_final_rmsnorm"},
        "stats": stats,
        "prompt_sha256": {name: hashlib.sha256(text.encode()).hexdigest() for name, text in cache.items()},
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    LOGGER.info(
        "Completed: states=%d trajectories=%d elapsed=%.1fs output=%s",
        stats["states"], stats["sequences"], time.monotonic() - started, args.output_dir,
    )


if __name__ == "__main__":
    main()
