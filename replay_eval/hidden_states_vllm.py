"""Extract replay hidden states through vLLM's native model executor.

This runner writes the NumPy record contract from ``replay_eval.hidden_states``
while using a vLLM worker hook to capture the base model's final-norm input. It
accepts both replay-runner curation output and prepared random-exit traces. For
prepared traces, ``response.input`` remains the authoritative rendered prompt.
"""

from __future__ import annotations

import argparse
import asyncio
import bisect
import hashlib
import json
import logging
import re
import time
import uuid
from pathlib import Path
from typing import Any, Iterator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOGGER = logging.getLogger("replay_eval.hidden_states_vllm")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="openai/gpt-oss-20b")
    parser.add_argument(
        "--model-key",
        default=None,
        help="Config model alias; defaults to each replay row's source model.",
    )
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "scripts/config.yml"
    )
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-embedding-tokens", type=int, default=16384)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--storage", choices=("numpy",), default="numpy")
    parser.add_argument("--save-workers", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def iter_runner_curation(path: Path) -> Iterator[dict[str, Any]]:
    """Read trajectories without prematurely applying one model's delimiters."""
    with path.open("r", encoding="utf-8") as handle:
        for source_row, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            logs = payload.get("additional_logs") or {}
            records = logs.get("curation_records") or []
            content = payload.get("output")
            if not isinstance(content, str) or not isinstance(records, list):
                continue
            method_results = logs.get("method_results") or []
            allow_over_16k = any(
                isinstance(result, dict)
                and str(result.get("method", "")).lower()
                in {"baseline_16k", "baseline_24k"}
                and bool(result.get("stopped_early"))
                for result in method_results
            )
            nested = payload.get("input") or {}
            yield {
                "input_format": "runner_curation",
                "source_row": source_row,
                "line_number": payload.get("line_number", source_row),
                "problem_id": nested.get("id") or nested.get("problem_id"),
                "question": nested.get("question", ""),
                "dataloader": payload.get("dataloader") or "math",
                "source_model": payload.get("model"),
                "content": content,
                "records": records,
                "allow_over_16k": allow_over_16k,
            }


def iter_prepared_traces(path: Path) -> Iterator[dict[str, Any]]:
    """Read authoritative random-exit inputs with stored rendered prompts."""
    with path.open("r", encoding="utf-8") as handle:
        for source_row, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            response = payload.get("response") or {}
            stored_input = response.get("input")
            content = response.get("content")
            if not isinstance(stored_input, str) or not isinstance(content, str):
                raise ValueError(
                    f"Prepared trace row {source_row} needs string-valued "
                    "response.input and response.content"
                )
            yield {
                "input_format": "prepared_trace",
                "source_row": source_row,
                "line_number": payload.get("line_number", source_row),
                "problem_id": payload.get("data_id") or payload.get("line_number"),
                "data_id": payload.get("data_id"),
                "question": payload.get("question", ""),
                "dataloader": payload.get("dataset") or "math",
                "source_model": payload.get("model"),
                "content": content,
                "stored_input": stored_input,
                "output_token_ids": payload.get("output_token_ids"),
                "allow_over_16k": True,
            }


def load_input_rows(path: Path) -> tuple[str, list[dict[str, Any]]]:
    """Detect runner-curation versus prepared random-exit JSONL input."""
    with path.open("r", encoding="utf-8") as handle:
        first = next((json.loads(line) for line in handle if line.strip()), None)
    if first is None:
        return "empty", []
    response = first.get("response") if isinstance(first, dict) else None
    if (
        isinstance(response, dict)
        and isinstance(response.get("input"), str)
        and isinstance(response.get("content"), str)
    ):
        return "prepared_trace", list(iter_prepared_traces(path))
    return "runner_curation", list(iter_runner_curation(path))


def positive_positions(records: list[Any]) -> list[int]:
    return sorted(
        {
            int(record["position"])
            for record in records
            if isinstance(record, dict)
            and isinstance(record.get("position"), int)
            and record["position"] > 0
        }
    )


def split_thinking(
    content: str, think_start_token: str, think_end_token: str
) -> tuple[str, list[tuple[int, int]], int]:
    """Flatten Harmony analysis segments and report removed control spans."""
    end = content.find(think_end_token) if think_end_token else -1
    thinking = content if end < 0 else content[:end]
    removed: list[tuple[int, int]] = []
    if think_start_token and thinking.startswith(think_start_token):
        removed.append((0, len(think_start_token)))

    if think_start_token.startswith("<|channel|>analysis"):
        transition = re.compile(
            re.escape("<|end|><|start|>assistant<|channel|>analysis")
            + r"(?: to=[^<]*)?"
            + re.escape("<|message|>")
        )
        removed.extend(
            (match.start(), match.end()) for match in transition.finditer(thinking)
        )

    removed.sort()
    for left, right in zip(removed, removed[1:]):
        if left[1] > right[0]:
            raise ValueError("Overlapping thinking-control spans")
    if removed:
        pieces = []
        cursor = 0
        for start, stop in removed:
            pieces.append(thinking[cursor:start])
            cursor = stop
        pieces.append(thinking[cursor:])
        thinking = "".join(pieces)
    return thinking, removed, len(content) if end < 0 else end


def remap_position(position: int, removed: list[tuple[int, int]]) -> int:
    removed_before = sum(
        max(0, min(position, stop) - start)
        for start, stop in removed
        if position > start
    )
    return max(0, position - removed_before)


def _model_id(value: str) -> str:
    provider, separator, model_id = value.partition(":")
    if separator:
        if provider != "vllm":
            raise ValueError(f"Expected a vLLM model id, got {value!r}")
        return model_id
    return value


def _messages(system: str, question: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]


def prepare_row(
    row: dict[str, Any],
    *,
    renderer: Any,
    system_prompt: str,
    think_start_token: str,
    think_end_token: str,
    max_embedding_tokens: int,
    max_model_len: int,
) -> dict[str, Any] | None:
    """Render one canonical thinking prefix and build character/token maps."""
    import torch

    original_positions = positive_positions(row["records"])
    if not original_positions:
        return None
    thinking, removed_spans, original_thinking_end = split_thinking(
        row["content"], think_start_token, think_end_token
    )
    position_pairs = [
        (position, remap_position(position, removed_spans))
        for position in original_positions
        if position <= original_thinking_end
    ]
    if not position_pairs:
        return None

    max_grid_position = min(max(position for _, position in position_pairs), len(thinking))
    reasoning = thinking[:max_grid_position]
    encoded = renderer.tokenizer(
        reasoning,
        add_special_tokens=False,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    reasoning_ids = encoded.input_ids[0]
    offsets = encoded.offset_mapping[0]
    original_token_count = int(reasoning_ids.numel())
    capped = (
        not row.get("allow_over_16k", False)
        and max_embedding_tokens > 0
        and original_token_count > max_embedding_tokens
    )
    if capped:
        reasoning_ids = reasoning_ids[:max_embedding_tokens]
        offsets = offsets[:max_embedding_tokens]
        reasoning = reasoning[: int(offsets[-1, 1])]

    messages = _messages(system_prompt, row["question"])
    thinking_prefix_ids = renderer.checkpoint(messages, "")
    full_ids = renderer.checkpoint(messages, reasoning)
    if full_ids[: len(thinking_prefix_ids)] != thinking_prefix_ids:
        raise ValueError("Assistant thinking render changed its canonical prompt prefix")
    output_ids = full_ids[len(thinking_prefix_ids) :]
    if output_ids != reasoning_ids.tolist():
        raise ValueError(
            "Assistant thinking tokens differ from standalone reasoning tokens; "
            "checkpoint character offsets would be ambiguous"
        )
    if len(full_ids) + 1 > max_model_len:
        raise ValueError(
            f"Canonical trace has {len(full_ids)} tokens, exceeding "
            f"--max-model-len={max_model_len} after reserving one scored token"
        )

    char_ends = [int(end) for _start, end in offsets.tolist()]
    max_position = len(reasoning)
    checkpoint_tokens = {
        str(original): bisect.bisect_right(char_ends, min(adjusted, max_position))
        for original, adjusted in position_pairs
        if adjusted <= max_grid_position
    }
    return {
        "row": row,
        "full_token_ids": full_ids,
        "reasoning_start": len(thinking_prefix_ids),
        "reasoning_ids": reasoning_ids.to(dtype=torch.long, device="cpu"),
        "char_ends": char_ends,
        "checkpoint_tokens": checkpoint_tokens,
        "original_token_count": original_token_count,
        "capped": capped,
        "max_grid_position": max_grid_position,
        "stripped_chars": sum(stop - start for start, stop in removed_spans),
    }


def prepare_prepared_row(
    row: dict[str, Any],
    *,
    renderer: Any,
    system_prompt: str,
    max_embedding_tokens: int,
    max_model_len: int,
) -> dict[str, Any]:
    """Render a prepared trace while retaining its original stored prompt."""
    import torch

    messages = _messages(system_prompt, row["question"])
    stored_output_ids = row.get("output_token_ids")
    output_token_ids = (
        [int(value) for value in stored_output_ids]
        if isinstance(stored_output_ids, list)
        else None
    )
    # Older random-exit datasets contain an empty response.input. They do not
    # record the original GPT-OSS template date, so render a fresh prompt when
    # there is no stored prompt to preserve.
    trace = renderer.full_trace(
        messages,
        row["content"],
        stored_input=row["stored_input"] or None,
        output_token_ids=output_token_ids,
    )
    original_token_count = len(trace.output_token_ids)
    count = original_token_count
    capped = max_embedding_tokens > 0 and count > max_embedding_tokens
    if capped:
        count = max_embedding_tokens
    output_ids = list(trace.output_token_ids[:count])
    full_ids = [*trace.source_token_ids, *output_ids]
    if len(full_ids) + 1 > max_model_len:
        raise ValueError(
            f"Canonical trace has {len(full_ids)} tokens, exceeding "
            f"--max-model-len={max_model_len} after reserving one scored token"
        )
    return {
        "row": row,
        "full_token_ids": full_ids,
        "reasoning_start": len(trace.source_token_ids),
        "reasoning_ids": torch.tensor(output_ids, dtype=torch.long),
        "char_ends": [],
        "checkpoint_tokens": {},
        "original_token_count": original_token_count,
        "capped": capped,
        "max_grid_position": len(row["content"]),
        "stripped_chars": 0,
    }


async def capture_pre_final_states(
    engine: Any, prepared: dict[str, Any], example_id: str
) -> Any:
    """Run one vLLM prefill and return assistant-token final-norm inputs."""
    import torch
    from vllm import SamplingParams
    from replay_eval.v2.engine.vllm_engine import _consume_final

    live_engine = getattr(engine, "_engine", None)
    if live_engine is None or not callable(getattr(live_engine, "collective_rpc", None)):
        raise RuntimeError("The vLLM engine does not expose collective_rpc")
    temp_dir = Path("/tmp/replay-hidden-states-vllm")
    temp_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{example_id}-{uuid.uuid4().hex}"
    raw_path = temp_dir / f"{stem}-raw.pt"
    normalized_path = temp_dir / f"{stem}-normalized.pt"
    capture_active = False
    try:
        await live_engine.collective_rpc("replay_v2_begin_hidden_capture")
        capture_active = True
        await _consume_final(
            live_engine.generate(
                {"prompt_token_ids": list(prepared["full_token_ids"])},
                SamplingParams(max_tokens=1, temperature=0.0),
                request_id=f"replay-hidden-{uuid.uuid4().hex}",
            )
        )
        await live_engine.collective_rpc(
            "replay_v2_finish_hidden_capture",
            args=(str(raw_path), str(normalized_path), len(prepared["full_token_ids"])),
        )
        capture_active = False
        raw = torch.load(raw_path, map_location="cpu", weights_only=True)
        start = int(prepared["reasoning_start"])
        count = int(prepared["reasoning_ids"].numel())
        states = raw[start : start + count].contiguous()
        if states.shape[0] != count:
            raise RuntimeError(f"Captured {states.shape[0]} states, expected {count}")
        return states
    except BaseException:
        if capture_active:
            await live_engine.collective_rpc("replay_v2_cancel_hidden_capture")
        raise
    finally:
        raw_path.unlink(missing_ok=True)
        normalized_path.unlink(missing_ok=True)


def save_record(
    output_dir: Path,
    item_index: int,
    prepared: dict[str, Any],
    states: Any,
    model: str,
) -> dict[str, Any]:
    import numpy as np

    row = prepared["row"]
    stem = f"embedding_{item_index:06d}"
    hidden_path = output_dir / f"{stem}.npy"
    metadata_path = output_dir / f"{stem}.json"
    states = states.to(device="cpu").half()
    np.save(hidden_path, states.numpy())
    prompt_tokens = int(prepared["reasoning_start"])
    token_count = int(prepared["reasoning_ids"].numel())
    metadata = {
        "problem_id": row["problem_id"] or row["line_number"],
        "line_number": row["line_number"],
        "source_row": row["source_row"],
        "source_model": row.get("source_model"),
        "data_id": row.get("data_id"),
        "input_format": row.get("input_format", "runner_curation"),
        "embedding_model": model,
        "hidden_representation": "pre_final_rmsnorm",
        "assistant_token_ids": prepared["reasoning_ids"].tolist(),
        "position_ids": list(range(prompt_tokens, prompt_tokens + token_count)),
        "assistant_char_ends": prepared["char_ends"],
        "prompt_tokens": prompt_tokens,
        "checkpoint_token_indices": prepared["checkpoint_tokens"],
        "original_assistant_tokens": prepared["original_token_count"],
        "embedding_was_capped": prepared["capped"],
        "max_curation_grid_position": prepared["max_grid_position"],
        "thinking_prefix_chars_stripped": prepared["stripped_chars"],
        "allow_over_16k": bool(row.get("allow_over_16k")),
        "tensor_dtype": "torch.float16",
        "tensor_shape": [int(states.shape[0]), int(states.shape[1])],
        "padding_tokens_discarded": 0,
    }
    if row.get("input_format") == "prepared_trace":
        metadata.update(
            {
                "stored_input_sha256": hashlib.sha256(
                    row["stored_input"].encode()
                ).hexdigest(),
                "response_content_sha256": hashlib.sha256(
                    row["content"].encode()
                ).hexdigest(),
            }
        )
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    return {
        "problem_id": metadata["problem_id"],
        "line_number": row["line_number"],
        "embedding_key": str(row["line_number"]),
        "embedding_file": hidden_path.name,
        "metadata_file": metadata_path.name,
    }


async def run(args: argparse.Namespace) -> None:
    import torch
    import yaml
    from replay_eval.v2.engine.vllm_engine import AsyncVLLMEngine
    from replay_eval.v2.rendering.hf import CanonicalRenderer
    from scripts.config_loader import resolve_model_config

    if args.batch_size != 1:
        raise ValueError("vLLM hidden capture currently requires --batch-size 1")
    if args.save_workers < 1:
        raise ValueError("--save-workers must be at least 1")

    config = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    prompt_paths = (config.get("system_prompt") or {}).get("dataset") or {}
    input_format, rows = load_input_rows(args.input)
    if input_format == "runner_curation":
        rows = [row for row in rows if positive_positions(row["records"])]
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("Input contains no extractable trajectories")

    model = _model_id(args.model)
    engine = AsyncVLLMEngine(
        model,
        max_concurrency=1,
        max_logprobs=50,
        engine_kwargs={
            "dtype": args.dtype,
            "max_model_len": args.max_model_len,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_num_seqs": 1,
            "enable_prefix_caching": False,
            "enforce_eager": True,
            "worker_extension_cls": "replay_eval.v2.engine.vllm_hidden_states.HiddenStateWorkerExtension",
        },
    )
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "index.json"
    index: list[dict[str, Any]] = []
    states_count = 0
    completed_source_rows: set[int] = set()
    if index_path.exists():
        candidate_index = json.loads(index_path.read_text(encoding="utf-8"))
        for item in candidate_index:
            metadata_path = output_dir / item["metadata_file"]
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if "thinking_prefix_chars_stripped" not in metadata:
                raise ValueError(
                    "Existing output is not from hidden_states_vllm; use a new "
                    "output directory or move the old partial extraction"
                )
            completed_source_rows.add(int(metadata["source_row"]))
            states_count += int(metadata["tensor_shape"][0])
        index = candidate_index
        rows = [row for row in rows if row["source_row"] not in completed_source_rows]
        LOGGER.info(
            "Resuming after %d trajectories; %d remain", len(index), len(rows)
        )

    prompt_cache: dict[str, str] = {}
    renderer_cache: dict[str, tuple[Any, str, str]] = {}
    started = time.monotonic()

    LOGGER.info("Starting vLLM model=%s trajectories=%d", model, len(rows))
    await engine.start()
    LOGGER.info("vLLM loaded in %.1fs", engine.load_seconds)
    try:
        try:
            from tqdm import tqdm
            iterator = tqdm(rows, desc="Embedding trajectories", unit="row", dynamic_ncols=True)
        except ImportError:
            iterator = rows
        for row in iterator:
            alias = args.model_key or row.get("source_model")
            if not alias:
                raise ValueError(f"Replay row {row['source_row']} needs --model-key")
            if alias not in renderer_cache:
                model_config = resolve_model_config(config, alias)
                configured_model = _model_id(str(model_config["id"]))
                if configured_model != model:
                    raise ValueError(
                        f"Row model {alias!r} resolves to {configured_model!r}, "
                        f"but --model is {model!r}"
                    )
                chat = model_config.get("chat_template") or {}
                start_token = chat.get("think_start_token") or "<think>"
                end_token = chat.get("think_end_token") or "</think>"
                renderer_cache[alias] = (
                    CanonicalRenderer(
                        engine.tokenizer,
                        model_id=model,
                        template_kwargs=dict(chat.get("kwargs") or {}),
                        think_start_token=start_token,
                        think_end_token=end_token,
                    ),
                    start_token,
                    end_token,
                )
            renderer, start_token, end_token = renderer_cache[alias]
            dataloader = row["dataloader"]
            if dataloader not in prompt_cache:
                prompt_path = prompt_paths.get(dataloader)
                if not prompt_path:
                    raise ValueError(f"No system prompt configured for {dataloader!r}")
                path = Path(prompt_path)
                if not path.is_absolute():
                    path = PROJECT_ROOT / path
                prompt_cache[dataloader] = path.read_text(encoding="utf-8").strip()
            if row["input_format"] == "prepared_trace":
                prepared = prepare_prepared_row(
                    row,
                    renderer=renderer,
                    system_prompt=prompt_cache[dataloader],
                    max_embedding_tokens=args.max_embedding_tokens,
                    max_model_len=args.max_model_len,
                )
            else:
                prepared = prepare_row(
                    row,
                    renderer=renderer,
                    system_prompt=prompt_cache[dataloader],
                    think_start_token=start_token,
                    think_end_token=end_token,
                    max_embedding_tokens=args.max_embedding_tokens,
                    max_model_len=args.max_model_len,
                )
            if prepared is None:
                continue
            states = await capture_pre_final_states(engine, prepared, str(row["source_row"]))
            index.append(save_record(output_dir, len(index), prepared, states, model))
            states_count += int(states.shape[0])
            (output_dir / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    finally:
        await engine.shutdown()

    manifest = {
        "format_version": 2,
        "model": model,
        "source": str(args.input),
        "input_format": input_format,
        "batch_size": 1,
        "storage": "numpy",
        "engine": "vllm",
        "tensor": {
            "dtype": "torch.float16",
            "shape": "(T, H)",
            "hidden_representation": "pre_final_rmsnorm",
        },
        "stats": {"sequences": len(index), "states": states_count, "shards": len(index)},
        "prompt_sha256": {
            name: hashlib.sha256(text.encode()).hexdigest()
            for name, text in prompt_cache.items()
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    LOGGER.info(
        "Completed: states=%d trajectories=%d elapsed=%.1fs output=%s",
        states_count,
        len(index),
        time.monotonic() - started,
        output_dir,
    )
    if torch.cuda.is_available():
        LOGGER.info(
            "Peak CUDA allocated: %.2f GB",
            torch.cuda.max_memory_allocated() / 2**30,
        )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    asyncio.run(run(parse_args()))


if __name__ == "__main__":
    main()
