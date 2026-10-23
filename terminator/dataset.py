from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from terminator.hf_runtime import load_model_and_tokenizer


def _load_labeled_records(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _truncate_for_thinking_budget(
    record: dict[str, Any], max_thinking_length: int
) -> dict[str, Any] | None:
    """Cap thinking tokens at `max_thinking_length`; drop records whose exit
    position falls outside the retained window (can't supervise an exit we
    truncated away)."""
    if record["exit_token_idx"] >= max_thinking_length:
        return None
    if record["num_thinking_tokens"] <= max_thinking_length:
        return record
    truncated = dict(record)
    truncated["thinking_token_ids"] = record["thinking_token_ids"][:max_thinking_length]
    truncated["labels"] = record["labels"][:max_thinking_length]
    truncated["loss_mask"] = record["loss_mask"][:max_thinking_length]
    truncated["num_thinking_tokens"] = max_thinking_length
    return truncated


def compute_hidden_states_for_record(loaded, record: dict[str, Any]):
    """Run one no-grad forward pass and return final-layer hidden states
    restricted to the thinking-token positions, shape (num_thinking_tokens, hidden_size)."""
    import torch

    model, tokenizer = loaded.model, loaded.tokenizer
    prompt_ids = tokenizer(record["prompt"], add_special_tokens=False).input_ids
    thinking_ids = record["thinking_token_ids"]
    input_ids = torch.tensor([prompt_ids + thinking_ids], device=loaded.device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, output_hidden_states=True)

    last_hidden = outputs.hidden_states[-1][0]  # (seq_len, hidden_size)
    thinking_hidden = last_hidden[len(prompt_ids) : len(prompt_ids) + len(thinking_ids)]
    return thinking_hidden.detach().to(dtype=torch.float16, device="cpu")


def precompute_shards(
    loaded,
    labeled_records: list[dict[str, Any]],
    output_dir: Path,
    shard_size: int = 256,
    max_thinking_length: int = 8192,
) -> dict[str, Any]:
    import torch

    output_dir.mkdir(parents=True, exist_ok=True)
    index: list[dict[str, Any]] = []
    shard: list[dict[str, Any]] = []
    shard_index = 0
    skipped = 0

    def flush_shard():
        nonlocal shard, shard_index
        if not shard:
            return
        shard_path = output_dir / f"shard_{shard_index:05d}.pt"
        torch.save(shard, shard_path)
        for position, item in enumerate(shard):
            index.append(
                {
                    "problem_id": item["problem_id"],
                    "shard_file": shard_path.name,
                    "position": position,
                }
            )
        shard_index += 1
        shard = []

    for record in labeled_records:
        capped = _truncate_for_thinking_budget(record, max_thinking_length)
        if capped is None:
            skipped += 1
            continue
        hidden_states = compute_hidden_states_for_record(loaded, capped)
        shard.append(
            {
                "problem_id": capped["problem_id"],
                "hidden_states": hidden_states,
                "labels": torch.tensor(capped["labels"], dtype=torch.long),
                "loss_mask": torch.tensor(capped["loss_mask"], dtype=torch.long),
            }
        )
        if len(shard) >= shard_size:
            flush_shard()
    flush_shard()

    index_path = output_dir / "index.json"
    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    return {"kept": len(index), "skipped": skipped, "shards": shard_index}


class TerminatorHiddenStateDataset:
    """Map-style dataset reading precomputed hidden-state shards.

    Shards are cached in memory the first time they're touched (pilot-scale
    data: a handful of shards, not worth an eviction policy yet).
    """

    def __init__(self, shard_dir: Path):
        self.shard_dir = Path(shard_dir)
        index_path = self.shard_dir / "index.json"
        self._index = json.loads(index_path.read_text(encoding="utf-8"))
        self._shard_cache: dict[str, list[dict[str, Any]]] = {}

    def __len__(self) -> int:
        return len(self._index)

    def _get_shard(self, shard_file: str) -> list[dict[str, Any]]:
        if shard_file not in self._shard_cache:
            import torch

            self._shard_cache[shard_file] = torch.load(
                self.shard_dir / shard_file, weights_only=False
            )
        return self._shard_cache[shard_file]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        entry = self._index[idx]
        shard = self._get_shard(entry["shard_file"])
        return shard[entry["position"]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute frozen hidden-state shards.")
    parser.add_argument("--input", type=Path, required=True, help="Labeled JSONL (labeling.py output).")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--shard-size", type=int, default=256)
    parser.add_argument("--max-thinking-length", type=int, default=8192)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = _load_labeled_records(args.input)
    loaded = load_model_and_tokenizer(args.model, device=args.device)
    stats = precompute_shards(
        loaded,
        records,
        args.output_dir,
        shard_size=args.shard_size,
        max_thinking_length=args.max_thinking_length,
    )
    print(
        f"Precomputed {stats['kept']} records into {stats['shards']} shard(s) "
        f"({stats['skipped']} skipped over the {args.max_thinking_length}-token budget) "
        f"-> {args.output_dir}"
    )


if __name__ == "__main__":
    main()