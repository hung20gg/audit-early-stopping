from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_jsonl(path: Path) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows[row["line_number"]] = row
    return rows


def compression_report(vanilla_jsonl: Path, terminator_jsonl: Path) -> dict[str, Any]:
    vanilla_rows = _load_jsonl(vanilla_jsonl)
    terminator_rows = _load_jsonl(terminator_jsonl)
    shared_ids = sorted(set(vanilla_rows) & set(terminator_rows))

    compression_ratios = []
    per_problem = []
    vanilla_correct = 0
    terminator_correct = 0
    scored = 0

    for line_number in shared_ids:
        vanilla_row = vanilla_rows[line_number]
        terminator_row = terminator_rows[line_number]
        vanilla_tokens = vanilla_row.get("num_token") or 0
        terminator_tokens = terminator_row.get("num_token") or 0
        cr = terminator_tokens / vanilla_tokens if vanilla_tokens else None
        if cr is not None:
            compression_ratios.append(cr)
        per_problem.append(
            {
                "line_number": line_number,
                "vanilla_tokens": vanilla_tokens,
                "terminator_tokens": terminator_tokens,
                "compression_ratio": cr,
            }
        )
        if "correct" in vanilla_row and "correct" in terminator_row:
            scored += 1
            vanilla_correct += bool(vanilla_row["correct"])
            terminator_correct += bool(terminator_row["correct"])

    return {
        "num_shared_records": len(shared_ids),
        "mean_compression_ratio": statistics.fmean(compression_ratios) if compression_ratios else None,
        "median_compression_ratio": statistics.median(compression_ratios) if compression_ratios else None,
        "vanilla_accuracy": vanilla_correct / scored if scored else None,
        "terminator_accuracy": terminator_correct / scored if scored else None,
        "accuracy_delta": (terminator_correct - vanilla_correct) / scored if scored else None,
        "per_problem": per_problem,
    }


def probe_metrics(base_model: str, probe_checkpoint: Path, test_shards: Path, device: str = "auto") -> dict[str, Any]:
    import torch

    from terminator.dataset import TerminatorHiddenStateDataset
    from terminator.hf_runtime import load_model_and_tokenizer
    from terminator.probe import build_probe_from_loaded_model
    from terminator.train import evaluate_probe

    loaded = load_model_and_tokenizer(base_model, device=device)
    probe = build_probe_from_loaded_model(loaded).to(loaded.device)
    probe.load_state_dict(torch.load(probe_checkpoint, map_location=loaded.device))

    test_dataset = TerminatorHiddenStateDataset(test_shards)
    return evaluate_probe(probe, test_dataset)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained TERMINATOR probe / agent run.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    probe_parser = subparsers.add_parser("probe", help="Held-out probe metrics (Macro-F1, exit error).")
    probe_parser.add_argument("--base-model", required=True)
    probe_parser.add_argument("--probe-checkpoint", type=Path, required=True)
    probe_parser.add_argument("--test-shards", type=Path, required=True)
    probe_parser.add_argument("--device", default="auto")

    compression_parser = subparsers.add_parser("compression", help="Task-level CR/accuracy comparison.")
    compression_parser.add_argument("--vanilla-jsonl", type=Path, required=True)
    compression_parser.add_argument("--terminator-jsonl", type=Path, required=True)
    compression_parser.add_argument("--output", type=Path, default=None)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "probe":
        metrics = probe_metrics(args.base_model, args.probe_checkpoint, args.test_shards, device=args.device)
        print(json.dumps(metrics, indent=2))
    else:
        report = compression_report(args.vanilla_jsonl, args.terminator_jsonl)
        summary = {k: v for k, v in report.items() if k != "per_problem"}
        print(json.dumps(summary, indent=2))
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
