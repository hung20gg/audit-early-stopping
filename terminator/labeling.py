from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluator.math500 import math_equal

MIN_EXIT_TOKEN_INDEX = 3  # reject exits suspiciously close to the very start


def token_count(tokenizer, text: str) -> int:
    if not text:
        return 0
    return len(tokenizer(text, add_special_tokens=False).input_ids)


def tokenize_thinking(tokenizer, reasoning_text: str) -> list[int]:
    if not reasoning_text:
        return []
    return tokenizer(reasoning_text, add_special_tokens=False).input_ids


def char_to_exit_token_idx(tokenizer, reasoning_text: str, span_end_char: int) -> int:
    """`exit_idx = len(tokenizer(reasoning_text[:span_end]).input_ids) - 1`,
    per the notes -- the index of the last token inside the derivation span."""
    prefix_len = token_count(tokenizer, reasoning_text[:span_end_char])
    return max(0, prefix_len - 1)


def build_labels(num_thinking_tokens: int, exit_token_idx: int) -> list[int]:
    """labels[i] = 0 for i < exit_token_idx, else 1 (appendix rule)."""
    exit_token_idx = min(exit_token_idx, num_thinking_tokens)
    return [0] * exit_token_idx + [1] * (num_thinking_tokens - exit_token_idx)


# --------------------------------------------------------------------------
# Automatic QC checks. Each returns True (passed) / False (failed); labeling
# a record collects every failure name so the QC report shows *why* records
# were dropped, not just how many.
# --------------------------------------------------------------------------


def check_span_unique(reasoning_text: str, span: str) -> bool:
    return bool(span) and reasoning_text.count(span) == 1


def check_exit_in_bounds(exit_token_idx: int, num_thinking_tokens: int) -> bool:
    return 0 <= exit_token_idx < num_thinking_tokens


def check_single_transition(labels: list[int]) -> bool:
    transitions = sum(1 for a, b in zip(labels, labels[1:]) if a != b)
    return transitions <= 1


def check_both_classes_present(labels: list[int]) -> bool:
    return 0 in labels and 1 in labels


def check_not_too_early(exit_token_idx: int, min_index: int = MIN_EXIT_TOKEN_INDEX) -> bool:
    return exit_token_idx >= min_index


def check_answer_agreement(final_solution: str, eventual_answer: str) -> bool:
    if not final_solution or not eventual_answer:
        return False
    return math_equal(final_solution, eventual_answer)


def run_qc_checks(record: dict[str, Any]) -> list[str]:
    """Return the list of failed check names (empty list == record passes)."""
    failures = []
    if not check_span_unique(record["reasoning_text"], record["answer_span"]):
        failures.append("span_unique")
    if not check_exit_in_bounds(record["exit_token_idx"], record["num_thinking_tokens"]):
        failures.append("exit_in_bounds")
    if not check_single_transition(record["labels"]):
        failures.append("single_transition")
    if not check_both_classes_present(record["labels"]):
        failures.append("both_classes_present")
    if not check_not_too_early(record["exit_token_idx"]):
        failures.append("not_too_early")
    if not check_answer_agreement(record["final_solution"], record["eventual_answer"]):
        failures.append("answer_agreement")
    return failures


def label_record(tokenizer, curated_record: dict[str, Any]) -> dict[str, Any]:
    """Attach token-level labels and QC results to one curated record.

    Always returns a record (never drops silently) -- `qc_passed`/`qc_failures`
    tell the caller whether to keep it, so failure reasons stay auditable.
    """
    reasoning_text = curated_record["reasoning_text"]
    thinking_token_ids = tokenize_thinking(tokenizer, reasoning_text)
    exit_token_idx = char_to_exit_token_idx(
        tokenizer, reasoning_text, curated_record["answer_span_char_end"]
    )
    labels = build_labels(len(thinking_token_ids), exit_token_idx)
    loss_mask = [1] * len(labels)

    labeled = {
        **curated_record,
        "thinking_token_ids": thinking_token_ids,
        "num_thinking_tokens": len(thinking_token_ids),
        "exit_token_idx": exit_token_idx,
        "labels": labels,
        "loss_mask": loss_mask,
    }
    failures = run_qc_checks(labeled)
    labeled["qc_passed"] = not failures
    labeled["qc_failures"] = failures
    return labeled


def label_file(
    tokenizer,
    input_path: Path,
    output_path: Path,
    audit_sample_path: Path | None = None,
    audit_sample_size: int = 200,
    seed: int = 1337,
) -> dict[str, Any]:
    kept: list[dict[str, Any]] = []
    failure_counts: dict[str, int] = {}
    total = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open("r", encoding="utf-8") as infile, output_path.open(
        "w", encoding="utf-8"
    ) as outfile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            total += 1
            curated_record = json.loads(line)
            labeled = label_record(tokenizer, curated_record)
            if labeled["qc_passed"]:
                kept.append(labeled)
                outfile.write(json.dumps(labeled, ensure_ascii=False) + "\n")
            else:
                for failure in labeled["qc_failures"]:
                    failure_counts[failure] = failure_counts.get(failure, 0) + 1

    if audit_sample_path is not None and kept:
        sample_size = min(audit_sample_size, len(kept))
        sample = random.Random(seed).sample(kept, sample_size)
        audit_sample_path.parent.mkdir(parents=True, exist_ok=True)
        with audit_sample_path.open("w", encoding="utf-8") as audit_file:
            for record in sample:
                audit_file.write(
                    json.dumps(
                        {
                            "problem_id": record["problem_id"],
                            "answer_span": record["answer_span"],
                            "eventual_answer": record["eventual_answer"],
                            "exit_token_idx": record["exit_token_idx"],
                            "num_thinking_tokens": record["num_thinking_tokens"],
                            "audit_label": None,  # fill in: correct / late_but_safe /
                            # premature / only_textual_occurrence / ambiguous / invalid
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    return {
        "total": total,
        "kept": len(kept),
        "dropped": total - len(kept),
        "acceptance_rate": len(kept) / total if total else 0.0,
        "failure_counts": failure_counts,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build token-level TERMINATOR labels.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer-model", required=True, help="Same model id as generation.")
    parser.add_argument("--audit-sample", type=Path, default=None)
    parser.add_argument("--audit-sample-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_model)
    stats = label_file(
        tokenizer,
        args.input,
        args.output,
        audit_sample_path=args.audit_sample,
        audit_sample_size=args.audit_sample_size,
        seed=args.seed,
    )
    qc_report_path = args.output.parent / f"{args.output.stem}_qc_report.json"
    qc_report_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(
        f"Kept {stats['kept']}/{stats['total']} ({stats['acceptance_rate']:.1%}) "
        f"-> {args.output}; QC report -> {qc_report_path}"
    )


if __name__ == "__main__":
    main()