import argparse
import json
import re
from pathlib import Path
from typing import Iterable

DATASET_NAME = "fingertap/GPQA-Diamond"
DEFAULT_SPLIT = "test"
CHOICE_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def standardize_embedded_choices(question: str) -> str:
    """Make embedded choices easier to read when the dataset uses a), b), c), d)."""

    def replace_choice_marker(match: re.Match) -> str:
        prefix = match.group(1)
        label = match.group(2).upper()
        separator = "\n" if prefix.strip() else ""
        return f"{separator}{label}. "

    question = question.strip()
    question = re.sub(r"(?i)(^|\s)([a-d])\)\s*", replace_choice_marker, question)
    return question


def normalize_answer(record: dict) -> str:
    answer = str(record["answer"]).strip().upper()

    if len(answer) == 1 and answer in CHOICE_LABELS:
        return answer

    raise ValueError(f"Invalid GPQA-Diamond answer: {record.get('answer')}")


def normalize_record(record: dict) -> dict:
    """Return the common evaluation schema used by dataloader/README.md."""
    question = standardize_embedded_choices(str(record["question"]))

    return {
        "question": f"{question}\n\nAnswer with only the correct option letter.",
        "answer": normalize_answer(record),
    }


def load_gpqa_diamond(split: str = DEFAULT_SPLIT, limit: int | None = None) -> list[dict]:
    if split != "test":
        raise ValueError("GPQA-Diamond loader is configured for test rows only.")

    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Missing dependency 'datasets'. Install dependencies with "
            "`pip install -r requirements.txt`."
        ) from exc

    dataset = load_dataset(DATASET_NAME, split=split)
    records = [normalize_record(record) for record in dataset]

    if limit is not None:
        return records[:limit]
    return records


def write_jsonl(records: Iterable[dict], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download GPQA-Diamond from Hugging Face and write evaluation JSONL."
    )
    parser.add_argument(
        "--split",
        default=DEFAULT_SPLIT,
        help="Hugging Face split to load. This loader is set for test only.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("data/gpqa_diamond_test.jsonl"),
        help="Destination JSONL path.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of rows to write.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_gpqa_diamond(split=args.split, limit=args.limit)

    write_jsonl(records, args.output_file)
    print(f"Wrote {len(records)} records to {args.output_file}")


if __name__ == "__main__":
    main()
