import argparse
import json
from pathlib import Path
from typing import Iterable

DATASET_NAME = "openai/gsm8k"
TINY_DATASET_NAME = "tinyBenchmarks/tinyGSM8k"
DATASET_CONFIG = "main"
DEFAULT_SPLIT = "test"


def extract_final_answer(answer: str) -> str:
    if "####" not in answer:
        return answer.strip()
    return answer.rsplit("####", 1)[-1].strip()


def normalize_record(record: dict) -> dict:
    """Return the common evaluation schema used by dataloader/README.md."""
    return {
        "question": str(record["question"]),
        "answer": extract_final_answer(str(record["answer"])),
    }


def load_gsm8k(split: str = DEFAULT_SPLIT, limit: int | None = None) -> list[dict]:
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Missing dependency 'datasets'. Install dependencies with "
            "`pip install -r requirements.txt`."
        ) from exc

    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split=split)
    records = [normalize_record(record) for record in dataset]
    if limit is not None:
        return records[:limit]
    return records


def load_tiny_gsm8k(split: str = DEFAULT_SPLIT, limit: int | None = None) -> list[dict]:
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Missing dependency 'datasets'. Install dependencies with "
            "`pip install -r requirements.txt`."
        ) from exc

    dataset = load_dataset(TINY_DATASET_NAME, DATASET_CONFIG, split=split)
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
        description="Download GSM8K from Hugging Face and write evaluation JSONL."
    )
    parser.add_argument(
        "--split",
        default=DEFAULT_SPLIT,
        help="Hugging Face split to load. Defaults to the GSM8K test split.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("data/gsm8k_test.jsonl"),
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
    records = load_gsm8k(split=args.split, limit=args.limit)

    write_jsonl(records, args.output_file)
    print(f"Wrote {len(records)} records to {args.output_file}")


if __name__ == "__main__":
    main()
