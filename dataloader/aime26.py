import argparse
import json
from pathlib import Path
from typing import Iterable

DATASET_NAME = "math-ai/aime26"
DEFAULT_SPLIT = "test"


def get_field(record: dict, field_names: tuple[str, ...]) -> str:
    for field_name in field_names:
        value = record.get(field_name)
        if value is not None:
            return str(value)
    raise ValueError(f"Record must contain one of these fields: {field_names}")


def normalize_record(record: dict) -> dict:
    """Return the common evaluation schema used by dataloader/README.md."""
    return {
        "question": get_field(record, ("problem", "question", "prompt", "input")),
        "answer": get_field(record, ("answer", "final_answer", "solution")),
    }


def load_aime26(split: str = DEFAULT_SPLIT, limit: int | None = None) -> list[dict]:
    if split != "test":
        raise ValueError("AIME26 loader is configured for test rows only.")

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
        description="Download AIME26 from Hugging Face and write evaluation JSONL."
    )
    parser.add_argument(
        "--split",
        default=DEFAULT_SPLIT,
        help="Hugging Face split to load. This loader is set for test only.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("data/aime26_test.jsonl"),
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
    records = load_aime26(split=args.split, limit=args.limit)

    write_jsonl(records, args.output_file)
    print(f"Wrote {len(records)} records to {args.output_file}")


if __name__ == "__main__":
    main()
