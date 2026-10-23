import argparse
import json
from pathlib import Path
from typing import Iterable

DATASET_NAME = "TIGER-Lab/MMLU-Pro"
DEFAULT_SPLIT = "test"
CHOICE_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def format_choices(options: list) -> str:
    lines = []

    for index, option in enumerate(options):
        if index >= len(CHOICE_LABELS):
            raise ValueError(f"Too many answer options: {len(options)}")

        label = CHOICE_LABELS[index]
        lines.append(f"{label}. {str(option).strip()}")

    return "\n".join(lines)


def flatten_question(question: str, options: list) -> str:
    return (
        f"{question.strip()}\n\n"
        f"Choices:\n{format_choices(options)}\n\n"
        "Answer with only the correct option letter."
    )


def normalize_answer(record: dict) -> str:
    answer = str(record["answer"]).strip().upper()

    if len(answer) == 1 and answer in CHOICE_LABELS:
        return answer

    answer_index = record.get("answer_index")
    if isinstance(answer_index, int):
        if answer_index < 0 or answer_index >= len(CHOICE_LABELS):
            raise ValueError(f"Invalid answer_index: {answer_index}")
        return CHOICE_LABELS[answer_index]

    raise ValueError(f"Invalid MMLU-Pro answer: {record.get('answer')}")


def normalize_record(record: dict) -> dict:
    """Return the common evaluation schema used by dataloader/README.md."""
    return {
        "question": flatten_question(str(record["question"]), record["options"]),
        "answer": normalize_answer(record),
    }


def load_mmlu_pro(split: str = DEFAULT_SPLIT, limit: int | None = None) -> list[dict]:
    if split != "test":
        raise ValueError("MMLU-Pro loader is configured for test rows only.")

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


def load_mmlu_pro_val(split: str = DEFAULT_SPLIT, limit: int | None = None) -> list[dict]:
    if split != "test":
        raise ValueError("MMLU-Pro loader is configured for test rows only.")

    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Missing dependency 'datasets'. Install dependencies with "
            "`pip install -r requirements.txt`."
        ) from exc

    dataset = load_dataset(DATASET_NAME, split="validation")
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
        description="Download MMLU-Pro from Hugging Face and write evaluation JSONL."
    )
    parser.add_argument(
        "--split",
        default=DEFAULT_SPLIT,
        help="Hugging Face split to load. This loader is set for test only.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("data/mmlu_pro_test.jsonl"),
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
    records = load_mmlu_pro(split=args.split, limit=args.limit)

    write_jsonl(records, args.output_file)
    print(f"Wrote {len(records)} records to {args.output_file}")


if __name__ == "__main__":
    main()
