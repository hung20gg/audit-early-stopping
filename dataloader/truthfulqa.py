import argparse
import json
from pathlib import Path
from typing import Iterable


DATASET_NAME = "truthfulqa/truthful_qa"
DATASET_CONFIG = "multiple_choice"
DEFAULT_SPLIT = "validation"
CHOICE_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def format_choices(choices: list) -> str:
    if len(choices) > len(CHOICE_LABELS):
        raise ValueError(f"Too many answer options: {len(choices)}")

    return "\n".join(
        f"{CHOICE_LABELS[index]}. {str(choice).strip()}"
        for index, choice in enumerate(choices)
    )


def normalize_answer(choices: list, labels: list) -> str:
    if len(choices) != len(labels):
        raise ValueError(
            "TruthfulQA mc2_targets choices and labels must have the same length."
        )
    if len(choices) > len(CHOICE_LABELS):
        raise ValueError(f"Too many answer options: {len(choices)}")

    answer_labels = []
    for index, label in enumerate(labels):
        if label not in (0, 1, False, True):
            raise ValueError(f"Invalid TruthfulQA label at index {index}: {label!r}")
        if bool(label):
            answer_labels.append(CHOICE_LABELS[index])

    if not answer_labels:
        raise ValueError("TruthfulQA record has no correct mc2_targets choices.")
    return ",".join(answer_labels)


def normalize_record(record: dict) -> dict:
    """Convert a TruthfulQA multiple-choice row to the common evaluation schema."""
    targets = record["mc2_targets"]
    choices = targets["choices"]
    labels = targets["labels"]
    question = str(record["question"]).strip()

    return {
        "question": f"{question}\n\nChoices:\n{format_choices(choices)}",
        "answer": normalize_answer(choices, labels),
    }


def load_truthfulqa(
    split: str = DEFAULT_SPLIT, limit: int | None = None
) -> list[dict]:
    # TruthfulQA only publishes a validation split. Accepting the runner's default
    # "test" keeps the dataset usable without an otherwise surprising extra flag.
    if split == "test":
        split = DEFAULT_SPLIT
    if split != DEFAULT_SPLIT:
        raise ValueError("TruthfulQA loader is configured for validation rows only.")
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
    return records[:limit] if limit is not None else records


def write_jsonl(records: Iterable[dict], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download TruthfulQA from Hugging Face and write evaluation JSONL."
    )
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("data/truthfulqa_validation.jsonl"),
    )
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_truthfulqa(split=args.split, limit=args.limit)
    write_jsonl(records, args.output_file)
    print(f"Wrote {len(records)} records to {args.output_file}")


if __name__ == "__main__":
    main()
