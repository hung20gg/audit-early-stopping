import argparse
import json
from pathlib import Path
from typing import Iterable


DATASET_NAME = "pminervini/HaluEval"
DATASET_CONFIG = "general"
DEFAULT_SPLIT = "data"
VALID_LABELS = {"yes", "no"}


def normalize_answer(value: object) -> str:
    """Normalize HaluEval's hallucination label to ``yes`` or ``no``."""
    answer = str(value).strip().lower()
    if answer not in VALID_LABELS:
        raise ValueError(
            "HaluEval hallucination labels must be 'yes' or 'no', "
            f"got {value!r}."
        )
    return answer


def normalize_record(record: dict) -> dict:
    """Convert a HaluEval general row to the common evaluation schema."""
    user_query = str(record["user_query"]).strip()
    chatgpt_response = str(record["chatgpt_response"]).strip()

    return {
        "question": (
            f"User query:\n{user_query}\n\n"
            f"ChatGPT response:\n{chatgpt_response}"
        ),
        "answer": normalize_answer(record["hallucination"]),
    }


def load_halueval(split: str = DEFAULT_SPLIT, limit: int | None = None) -> list[dict]:
    # The runner defaults to ``test``, while this HaluEval subset publishes its
    # rows under ``data``. Accepting the runner default avoids requiring a
    # dataset-specific CLI override for normal evaluation runs.
    if split == "test":
        split = DEFAULT_SPLIT
    if split != DEFAULT_SPLIT:
        raise ValueError("HaluEval general loader is configured for data rows only.")
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
        description="Download HaluEval general rows from Hugging Face and write JSONL."
    )
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("data/halueval_general_data.jsonl"),
    )
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_halueval(split=args.split, limit=args.limit)
    write_jsonl(records, args.output_file)
    print(f"Wrote {len(records)} records to {args.output_file}")


if __name__ == "__main__":
    main()
