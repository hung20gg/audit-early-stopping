import argparse
import datetime
import json
from pathlib import Path
from typing import Iterable

DATASET_NAME = "bzantium/livecodebench"
DEFAULT_SPLIT = "test"
DEFAULT_VERSION_TAG = "release_v6"


def make_json_serializable(obj):
    """Recursively convert datetime/date objects to ISO format strings."""
    if isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_json_serializable(x) for x in obj]
    return obj


def build_question(record: dict) -> str:
    """Build a self-contained problem statement from LiveCodeBench fields."""
    parts: list[str] = []

    title = str(record.get("question_title") or "").strip()
    if title:
        parts.append(f"## {title}")

    content = str(record.get("question_content") or "").strip()
    if content:
        parts.append(content)

    starter = str(record.get("starter_code") or "").strip()
    if starter:
        parts.append(f"### Starter Code\n```python\n{starter}\n```")

    return "\n\n".join(parts)


def build_answer(record: dict) -> str:
    """Pack test harness data into a JSON string for the code evaluator."""
    metadata = record.get("metadata", record.get("meta_data", {}))
    return json.dumps(
        {
            "public_test_cases": str(record.get("public_test_cases") or "[]"),
            "private_test_cases": str(record.get("private_test_cases") or "[]"),
            "starter_code": str(record.get("starter_code") or ""),
            "metadata": metadata or {},
        },
        ensure_ascii=False,
    )


def normalize_record(record: dict) -> dict:
    """Return the common evaluation schema used by dataloader/README.md."""
    normalized = {}
    if "question" not in record or not record["question"]:
        normalized["question"] = build_question(record)
    else:
        normalized["question"] = record["question"]
        
    if "answer" not in record or not record["answer"]:
        normalized["answer"] = build_answer(record)
    else:
        normalized["answer"] = record["answer"]

    # Preserve all other fields, ensuring they are JSON serializable
    for k, v in record.items():
        if k not in normalized:
            normalized[k] = make_json_serializable(v)
    return normalized


def load_livecodebench(
    split: str = DEFAULT_SPLIT,
    limit: int | None = None,
    version_tag: str = DEFAULT_VERSION_TAG,
) -> list[dict]:
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")

    # Local fallback for offline/slow networks
    local_file = Path(f"data/livecodebench_{version_tag}_{split}.jsonl")
    if local_file.exists():
        records = []
        with open(local_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    raw_record = json.loads(line)
                    if "question" in raw_record and "answer" in raw_record:
                        records.append(raw_record)
                    else:
                        records.append(normalize_record(raw_record))
        if limit is not None:
            return records[:limit]
        return records

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Missing dependency 'datasets'. Install dependencies with "
            "`pip install -r requirements.txt`."
        ) from exc

    dataset = load_dataset(DATASET_NAME, version_tag, split=split)
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
        description="Download LiveCodeBench from Hugging Face and write evaluation JSONL."
    )
    parser.add_argument(
        "--split",
        default=DEFAULT_SPLIT,
        help="Hugging Face split to load.",
    )
    parser.add_argument(
        "--version-tag",
        default=DEFAULT_VERSION_TAG,
        help="LiveCodeBench version tag (e.g. release_v1 .. release_v5).",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("data/livecodebench_test.jsonl"),
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
    records = load_livecodebench(
        split=args.split,
        limit=args.limit,
        version_tag=args.version_tag,
    )

    write_jsonl(records, args.output_file)
    print(f"Wrote {len(records)} records to {args.output_file}")


if __name__ == "__main__":
    main()
