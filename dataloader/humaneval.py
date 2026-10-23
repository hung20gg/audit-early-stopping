import argparse
import json
import re
from pathlib import Path
from typing import Iterable

DATASET_NAME = "openai/openai_humaneval"
DEFAULT_SPLIT = "test"


def extract_doctests(prompt: str) -> list[dict]:
    """
    Returns a list of ``{"input": "<expression>", "output": "<expected>"}``
    dicts that can be used as lightweight public test cases for early-exit
    verification without touching the hidden ``check()`` harness.
    """
    tests: list[dict] = []
    lines = prompt.splitlines()
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped.startswith(">>> "):
            expression = stripped[4:]  # strip ">>> " prefix
            i += 1
            # Continuation lines ("... " prefix in multi-line expressions)
            while i < len(lines) and lines[i].strip().startswith("... "):
                expression += "\n" + lines[i].strip()[4:]
                i += 1
            # Collect expected output lines until next >>>, blank, or docstring end
            output_lines: list[str] = []
            while i < len(lines):
                s = lines[i].strip()
                if (
                    not s
                    or s.startswith(">>>")
                    or s.startswith('"""')
                    or s.startswith("'''")
                ):
                    break
                output_lines.append(s)
                i += 1
            if output_lines:
                tests.append({"input": expression, "output": "\n".join(output_lines)})
        else:
            i += 1
    return tests


def normalize_record(record: dict) -> dict:
    """Return the common evaluation schema used by dataloader/README.md.

    For code datasets the ``question`` is the function prompt and ``answer``
    carries the test harness needed by the code evaluator.
    """
    prompt_text = str(record["prompt"])
    public_tests = extract_doctests(prompt_text)
    normalized = {
        "question": prompt_text,
        "answer": json.dumps(
            {
                "test": str(record["test"]),
                "entry_point": str(record["entry_point"]),
                "canonical_solution": str(record.get("canonical_solution", "")),
                "public_tests": public_tests,
            },
            ensure_ascii=False,
        ),
    }
    # Extract inline >>> doctests as structured public test cases
    if public_tests:
        normalized["public_tests"] = public_tests

    for k, v in record.items():
        if k not in normalized:
            normalized[k] = v
    return normalized


def load_humaneval(split: str = DEFAULT_SPLIT, limit: int | None = None) -> list[dict]:
    if split != "test":
        raise ValueError("HumanEval loader is configured for test rows only.")

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
        description="Download HumanEval from Hugging Face and write evaluation JSONL."
    )
    parser.add_argument(
        "--split",
        default=DEFAULT_SPLIT,
        help="Hugging Face split to load. This loader is set for test only.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("data/humaneval_test.jsonl"),
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
    records = load_humaneval(split=args.split, limit=args.limit)

    write_jsonl(records, args.output_file)
    print(f"Wrote {len(records)} records to {args.output_file}")


if __name__ == "__main__":
    main()
