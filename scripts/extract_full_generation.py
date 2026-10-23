import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract reusable full-generation prompt/content rows from runner JSONL."
    )
    parser.add_argument("input_file", help="Runner JSONL containing additional_logs.full_generation_response.")
    parser.add_argument("output_file", help="Output JSONL with id, input, and content.")
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip rows without full_generation_response instead of failing.",
    )
    return parser.parse_args()


def iter_jsonl(path: str | Path):
    with open(path, "r", encoding="utf-8") as input_file:
        for source_line_number, line in enumerate(input_file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield source_line_number, json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {source_line_number}: {exc}") from exc


def extract_row(row: dict[str, Any], source_line_number: int) -> dict[str, Any] | None:
    additional_logs = row.get("additional_logs", {})
    full_generation = additional_logs.get("full_generation_response")
    if not isinstance(full_generation, dict):
        return None

    input_text = full_generation.get("input")
    content = full_generation.get("content")
    if not isinstance(input_text, str) or not isinstance(content, str):
        return None

    row_id = row.get("id", row.get("line_number", source_line_number))
    return {
        "id": row_id,
        "input": input_text,
        "content": content,
    }


def extract_file(input_path: str | Path, output_path: str | Path, *, skip_missing: bool) -> int:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    with open(output_path, "w", encoding="utf-8") as output_file:
        for source_line_number, row in iter_jsonl(input_path):
            extracted = extract_row(row, source_line_number)
            if extracted is None:
                if skip_missing:
                    continue
                raise ValueError(
                    "Missing additional_logs.full_generation_response with string "
                    f"input/content on line {source_line_number}."
                )
            output_file.write(json.dumps(extracted, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    args = parse_args()
    count = extract_file(
        args.input_file,
        args.output_file,
        skip_missing=args.skip_missing,
    )
    print(f"Extracted {count} rows to {args.output_file}")


if __name__ == "__main__":
    main()
