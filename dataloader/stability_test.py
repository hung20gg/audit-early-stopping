import argparse
import json
from pathlib import Path

DEFAULT_SPLIT = "test"
DATA_FILE = Path(__file__).resolve().parents[1] / "data" / "stability_test.json"


def load_stability_test(
    split: str = DEFAULT_SPLIT,
    limit: int | None = None,
    data_file: str | Path = DATA_FILE,
) -> list[dict]:
    """Load the checked-in AIME26/MATH-500 stability-test sample."""
    if split != DEFAULT_SPLIT:
        raise ValueError("stability_test loader is configured for test rows only.")
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")

    with Path(data_file).open("r", encoding="utf-8") as input_file:
        records = json.load(input_file)

    if not isinstance(records, list):
        raise ValueError(f"Expected a JSON list in {data_file}")

    for index, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise ValueError(f"Record {index} in {data_file} must be an object")
        if not isinstance(record.get("question"), str) or not isinstance(
            record.get("answer"), str
        ):
            raise ValueError(
                f"Record {index} in {data_file} must contain string question and answer"
            )

    if limit is not None:
        return records[:limit]
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load the stability-test JSON sample.")
    parser.add_argument(
        "--limit", type=int, default=None, help="Optional maximum number of rows to print."
    )
    parser.add_argument(
        "--data-file", type=Path, default=DATA_FILE, help="Stability-test JSON path."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_stability_test(limit=args.limit, data_file=args.data_file)
    print(json.dumps(records, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
