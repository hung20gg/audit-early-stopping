#!/usr/bin/env python
"""Re-evaluate LeetCode rows against full LiveCodeBench tests.

The matching dataset release is loaded so private tests do not need to be
embedded in the runner JSONL.  Records for every platform other than exactly
``leetcode`` are copied without modification.
"""

import argparse
import json
import os
import stat
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataloader.livecodebench import (  # noqa: E402
    DEFAULT_SPLIT,
    DEFAULT_VERSION_TAG,
    build_answer,
    load_livecodebench,
)
from evaluator.code_exec_fix import code_exec_equal  # noqa: E402


DEFAULT_THREADS = min(16, max(4, os.cpu_count() or 1))


@dataclass
class Stats:
    rows: int = 0
    non_leetcode: int = 0
    retained: int = 0
    evaluated: int = 0
    changed: int = 0
    invalid_submission: int = 0

    def add(self, other: "Stats") -> None:
        for name in self.__dataclass_fields__:
            setattr(self, name, getattr(self, name) + getattr(other, name))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Runner output JSONL to scan.")
    destination = parser.add_mutually_exclusive_group()
    destination.add_argument(
        "--output",
        type=Path,
        help="Output JSONL (default: INPUT.code_exec_fix.jsonl).",
    )
    destination.add_argument(
        "--in-place",
        action="store_true",
        help="Atomically replace the input JSONL.",
    )
    parser.add_argument(
        "--rerun",
        choices=("missing", "failed", "all"),
        default="all",
        help=(
            "Which statuses to evaluate: missing only, false/missing, or all "
            "(default: all)."
        ),
    )
    parser.add_argument(
        "--split",
        default=DEFAULT_SPLIT,
        help=f"LiveCodeBench split used by the run (default: {DEFAULT_SPLIT}).",
    )
    parser.add_argument(
        "--version-tag",
        default=DEFAULT_VERSION_TAG,
        help=(
            "LiveCodeBench release used by the run "
            f"(default: {DEFAULT_VERSION_TAG})."
        ),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=1,
        help="Per-submission timeout in seconds (default: 1).",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=DEFAULT_THREADS,
        help=(
            "Number of rows to process concurrently "
            f"(default: {DEFAULT_THREADS}, selected automatically)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Evaluate and print statistics without writing output.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_number}")
            rows.append(row)
    return rows


def get_reference(dataset: list[dict], line_number: object) -> str:
    """Build a full public/private reference for a one-based dataset row."""
    if not isinstance(line_number, int) or isinstance(line_number, bool):
        raise ValueError(f"Record has invalid line_number: {line_number!r}")
    if line_number < 1 or line_number > len(dataset):
        raise IndexError(
            f"line_number {line_number} is outside loaded dataset range "
            f"1..{len(dataset)}"
        )
    return build_answer(dataset[line_number - 1])


def should_evaluate(value: object, rerun: str) -> bool:
    if rerun == "all":
        return True
    if rerun == "missing":
        return not isinstance(value, bool)
    return value is not True


def update_status(
    container: dict,
    answer_field: str,
    reference: str | dict,
    *,
    rerun: str,
    timeout: float,
) -> Stats:
    stats = Stats()
    if not should_evaluate(container.get("correct"), rerun):
        stats.retained = 1
        return stats

    answer = container.get(answer_field)
    if not isinstance(answer, str) or not answer.strip():
        stats.invalid_submission = 1
        return stats

    old_value = container.get("correct")
    new_value = code_exec_equal(answer, reference, timeout=timeout)
    container["correct"] = new_value
    stats.evaluated = 1
    stats.changed = int(old_value is not new_value)
    return stats


def reevaluate_row(
    row: dict,
    dataset: list[dict],
    *,
    rerun: str = "all",
    timeout: float = 1,
) -> tuple[dict, Stats]:
    stats = Stats(rows=1)
    input_record = row.get("input")
    if not isinstance(input_record, dict) or input_record.get("platform") != "leetcode":
        stats.non_leetcode = 1
        return row, stats

    reference = get_reference(dataset, row.get("line_number"))
    stats.add(
        update_status(
            row,
            "output",
            reference,
            rerun=rerun,
            timeout=timeout,
        )
    )

    method_results = row.get("method_results")
    if isinstance(method_results, list):
        for method_result in method_results:
            if not isinstance(method_result, dict):
                continue
            stats.add(
                update_status(
                    method_result,
                    "final_answer",
                    reference,
                    rerun=rerun,
                    timeout=timeout,
                )
            )
    return row, stats


def write_jsonl_atomic(path: Path, rows: list[dict], mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            for row in rows:
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
            output.flush()
            os.fsync(output.fileno())
        os.chmod(temporary_path, mode)
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def main() -> None:
    args = parse_args()
    if args.timeout <= 0:
        raise ValueError("--timeout must be > 0")
    if args.threads < 1:
        raise ValueError("--threads must be >= 1")

    input_path = args.input.resolve()
    rows = load_jsonl(input_path)
    print(
        f"Loading LiveCodeBench split={args.split!r} "
        f"version_tag={args.version_tag!r}..."
    )
    dataset = load_livecodebench(
        split=args.split,
        version_tag=args.version_tag,
    )
    print(f"Loaded {len(dataset)} dataset records with private tests")

    def evaluate(row: dict) -> tuple[dict, Stats]:
        return reevaluate_row(
            row,
            dataset,
            rerun=args.rerun,
            timeout=args.timeout,
        )

    if args.threads == 1:
        results = [
            evaluate(row)
            for row in tqdm(rows, desc="Evaluating", unit="row")
        ]
    else:
        with ThreadPoolExecutor(max_workers=args.threads) as executor:
            futures = [executor.submit(evaluate, row) for row in rows]
            results = []
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"Evaluating ({args.threads} workers)",
                unit="row",
            ):
                results.append(future.result())

    updated_rows = [row for row, _ in results]
    total = Stats()
    for _, stats in results:
        total.add(stats)

    if not args.dry_run:
        if args.in_place:
            output_path = input_path
        elif args.output:
            output_path = args.output.resolve()
        else:
            output_path = input_path.with_name(
                f"{input_path.stem}.code_exec_fix{input_path.suffix}"
            )
        write_jsonl_atomic(
            output_path,
            updated_rows,
            mode=stat.S_IMODE(input_path.stat().st_mode),
        )
        print(f"Wrote {output_path}")

    print(
        f"rows={total.rows} non_leetcode={total.non_leetcode} "
        f"evaluated={total.evaluated} retained={total.retained} "
        f"changed={total.changed} invalid_submission={total.invalid_submission}"
    )


if __name__ == "__main__":
    main()
