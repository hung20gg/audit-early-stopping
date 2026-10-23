#!/usr/bin/env python
"""Add top-level correctness to flat random-exit continuation JSONL rows."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Callable, Iterable

from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.config_loader import import_config_entry, load_config  # noqa: E402


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for file_line, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {path} at file line {file_line}: {error}"
                ) from error
            if not isinstance(row, dict):
                raise ValueError(
                    f"Expected an object in {path} at file line {file_line}"
                )
            rows.append(row)
    return rows


def reference_answers(
    prepared_rows: Iterable[dict[str, Any]],
) -> dict[str, tuple[int, Any]]:
    references: dict[str, tuple[int, Any]] = {}
    for file_line, row in enumerate(prepared_rows, start=1):
        data_id = row.get("data_id")
        line_number = row.get("line_number")
        if not isinstance(data_id, str) or not data_id:
            raise ValueError(f"Prepared input row {file_line} has no data_id")
        if data_id in references:
            raise ValueError(f"Prepared input contains duplicate data_id {data_id!r}")
        if "answer" not in row:
            raise ValueError(f"Prepared input row {data_id!r} has no answer")
        references[data_id] = (line_number, row["answer"])
    return references


def evaluate_rows(
    rows: Iterable[dict[str, Any]],
    references: dict[str, tuple[int, Any]],
    evaluator: Callable[[Any, Any], bool],
    *,
    empty_only: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    evaluated: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    stats = {
        "rows": 0,
        "evaluated": 0,
        "preserved": 0,
        "correct": 0,
        "incorrect": 0,
        "failed_generation": 0,
    }
    for file_line, source in enumerate(
        tqdm(rows, desc="Evaluating", unit="row"), start=1
    ):
        row = dict(source)
        key = row.get("key")
        if not isinstance(key, str) or not key:
            raise ValueError(f"Continuation row {file_line} has no key")
        if key in seen_keys:
            raise ValueError(f"Continuation input contains duplicate key {key!r}")
        seen_keys.add(key)
        data_id = row.get("data_id")
        if data_id not in references:
            raise ValueError(
                f"Continuation {key!r} has no prepared reference for {data_id!r}"
            )
        reference_line, answer = references[data_id]
        if row.get("line_number") != reference_line:
            raise ValueError(
                f"Continuation {key!r} line_number conflicts with prepared input"
            )

        stats["rows"] += 1
        if empty_only and isinstance(row.get("correct"), bool):
            stats["preserved"] += 1
        else:
            output = row.get("output")
            if not row.get("ok") or not isinstance(output, str) or not output.strip():
                row["correct"] = False
                stats["failed_generation"] += 1
            else:
                row["correct"] = bool(evaluator(output, answer))
            stats["evaluated"] += 1
        stats["correct" if row.get("correct") is True else "incorrect"] += 1
        evaluated.append(row)
    return evaluated, stats


def write_jsonl_atomic(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", text=True
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            os.fchmod(handle.fileno(), 0o644)
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}.evaluated.jsonl")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Flat random-exit JSONL")
    parser.add_argument(
        "--prepared-input",
        type=Path,
        required=True,
        help="Prepared trajectory JSONL containing data_id and answer",
    )
    destination = parser.add_mutually_exclusive_group()
    destination.add_argument("--output", type=Path)
    destination.add_argument(
        "--in-place",
        action="store_true",
        help=(
            "Atomically replace the input JSONL with reevaluated rows. Useful "
            "when the input is already a *.evaluated.jsonl file."
        ),
    )
    parser.add_argument("--config", default=str(PROJECT_ROOT / "scripts/config.yml"))
    parser.add_argument("--evaluator", default="math500")
    parser.add_argument(
        "--empty-only",
        action="store_true",
        help="Preserve rows that already contain a boolean correct field",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Evaluate only the first N rows (useful for validation)",
    )
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output = (
        args.input
        if args.in_place
        else (args.output or default_output_path(args.input))
    )
    config = load_config(args.config)
    evaluator = import_config_entry(config, "evaluator", args.evaluator)
    rows = read_jsonl(args.input)
    if args.limit is not None:
        rows = rows[: args.limit]
    references = reference_answers(read_jsonl(args.prepared_input))
    evaluated, stats = evaluate_rows(
        rows, references, evaluator, empty_only=args.empty_only
    )
    write_jsonl_atomic(output, evaluated)
    print(
        f"Wrote {stats['rows']:,} rows to {output} "
        f"(evaluated={stats['evaluated']:,}, preserved={stats['preserved']:,}, "
        f"correct={stats['correct']:,}, incorrect={stats['incorrect']:,}, "
        f"failed_generation={stats['failed_generation']:,})"
    )


if __name__ == "__main__":
    main()
