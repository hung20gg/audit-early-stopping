#!/usr/bin/env python
"""Rerun LiveCodeBench code evaluation for existing runner output JSONL files."""

import argparse
import json
import os
import stat
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

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


DEFAULT_NUM_THREADS = min(16, max(4, os.cpu_count() or 1))


def parse_methods(value: str) -> set[str] | None:
    methods = {name.strip() for name in value.split(",") if name.strip()}
    if not methods:
        raise argparse.ArgumentTypeError("Specify at least one method or 'all'")
    if "all" in methods:
        if len(methods) != 1:
            raise argparse.ArgumentTypeError("'all' cannot be combined with method names")
        return None
    return methods


@dataclass
class StatusStats:
    evaluated: int = 0
    unchanged: int = 0
    false_to_true: int = 0
    true_to_false: int = 0
    added: int = 0
    final_true: int = 0
    final_false: int = 0

    @property
    def changed(self) -> int:
        return self.false_to_true + self.true_to_false + self.added

    def observe(self, old_value: object, new_value: bool) -> None:
        self.evaluated += 1
        self.final_true += int(new_value)
        self.final_false += int(not new_value)
        if not isinstance(old_value, bool):
            self.added += 1
        elif old_value == new_value:
            self.unchanged += 1
        elif new_value:
            self.false_to_true += 1
        else:
            self.true_to_false += 1

    def merge(self, other: "StatusStats") -> None:
        for name in (
            "evaluated",
            "unchanged",
            "false_to_true",
            "true_to_false",
            "added",
            "final_true",
            "final_false",
        ):
            setattr(self, name, getattr(self, name) + getattr(other, name))


@dataclass
class EvaluationStats:
    records: int = 0
    skipped_records: int = 0
    top_level: StatusStats = field(default_factory=StatusStats)
    method_results: StatusStats = field(default_factory=StatusStats)

    @property
    def changed(self) -> int:
        return self.top_level.changed + self.method_results.changed

    def merge(self, other: "EvaluationStats") -> None:
        self.records += other.records
        self.skipped_records += other.skipped_records
        self.top_level.merge(other.top_level)
        self.method_results.merge(other.method_results)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rerun the current code_exec evaluator against private LiveCodeBench "
            "tests and atomically update correctness fields in JSONL output files."
        )
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="JSONL output file(s), or directories containing JSONL files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "examples/output/livecodebench-fixed",
        help=(
            "Directory for reevaluated JSONL files; source files are never modified "
            "(default: examples/output/livecodebench-fixed)."
        ),
    )
    parser.add_argument(
        "--split",
        default=DEFAULT_SPLIT,
        help=f"LiveCodeBench split used by the original run (default: {DEFAULT_SPLIT}).",
    )
    parser.add_argument(
        "--version-tag",
        default=DEFAULT_VERSION_TAG,
        help=(
            "LiveCodeBench version used by the original run "
            f"(default: {DEFAULT_VERSION_TAG})."
        ),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=1,
        help="Per-submission evaluator timeout in seconds (default: 1).",
    )
    parser.add_argument(
        "--num-thread",
        "--num-threads",
        "--threads",
        dest="num_thread",
        type=int,
        default=DEFAULT_NUM_THREADS,
        help=(
            "Number of records to evaluate concurrently "
            f"(default: {DEFAULT_NUM_THREADS}, selected automatically)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Evaluate and report changes without rewriting files.",
    )
    parser.add_argument(
        "--methods",
        type=parse_methods,
        default=None,
        help=(
            "Comma-separated method names to evaluate (default: all). "
            "Use 'top-level' to include the record's output when filtering. "
            "Unselected correctness fields are preserved."
        ),
    )
    parser.add_argument(
        "--empty-only",
        action="store_true",
        help=(
            "Evaluate only submissions that do not have a 'correct' key; "
            "existing true, false, and null statuses are preserved."
        ),
    )
    return parser.parse_args()


def discover_jsonl_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(path.glob("*.jsonl")))
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(f"Path does not exist: {path}")

    unique_files = list(dict.fromkeys(path.resolve() for path in files))
    if not unique_files:
        raise ValueError("No JSONL files found")
    return unique_files


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as source:
        for file_line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path} on file line {file_line_number}: {exc}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected a JSON object in {path} on file line {file_line_number}"
                )
            records.append(record)
    return records


def get_reference(dataset: list[dict], line_number: object) -> str | dict:
    if not isinstance(line_number, int) or isinstance(line_number, bool):
        raise ValueError(f"Record has invalid line_number: {line_number!r}")
    if line_number < 1 or line_number > len(dataset):
        raise IndexError(
            f"line_number {line_number} is outside loaded dataset range 1..{len(dataset)}"
        )

    dataset_record = dataset[line_number - 1]
    # Rebuild from the complete dataset row. Older normalized answers could
    # omit `metadata` because LiveCodeBench has used both `metadata` and
    # `meta_data`; functional tests need metadata.func_name.
    return build_answer(dataset_record)


def reevaluate_record(
    record: dict,
    dataset: list[dict],
    evaluator: Callable[..., bool] = code_exec_equal,
    timeout: float = 1,
    empty_only: bool = False,
    methods: set[str] | None = None,
) -> tuple[dict, EvaluationStats]:
    stats = EvaluationStats(records=1)

    if not isinstance(record.get("output"), str):
        stats.skipped_records = 1
        return record, stats

    reference = None

    def evaluate(prediction: str) -> bool:
        nonlocal reference
        if reference is None:
            reference = get_reference(dataset, record.get("line_number"))
        return evaluator(prediction, reference, timeout=timeout)

    if (methods is None or "top-level" in methods) and (
        not empty_only or "correct" not in record
    ):
        correct = evaluate(record["output"])
        stats.top_level.observe(record.get("correct"), correct)
        record["correct"] = correct

    additional_logs = record.get("additional_logs")
    nested_method_results = (
        additional_logs.get("method_results")
        if isinstance(additional_logs, dict)
        else None
    )
    method_result_lists = []
    direct_method_results = record.get("method_results")
    if isinstance(direct_method_results, list):
        method_result_lists.append(direct_method_results)
    if (
        isinstance(nested_method_results, list)
        and nested_method_results is not direct_method_results
    ):
        method_result_lists.append(nested_method_results)

    for method_results in method_result_lists:
        for method_result in method_results:
            if not isinstance(method_result, dict) or "final_answer" not in method_result:
                continue
            if methods is not None and method_result.get("method") not in methods:
                continue
            if empty_only and "correct" in method_result:
                continue
            method_correct = evaluate(method_result["final_answer"])
            stats.method_results.observe(method_result.get("correct"), method_correct)
            method_result["correct"] = method_correct

    return record, stats


def write_jsonl_atomic(
    path: Path,
    records: list[dict],
    *,
    mode: int = 0o644,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as output:
            for record in records:
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
            os.fsync(output.fileno())
        os.chmod(temporary_path, mode)
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def process_file(
    path: Path,
    output_path: Path,
    dataset: list[dict],
    *,
    evaluator: Callable[..., bool] = code_exec_equal,
    timeout: float = 1,
    num_thread: int = 1,
    dry_run: bool = False,
    empty_only: bool = False,
    methods: set[str] | None = None,
) -> EvaluationStats:
    records = load_jsonl(path)

    def evaluate(record: dict) -> tuple[dict, EvaluationStats]:
        return reevaluate_record(
            record,
            dataset,
            evaluator=evaluator,
            timeout=timeout,
            empty_only=empty_only,
            methods=methods,
        )

    if num_thread == 1:
        evaluated = [
            evaluate(record)
            for record in tqdm(
                records,
                desc=f"Evaluating {path.name}",
                unit="record",
            )
        ]
    else:
        with ThreadPoolExecutor(max_workers=num_thread) as executor:
            evaluated = list(
                tqdm(
                    executor.map(evaluate, records),
                    total=len(records),
                    desc=f"Evaluating {path.name} ({num_thread} workers)",
                    unit="record",
                )
            )

    updated_records = [record for record, _ in evaluated]
    stats = EvaluationStats()
    for _, record_stats in evaluated:
        stats.merge(record_stats)
    if not dry_run:
        source_mode = stat.S_IMODE(path.stat().st_mode)
        write_jsonl_atomic(output_path, updated_records, mode=source_mode)
    return stats


def format_status_stats(label: str, stats: StatusStats) -> str:
    return (
        f"  {label}: evaluated={stats.evaluated}, changed={stats.changed} "
        f"(false->true={stats.false_to_true}, true->false={stats.true_to_false}, "
        f"added={stats.added}), unchanged={stats.unchanged}, "
        f"final=true:{stats.final_true}/false:{stats.final_false}"
    )


def print_stats(label: str, stats: EvaluationStats, action: str) -> None:
    print(
        f"{label}: records={stats.records}, skipped={stats.skipped_records}, "
        f"statuses_{action}={stats.changed}"
    )
    print(format_status_stats("top-level", stats.top_level))
    print(format_status_stats("method-results", stats.method_results))


def main() -> None:
    args = parse_args()
    if args.timeout <= 0:
        raise ValueError("--timeout must be > 0")
    if args.num_thread < 1:
        raise ValueError("--num-thread must be >= 1")

    files = discover_jsonl_files(args.paths)
    output_dir = args.output_dir.resolve()
    output_paths = [output_dir / path.name for path in files]
    duplicate_names = {
        path.name for path in output_paths if sum(p.name == path.name for p in output_paths) > 1
    }
    if duplicate_names:
        raise ValueError(
            "Input files would collide in --output-dir: "
            + ", ".join(sorted(duplicate_names))
        )
    source_paths = set(files)
    if any(path.resolve() in source_paths for path in output_paths):
        raise ValueError("--output-dir must not resolve to a source file directory")

    print(
        f"Loading LiveCodeBench split={args.split!r} "
        f"version_tag={args.version_tag!r}..."
    )
    dataset = load_livecodebench(
        split=args.split,
        version_tag=args.version_tag,
    )
    print(f"Loaded {len(dataset)} dataset records")

    total_stats = EvaluationStats()
    action = "would_update" if args.dry_run else "updated"
    for path, output_path in zip(files, output_paths):
        stats = process_file(
            path,
            output_path,
            dataset,
            timeout=args.timeout,
            num_thread=args.num_thread,
            dry_run=args.dry_run,
            empty_only=args.empty_only,
            methods=args.methods,
        )
        total_stats.merge(stats)
        destination = f" -> {output_path}" if not args.dry_run else ""
        print_stats(f"{path}{destination}", stats, action)

    print_stats(f"TOTAL files={len(files)}", total_stats, action)


if __name__ == "__main__":
    main()
