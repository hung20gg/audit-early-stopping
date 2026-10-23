#!/usr/bin/env python3
"""Add top-level model and replay-eval agent keys to a trace JSONL file."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Source trace JSONL file.")
    parser.add_argument(
        "--model",
        default="gpt_oss_20b_high",
        help="Model config key to store (default: gpt_oss_20b_high).",
    )
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--output", type=Path, help="Write a patched copy.")
    destination.add_argument(
        "--in-place", action="store_true", help="Atomically replace the source."
    )
    parser.add_argument(
        "--backup-suffix",
        default="",
        help="With --in-place, copy the original to INPUT<suffix> before patching.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            rows.append(row)
    return rows


def add_model_key(rows: list[dict[str, Any]], model: str) -> int:
    model = model.strip()
    if not model:
        raise ValueError("model must not be empty")
    changed = 0
    for row in rows:
        if row.get("model") != model or row.get("agent") != "replay_eval":
            row["model"] = model
            row["agent"] = "replay_eval"
            changed += 1
    return changed


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        if path.exists():
            shutil.copymode(path, temporary_name)
        os.replace(temporary_name, path)
    except BaseException:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
        raise


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"Input JSONL does not exist: {args.input}")
    target = args.input if args.in_place else args.output
    assert target is not None
    if target.resolve() == args.input.resolve() and not args.in_place:
        raise ValueError("--output must differ from input; use --in-place")
    if args.backup_suffix and not args.in_place:
        raise ValueError("--backup-suffix requires --in-place")

    rows = load_jsonl(args.input)
    changed = add_model_key(rows, args.model)
    if args.backup_suffix:
        shutil.copy2(
            args.input, args.input.with_name(args.input.name + args.backup_suffix)
        )
    write_jsonl_atomic(target, rows)
    print(
        f"Patched {changed}/{len(rows)} records in {target} "
        f"with model={args.model!r}, agent='replay_eval'"
    )


if __name__ == "__main__":
    main()

