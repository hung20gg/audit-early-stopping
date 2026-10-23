from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from random_exit.sampling import (
    build_or_load_manifest,
    model_run_line_numbers,
    prepare_model_run,
    read_jsonl,
    write_prepared_run,
)
from replay_eval.v2.config.schema import RandomExitConfig, load_random_exit_config
from scripts.config_loader import import_config_entry, load_config


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "scripts" / "config.yml"


def _positive(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return parsed


def _parse_bool(value: str) -> bool:
    normalized = value.strip().casefold()
    if normalized in {"true", "1", "yes", "on"}:
        return True
    if normalized in {"false", "0", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create/reuse a permanent dataset sample manifest and extract matching "
            "trajectories from existing model runs."
        )
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--dataset", required=True, help="Dataloader/config key")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        metavar="MODEL=JSONL",
        help="Existing model run; repeat for each model.",
    )
    parser.add_argument(
        "--input-file",
        action="append",
        default=[],
        type=Path,
        help="Existing run whose model identity is stored in its rows.",
    )
    parser.add_argument("--random-exit", type=_parse_bool, default=None)
    parser.add_argument("--random-exit-sample-size", type=_positive, default=None)
    parser.add_argument("--random-exit-sample-seed", type=int, default=None)
    parser.add_argument("--random-exit-position-seed", type=int, default=None)
    parser.add_argument("--random-exit-spacing", type=_positive, default=None)
    parser.add_argument("--random-exit-max-positions", type=_positive, default=None)
    parser.add_argument("--random-exit-num-runs", type=_positive, default=None)
    parser.add_argument(
        "--random-exit-min-valid-positions", type=_positive, default=None
    )
    return parser.parse_args(argv)


def resolve_random_exit_config(args: argparse.Namespace) -> RandomExitConfig:
    config = load_random_exit_config(args.config)
    if args.random_exit is not None:
        config.enabled = args.random_exit
    mapping = {
        "sample_size": "random_exit_sample_size",
        "sample_seed": "random_exit_sample_seed",
        "position_seed": "random_exit_position_seed",
        "spacing": "random_exit_spacing",
        "max_positions": "random_exit_max_positions",
        "num_runs": "random_exit_num_runs",
        "min_valid_positions": "random_exit_min_valid_positions",
    }
    for field, argument in mapping.items():
        value = getattr(args, argument)
        if value is not None:
            setattr(config, field, value)
    config.validate()
    return config


def _runs(args: argparse.Namespace) -> list[tuple[str | None, Path]]:
    values: list[tuple[str | None, Path]] = []
    for specification in args.run:
        model, separator, path = specification.partition("=")
        if not separator or not model or not path:
            raise ValueError(f"--run must be MODEL=JSONL, got {specification!r}")
        values.append((model, Path(path)))
    values.extend((None, path) for path in args.input_file)
    if not values:
        raise ValueError("At least one --run or --input-file is required")
    return values


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    random_config = resolve_random_exit_config(args)
    shared_config = load_config(args.config)
    dataloader = import_config_entry(shared_config, "dataloader", args.dataset)
    dataset_records: list[dict[str, Any]] = list(
        dataloader(split=args.split, limit=None)
    )
    manifest_path = args.manifest or (
        args.output_dir / f"{args.dataset}.{args.split}.manifest.json"
    )
    loaded_runs = [
        (explicit_model, input_path, read_jsonl(input_path))
        for explicit_model, input_path in _runs(args)
    ]
    available_sets = [
        model_run_line_numbers(records, dataset=args.dataset)
        for _model, _path, records in loaded_runs
    ]
    common_line_numbers = set(range(1, len(dataset_records) + 1))
    for available in available_sets:
        common_line_numbers.intersection_update(available)
    if not common_line_numbers:
        raise ValueError("No dataset records are present in every supplied model run")
    skipped_line_numbers = sorted(
        set(range(1, len(dataset_records) + 1)) - common_line_numbers
    )
    print(
        f"Common records across {len(loaded_runs)} runs: "
        f"{len(common_line_numbers)}/{len(dataset_records)} "
        f"({len(skipped_line_numbers)} skipped)"
    )
    if skipped_line_numbers:
        preview = ", ".join(
            f"{args.dataset}:{line_number}" for line_number in skipped_line_numbers[:20]
        )
        suffix = (
            ""
            if len(skipped_line_numbers) <= 20
            else f" (and {len(skipped_line_numbers) - 20} more)"
        )
        print(f"Skipped IDs absent from at least one run: {preview}{suffix}")
    manifest = build_or_load_manifest(
        manifest_path,
        dataset_records,
        dataset=args.dataset,
        split=args.split,
        config=random_config,
        eligible_line_numbers=common_line_numbers,
        migrate_legacy_eligibility=True,
    )
    for explicit_model, input_path, input_records in loaded_runs:
        try:
            rows, metadata = prepare_model_run(
                input_records, manifest, model=explicit_model
            )
        except Exception as exc:
            label = explicit_model or "<model from rows>"
            raise ValueError(
                f"Cannot prepare model {label!r} from {input_path}: {exc}"
            ) from exc
        model = str(metadata["model"]).replace("/", "_").replace(":", "_")
        output_path = args.output_dir / f"{args.dataset}.{model}.jsonl"
        write_prepared_run(output_path, rows, metadata)
        print(f"Wrote {len(rows)} records to {output_path}")
    print(f"Manifest: {manifest_path} ({manifest['identity']})")


if __name__ == "__main__":
    main()
