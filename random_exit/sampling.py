from __future__ import annotations

import hashlib
import json
import random
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

from replay_eval.v2.config.schema import RandomExitConfig


MANIFEST_VERSION = 1


def metadata_path(path: str | Path) -> Path:
    return Path(path).with_suffix(".metadata.json")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def manifest_identity(manifest: dict[str, Any]) -> str:
    payload = {key: value for key, value in manifest.items() if key != "identity"}
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def dataset_fingerprint(records: Iterable[dict[str, Any]]) -> str:
    return hashlib.sha256(
        _canonical_json(list(records)).encode("utf-8")
    ).hexdigest()


def benchmark_data_id(dataset: str, line_number: int) -> str:
    return f"{dataset}:{line_number}"


def model_run_line_numbers(
    records: Iterable[dict[str, Any]], *, dataset: str | None = None
) -> set[int]:
    source = list(records)
    run_datasets = {
        str(value)
        for record in source
        if isinstance(record, dict)
        for value in (record.get("dataloader") or record.get("dataset"),)
        if value is not None
    }
    if len(run_datasets) > 1:
        raise ValueError(f"Model run mixes dataset identities: {sorted(run_datasets)!r}")
    if dataset is not None and run_datasets and dataset not in run_datasets:
        raise ValueError(
            f"Model run dataset {next(iter(run_datasets))!r} conflicts with "
            f"manifest dataset {dataset!r}"
        )
    line_numbers: set[int] = set()
    for fallback, record in enumerate(source, start=1):
        if not isinstance(record, dict):
            raise TypeError(f"Model run record {fallback} is not an object")
        number = _record_number(record, fallback)
        if number in line_numbers:
            raise ValueError(f"Model run has duplicate line_number {number}")
        line_numbers.add(number)
    return line_numbers


def build_manifest(
    records: list[dict[str, Any]],
    *,
    dataset: str,
    split: str,
    config: RandomExitConfig,
    eligible_line_numbers: Iterable[int] | None = None,
) -> dict[str, Any]:
    config.validate()
    dataset_size = len(records)
    if eligible_line_numbers is None:
        candidates = list(range(1, dataset_size + 1))
    else:
        candidates = sorted(set(eligible_line_numbers))
        invalid = [value for value in candidates if not 1 <= value <= dataset_size]
        if invalid:
            raise ValueError(
                "Eligible model-run line numbers fall outside the dataset: "
                + ", ".join(str(value) for value in invalid[:20])
            )
    count = min(config.sample_size, len(candidates))
    line_numbers = candidates
    if len(candidates) > config.sample_size:
        line_numbers = sorted(
            random.Random(config.sample_seed).sample(candidates, count)
        )
    items = [
        {
            "data_id": benchmark_data_id(dataset, line_number),
            "line_number": line_number,
        }
        for line_number in line_numbers
    ]
    manifest = {
        "version": MANIFEST_VERSION,
        "dataset": dataset,
        "split": split,
        "dataset_size": dataset_size,
        "dataset_fingerprint": dataset_fingerprint(records),
        "sample_size": config.sample_size,
        "sample_seed": config.sample_seed,
        "candidate_size": len(candidates),
        "candidate_fingerprint": hashlib.sha256(
            _canonical_json(candidates).encode("utf-8")
        ).hexdigest(),
        "items": items,
    }
    manifest["identity"] = manifest_identity(manifest)
    return manifest


def validate_manifest(
    manifest: dict[str, Any],
    *,
    dataset: str,
    split: str,
    dataset_size: int,
    config: RandomExitConfig,
    records: list[dict[str, Any]] | None = None,
    eligible_line_numbers: Iterable[int] | None = None,
) -> None:
    expected = {
        "version": MANIFEST_VERSION,
        "dataset": dataset,
        "split": split,
        "dataset_size": dataset_size,
        "dataset_fingerprint": (
            dataset_fingerprint(records)
            if records is not None
            else manifest.get("dataset_fingerprint")
        ),
        "sample_size": config.sample_size,
        "sample_seed": config.sample_seed,
    }
    conflicts = {
        key: (manifest.get(key), value)
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if conflicts:
        details = ", ".join(
            f"{key}: manifest={old!r}, requested={new!r}"
            for key, (old, new) in conflicts.items()
        )
        raise ValueError(f"Existing random-exit manifest conflicts with input: {details}")
    identity = manifest.get("identity")
    actual_identity = manifest_identity(manifest)
    if identity != actual_identity:
        raise ValueError(
            "Existing random-exit manifest identity does not match its contents"
        )
    items = manifest.get("items")
    if not isinstance(items, list):
        raise ValueError("Existing random-exit manifest items must be a list")
    candidate_size = manifest.get("candidate_size", dataset_size)
    if (
        not isinstance(candidate_size, int)
        or isinstance(candidate_size, bool)
        or not 0 <= candidate_size <= dataset_size
    ):
        raise ValueError("Existing random-exit manifest has invalid candidate_size")
    expected_count = min(config.sample_size, candidate_size)
    if len(items) != expected_count:
        raise ValueError(
            f"Existing random-exit manifest has {len(items)} items; expected {expected_count}"
        )
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Existing random-exit manifest contains a non-object item")
        line_number = item.get("line_number")
        data_id = item.get("data_id")
        expected_id = (
            benchmark_data_id(dataset, line_number)
            if isinstance(line_number, int) and not isinstance(line_number, bool)
            else None
        )
        if (
            expected_id is None
            or not 1 <= line_number <= dataset_size
            or data_id != expected_id
            or data_id in seen
        ):
            raise ValueError(f"Invalid random-exit manifest item: {item!r}")
        seen.add(data_id)
    if eligible_line_numbers is not None:
        eligible = set(eligible_line_numbers)
        unavailable = [
            item["data_id"]
            for item in items
            if item["line_number"] not in eligible
        ]
        if unavailable:
            preview = ", ".join(unavailable[:20])
            suffix = (
                "" if len(unavailable) <= 20 else f" (and {len(unavailable) - 20} more)"
            )
            raise ValueError(
                "Existing permanent manifest contains IDs unavailable in one or "
                f"more supplied model runs: {preview}{suffix}"
            )


def build_or_load_manifest(
    path: str | Path,
    records: Iterable[dict[str, Any]],
    *,
    dataset: str,
    split: str,
    config: RandomExitConfig,
    eligible_line_numbers: Iterable[int] | None = None,
    migrate_legacy_eligibility: bool = False,
) -> dict[str, Any]:
    path = Path(path)
    materialized = list(records)
    eligible = (
        None if eligible_line_numbers is None else set(eligible_line_numbers)
    )
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if not isinstance(manifest, dict):
            raise ValueError("Existing random-exit manifest must be a JSON object")
        if migrate_legacy_eligibility and "candidate_size" not in manifest:
            validate_manifest(
                manifest,
                dataset=dataset,
                split=split,
                dataset_size=len(materialized),
                config=config,
                records=materialized,
            )
            manifest = build_manifest(
                materialized,
                dataset=dataset,
                split=split,
                config=config,
                eligible_line_numbers=eligible,
            )
            with path.open("w", encoding="utf-8") as handle:
                json.dump(manifest, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
            return manifest
        existing_sample_size = manifest.get("sample_size")
        if (
            isinstance(existing_sample_size, int)
            and not isinstance(existing_sample_size, bool)
            and config.sample_size > existing_sample_size
        ):
            existing_config = replace(config, sample_size=existing_sample_size)
            validate_manifest(
                manifest,
                dataset=dataset,
                split=split,
                dataset_size=len(materialized),
                config=existing_config,
                records=materialized,
                eligible_line_numbers=eligible,
            )
            candidates = (
                list(range(1, len(materialized) + 1))
                if eligible is None
                else sorted(eligible)
            )
            candidate_fingerprint = hashlib.sha256(
                _canonical_json(candidates).encode("utf-8")
            ).hexdigest()
            selected = {int(item["line_number"]) for item in manifest["items"]}
            remaining = [value for value in candidates if value not in selected]
            target_count = min(config.sample_size, len(candidates))
            additional_count = target_count - len(manifest["items"])
            additional = sorted(
                random.Random(config.sample_seed).sample(remaining, additional_count)
            )
            manifest["sample_size"] = config.sample_size
            manifest["candidate_size"] = len(candidates)
            manifest["candidate_fingerprint"] = candidate_fingerprint
            manifest["items"].extend(
                {
                    "data_id": benchmark_data_id(dataset, line_number),
                    "line_number": line_number,
                }
                for line_number in additional
            )
            manifest["identity"] = manifest_identity(manifest)
            temporary = path.with_suffix(path.suffix + ".tmp")
            with temporary.open("w", encoding="utf-8") as handle:
                json.dump(manifest, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
            temporary.replace(path)
            return manifest
        validate_manifest(
            manifest,
            dataset=dataset,
            split=split,
            dataset_size=len(materialized),
            config=config,
            records=materialized,
            eligible_line_numbers=eligible,
        )
        return manifest
    manifest = build_manifest(
        materialized,
        dataset=dataset,
        split=split,
        config=config,
        eligible_line_numbers=eligible,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return manifest


def _record_number(record: dict[str, Any], fallback: int) -> int:
    nested = record.get("input")
    nested = nested if isinstance(nested, dict) else {}
    value = record.get("line_number", record.get("id", nested.get("id", fallback)))
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return fallback


def _trajectory_payload(record: dict[str, Any]) -> dict[str, Any]:
    for key in ("response", "trajectory", "generation_response"):
        value = record.get(key)
        if isinstance(value, dict):
            return value
    if isinstance(record.get("output"), str):
        return {
            "input": record.get("prompt_input", ""),
            "content": record["output"],
            "finish_reason": record.get("finish_reason"),
            "prompt_tokens": record.get("prompt_tokens"),
            "completion_tokens": record.get("completion_tokens")
            or record.get("num_token"),
        }
    raise ValueError("sampled model record has no supplied trajectory")


def _field(record: dict[str, Any], names: tuple[str, ...]) -> Any:
    nested = record.get("input")
    nested = nested if isinstance(nested, dict) else {}
    for source in (record, nested):
        for name in names:
            value = source.get(name)
            if value is not None:
                return value
    return None


def _model(record: dict[str, Any], model: str | None) -> str:
    value = model or record.get("model") or record.get("model_name")
    if not value:
        raise ValueError("model identity is missing; pass model= or store model in the run")
    return str(value)


def prepare_model_run(
    records: Iterable[dict[str, Any]],
    manifest: dict[str, Any],
    *,
    model: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source = list(records)
    model_run_line_numbers(source, dataset=str(manifest.get("dataset")))
    by_number: dict[int, dict[str, Any]] = {}
    for fallback, record in enumerate(source, start=1):
        if not isinstance(record, dict):
            raise TypeError(f"Model run record {fallback} is not an object")
        number = _record_number(record, fallback)
        if number in by_number:
            raise ValueError(f"Model run has duplicate line_number {number}")
        by_number[number] = record

    prepared: list[dict[str, Any]] = []
    prompt_counts = {"stored": 0, "reconstructed_during_replay": 0}
    missing: list[str] = []
    model_identity: str | None = model
    for item in manifest["items"]:
        line_number = int(item["line_number"])
        record = by_number.get(line_number)
        if record is None:
            missing.append(str(item["data_id"]))
            continue
        trajectory = _trajectory_payload(record)
        record_model = _model(record, model)
        if model_identity is not None and record_model != model_identity:
            raise ValueError(
                f"Model run mixes model identities: {model_identity!r} and {record_model!r}"
            )
        model_identity = record_model
        stored_input = trajectory.get("input")
        if isinstance(stored_input, str) and stored_input:
            prompt_counts["stored"] += 1
        else:
            prompt_counts["reconstructed_during_replay"] += 1
        response = {
            key: trajectory.get(key)
            for key in (
                "input",
                "content",
                "finish_reason",
                "prompt_tokens",
                "completion_tokens",
            )
            if trajectory.get(key) is not None
        }
        row = {
            "data_id": item["data_id"],
            "line_number": line_number,
            "dataset": manifest["dataset"],
            "question": _field(record, ("question", "problem", "prompt")),
            "answer": _field(record, ("answer", "final_answer", "solution")),
            "model": model_identity,
            "response": response,
        }
        for key in (
            "output_token_ids",
            "reasoning_token_count",
            "thinking_token_count",
        ):
            if record.get(key) is not None:
                row[key] = record[key]
        prepared.append(row)
    if missing:
        preview = ", ".join(missing[:20])
        suffix = "" if len(missing) <= 20 else f" (and {len(missing) - 20} more)"
        raise ValueError(f"Model run is missing manifest IDs: {preview}{suffix}")
    metadata = {
        "version": 1,
        "kind": "random_exit_prepared_input",
        "manifest_identity": manifest["identity"],
        "dataset": manifest["dataset"],
        "split": manifest["split"],
        "model": model_identity,
        "records": len(prepared),
        "prompt_provenance": prompt_counts,
    }
    return prepared, metadata


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"JSONL line {line_number} is not an object")
            values.append(value)
    return values


def write_prepared_run(
    output_path: str | Path,
    rows: Iterable[dict[str, Any]],
    metadata: dict[str, Any],
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with metadata_path(output_path).open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
