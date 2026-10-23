#!/usr/bin/env python
"""Analyze evaluated random-exit continuations for within-trajectory stability."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


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
                raise ValueError(f"Invalid JSON in {path} at line {file_line}: {error}") from error
            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object in {path} at line {file_line}")
            rows.append(row)
    return rows


def discover_runs(
    output_dir: str | Path = "output/random_exit",
    prepared_dir: str | Path = "data/random_exit",
    *,
    benchmarks: Iterable[str] | None = None,
    models: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Recursively find raw runs and their evaluated/prepared/metadata companions."""
    output_dir = Path(output_dir)
    prepared_dir = Path(prepared_dir)
    benchmark_filter = set(benchmarks) if benchmarks else None
    model_filter = set(models) if models else None
    found: list[dict[str, Any]] = []
    for raw in sorted(output_dir.rglob("*.jsonl")):
        if raw.name.endswith(".evaluated.jsonl") or "analysis" in raw.parts:
            continue
        first_row: dict[str, Any] | None = None
        with raw.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid first JSON row in {raw}: {error}") from error
                if isinstance(value, dict):
                    first_row = value
                break
        if first_row is None:
            continue
        data_id = first_row.get("data_id")
        model = first_row.get("model")
        benchmark = data_id.partition(":")[0] if isinstance(data_id, str) else ""
        if not benchmark or not isinstance(model, str) or not model:
            raise ValueError(f"Cannot identify benchmark/model from first row of {raw}")
        if benchmark_filter is not None and benchmark not in benchmark_filter:
            continue
        if model_filter is not None and model not in model_filter:
            continue
        found.append(
            {
                "benchmark": benchmark,
                "model": model,
                "raw": raw,
                "evaluated": raw.with_name(f"{raw.stem}.evaluated.jsonl"),
                "prepared": prepared_dir / benchmark / f"{benchmark}.{model}.jsonl",
                "metadata": raw.with_suffix(".metadata.json"),
            }
        )
    if not found:
        raise ValueError(f"No random-exit runs found under {output_dir}")
    result = pd.DataFrame(found)
    duplicates = result.duplicated(["benchmark", "model"], keep=False)
    if duplicates.any():
        values = result.loc[duplicates, ["benchmark", "model", "raw"]]
        raise ValueError(f"Multiple raw files identify the same run:\n{values.to_string(index=False)}")
    return result


def validate_companions(runs: pd.DataFrame, *, require_evaluated: bool = True) -> None:
    columns = ["raw", "prepared", "metadata"]
    if require_evaluated:
        columns.append("evaluated")
    missing = [str(path) for column in columns for path in runs[column] if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError("Missing random-exit companion files:\n" + "\n".join(missing))


def pava_binomial(successes: Sequence[int], trials: Sequence[int]) -> np.ndarray:
    """Binomial MLE under nondecreasing probabilities via weighted PAVA."""
    x = np.asarray(successes, dtype=float)
    n = np.asarray(trials, dtype=float)
    if x.ndim != 1 or n.shape != x.shape or len(x) == 0:
        raise ValueError("successes and trials must be nonempty one-dimensional arrays")
    if np.any(n <= 0) or np.any(x < 0) or np.any(x > n):
        raise ValueError("Binomial counts must satisfy 0 <= successes <= trials")
    blocks: list[list[float]] = []
    for index, (success, total) in enumerate(zip(x, n, strict=True)):
        blocks.append([float(index), float(index), success, total])
        while len(blocks) >= 2:
            left, right = blocks[-2], blocks[-1]
            if left[2] / left[3] <= right[2] / right[3]:
                break
            blocks[-2:] = [[left[0], right[1], left[2] + right[2], left[3] + right[3]]]
    fitted = np.empty(len(x), dtype=float)
    for start, end, success, total in blocks:
        fitted[int(start) : int(end) + 1] = success / total
    return fitted


def binomial_log_likelihood(successes: Sequence[int], trials: Sequence[int], probabilities: Sequence[float]) -> float:
    x = np.asarray(successes, dtype=float)
    n = np.asarray(trials, dtype=float)
    p = np.asarray(probabilities, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        success_term = np.where(x == 0, 0.0, x * np.log(p))
        failure_term = np.where(n == x, 0.0, (n - x) * np.log1p(-p))
    return float(np.sum(success_term + failure_term))


def trajectory_statistics(successes: Sequence[int], trials: Sequence[int]) -> dict[str, Any]:
    x = np.asarray(successes, dtype=int)
    n = np.asarray(trials, dtype=int)
    observed = x / n
    fitted = pava_binomial(x, n)
    discrepancy = 2.0 * (
        binomial_log_likelihood(x, n, observed)
        - binomial_log_likelihood(x, n, fitted)
    ) / len(x)
    pair_drops = [max(float(observed[a] - observed[b]), 0.0)
                  for a in range(len(x)) for b in range(a + 1, len(x))]
    return {
        "d": max(float(discrepancy), 0.0),
        "a": float(np.mean(pair_drops)) if pair_drops else 0.0,
        "observed_probabilities": observed,
        "isotonic_probabilities": fitted,
    }


def load_run(
    run: pd.Series,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame, pd.DataFrame]:
    metadata = json.loads(Path(run.metadata).read_text(encoding="utf-8"))
    settings = metadata.get("resolved", {}).get("random_exit", {})
    num_runs = int(settings.get("num_runs", 0))
    min_positions = int(settings.get("min_valid_positions", 0))
    if num_runs < 1 or min_positions < 1:
        raise ValueError(f"Invalid random-exit settings in {run.metadata}")

    prepared_records = read_jsonl(run.prepared)
    manifest_ids = [record.get("data_id") for record in prepared_records]
    if any(not isinstance(value, str) for value in manifest_ids) or len(set(manifest_ids)) != len(manifest_ids):
        raise ValueError(f"Prepared input has invalid or duplicate data_id values: {run.prepared}")

    records = read_jsonl(run.evaluated)
    rows = pd.DataFrame(records)
    required = {"key", "data_id", "line_number", "model", "position", "num_run", "ok", "correct"}
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"{run.evaluated} is missing columns: {', '.join(missing)}")
    if rows["key"].duplicated().any():
        raise ValueError(f"Duplicate continuation keys in {run.evaluated}")
    if not rows["correct"].map(lambda value: isinstance(value, (bool, np.bool_))).all():
        raise ValueError(f"Every row in {run.evaluated} must have boolean correct")
    unknown = sorted(set(rows["data_id"]) - set(manifest_ids))
    if unknown:
        raise ValueError(f"Evaluated rows are absent from prepared manifest: {unknown[:10]}")
    rows["benchmark"] = str(run.benchmark)
    rows["model"] = str(run.model)
    rows["trajectory_id"] = rows["benchmark"] + "::" + rows["model"] + "::" + rows["data_id"]

    position_rows: list[dict[str, Any]] = []
    expected_runs = set(range(num_runs))
    for (trajectory_id, data_id, position), group in rows.groupby(
        ["trajectory_id", "data_id", "position"], sort=False
    ):
        run_numbers = set(int(value) for value in group["num_run"])
        complete = len(group) == num_runs and run_numbers == expected_runs and bool(group["ok"].eq(True).all())
        position_rows.append(
            {
                "benchmark": str(run.benchmark),
                "model": str(run.model),
                "trajectory_id": trajectory_id,
                "data_id": data_id,
                "position": int(position),
                "observed_runs": len(group),
                "successful_runs": int(group["ok"].eq(True).sum()),
                "expected_runs": num_runs,
                "complete": complete,
                "successes": int(group["correct"].sum()) if complete else np.nan,
            }
        )
    positions = pd.DataFrame(position_rows)
    complete_counts = positions.loc[positions["complete"]].groupby("trajectory_id").size()
    eligible = set(complete_counts[complete_counts >= min_positions].index)

    trajectory_rows: list[dict[str, Any]] = []
    for trajectory_id, group in positions.loc[
        positions["complete"] & positions["trajectory_id"].isin(eligible)
    ].groupby("trajectory_id", sort=False):
        group = group.sort_values("position")
        stats = trajectory_statistics(
            group["successes"].astype(int).to_numpy(),
            group["expected_runs"].astype(int).to_numpy(),
        )
        trajectory_rows.append(
            {
                "benchmark": str(run.benchmark),
                "model": str(run.model),
                "trajectory_id": trajectory_id,
                "data_id": group["data_id"].iloc[0],
                "num_positions": len(group),
                "d": stats["d"],
                "a": stats["a"],
                "successes": group["successes"].astype(int).to_list(),
                "trials": group["expected_runs"].astype(int).to_list(),
                "isotonic_probabilities": stats["isotonic_probabilities"].tolist(),
            }
        )
    trajectories = pd.DataFrame(trajectory_rows)
    manifest_count = len(manifest_ids)
    eligible_count = len(trajectories)
    completeness = pd.DataFrame([{
        "benchmark": str(run.benchmark),
        "model": str(run.model),
        "manifest_trajectories": manifest_count,
        "observed_trajectories": int(positions["trajectory_id"].nunique()),
        "eligible_trajectories": eligible_count,
        "excluded_trajectories": manifest_count - eligible_count,
        "excluded_fraction": (manifest_count - eligible_count) / manifest_count if manifest_count else np.nan,
        "complete_positions": int(positions["complete"].sum()),
        "observed_rows": len(rows),
        "num_runs": num_runs,
        "min_valid_positions": min_positions,
    }])
    return trajectories, metadata, completeness, positions


def macro_average(values: pd.DataFrame, column: str) -> tuple[float, pd.DataFrame, pd.DataFrame]:
    cells = values.groupby(["benchmark", "model"], as_index=False)[column].mean()
    models = cells.groupby("model", as_index=False)[column].mean()
    return float(models[column].mean()), cells, models


def parametric_monotone_bootstrap(
    trajectories: pd.DataFrame,
    *,
    replicates: int = 10_000,
    seed: int = 0,
    trajectory_sample_replicates: int = 100,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if trajectories.empty:
        raise ValueError("No eligible trajectories for bootstrap")
    rng = np.random.default_rng(seed)
    records: list[dict[str, Any]] = []
    trajectory_sample: list[dict[str, Any]] = []
    null_a_sums = np.zeros(len(trajectories), dtype=float)
    null_d_sums = np.zeros(len(trajectories), dtype=float)
    for replicate in range(replicates):
        simulated: list[dict[str, Any]] = []
        for row_index, row in enumerate(trajectories.itertuples(index=False)):
            trials = np.asarray(row.trials, dtype=int)
            successes = rng.binomial(trials, np.asarray(row.isotonic_probabilities))
            stats = trajectory_statistics(successes, trials)
            null_a_sums[row_index] += stats["a"]
            null_d_sums[row_index] += stats["d"]
            if replicate < trajectory_sample_replicates:
                trajectory_sample.append(
                    {
                        "replicate": replicate,
                        "trajectory_id": row.trajectory_id,
                        "d": stats["d"],
                        "a": stats["a"],
                    }
                )
            simulated.append({"benchmark": row.benchmark, "model": row.model,
                              "d": stats["d"], "a": stats["a"]})
        simulated_frame = pd.DataFrame(simulated)
        d_value, _cells, _models = macro_average(simulated_frame, "d")
        a_value, _cells, _models = macro_average(simulated_frame, "a")
        records.append({"replicate": replicate, "d": d_value, "a": a_value})
    null_means = trajectories[["trajectory_id"]].copy()
    null_means["a_null_mean"] = null_a_sums / replicates
    null_means["d_null_mean"] = null_d_sums / replicates
    return pd.DataFrame(records), null_means, pd.DataFrame(trajectory_sample)


def question_bootstrap(
    trajectories: pd.DataFrame,
    *,
    replicates: int = 10_000,
    seed: int = 1,
) -> pd.DataFrame:
    required = {"benchmark", "model", "data_id", "d", "a", "a_null_mean"}
    if missing := sorted(required - set(trajectories.columns)):
        raise ValueError(f"Question bootstrap is missing columns: {missing}")
    rng = np.random.default_rng(seed)
    benchmark_ids = {
        benchmark: np.asarray(sorted(group["data_id"].unique()), dtype=object)
        for benchmark, group in trajectories.groupby("benchmark")
    }
    records: list[dict[str, Any]] = []
    for replicate in range(replicates):
        pieces: list[pd.DataFrame] = []
        for benchmark, identifiers in benchmark_ids.items():
            draws = rng.choice(identifiers, size=len(identifiers), replace=True)
            source = trajectories.loc[trajectories["benchmark"] == benchmark]
            for draw_index, data_id in enumerate(draws):
                selected = source.loc[source["data_id"] == data_id].copy()
                selected["bootstrap_question"] = draw_index
                pieces.append(selected)
        sampled = pd.concat(pieces, ignore_index=True)
        d_value, _cells, _models = macro_average(sampled, "d")
        a_value, _cells, _models = macro_average(sampled, "a")
        excess_value, _cells, _models = macro_average(
            sampled.assign(a_excess=sampled["a"] - sampled["a_null_mean"]), "a_excess"
        )
        records.append({"replicate": replicate, "d": d_value, "a": a_value,
                        "a_excess": excess_value})
    return pd.DataFrame(records)


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(values) - rank) * values[index])
        adjusted[index] = min(running, 1.0)
    return adjusted


def summarize(
    trajectories: pd.DataFrame,
    null_bootstrap: pd.DataFrame,
    question_samples: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    observed_d, cells_d, models_d = macro_average(trajectories, "d")
    observed_a, cells_a, models_a = macro_average(trajectories, "a")
    null_a = float(null_bootstrap["a"].mean())
    p_d = (1 + int((null_bootstrap["d"] >= observed_d).sum())) / (len(null_bootstrap) + 1)
    p_a = (1 + int((null_bootstrap["a"] >= observed_a).sum())) / (len(null_bootstrap) + 1)
    headline = pd.DataFrame([{
        "d_overall": observed_d,
        "d_ci_low": question_samples["d"].quantile(0.025),
        "d_ci_high": question_samples["d"].quantile(0.975),
        "p_d": p_d,
        "a_observed": observed_a,
        "a_null_expected": null_a,
        "a_excess": observed_a - null_a,
        "a_excess_ci_low": question_samples["a_excess"].quantile(0.025),
        "a_excess_ci_high": question_samples["a_excess"].quantile(0.975),
        "p_a": p_a,
        "eligible_trajectories": len(trajectories),
    }])
    cells = cells_d.merge(cells_a, on=["benchmark", "model"])
    models = models_d.merge(models_a, on="model")
    return headline, cells, models


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("output/random_exit"))
    parser.add_argument("--prepared-dir", type=Path, default=Path("data/random_exit"))
    parser.add_argument("--table-dir", type=Path, default=Path("output/random_exit/analysis/stability"))
    parser.add_argument("--benchmark", action="append")
    parser.add_argument("--model", action="append")
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--question-bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.bootstrap_replicates < 1 or args.question_bootstrap_replicates < 1:
        parser.error("bootstrap replicate counts must be positive")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    runs = discover_runs(args.output_dir, args.prepared_dir,
                         benchmarks=args.benchmark, models=args.model)
    validate_companions(runs)
    trajectory_parts: list[pd.DataFrame] = []
    completeness_parts: list[pd.DataFrame] = []
    for run in runs.itertuples(index=False):
        trajectories, _metadata, completeness, _positions = load_run(
            pd.Series(run._asdict())
        )
        trajectory_parts.append(trajectories)
        completeness_parts.append(completeness)
    trajectories = pd.concat(trajectory_parts, ignore_index=True)
    if trajectories.empty:
        raise ValueError("No trajectories meet min_valid_positions")
    null_bootstrap, null_means, null_trajectory_sample = parametric_monotone_bootstrap(
        trajectories, replicates=args.bootstrap_replicates, seed=args.seed
    )
    trajectories = trajectories.merge(null_means, on="trajectory_id", validate="one_to_one")
    question_samples = question_bootstrap(
        trajectories, replicates=args.question_bootstrap_replicates, seed=args.seed + 1
    )
    headline, cells, models = summarize(trajectories, null_bootstrap, question_samples)
    args.table_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "runs": runs,
        "completeness": pd.concat(completeness_parts, ignore_index=True),
        "trajectories": trajectories.drop(columns=["successes", "trials", "isotonic_probabilities"]),
        "benchmark_model": cells,
        "models": models,
        "headline": headline,
        "monotone_bootstrap": null_bootstrap,
        "monotone_trajectory_sample": null_trajectory_sample,
        "question_bootstrap": question_samples,
    }
    for name, table in tables.items():
        table.to_csv(args.table_dir / f"{name}.csv", index=False)
    print(headline.to_string(index=False))
    print(f"Wrote {len(tables)} tables to {args.table_dir}")


if __name__ == "__main__":
    main()
