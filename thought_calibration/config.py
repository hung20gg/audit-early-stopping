"""Strict YAML configuration for Thought Calibration training."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULTS: dict[str, Any] = {
    "base_model": None,
    "datasets": [],
    "output_dir": None,
    "output_policy": "error",
    "features": {
        "source": "pre_final_rmsnorm",
        "pooling": "segment_mean",
        "record_method": "curation_grid",
        "label_policy": "cumulative_first_correct",
        "duplicate_label_policy": "require_agreement",
        "empty_segment_policy": "skip",
    },
    "split": {"train_fraction": 0.50, "calibration_fraction": 0.45, "test_fraction": 0.05, "seed": 42},
    "standard_scaler": {"with_mean": True, "with_std": True},
    "pca": {"n_components": 256, "svd_solver": "auto", "whiten": False, "random_state": 42},
    "logistic_regression": {
        "solver": "lbfgs", "C": 1.0, "max_iter": 2000, "class_weight": None,
        "tol": 0.0001, "random_state": 42,
    },
    "smoothing": {"window": 10},
    "ltt": {
        "risk_tolerances": [0.05, 0.1], "failure_probability": 0.05,
        "threshold_grid": {"minimum": 0.5, "maximum": 0.99, "count": 100},
        "comparison": ">=", "loss": "binary_stop", "fallback": "full_budget",
    },
    "loading": {"batch_size": 1, "workers": 0, "cache": True},
    "logging": {"level": "INFO"},
    "numeric": {"dtype": "auto"},
}


def _merge_strict(default: Any, supplied: Any, path: str = "") -> Any:
    if isinstance(default, dict):
        if not isinstance(supplied, dict):
            raise ValueError(f"Configuration {path or '<root>'} must be a mapping")
        unknown = set(supplied) - set(default)
        if unknown:
            names = ", ".join(sorted(f"{path}.{key}".strip(".") for key in unknown))
            raise ValueError(f"Unknown configuration key(s): {names}")
        result = deepcopy(default)
        for key, value in supplied.items():
            result[key] = _merge_strict(default[key], value, f"{path}.{key}".strip("."))
        return result
    return deepcopy(supplied)


def _resolve_path(value: str, base: Path) -> str:
    path = Path(value).expanduser()
    return str(path if path.is_absolute() else (base / path).resolve())


def load_config(path: str | Path, *, require_paths: bool = True) -> dict[str, Any]:
    config_path = Path(path).resolve()
    supplied = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    required = {"base_model", "datasets", "output_dir"}
    missing_top_level = sorted(required - set(supplied))
    if missing_top_level:
        raise ValueError(f"Missing required configuration keys: {missing_top_level}")
    config = _merge_strict(DEFAULTS, supplied)
    if not isinstance(config["base_model"], str) or not config["base_model"].strip():
        raise ValueError("base_model must be a non-empty model identifier or local path")
    if not isinstance(config["datasets"], list) or not config["datasets"]:
        raise ValueError("datasets must be a non-empty list")
    allowed_dataset_keys = {"shards", "curation_jsonl"}
    for index, dataset in enumerate(config["datasets"]):
        if not isinstance(dataset, dict):
            raise ValueError(f"datasets[{index}] must be a mapping")
        unknown = set(dataset) - allowed_dataset_keys
        missing = allowed_dataset_keys - set(dataset)
        if unknown or missing:
            raise ValueError(
                f"datasets[{index}] invalid keys; missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        for key in ("shards", "curation_jsonl"):
            dataset[key] = _resolve_path(str(dataset[key]), PROJECT_ROOT)
            if require_paths and not Path(dataset[key]).exists():
                raise ValueError(f"Dataset path does not exist: {dataset[key]}")
    config["output_dir"] = _resolve_path(str(config["output_dir"]), PROJECT_ROOT)

    split = config["split"]
    fractions = [split["train_fraction"], split["calibration_fraction"], split["test_fraction"]]
    if any(not isinstance(value, (int, float)) or value <= 0 for value in fractions):
        raise ValueError("Split fractions must be positive numbers")
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("Split fractions must sum to 1")
    if config["output_policy"] not in {"error", "overwrite", "resume"}:
        raise ValueError("output_policy must be error, overwrite, or resume")
    feature = config["features"]
    supported = {
        "source": {"pre_final_rmsnorm"}, "pooling": {"segment_mean"},
        "label_policy": {"cumulative_first_correct"},
        "duplicate_label_policy": {"require_agreement"}, "empty_segment_policy": {"skip", "error"},
    }
    for key, choices in supported.items():
        if feature[key] not in choices:
            raise ValueError(f"Unsupported features.{key}: {feature[key]}")
    if feature["record_method"] is not None and not isinstance(feature["record_method"], str):
        raise ValueError("features.record_method must be a string or null")
    if config["numeric"]["dtype"] not in {"auto", "float16", "float32", "float64"}:
        raise ValueError("numeric.dtype must be auto, float16, float32, or float64")
    if config["smoothing"]["window"] < 1:
        raise ValueError("smoothing.window must be >= 1")
    loading = config["loading"]
    if not isinstance(loading["batch_size"], int) or loading["batch_size"] < 1:
        raise ValueError("loading.batch_size must be an integer >= 1")
    if not isinstance(loading["workers"], int) or loading["workers"] < 0:
        raise ValueError("loading.workers must be an integer >= 0")
    if not isinstance(loading["cache"], bool):
        raise ValueError("loading.cache must be boolean")
    ltt = config["ltt"]
    if ltt["comparison"] not in {">=", ">"} or ltt["loss"] != "binary_stop" or ltt["fallback"] != "full_budget":
        raise ValueError("Unsupported LTT comparison, loss, or fallback")
    grid = ltt["threshold_grid"]
    if grid["count"] < 1 or not 0 <= grid["minimum"] <= grid["maximum"] <= 1:
        raise ValueError("Invalid LTT threshold grid")
    if not 0 < ltt["failure_probability"] < 1 or any(
        not 0 < value < 1 for value in ltt["risk_tolerances"]
    ):
        raise ValueError("LTT probabilities must be in (0, 1)")
    return config
