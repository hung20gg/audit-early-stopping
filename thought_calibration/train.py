"""Train, calibrate, and evaluate a Thought Calibration linear probe."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from sklearn.metrics import accuracy_score, roc_auc_score

from .calibration import calibrate_ltt, moving_average, select_stop
from .config import load_config
from .dataset import ThoughtTrajectory, load_trajectories, split_trajectories
from .model import ThoughtCalibrationModel

LOGGER = logging.getLogger(__name__)


def _load_base_model_config(base_model: str) -> tuple[Any, int]:
    """Load only the Hugging Face config and resolve its text hidden size."""
    from transformers import AutoConfig

    model_config = AutoConfig.from_pretrained(base_model)
    text_config = getattr(model_config, "text_config", None)
    hidden_size = getattr(text_config, "hidden_size", None)
    if hidden_size is None:
        hidden_size = getattr(model_config, "hidden_size", None)
    if not isinstance(hidden_size, int) or hidden_size < 1:
        raise ValueError(f"Could not resolve hidden_size from base model config: {base_model}")
    return model_config, hidden_size


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _stack(trajectories: list[ThoughtTrajectory]) -> tuple[np.ndarray, np.ndarray]:
    if not trajectories:
        raise ValueError("A required data split is empty")
    return np.concatenate([item.features for item in trajectories]), np.concatenate([item.labels for item in trajectories])


def _score_traces(model: ThoughtCalibrationModel, trajectories: list[ThoughtTrajectory], window: int) -> list[dict[str, Any]]:
    output = []
    for trajectory in trajectories:
        scores = model.predict_proba(trajectory.features)
        output.append(
            {
                "problem_id": trajectory.problem_id,
                "scores": scores,
                "smoothed_scores": moving_average(scores, window),
                "labels": trajectory.labels,
                "token_boundaries": trajectory.token_boundaries,
            }
        )
    return output


def _classification_metrics(traces: list[dict[str, Any]]) -> dict[str, float | None]:
    labels = np.concatenate([trace["labels"] for trace in traces])
    scores = np.concatenate([trace["scores"] for trace in traces])
    auroc = float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else None
    return {"auroc": auroc, "accuracy": float(accuracy_score(labels, scores >= 0.5)), "checkpoints": int(len(labels))}


def _stopping_metrics(traces: list[dict[str, Any]], threshold: float | None, comparison: str) -> dict[str, Any]:
    correct, tokens, savings, early, fallback = [], [], [], [], []
    for trace in traces:
        if threshold is None:
            index, did_stop = len(trace["labels"]) - 1, False
            did_fallback = True
        else:
            index, did_stop = select_stop(trace["smoothed_scores"], threshold, comparison)
            did_fallback = not did_stop
        stop_tokens = int(trace["token_boundaries"][index])
        final_tokens = int(trace["token_boundaries"][-1])
        correct.append(int(trace["labels"][index]))
        tokens.append(stop_tokens)
        savings.append(final_tokens - stop_tokens)
        early.append(did_stop and index < len(trace["labels"]) - 1)
        fallback.append(did_fallback)
    accuracy = float(np.mean(correct))
    return {
        "accuracy": accuracy,
        "error_rate": 1.0 - accuracy,
        "mean_stop_tokens": float(np.mean(tokens)),
        "mean_token_savings": float(np.mean(savings)),
        "early_stop_rate": float(np.mean(early)),
        "fallback_rate": float(np.mean(fallback)),
        "trajectories": len(traces),
    }


def _prepare_output(path: Path, policy: str) -> bool:
    artifacts = [
        "model.joblib", "config.json", "thresholds.json", "splits.json",
        "metrics.json", "config.resolved.yml",
    ]
    if policy == "resume" and all((path / name).exists() for name in artifacts):
        LOGGER.info("All artifacts already exist; resume completed without retraining")
        return False
    if path.exists() and any(path.iterdir()):
        if policy == "error":
            raise FileExistsError(f"Output directory is not empty: {path}")
        if policy == "overwrite":
            # Replace only artifacts owned by this command; preserve unrelated files.
            for name in artifacts:
                artifact = path / name
                if artifact.is_file() or artifact.is_symlink():
                    artifact.unlink()
    path.mkdir(parents=True, exist_ok=True)
    return True


def run(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    logging.basicConfig(level=getattr(logging, config["logging"]["level"].upper()))
    output_dir = Path(config["output_dir"])
    if not _prepare_output(output_dir, config["output_policy"]):
        return json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))

    _base_config, base_hidden_size = _load_base_model_config(config["base_model"])
    LOGGER.info(
        "Loaded base model config only: model=%s hidden_size=%d",
        config["base_model"], base_hidden_size,
    )

    feature = config["features"]
    trajectories, counters, data_metadata = load_trajectories(
        config["datasets"],
        batch_size=config["loading"]["batch_size"],
        workers=config["loading"]["workers"],
        cache=config["loading"]["cache"],
        record_method=feature["record_method"],
        duplicate_label_policy=feature["duplicate_label_policy"],
        empty_segment_policy=feature["empty_segment_policy"],
        label_policy=feature["label_policy"],
        pooling=feature["pooling"],
        dtype=config["numeric"]["dtype"],
    )
    if data_metadata["hidden_size"] != base_hidden_size:
        raise ValueError(
            f"Embedding hidden size {data_metadata['hidden_size']} does not match "
            f"base model {config['base_model']!r} hidden size {base_hidden_size}"
        )
    split_cfg = config["split"]
    splits = split_trajectories(trajectories, **split_cfg)
    for name, values in splits.items():
        if not values:
            raise ValueError(f"Deterministic split {name!r} is empty; provide more problems or change split.seed")
    split_artifact = {
        "seed": split_cfg["seed"],
        "fractions": {key: split_cfg[f"{key}_fraction"] for key in ("train", "calibration", "test")},
        "problems": {name: [item.problem_id for item in values] for name, values in splits.items()},
        "dataset_counters": counters.as_dict(),
    }
    _write_json(output_dir / "splits.json", split_artifact)

    train_features, train_labels = _stack(splits["train"])
    if len(np.unique(train_labels)) != 2:
        raise ValueError("Training split does not contain both label classes")
    metadata = {
        "base_model": config["base_model"],
        "feature_source": feature["source"], "hidden_size": data_metadata["hidden_size"],
        "pooling": feature["pooling"], "preprocessing": "standard_scaler_then_pca",
        "embedding_model": data_metadata["embedding_model"],
        "source_model": data_metadata["source_model"],
    }
    model = ThoughtCalibrationModel(
        metadata=metadata,
        scaler_options=config["standard_scaler"], pca_options=config["pca"],
        probe_options=config["logistic_regression"],
    ).fit(train_features, train_labels)
    model.save(output_dir / "model.joblib")

    window = config["smoothing"]["window"]
    calibration_traces = _score_traces(model, splits["calibration"], window)
    grid_cfg = config["ltt"]["threshold_grid"]
    grid = np.linspace(grid_cfg["maximum"], grid_cfg["minimum"], grid_cfg["count"])
    thresholds = calibrate_ltt(
        calibration_traces, risk_tolerances=config["ltt"]["risk_tolerances"],
        failure_probability=config["ltt"]["failure_probability"], thresholds=grid,
        comparison=config["ltt"]["comparison"],
    )
    threshold_artifact = {
        "smoothing_window": window, "comparison": config["ltt"]["comparison"],
        "loss": config["ltt"]["loss"], "failure_probability": config["ltt"]["failure_probability"],
        "by_risk_tolerance": thresholds,
    }
    _write_json(output_dir / "thresholds.json", threshold_artifact)
    model_config = {
        "architectures": ["ThoughtCalibrationModule"],
        "model_type": "thought_calibration",
        "model_version": ThoughtCalibrationModel.MODEL_VERSION,
        "base_model": config["base_model"],
        "hidden_size": base_hidden_size,
        "feature_source": feature["source"],
        "pooling": feature["pooling"],
        "preprocessing": "standard_scaler_then_pca",
        "pca_components": int(model.pca.n_components_),
        "smoothing_window": window,
        "threshold_comparison": config["ltt"]["comparison"],
        "thresholds": {
            risk: result["selected_threshold"] for risk, result in thresholds.items()
        },
        "fallback_rule": config["ltt"]["fallback"],
        "model_file": "model.joblib",
    }
    _write_json(output_dir / "config.json", model_config)

    test_traces = _score_traces(model, splits["test"], window)
    metrics = {
        "classification": _classification_metrics(test_traces),
        "stopping": {
            key: _stopping_metrics(test_traces, result["selected_threshold"], config["ltt"]["comparison"])
            for key, result in thresholds.items()
        },
    }
    _write_json(output_dir / "metrics.json", metrics)
    (output_dir / "config.resolved.yml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = run(args.config)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
