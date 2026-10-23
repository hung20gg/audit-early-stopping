"""Adapt dense replay embeddings into checkpoint-level trajectories."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from terminator.dataset import ReplayEmbeddingDataset

LOGGER = logging.getLogger(__name__)


@dataclass
class DatasetCounters:
    input_trajectories: int = 0
    kept_trajectories: int = 0
    never_correct_trajectories: int = 0
    empty_trajectories: int = 0
    zero_token_boundaries: int = 0
    filtered_records: int = 0
    missing_records: int = 0

    def as_dict(self) -> dict[str, int]:
        return dict(vars(self))

    def add(self, other: "DatasetCounters") -> None:
        for name in vars(self):
            setattr(self, name, getattr(self, name) + getattr(other, name))


@dataclass
class ThoughtTrajectory:
    """A complete trajectory of pooled features and cumulative labels."""

    problem_id: str
    features: np.ndarray
    labels: np.ndarray
    token_boundaries: np.ndarray
    checkpoint_positions: np.ndarray
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.features = np.asarray(self.features)
        self.labels = np.asarray(self.labels, dtype=np.int64)
        self.token_boundaries = np.asarray(self.token_boundaries, dtype=np.int64)
        self.checkpoint_positions = np.asarray(self.checkpoint_positions, dtype=np.int64)
        length = len(self.labels)
        if self.features.ndim != 2 or not all(
            len(value) == length
            for value in (self.features, self.token_boundaries, self.checkpoint_positions)
        ):
            raise ValueError("Trajectory arrays must have one row per checkpoint")


def _token_boundary(checkpoint_map: dict[Any, Any], position: int) -> int | None:
    value = checkpoint_map.get(str(position), checkpoint_map.get(position))
    return None if value is None else int(value)


def pool_replay_item(
    item: dict[str, Any],
    *,
    source: str = "",
    record_method: str | None = "curation_grid",
    duplicate_label_policy: str = "require_agreement",
    empty_segment_policy: str = "skip",
    label_policy: str = "cumulative_first_correct",
    pooling: str = "segment_mean",
    dtype: str = "auto",
) -> tuple[ThoughtTrajectory | None, DatasetCounters]:
    """Pool one item without modifying its source tensor."""
    counters = DatasetCounters(input_trajectories=1)
    if duplicate_label_policy != "require_agreement":
        raise ValueError(f"Unsupported duplicate_label_policy: {duplicate_label_policy}")
    if empty_segment_policy not in {"skip", "error"}:
        raise ValueError(f"Unsupported empty_segment_policy: {empty_segment_policy}")
    if label_policy != "cumulative_first_correct":
        raise ValueError(f"Unsupported label_policy: {label_policy}")
    if pooling != "segment_mean":
        raise ValueError(f"Unsupported pooling: {pooling}")

    states = item["hidden_states"]
    # np.asarray on torch tensors can alias memory; all operations below are read-only.
    states_np = states.detach().cpu().numpy() if hasattr(states, "detach") else np.asarray(states)
    if states_np.ndim != 2:
        raise ValueError("hidden_states must have shape [tokens, hidden_size]")
    checkpoint_map = item.get("checkpoint_token_indices") or {}
    records_by_position = item.get("records_by_position") or {}
    candidates: list[tuple[int, int, int]] = []
    for raw_position, records in records_by_position.items():
        position = int(raw_position)
        boundary = _token_boundary(checkpoint_map, position)
        if boundary is None:
            counters.missing_records += 1
            continue
        selected = [
            record for record in records
            if isinstance(record, dict)
            and (record_method is None or record.get("method") == record_method)
        ]
        counters.filtered_records += len(records) - len(selected)
        if not selected:
            counters.missing_records += 1
            continue
        if any("correct" not in record for record in selected):
            raise ValueError(f"Checkpoint {position} has a record without 'correct'")
        labels = {bool(record["correct"]) for record in selected}
        if len(labels) != 1:
            raise ValueError(f"Duplicate labels disagree at checkpoint {position}")
        if boundary < 0 or boundary > len(states_np):
            raise ValueError(
                f"Checkpoint {position} maps to invalid token boundary {boundary} "
                f"for {len(states_np)} states"
            )
        candidates.append((boundary, position, int(labels.pop())))

    # Generation order is token boundary first; character position breaks ties.
    candidates.sort(key=lambda value: (value[0], value[1]))
    pooled, observed, boundaries, positions = [], [], [], []
    start = 0
    for end, position, label in candidates:
        if end <= start:
            counters.zero_token_boundaries += 1
            if empty_segment_policy == "error":
                raise ValueError(f"Zero-token segment at checkpoint {position}")
            continue
        pooled.append(np.asarray(states_np[start:end], dtype=np.float64).mean(axis=0))
        observed.append(label)
        boundaries.append(end)
        positions.append(position)
        start = end
    if not pooled:
        counters.empty_trajectories = 1
        return None, counters
    if not any(observed):
        counters.never_correct_trajectories = 1
        return None, counters
    first_correct = observed.index(1)
    labels = np.zeros(len(observed), dtype=np.int64)
    labels[first_correct:] = 1
    problem = item.get("problem_id", item.get("line_number"))
    qualified_id = f"{source}:{problem}" if source else str(problem)
    feature_dtype = states_np.dtype if dtype == "auto" else np.dtype(dtype)
    trajectory = ThoughtTrajectory(
        problem_id=qualified_id,
        features=np.asarray(pooled, dtype=feature_dtype),
        labels=labels,
        token_boundaries=np.asarray(boundaries),
        checkpoint_positions=np.asarray(positions),
        source=source,
        metadata={
            "embedding_model": item.get("embedding_model"),
            "source_model": item.get("source_model"),
            "line_number": item.get("line_number"),
            "split_problem_id": str(problem),
        },
    )
    counters.kept_trajectories = 1
    return trajectory, counters


def load_trajectories(
    datasets: Iterable[dict[str, Any]],
    *,
    batch_size: int = 1,
    workers: int = 0,
    cache: bool = True,
    **adapter_options: Any,
) -> tuple[list[ThoughtTrajectory], DatasetCounters, dict[str, Any]]:
    """Load configured ReplayEmbeddingDatasets and validate compatibility."""
    if batch_size < 1 or workers < 0:
        raise ValueError("Loading batch_size must be >= 1 and workers must be >= 0")
    trajectories: list[ThoughtTrajectory] = []
    total = DatasetCounters()
    hidden_sizes: set[int] = set()
    embedding_models: set[str] = set()
    source_models: set[str] = set()
    for dataset_index, spec in enumerate(datasets):
        name = f"dataset_{dataset_index:03d}"
        dataset = ReplayEmbeddingDataset(Path(spec["shards"]), Path(spec["curation_jsonl"]))
        if workers:
            from torch.utils.data import DataLoader

            batches = DataLoader(
                dataset, batch_size=batch_size, num_workers=workers,
                collate_fn=list, persistent_workers=bool(cache),
            )
        else:
            batches = (
                [dataset[index] for index in range(start, min(start + batch_size, len(dataset)))]
                for start in range(0, len(dataset), batch_size)
            )
        for batch in batches:
            for item in batch:
                trajectory, counters = pool_replay_item(item, source=name, **adapter_options)
                total.add(counters)
                if trajectory is not None:
                    trajectories.append(trajectory)
                    hidden_sizes.add(trajectory.features.shape[1])
                    model = trajectory.metadata.get("embedding_model")
                    if model:
                        embedding_models.add(str(model))
                    source_model = trajectory.metadata.get("source_model")
                    if source_model:
                        source_models.add(str(source_model))
            if not cache and workers == 0:
                dataset._shard_cache.clear()
                dataset._metadata_cache.clear()
    if len(hidden_sizes) > 1:
        raise ValueError(f"Incompatible hidden sizes across datasets: {sorted(hidden_sizes)}")
    if len(embedding_models) > 1:
        raise ValueError(f"Incompatible embedding models across datasets: {sorted(embedding_models)}")
    if len(source_models) > 1:
        raise ValueError(f"Incompatible source models across datasets: {sorted(source_models)}")
    if not trajectories:
        raise ValueError("No usable trajectories remain after label filtering")
    metadata = {
        "hidden_size": next(iter(hidden_sizes)),
        "embedding_model": next(iter(embedding_models), None),
        "source_model": next(iter(source_models), None),
        "counters": total.as_dict(),
    }
    LOGGER.info("Loaded trajectories: %s", metadata)
    return trajectories, total, metadata


def split_trajectories(
    trajectories: Iterable[ThoughtTrajectory],
    *,
    train_fraction: float = 0.50,
    calibration_fraction: float = 0.45,
    test_fraction: float = 0.05,
    seed: int = 0,
) -> dict[str, list[ThoughtTrajectory]]:
    """Assign whole problems by a stable seeded hash (independent of input order)."""
    if min(train_fraction, calibration_fraction, test_fraction) <= 0:
        raise ValueError("Split fractions must be positive")
    if not np.isclose(train_fraction + calibration_fraction + test_fraction, 1.0):
        raise ValueError("Split fractions must sum to 1")
    splits = {"train": [], "calibration": [], "test": []}
    train_edge = train_fraction
    calibration_edge = train_fraction + calibration_fraction
    assignments: dict[str, str] = {}
    for trajectory in trajectories:
        split_problem_id = str(trajectory.metadata.get("split_problem_id", trajectory.problem_id))
        digest = hashlib.sha256(f"{seed}:{split_problem_id}".encode()).digest()
        value = int.from_bytes(digest[:8], "big") / 2**64
        split = "train" if value < train_edge else "calibration" if value < calibration_edge else "test"
        previous = assignments.setdefault(split_problem_id, split)
        if previous != split:  # Defensive: should be impossible for stable IDs.
            raise RuntimeError(f"Problem leaked across splits: {split_problem_id}")
        splits[split].append(trajectory)
    return splits
