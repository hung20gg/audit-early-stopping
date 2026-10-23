"""Trajectory smoothing, stopping simulation, and fixed-sequence LTT."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
from scipy.stats import binom


def moving_average(scores: Iterable[float], window: int = 10) -> np.ndarray:
    values = np.asarray(list(scores), dtype=np.float64)
    if window < 1:
        raise ValueError("Smoothing window must be >= 1")
    if not len(values):
        return values
    cumulative = np.cumsum(np.insert(values, 0, 0.0))
    result = np.empty_like(values)
    for index in range(len(values)):
        start = max(0, index - window + 1)
        result[index] = (cumulative[index + 1] - cumulative[start]) / (index - start + 1)
    return result


def select_stop(scores: Iterable[float], threshold: float, comparison: str = ">=") -> tuple[int, bool]:
    values = np.asarray(list(scores), dtype=np.float64)
    if not len(values):
        raise ValueError("Cannot stop on an empty trajectory")
    if comparison == ">=":
        matches = np.flatnonzero(values >= threshold)
    elif comparison == ">":
        matches = np.flatnonzero(values > threshold)
    else:
        raise ValueError(f"Unsupported threshold comparison: {comparison}")
    return (int(matches[0]), True) if len(matches) else (len(values) - 1, False)


def calibrate_ltt(
    traces: Iterable[dict[str, Any]],
    *,
    risk_tolerances: Iterable[float],
    failure_probability: float,
    thresholds: Iterable[float],
    comparison: str = ">=",
) -> dict[str, Any]:
    """Calibrate binary stop loss with a binomial-CDF fixed sequence test."""
    traces = list(traces)
    if not traces:
        raise ValueError("Calibration split has no trajectories")
    grid = sorted({float(value) for value in thresholds}, reverse=True)
    if not grid:
        raise ValueError("Threshold grid is empty")
    if not 0 < failure_probability < 1:
        raise ValueError("failure_probability must be in (0, 1)")
    output: dict[str, Any] = {}
    for tolerance in risk_tolerances:
        tolerance = float(tolerance)
        if not 0 < tolerance < 1:
            raise ValueError("Risk tolerances must be in (0, 1)")
        candidates = []
        selected: float | None = None
        sequence_active = True
        for threshold in grid:
            losses = []
            early = []
            for trace in traces:
                index, stopped_early = select_stop(trace["smoothed_scores"], threshold, comparison)
                losses.append(1 - int(trace["labels"][index]))
                early.append(stopped_early)
            errors = int(sum(losses))
            p_value = float(binom.cdf(errors, len(losses), tolerance))
            rejected = p_value <= failure_probability
            accepted_in_sequence = bool(sequence_active and rejected)
            candidates.append(
                {
                    "threshold": threshold,
                    "p_value": p_value,
                    "empirical_risk": errors / len(losses),
                    "errors": errors,
                    "trajectories": len(losses),
                    "early_stop_rate": float(np.mean(early)),
                    "null_rejected": bool(rejected),
                    "accepted_in_fixed_sequence": accepted_in_sequence,
                }
            )
            if accepted_in_sequence:
                selected = threshold
            else:
                sequence_active = False
        output[str(tolerance)] = {
            "risk_tolerance": tolerance,
            "selected_threshold": selected,
            "fallback": selected is None,
            "fallback_rule": "full_budget",
            "candidates": candidates,
        }
    return output
