"""CodeStop-specific scoring helpers."""

import math
from itertools import pairwise
from typing import Sequence

from replay_eval.utils.deer import DeerProbeScore
from replay_eval.utils.deer import probability_score


def compute_ramping_threshold(
    step_index: int,
    ramp_steps: int | None,
    ramp_min: float | None,
    ramp_max: float,
) -> float:
    if ramp_steps is None:
        return float(ramp_max)
    alpha = min(1.0, step_index / ramp_steps) if ramp_steps > 0 else 1.0
    minimum = float(ramp_min if ramp_min is not None else ramp_max)
    return minimum + (float(ramp_max) - minimum) * alpha


def compute_degeneration_score(
    positions: Sequence[int],
    probabilities: Sequence[float],
    *,
    use_log: bool = True,
) -> float:
    if len(positions) < 3 or len(probabilities) < 3:
        return 0.0
    tk = positions[-1]
    probs = [math.log(max(p, 1e-12)) for p in probabilities] if use_log else list(probabilities)

    def value(previous: float, current: float) -> int:
        return 1 if 2 * current - previous < 0.55 else 0

    def weight(position: int) -> float:
        if position <= 0:
            return 1.0
        return math.log1p(tk / position - 1.0) + 1.0

    total = 0.0
    for (_, previous), (position, current) in pairwise(zip(positions, probs)):
        if previous > current:
            total += value(previous, current) * weight(position)
    return total


CodeStopProbeScore = DeerProbeScore
