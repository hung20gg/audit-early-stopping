"""DEER scoring and temporary-answer probe helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

from methods.response import Response


@dataclass(frozen=True)
class DeerProbeScore:
    total_prob_max: float
    token_probs: list[float]
    ended_with_think: bool


@dataclass(frozen=True)
class TopLogprobEntropy:
    entropy: float
    tail_mass: float
    covered_mass: float


def best_token_logprobs(response: Response) -> list[float]:
    token_logprobs: list[float] = []
    for item in response.logprobs:
        candidates = [
            top.logprob for top in item.top_logprobs if top.logprob is not None
        ]
        if item.logprob is not None:
            candidates.append(item.logprob)
        if candidates:
            token_logprobs.append(max(candidates))
    return token_logprobs


def mean_token_probability(response: Response) -> float:
    token_logprobs = best_token_logprobs(response)
    if not token_logprobs:
        return 0.0
    return math.exp(sum(token_logprobs) / len(token_logprobs))


def residual_bucket_entropy(logprobs: Sequence[float]) -> TopLogprobEntropy:
    """Approximate entropy from returned top logprobs plus one residual bucket.

    The returned logprobs are treated as original probabilities, not normalized
    over the returned set. Entropy is measured in nats.
    """
    probabilities = [math.exp(logprob) for logprob in logprobs]
    covered_mass = min(1.0, max(0.0, sum(probabilities)))
    tail_mass = max(0.0, 1.0 - covered_mass)
    entropy = -sum(
        probability * math.log(probability)
        for probability in probabilities
        if probability > 0.0
    )
    if tail_mass > 0.0:
        entropy -= tail_mass * math.log(tail_mass)
    return TopLogprobEntropy(
        entropy=entropy,
        tail_mass=tail_mass,
        covered_mass=covered_mass,
    )


def probability_score(
    token_logprobs: Sequence[float],
    token_texts: Sequence[str],
    *,
    think_end_token: str,
    ewt: bool = True,
) -> DeerProbeScore:
    ended_with_think = any(
        think_end_token and think_end_token in token for token in token_texts
    )
    probs = [math.exp(logprob) for logprob in token_logprobs]
    if ended_with_think:
        end_index = next(
            index
            for index, token in enumerate(token_texts)
            if think_end_token and think_end_token in token
        )
        probs = probs[:end_index]
    if not probs:
        return DeerProbeScore(
            total_prob_max=0.0,
            token_probs=[],
            ended_with_think=ended_with_think,
        )
    if ewt:
        total_prob_max = math.exp(
            sum(math.log(max(prob, 1e-12)) for prob in probs) / len(probs)
        )
    else:
        total_prob_max = sum(probs) / len(probs)
    return DeerProbeScore(
        total_prob_max=total_prob_max,
        token_probs=list(probs),
        ended_with_think=ended_with_think,
    )


def answer_markers(*, is_code: bool, think_end_token: str) -> tuple[str, ...]:
    if is_code:
        return (think_end_token,)
    return (
        " }",
        "}",
        "}.",
        "}.\n",
        "}\\",
        "}}",
        ")}",
        ")}.",
        ")}\n",
        think_end_token,
    )


def first_answer_marker_char_index(content: str, markers: Sequence[str]) -> int:
    indices = []
    for marker in markers:
        if not marker:
            continue
        index = content.find(marker)
        if index != -1:
            indices.append(index)
    return min(indices) if indices else len(content)


def first_answer_marker_token_index(
    token_texts: Sequence[str],
    markers: Sequence[str],
) -> int:
    marker_char_index = first_answer_marker_char_index("".join(token_texts), markers)
    cursor = 0
    for index, token in enumerate(token_texts):
        if cursor >= marker_char_index:
            return index
        cursor += len(token)
        if cursor > marker_char_index:
            return index + 1
    return len(token_texts)


def slice_temp_answer_probe(
    response: Response,
    *,
    probe_tokens: int,
    token_counter: Callable[[str], int | None],
    is_code: bool,
    think_end_token: str,
) -> Response:
    markers = answer_markers(is_code=is_code, think_end_token=think_end_token)
    token_texts = response.token_texts()
    if token_texts:
        marker_token_index = first_answer_marker_token_index(token_texts, markers)
        token_end = min(marker_token_index, probe_tokens, len(token_texts))
        return response.token_slice(0, token_end)

    marker_char_index = first_answer_marker_char_index(response.content, markers)
    content = response.content[:marker_char_index]
    return Response(
        input=response.input,
        content=content,
        finish_reason=response.finish_reason,
        completion_tokens=min(
            probe_tokens,
            token_counter(content) or probe_tokens,
        ),
    )
