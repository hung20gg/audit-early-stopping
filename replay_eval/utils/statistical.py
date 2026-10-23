"""Statistical uncertainty-density early-exit utilities."""

from __future__ import annotations

import re
from dataclasses import dataclass
from math import ceil
from typing import Sequence


DEFAULT_UNCERTAINTY_KEYWORDS: tuple[tuple[str, ...], ...] = (
    ("not", "possible"),
    ("impossible",),
    ("cannot",),
    ("hard",),
    ("guess",),
    ("assum",),
    ("forgot",),
    ("intend",),
    ("missing",),
    ("insufficient",),
    ("incomplete",),
    ("without",),
    ("additional",),
    ("lack",),
    ("no", "data"),
    ("no", "info"),
    ("not", "helpful"),
    ("not", "give"),
    ("not", "specif"),
    ("not", "recall"),
    ("not", "provide"),
    ("not", "include"),
    ("not", "enough"),
    ("not", "access"),
    ("absence",),
    ("ambiguous",),
    ("vague",),
    ("maybe",),
    ("perhaps",),
    ("alternatively",),
    ("wait",),
)


WORD_PATTERN = re.compile(r"[a-z]+")


@dataclass(frozen=True)
class StatisticalUncertaintyScore:
    density: float
    count: int
    length: int
    arrivals: tuple[int, ...]


def tokenize_uncertainty_text(text: str) -> list[str]:
    return WORD_PATTERN.findall(text.lower())


def normalize_uncertainty_keywords(
    keywords: Sequence[str | Sequence[str]] | None = None,
) -> tuple[tuple[str, ...], ...]:
    if keywords is None:
        return DEFAULT_UNCERTAINTY_KEYWORDS
    normalized: list[tuple[str, ...]] = []
    for keyword in keywords:
        if isinstance(keyword, str):
            terms = tuple(tokenize_uncertainty_text(keyword))
        else:
            terms = tuple(str(term).lower().strip() for term in keyword if str(term).strip())
        if terms:
            normalized.append(terms)
    return tuple(normalized)


def detect_uncertainty_arrivals(
    tokens: Sequence[str],
    uncertainty_keywords: Sequence[Sequence[str]],
    delta: int = 5,
) -> list[int]:
    arrivals: list[int] = []
    i = 0
    while i < len(tokens):
        best_match_len = 0
        for keyword in uncertainty_keywords:
            if _matches_keyword(tokens, i, keyword):
                best_match_len = max(best_match_len, len(keyword))
        if best_match_len > 0:
            arrivals.append(i)
            i += max(best_match_len, int(delta))
        else:
            i += 1
    return arrivals


def uncertainty_density_score(
    text: str,
    uncertainty_keywords: Sequence[Sequence[str]] | None = None,
    *,
    length: int | None = None,
    delta: int = 5,
) -> StatisticalUncertaintyScore:
    tokens = tokenize_uncertainty_text(text)
    keywords = normalize_uncertainty_keywords(uncertainty_keywords)
    arrivals = detect_uncertainty_arrivals(tokens, keywords, delta=delta)
    denominator = max(1, int(length if length is not None else len(tokens)))
    return StatisticalUncertaintyScore(
        density=len(arrivals) / denominator,
        count=len(arrivals),
        length=denominator,
        arrivals=tuple(arrivals),
    )


def calibrate_maxwise_conformal(
    calibration_traces: Sequence[Sequence[str]],
    uncertainty_keywords: Sequence[Sequence[str]] | None = None,
    *,
    B: int = 100,
    alpha: float = 0.05,
    delta: int = 5,
) -> float:
    keywords = normalize_uncertainty_keywords(uncertainty_keywords)
    max_scores: list[float] = []
    for tokens in calibration_traces:
        arrivals = detect_uncertainty_arrivals(tokens, keywords, delta=delta)
        trace_scores = [
            number_of_arrivals_before(arrivals, L) / L
            for L in range(int(B), len(tokens) + 1, int(B))
        ]
        max_scores.append(max(trace_scores) if trace_scores else 0.0)
    if not max_scores:
        return float(alpha)
    n = len(max_scores)
    q = min(1.0, ceil((n + 1) * (1 - alpha)) / n)
    return _quantile(max_scores, q)


def number_of_arrivals_before(arrivals: Sequence[int], L: int) -> int:
    return sum(1 for pos in arrivals if pos < L)


def _matches_keyword(tokens: Sequence[str], index: int, keyword: Sequence[str]) -> bool:
    if index + len(keyword) > len(tokens):
        return False
    for offset, term in enumerate(keyword):
        token = tokens[index + offset]
        if not token.startswith(term):
            return False
    return True


def _quantile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    # statistics.quantiles excludes endpoints; nearest-rank is simpler and
    # matches the conformal upper quantile intent here.
    index = min(len(ordered) - 1, max(0, ceil(q * len(ordered)) - 1))
    return ordered[index]
