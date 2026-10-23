"""Pairwise semantic agreement for independently sampled random exits.

Answer classes are formed only when the benchmark comparator is an equivalence
relation on the observed samples. Nontransitive comparisons remain pairwise
statistics and are explicitly flagged instead of being silently clustered.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any, Iterable

from evaluator.math500 import extract_final_answer, math_equal
from evaluator.truthfulqa import multiple_choice_equal, normalize_options

SUPPORTED_PAIRWISE_BENCHMARKS = frozenset({"truthfulqa", "math500"})


def _valid_answer(output: str, benchmark: str) -> bool:
    if benchmark == "truthfulqa":
        return normalize_options(output, require_box=True) is not None
    if benchmark == "math500":
        return bool(extract_final_answer(output))
    raise ValueError(f"No pairwise answer comparator for benchmark {benchmark!r}")


def same_answer(left: str, right: str, benchmark: str) -> bool:
    """Use the benchmark evaluator in both directions; never compare raw text."""
    if not (_valid_answer(left, benchmark) and _valid_answer(right, benchmark)):
        return False
    if benchmark == "truthfulqa":
        return multiple_choice_equal(left, right) and multiple_choice_equal(right, left)
    return bool(
        math_equal(left, right, include_percentage=False)
        and math_equal(right, left, include_percentage=False)
    )


def answer_consistency(rows: Iterable[dict[str, Any]], benchmark: str) -> dict[str, Any]:
    """Summarize one position, including wrong-answer convergence and diversity.

    Rows carry ``num_run``, ``output`` and boolean ``correct``. Invalid answers
    are excluded from pair comparisons and make all-sample consistency false.
    """
    if benchmark not in SUPPORTED_PAIRWISE_BENCHMARKS:
        raise ValueError(f"No pairwise answer comparator for benchmark {benchmark!r}")
    samples = sorted(rows, key=lambda row: int(row["num_run"]))
    slots = [int(row["num_run"]) for row in samples]
    if len(slots) != len(set(slots)):
        raise ValueError("Repeated sample slot at one exit position")
    valid = [index for index, row in enumerate(samples) if _valid_answer(str(row.get("output") or ""), benchmark)]
    pairs: dict[tuple[int, int], bool] = {}
    for a, b in combinations(valid, 2):
        pairs[a, b] = same_answer(str(samples[a]["output"]), str(samples[b]["output"]), benchmark)
    pair_values = list(pairs.values())
    nontransitive = any(
        sum((pairs[a, b], pairs[a, c], pairs[b, c])) == 2
        for a, b, c in combinations(valid, 3)
    )
    # For a valid equivalence relation, count classes through earlier members.
    representatives: list[int] = []
    if not nontransitive:
        for index in valid:
            if not any(pairs.get((min(index, rep), max(index, rep)), False) for rep in representatives):
                representatives.append(index)
    wrong = [index for index in valid if samples[index].get("correct") is False]
    wrong_pairs = [pairs[min(a, b), max(a, b)] for a, b in combinations(wrong, 2)]
    wrong_representatives: list[int] = []
    if not nontransitive:
        for index in wrong:
            if not any(pairs.get((min(index, rep), max(index, rep)), False) for rep in wrong_representatives):
                wrong_representatives.append(index)
    all_valid = len(valid) == len(samples) and bool(samples)
    unique = len(representatives) if all_valid and not nontransitive else None
    wrong_count = sum(row.get("correct") is False for row in samples)
    wrong_complete = len(wrong) == wrong_count
    wrong_unique = len(wrong_representatives) if wrong_complete and not nontransitive else None
    return {
        "sample_count": len(samples),
        "valid_answer_count": len(valid),
        "invalid_answer_count": len(samples) - len(valid),
        "unique_answer_count": unique,
        "pairwise_agreement": sum(pair_values) / len(pair_values) if pair_values else None,
        "all_answers_consistent": bool(all_valid and not nontransitive and unique == 1),
        "wrong_answer_count": wrong_count,
        "valid_wrong_answer_count": len(wrong),
        "wrong_unique_answer_count": wrong_unique,
        "wrong_pairwise_agreement": sum(wrong_pairs) / len(wrong_pairs) if wrong_pairs else None,
        "wrong_answers_converged": wrong_count >= 2 and wrong_unique == 1,
        "wrong_answers_diverse": wrong_count >= 2 and wrong_unique is not None and wrong_unique > 1,
        "nontransitive_comparator": nontransitive,
    }


def all_high_score_test(
    answers: Iterable[dict[str, Any]], scores: Iterable[dict[str, Any]],
    *, threshold: float, higher_is_better: bool = True,
) -> dict[str, Any]:
    """Join exactly one score per answer slot; test unanimity without gold leakage."""
    answers = list(answers)
    scores = list(scores)
    key = lambda row: (str(row["benchmark"]), str(row["model"]), str(row["data_id"]), int(row["position"]), int(row["num_run"]))
    score_by_key = {key(row): row for row in scores}
    if len(score_by_key) != len(scores):
        raise ValueError("Duplicate score for a sampled request")
    missing = [key(row) for row in answers if key(row) not in score_by_key]
    if missing:
        return {"supported": False, "reason": "missing_per_sample_scores", "missing_slots": len(missing)}
    for row in answers:
        score = score_by_key[key(row)]
        if row.get("key") != score.get("request_key"):
            raise ValueError(f"Score request key differs from saved exit: {key(row)}")
    high = [float(score_by_key[key(row)]["score"]) > threshold if higher_is_better else float(score_by_key[key(row)]["score"]) < threshold for row in answers]
    return {"supported": True, "all_scores_high": all(high) and bool(high), "high_score_count": sum(high), "sample_count": len(high)}
