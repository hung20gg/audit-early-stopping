from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from methods.response import Response
from evaluator.math500 import extract_final_answer

DEFAULT_ANNOTATOR_MODEL = "deepinfra:Qwen/Qwen3-32B"
MAX_IDENTIFY_RETRIES = 3


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """Best-effort extraction of the first balanced {...} JSON object in text.

    Annotation-LLM output is not guaranteed to be pure JSON (it may wrap the
    object in prose or a fenced code block), so this scans for the first
    balanced brace group instead of assuming `json.loads(text)` works.
    """
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for index in range(start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : index + 1]
                try:
                    parsed = json.loads(candidate)
                except json.JSONDecodeError:
                    return None
                return parsed if isinstance(parsed, dict) else None
    return None


def call_annotator(llm, prompt: str, **kwargs) -> str:
    raw = llm.invoke([{"role": "user", "content": prompt}], **kwargs)
    return Response.from_completion(raw).content


def extract_eventual_answer(llm, final_solution: str) -> dict[str, Any] | None:
    """Stage A: extract the model's own eventual answer from its final solution.

    This is deliberately *not* the ground-truth answer -- TERMINATOR learns to
    reproduce when the base model commits to its answer, not whether that
    answer is correct.
    """
    prefill_guess = extract_final_answer(final_solution)
    prompt = (
        "A reasoning model produced the following final solution. Extract the "
        "single answer value it committed to -- not the ground truth, just "
        "what this text asserts.\n\n"
        f"### Final solution\n{final_solution}\n\n"
        "Respond with only a JSON object:\n"
        '{"answer": "<value>", "answer_type": "number|fraction|choice|expression", '
        '"confidence": <0-1 float>}\n\n'
        f"A regex pre-extraction guessed: {prefill_guess!r}. Use it only if correct."
    )
    content = call_annotator(llm, prompt, temperature=0.0, max_completion_token=200)
    parsed = _extract_json_object(content)
    if not parsed or "answer" not in parsed:
        return None
    return parsed


def identify_span(
    llm,
    question: str,
    reasoning_text: str,
    eventual_answer: str,
    rejected_spans: list[str] | None = None,
) -> dict[str, Any] | None:
    """Stage B: find the earliest verbatim span that logically derives the
    eventual answer (not merely the first textual occurrence of its value)."""
    feedback = ""
    if rejected_spans:
        rejected_list = "\n".join(f"- {span!r}" for span in rejected_spans)
        feedback = (
            "\nThe following previously proposed spans were rejected (either "
            "not verbatim, only a mention/guess, or not the earliest "
            f"derivation). Do not repeat them:\n{rejected_list}\n"
        )

    prompt = (
        "A reasoning model, thinking through the question below, eventually "
        f"derives the answer {eventual_answer!r}. Find the earliest text span "
        "in its reasoning where that answer has actually been logically "
        "derived -- not merely mentioned, guessed, or restated from the "
        "question.\n\n"
        f"### Question\n{question}\n\n"
        f"### Reasoning\n{reasoning_text}\n"
        f"{feedback}\n"
        "The span must:\n"
        "1. be copied verbatim from the reasoning above;\n"
        "2. include enough preceding context to show a derivation, not a guess;\n"
        "3. end immediately after that derivation.\n\n"
        "Respond with only a JSON object:\n"
        '{"span": "<verbatim substring>", "is_logical_derivation": true|false}'
    )
    content = call_annotator(llm, prompt, temperature=0.0, max_completion_token=400)
    return _extract_json_object(content)


def verify_span(
    llm,
    reasoning_text: str,
    eventual_answer: str,
    span: str,
) -> dict[str, Any] | None:
    """Stage C: independent verification of the proposed span, using a
    different prompt so it isn't just re-confirming stage B's own framing."""
    prompt = (
        "You are auditing a candidate excerpt from a reasoning trace, checking "
        "whether it is the FIRST point where the trace logically derives a "
        "given answer (as opposed to merely mentioning the value, restating "
        "the question, or an abandoned attempt).\n\n"
        f"### Full reasoning\n{reasoning_text}\n\n"
        f"### Claimed answer\n{eventual_answer}\n\n"
        f"### Candidate excerpt\n{span}\n\n"
        "Respond with only a JSON object:\n"
        '{"contains_eventual_answer": true|false, "is_first_logical_arrival": '
        'true|false, "is_only_a_guess_or_mention": true|false, "reason": "<short>"}'
    )
    content = call_annotator(llm, prompt, temperature=0.0, max_completion_token=200)
    return _extract_json_object(content)


def _span_is_acceptable(reasoning_text: str, identified: dict[str, Any] | None) -> bool:
    if not identified or not identified.get("is_logical_derivation"):
        return False
    span = identified.get("span")
    return isinstance(span, str) and bool(span) and span in reasoning_text


def _verification_passed(verified: dict[str, Any] | None) -> bool:
    if not verified:
        return False
    return bool(
        verified.get("contains_eventual_answer")
        and verified.get("is_first_logical_arrival")
        and not verified.get("is_only_a_guess_or_mention")
    )


def curate_one(llm, raw_cot: dict[str, Any], max_retries: int = MAX_IDENTIFY_RETRIES) -> dict[str, Any] | None:
    """Run stages A-C for a single raw CoT record. Returns None (drop the
    trajectory) if no valid span is found within `max_retries` attempts."""
    reasoning_text = raw_cot["reasoning_text"]
    final_solution = raw_cot["final_solution"]

    extracted = extract_eventual_answer(llm, final_solution)
    if not extracted:
        return None
    eventual_answer = str(extracted["answer"])

    rejected_spans: list[str] = []
    for attempt in range(1, max_retries + 1):
        identified = identify_span(llm, raw_cot["prompt"], reasoning_text, eventual_answer, rejected_spans)
        if not _span_is_acceptable(reasoning_text, identified):
            if identified and isinstance(identified.get("span"), str):
                rejected_spans.append(identified["span"])
            continue

        span = identified["span"]
        verified = verify_span(llm, reasoning_text, eventual_answer, span)
        if _verification_passed(verified):
            span_start = reasoning_text.find(span)
            return {
                **raw_cot,
                "eventual_answer": eventual_answer,
                "answer_type": extracted.get("answer_type"),
                "answer_span": span,
                "answer_span_char_start": span_start,
                "answer_span_char_end": span_start + len(span),
                "curation_attempts": attempt,
                "verified": True,
            }
        rejected_spans.append(span)

    return None


def curate_file(llm, input_path: Path, output_path: Path, max_retries: int = MAX_IDENTIFY_RETRIES) -> dict[str, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    accepted = 0
    rejected = 0
    with input_path.open("r", encoding="utf-8") as infile, output_path.open(
        "w", encoding="utf-8"
    ) as outfile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            raw_cot = json.loads(line)
            curated = curate_one(llm, raw_cot, max_retries=max_retries)
            if curated is None:
                rejected += 1
                continue
            accepted += 1
            outfile.write(json.dumps(curated, ensure_ascii=False) + "\n")
    return {"accepted": accepted, "rejected": rejected}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Curate hindsight-optimal exit spans.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--annotator-model", default=DEFAULT_ANNOTATOR_MODEL)
    parser.add_argument("--max-retries", type=int, default=MAX_IDENTIFY_RETRIES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from llm import get_llm_wrapper

    llm = get_llm_wrapper(args.annotator_model)
    stats = curate_file(llm, args.input, args.output, max_retries=args.max_retries)
    total = stats["accepted"] + stats["rejected"]
    rate = stats["accepted"] / total if total else 0.0
    print(
        f"Curated {stats['accepted']}/{total} ({rate:.1%} acceptance) -> {args.output}"
    )


if __name__ == "__main__":
    main()
