"""Semantic reasoning-step splitting utilities."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    text: str
    is_enumerated: bool
    first_word: str
    semantic_type: str
    start: int
    end: int


@dataclass(frozen=True)
class StepSpan:
    text: str
    start: int
    end: int


def separate_steps(
    thoughts: str,
    min_step_chars: int = 200,
    max_step_chars: int = 1000,
) -> list[str]:
    return [
        span.text.strip()
        for span in separate_step_spans(thoughts, min_step_chars, max_step_chars)
        if span.text.strip()
    ]


def separate_step_spans(
    thoughts: str,
    min_step_chars: int = 200,
    max_step_chars: int = 1000,
) -> list[StepSpan]:
    raw_chunks = _paragraph_chunks(thoughts)
    if not raw_chunks:
        return []
    return _merge_chunks_semantically(
        [_analyze_chunk(text, start, end) for text, start, end in raw_chunks],
        min_step_chars,
        max_step_chars,
    )


def _paragraph_chunks(text: str) -> list[tuple[str, int, int]]:
    chunks: list[tuple[str, int, int]] = []
    for match in re.finditer(r"\S.*?(?=(?:\n\s*\n+)|\Z)", text, re.DOTALL):
        chunk = match.group(0).strip()
        if not chunk:
            continue
        leading = len(match.group(0)) - len(match.group(0).lstrip())
        trailing = len(match.group(0)) - len(match.group(0).rstrip())
        start = match.start() + leading
        end = match.end() - trailing
        chunks.append((text[start:end], start, end))
    return chunks


def _analyze_chunk(text: str, start: int, end: int) -> Chunk:
    tokens = text.split()
    first_word = tokens[0].strip(".,!?;:").lower() if tokens else ""
    is_enum = bool(
        re.match(
            r"^\s*(\d+[\.\)]|step\s+\d+|first|second|third|finally)\b",
            text,
            re.IGNORECASE,
        )
    )
    return Chunk(
        text=text,
        is_enumerated=is_enum,
        first_word=first_word,
        semantic_type=_classify_semantic_type(text, first_word),
        start=start,
        end=end,
    )


def _classify_semantic_type(text: str, first_word: str) -> str:
    text_lower = text.lower()
    if any(
        phrase in text_lower[:100]
        for phrase in (
            "need to solve",
            "problem states",
            "given that",
            "first,",
            "the problem says",
            "let me understand",
        )
    ):
        return "setup"
    if any(
        phrase in text_lower
        for phrase in (
            "calculate",
            "compute",
            " = ",
            "divided by",
            "multiply",
            "adding",
            "subtract",
            "total",
            "sum",
        )
    ) and not any(phrase in text_lower for phrase in ("let me", "should i", "wait")):
        return "calculate"
    if any(
        phrase in text_lower
        for phrase in (
            "wait",
            "but ",
            "however",
            "that doesn't make sense",
            "let me reconsider",
            "or maybe",
            "perhaps",
            "could it be",
            "i'm not sure",
            "ambiguous",
            "confusing",
            "mistake",
        )
    ):
        return "doubt"
    if any(
        phrase in text_lower
        for phrase in (
            "let me check",
            "double-check",
            "verify",
            "confirm",
            "seems correct",
            "looks right",
            "to make sure",
        )
    ):
        return "verify"
    if any(
        phrase in text_lower
        for phrase in (
            "therefore",
            "so the answer",
            "final answer",
            "in conclusion",
            "thus",
            "hence",
            "this gives us",
        )
    ):
        return "conclude"
    return "general"


def _merge_chunks_semantically(
    chunks: list[Chunk],
    min_chars: int,
    max_chars: int,
) -> list[StepSpan]:
    if not chunks:
        return []

    steps: list[StepSpan] = [
        StepSpan(chunks[0].text, chunks[0].start, chunks[0].end)
    ]
    current_type = chunks[0].semantic_type

    for chunk in chunks[1:]:
        prev = steps[-1]
        prev_len = len(prev.text)
        should_merge = False

        if chunk.is_enumerated and prev_len >= min_chars:
            should_merge = False
        elif chunk.semantic_type == current_type:
            if current_type in {"doubt", "verify"}:
                should_merge = prev_len + len(chunk.text) <= max_chars * 1.2
            elif current_type == "calculate":
                should_merge = prev_len < min_chars and prev_len + len(chunk.text) <= max_chars
            else:
                should_merge = prev_len < min_chars * 1.5
        elif _should_keep_together(current_type, chunk.semantic_type):
            should_merge = prev_len + len(chunk.text) <= max_chars
        elif _is_major_transition(current_type, chunk.semantic_type):
            should_merge = prev_len < min_chars * 0.5
        elif prev_len < min_chars * 0.6:
            should_merge = True
        elif len(chunk.text) < min_chars * 0.4 and not chunk.is_enumerated:
            should_merge = prev_len + len(chunk.text) <= max_chars

        if should_merge:
            steps[-1] = StepSpan(
                prev.text + "\n\n" + chunk.text,
                prev.start,
                chunk.end,
            )
        else:
            steps.append(StepSpan(chunk.text, chunk.start, chunk.end))
            current_type = chunk.semantic_type

    if len(steps) >= 2 and len(steps[-1].text) < min_chars * 0.5:
        previous = steps[-2]
        last = steps[-1]
        steps[-2] = StepSpan(
            previous.text + "\n\n" + last.text,
            previous.start,
            last.end,
        )
        steps.pop()

    return steps


def _should_keep_together(type1: str, type2: str) -> bool:
    return (type1, type2) in {
        ("setup", "calculate"),
        ("calculate", "verify"),
        ("doubt", "doubt"),
        ("verify", "verify"),
    }


def _is_major_transition(type1: str, type2: str) -> bool:
    return (type1, type2) in {
        ("doubt", "calculate"),
        ("verify", "conclude"),
        ("calculate", "doubt"),
        ("setup", "doubt"),
    }
