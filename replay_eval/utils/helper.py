import re
from logging import Logger
from collections.abc import Mapping
from typing import Any, Callable

from methods.response import Response


def is_finished_thinking_base(
    generated_text: str,
    think_start_token: str = "<think>",
    think_end_token: str = "</think>",
) -> bool:
    pattern = re.escape(think_start_token) + r".*?" + re.escape(think_end_token)
    return re.search(pattern, generated_text, re.DOTALL) is not None


def break_thinking_base(
    generated_text: str,
    think_start_token: str = "<think>",
    think_end_token: str = "</think>",
) -> str:
    if think_end_token not in generated_text:
        return f"You are out of time. Please provide your final answer now. {think_end_token}"
    return ""


def int_or_none(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def drop_internal_kwargs(completion_kwargs: dict[str, Any]) -> None:
    completion_kwargs.pop("answer", None)
    completion_kwargs.pop("evaluator", None)
    completion_kwargs.pop("evaluator_name", None)
    completion_kwargs.pop("dataloader", None)
    completion_kwargs.pop("dataset", None)
    completion_kwargs.pop("is_code", None)
    completion_kwargs.pop("reuse_generation_response", None)
    completion_kwargs.pop("reuse_generation_input", None)
    completion_kwargs.pop("reuse_generation_content", None)
    completion_kwargs.pop("verify_batch_token", None)
    completion_kwargs.pop("final_answer_batch_size", None)


def completion_kwargs_for_phase(
    phase: str,
    base_kwargs: dict[str, Any],
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Translate internal config keys into provider-safe completion kwargs."""
    completion_kwargs = {"logprobs": 1, **base_kwargs, **kwargs}
    completion_kwargs["logprobs"] = max(1, int(completion_kwargs.get("logprobs") or 1))
    thinking_tokens = completion_kwargs.pop("max_thinking_completion_token", None)
    temp_answer_tokens = completion_kwargs.pop("max_temp_answer_completion_token", None)
    final_answer_tokens = completion_kwargs.pop("max_final_answer_completion_token", None)
    final_tokens = completion_kwargs.pop("max_completion_token", None)
    drop_internal_kwargs(completion_kwargs)
    completion_kwargs.pop("stop", None)
    completion_kwargs.pop("stop_token", None)

    if phase == "thinking":
        phase_tokens = thinking_tokens
    elif phase == "temp_answer":
        phase_tokens = temp_answer_tokens or final_answer_tokens or final_tokens
    else:
        phase_tokens = final_answer_tokens or final_tokens
    has_token_limit = (
        "max_competion_token" in completion_kwargs
        or "max_tokens" in completion_kwargs
    )
    if phase_tokens is not None and not has_token_limit:
        completion_kwargs["max_completion_token"] = phase_tokens

    return completion_kwargs


def full_generation_kwargs(
    base_kwargs: dict[str, Any],
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Build provider kwargs for one-shot full generation."""
    full_kwargs = {"logprobs": 1, **base_kwargs, **kwargs}
    full_kwargs["logprobs"] = max(1, int(full_kwargs.get("logprobs") or 1))
    drop_internal_kwargs(full_kwargs)

    if "max_tokens" not in full_kwargs:
        max_completion_token = full_kwargs.pop(
            "max_completion_token",
            full_kwargs.pop("max_competion_token", None),
        )
        full_kwargs.pop("max_thinking_completion_token", None)
        full_kwargs.pop("max_temp_answer_completion_token", None)
        full_kwargs.pop("max_final_answer_completion_token", None)
        if max_completion_token is not None:
            full_kwargs["max_tokens"] = max_completion_token
    else:
        full_kwargs.pop("max_thinking_completion_token", None)
        full_kwargs.pop("max_temp_answer_completion_token", None)
        full_kwargs.pop("max_final_answer_completion_token", None)

    full_kwargs.pop("stop", None)
    full_kwargs.pop("stop_token", None)
    return full_kwargs


def convert_chat_full_generation_to_replay_response(
    response: Response,
    *,
    template: str = "hermes",
    think_start_token: str = "<think>",
    think_end_token: str = "</think>",
) -> Response:
    """Normalize chat full-generation output into replay text-completion shape.

    Chat APIs commonly return content as ``<think>...</think>answer`` while the
    rendered prompt already ends with the assistant-side thinking opener for
    some templates. In that case replay must not see a duplicated
    ``<think>`` token in ``Response.content``.
    """
    # Some providers omit the thinking opener from returned content while
    # still emitting the configured boundary before the final channel.
    think_end_index = response.content.find(think_end_token)
    if think_end_index == -1:
        return response

    # Split think content and normal content, and reformat based on the template.
    if response.content.startswith(think_start_token):
        think_content = response.content[len(think_start_token):think_end_index]
    else:
        think_content = response.content[:think_end_index]
    normal_content = response.content[think_end_index + len(think_end_token):]

    # Apply template-specific formatting rules.
    normalized_template = (template or "hermes").lower()
    if (
        normalized_template in ["qwen3_xml", "deepseek_v4"]
        or think_start_token != "<think>"
    ):
        converted_content = f"{think_content}{think_end_token}{normal_content}"
    else:
        converted_content = (
            f"{think_start_token}{think_content}{think_end_token}{normal_content}"
        )

    if converted_content == response.content:
        return response

    response.content = converted_content
    return response


def normalize_stop_tokens(stop: Any) -> list[str]:
    if isinstance(stop, str):
        return [stop]
    if isinstance(stop, list):
        stop_tokens = []
        for item in stop:
            if isinstance(item, str):
                stop_tokens.append(item)
            elif isinstance(item, Mapping):
                value = item.get("text", item.get("token"))
                if isinstance(value, str):
                    stop_tokens.append(value)
        return stop_tokens
    if isinstance(stop, Mapping):
        value = stop.get("text", stop.get("token"))
        return [value] if isinstance(value, str) else []
    return []


def configured_stop_tokens(
    base_kwargs: dict[str, Any],
    kwargs: dict[str, Any],
) -> list[str]:
    completion_kwargs = {**base_kwargs, **kwargs}
    return normalize_stop_tokens(
        completion_kwargs.get("stop_token", completion_kwargs.get("stop"))
    )


def first_stop_match(content: str, stop_tokens: list[str]) -> tuple[int, str] | None:
    matches = [
        (content.find(stop_token), stop_token)
        for stop_token in stop_tokens
        if stop_token and content.find(stop_token) != -1
    ]
    if not matches:
        return None
    return min(matches, key=lambda item: (item[0], -len(item[1])))


def count_thinking_chunks(content: str, stop_tokens: list[str]) -> int:
    if not content:
        return 0
    if not stop_tokens:
        return 1

    chunk_count = 0
    cursor = 0
    while cursor < len(content):
        stop_match = first_stop_match(content[cursor:], stop_tokens)
        if stop_match is None:
            chunk_count += 1
            break

        stop_index, stop_token = stop_match
        cursor += stop_index + len(stop_token)
        chunk_count += 1

    return chunk_count


def slice_at_stop_token(
    response: Response,
    stop_tokens: list[str],
    token_counter: Callable[[str], int | None],
) -> Response:
    """Keep returned text through the first local stop marker, if present."""
    stop_match = first_stop_match(response.content, stop_tokens)
    if stop_match is None:
        return response

    stop_index, stop_token = stop_match
    end_char = stop_index + len(stop_token)
    if response.logprobs:
        return response.char_slice_from_tokens(0, end_char)
    content = response.content[:end_char]
    return Response(
        input=response.input,
        content=content,
        finish_reason=response.finish_reason,
        completion_tokens=token_counter(content),
    )


def is_generation_ended(
    response: Response,
    token_limit: int | None,
) -> bool:
    """Detect provider-side end before the fixed chunk budget is exhausted."""
    if response.finish_reason and response.finish_reason != "length":
        return True
    if token_limit is None:
        return False

    token_limit = int_or_none(token_limit)
    if token_limit is None:
        return False

    return response.completion_tokens is not None and response.completion_tokens < token_limit


def without_newline_stop(kwargs: dict[str, Any]) -> dict[str, Any]:
    updated_kwargs = dict(kwargs)
    stop = updated_kwargs.get("stop")
    if isinstance(stop, str):
        if not stop.strip():
            updated_kwargs.pop("stop")
        return updated_kwargs
    if isinstance(stop, list):
        filtered_stop = [
            item
            for item in stop
            if not (isinstance(item, str) and not item.strip())
        ]
        if filtered_stop:
            updated_kwargs["stop"] = filtered_stop
        else:
            updated_kwargs.pop("stop")
    updated_kwargs.pop("stop_token", None)
    return updated_kwargs


def finalize_kwargs_with_remaining_budget(
    generated_text: Response,
    kwargs: dict[str, Any],
    *,
    max_final_completion_request_tokens: int,
    logger: Logger,
) -> dict[str, Any]:
    finalize_kwargs = without_newline_stop(kwargs)
    final_budget = int_or_none(finalize_kwargs.get("max_completion_token"))
    if final_budget is None:
        return finalize_kwargs

    remaining_budget = final_budget - generated_text.token_count()
    capped_budget = min(max_final_completion_request_tokens, max(1, remaining_budget))
    finalize_kwargs["max_completion_token"] = capped_budget
    logger.info(
        "Finalization budget adjusted configured=%s generated_tokens=%s final_request_tokens=%s",
        final_budget,
        generated_text.token_count(),
        capped_budget,
    )
    return finalize_kwargs


def match_token_prefix(generated_prefix: str, token_texts: list[str]) -> int:
    full_content = "".join(token_texts)
    if not full_content.startswith(generated_prefix):
        raise ValueError(
            "Generated content does not match the full-generation token prefix. "
            f"full_chars={len(full_content)} prefix_chars={len(generated_prefix)}"
        )
    return len(generated_prefix)


def token_index_at_char(token_texts: list[str], char_index: int) -> int:
    cursor = 0
    for index, token in enumerate(token_texts):
        next_cursor = cursor + len(token)
        if next_cursor > char_index:
            return index
        cursor = next_cursor
    return len(token_texts)


def char_index_at_token(token_texts: list[str], token_index: int) -> int:
    return len("".join(token_texts[:token_index]))


def matched_reuse_prefix_chars(generated_prefix: str, full_content: str) -> int:
    if full_content.startswith(generated_prefix):
        return len(generated_prefix)

    common_length = 0
    for left, right in zip(generated_prefix, full_content):
        if left != right:
            break
        common_length += 1
    if generated_prefix and common_length == 0:
        raise ValueError(
            "Generated content does not match the reused generation prefix. "
            f"full_chars={len(full_content)} prefix_chars={len(generated_prefix)}"
        )
    return common_length


def content_token_slice(
    content: str,
    token_limit: int | None,
    token_counter: Callable[[str], int | None],
) -> tuple[str, int | None]:
    if token_limit is None:
        return content, token_counter(content)
    if token_limit <= 0 or not content:
        return "", 0

    full_token_count = token_counter(content)
    if full_token_count is not None and full_token_count <= token_limit:
        return content, full_token_count

    low = 0
    high = len(content)
    best_end = 0
    best_count = 0
    while low <= high:
        mid = (low + high) // 2
        candidate = content[:mid]
        token_count = token_counter(candidate)
        if token_count is None:
            break
        if token_count <= token_limit:
            best_end = mid
            best_count = token_count
            low = mid + 1
        else:
            high = mid - 1

    if best_end == 0 and full_token_count is None:
        best_end = min(len(content), token_limit)
        best_count = token_counter(content[:best_end])
    return content[:best_end], best_count
