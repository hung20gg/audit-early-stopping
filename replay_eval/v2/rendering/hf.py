from __future__ import annotations

from dataclasses import dataclass
import logging
import re
from typing import Any

from methods.chat_template import prepare_gemma4_messages


LOGGER = logging.getLogger(__name__)
_GPT_OSS_CHANNEL_FRAME = re.compile(
    r"<\|channel\|>[^<]*<\|message\|>",
)
_GPT_OSS_STRUCTURAL_TOKENS = (
    "<|start|>assistant",
    "<|channel|>",
    "<|message|>",
    "<|end|>",
    "<|return|>",
    "<|call|>",
)
_GPT_OSS_CURRENT_DATE = re.compile(
    r"(?<=\nCurrent date: )\d{4}-\d{2}-\d{2}(?=\n)"
)


@dataclass(frozen=True)
class RenderedTrace:
    source_token_ids: tuple[int, ...]
    full_token_ids: tuple[int, ...]
    output_token_ids: tuple[int, ...]
    thinking_token_count: int


def split_reasoning(
    content: str,
    think_end_token: str,
    think_start_token: str | None = None,
) -> tuple[str, str, bool]:
    if think_start_token:
        # Text-completion responses with skip_special_tokens=False may retain
        # GPT-OSS's structural analysis header. The chat template emits that
        # header from the structured ``thinking`` field, so passing it as part
        # of the field duplicates the channel and is explicitly rejected.
        assistant_header = f"<|start|>assistant{think_start_token}"
        candidate = content.lstrip()
        if candidate.startswith(assistant_header):
            content = candidate[len(assistant_header) :]
        elif candidate.startswith(think_start_token):
            content = candidate[len(think_start_token) :]
    # Harmony/GPT-OSS output can contain repeated channel boundaries while a
    # streamed prefix can end part-way through the final boundary. Other model
    # families retain the original first-boundary behavior.
    is_gpt_oss = think_start_token is not None
    index = (
        content.rfind(think_end_token)
        if is_gpt_oss
        else content.find(think_end_token)
    )
    if index < 0:
        thinking = (
            _strip_trailing_partial_marker(content, think_end_token)
            if is_gpt_oss
            else content
        )
        answer, finished = "", False
    else:
        thinking = content[:index]
        answer = content[index + len(think_end_token) :]
        finished = True
    if think_start_token:
        thinking = _clean_gpt_oss_field(thinking)
        answer = _clean_gpt_oss_field(answer)
    return thinking, answer, finished


def _strip_trailing_partial_marker(value: str, marker: str) -> str:
    """Drop the longest suffix that is a proper prefix of ``marker``."""
    if not value or not marker:
        return value
    maximum = min(len(value), len(marker) - 1)
    for length in range(maximum, 0, -1):
        if value.endswith(marker[:length]):
            return value[:-length]
    return value


def _clean_gpt_oss_field(value: str) -> str:
    """Remove Harmony framing while retaining the message payload."""
    value = _GPT_OSS_CHANNEL_FRAME.sub("", value)
    for token in _GPT_OSS_STRUCTURAL_TOKENS:
        value = value.replace(token, "")
    return value


class CanonicalRenderer:
    """Render all prompts through one Hugging Face tokenizer."""

    def __init__(
        self,
        tokenizer: Any,
        *,
        model_id: str,
        template_kwargs: dict[str, Any] | None = None,
        think_start_token: str = "<think>",
        think_end_token: str = "</think>",
    ) -> None:
        self.tokenizer = tokenizer
        self.model_id = model_id
        self.template_kwargs = dict(template_kwargs or {})
        self.think_start_token = think_start_token
        self.think_end_token = think_end_token

    @property
    def is_gpt_oss(self) -> bool:
        return "gpt-oss" in self.model_id.casefold()

    @property
    def is_gemma4(self) -> bool:
        return "gemma-4" in self.model_id.casefold().replace("_", "-")

    @property
    def uses_reasoning_content(self) -> bool:
        model_id = self.model_id.casefold().replace("_", ".")
        return "qwen3.5" in model_id

    @property
    def uses_inline_reasoning(self) -> bool:
        return not (self.is_gpt_oss or self.is_gemma4 or self.uses_reasoning_content)

    def source(
        self,
        messages: list[dict[str, Any]],
        *,
        enable_thinking: bool = True,
    ) -> list[int]:
        return self._apply(
            messages,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )

    def source_text(
        self,
        messages: list[dict[str, Any]],
        *,
        enable_thinking: bool = True,
    ) -> str:
        return self._apply_text(
            messages,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )

    def validate_response_input(
        self, messages: list[dict[str, Any]], stored_input: str
    ) -> list[int]:
        expected = self.source(messages)
        actual = list(self.tokenizer.encode(stored_input, add_special_tokens=False))
        if actual != expected:
            if self.is_gpt_oss:
                rendered_input = self.source_text(messages)
                if _same_gpt_oss_prompt_except_current_date(
                    stored_input, rendered_input
                ):
                    # Harmony's template defaults ``current_date`` to the day
                    # it is rendered. A replay on a later day must retain the
                    # prompt that was actually used for generation.
                    LOGGER.info(
                        "Using stored GPT-OSS prompt tokens after its rendered "
                        "current date changed"
                    )
                    return actual
            mismatch = next(
                (i for i, pair in enumerate(zip(actual, expected)) if pair[0] != pair[1]),
                min(len(actual), len(expected)),
            )
            raise ValueError(
                "Response.input is not the canonical HF-rendered prompt: "
                f"first mismatch at token {mismatch}; stored={len(actual)}, rendered={len(expected)}"
            )
        return expected

    def full_trace(
        self,
        messages: list[dict[str, Any]],
        content: str,
        *,
        stored_input: str | None = None,
        output_token_ids: list[int] | None = None,
    ) -> RenderedTrace:
        source_ids = (
            self.validate_response_input(messages, stored_input)
            if stored_input is not None
            else self.source(messages)
        )
        thinking, answer, finished = self._split_reasoning(content)
        if output_token_ids is not None:
            full_ids = [*source_ids, *output_token_ids]
        elif self.uses_inline_reasoning:
            # Replaying content as an assistant chat message can strip its
            # <think> block (Qwen3), or insert another opening marker. The
            # original generation is the validated prompt plus the raw output.
            full_ids = [
                *source_ids,
                *self.tokenizer.encode(content, add_special_tokens=False),
            ]
        else:
            assistant = self._assistant(thinking, answer, finished=finished)
            full_ids = self._apply_preserving_source_prefix(
                [*messages, assistant],
                source_messages=messages,
                source_ids=source_ids,
                add_generation_prompt=False,
                continue_final_message=True,
            )
        common = _common_prefix(source_ids, full_ids)
        if common != len(source_ids):
            raise ValueError(
                "Canonical assistant trace is not token-prefix consistent with the source prompt "
                f"({common}/{len(source_ids)} prompt tokens match)"
            )
        thinking_ids = self.checkpoint(
            messages, thinking, source_token_ids=source_ids
        )
        if thinking_ids[: len(source_ids)] != source_ids:
            raise ValueError("Checkpoint rendering changed the source prompt token prefix")
        return RenderedTrace(
            tuple(source_ids),
            tuple(full_ids),
            tuple(full_ids[len(source_ids) :]),
            len(thinking_ids) - len(source_ids),
        )

    def checkpoint(
        self,
        messages: list[dict[str, Any]],
        thinking_prefix: str,
        *,
        source_token_ids: list[int] | None = None,
    ) -> list[int]:
        if self.uses_inline_reasoning:
            source_ids = (
                source_token_ids
                if source_token_ids is not None
                else self.source(messages)
            )
            return [
                *source_ids,
                *self.tokenizer.encode(thinking_prefix, add_special_tokens=False),
            ]
        if self.uses_reasoning_content:
            # Qwen3.5 closes ``reasoning_content`` in its chat template even
            # with continue_final_message=True. Replay checkpoints must remain
            # the original generation prompt plus only the observed reasoning.
            return [
                *self.source(messages),
                *self.tokenizer.encode(thinking_prefix, add_special_tokens=False),
            ]
        thinking, answer, finished = self._split_reasoning(thinking_prefix)
        assistant = self._assistant(thinking, answer, finished=finished)
        continuation: bool | str = True
        if self.is_gpt_oss:
            continuation = "content" if finished else "thinking"
        elif self.is_gemma4:
            continuation = "content" if finished else "reasoning"
        source_ids = source_token_ids
        if self.is_gpt_oss and source_ids is None:
            source_ids = self.source(messages)
        rendered = self._apply_preserving_source_prefix(
            [*messages, assistant],
            source_messages=messages,
            source_ids=source_ids,
            add_generation_prompt=False,
            continue_final_message=continuation,
        )
        if self.is_gpt_oss:
            end_ids = list(
                self.tokenizer.encode(self.think_end_token, add_special_tokens=False)
            )
            boundary = _last_subsequence_start(rendered, end_ids)
            if boundary is not None:
                return rendered[:boundary]
        return rendered

    def _split_reasoning(self, content: str) -> tuple[str, str, bool]:
        if not self.is_gemma4:
            return split_reasoning(
                content,
                self.think_end_token,
                self.think_start_token if self.is_gpt_oss else None,
            )
        candidate = content.lstrip()
        if candidate.startswith(self.think_start_token):
            content = candidate[len(self.think_start_token) :]
        index = content.find(self.think_end_token)
        if index < 0:
            return content, "", False
        return (
            content[:index],
            content[index + len(self.think_end_token) :],
            True,
        )

    def _apply_preserving_source_prefix(
        self,
        messages: list[dict[str, Any]],
        *,
        source_messages: list[dict[str, Any]],
        source_ids: list[int] | None,
        **kwargs: Any,
    ) -> list[int]:
        rendered = self._apply_with_gpt_oss_retry(messages, **kwargs)
        if not self.is_gpt_oss or source_ids is None:
            return rendered
        if rendered[: len(source_ids)] == source_ids:
            return rendered

        # Some tokenizer versions encode the final prompt token differently
        # when an assistant continuation follows it, even though the rendered
        # strings have an exact prefix relationship. Keep the validated prompt
        # IDs authoritative and tokenize only the appended Harmony suffix.
        source_text = self._apply_text(
            source_messages,
            add_generation_prompt=True,
        )
        rendered_text = self._apply_text(messages, **kwargs)
        if not rendered_text.startswith(source_text):
            common = _common_prefix(source_ids, rendered)
            raise ValueError(
                "GPT-OSS assistant rendering changed the source prompt text: "
                f"token prefix={common}/{len(source_ids)}"
            )
        suffix = rendered_text[len(source_text) :]
        LOGGER.warning(
            "Realigning GPT-OSS assistant tokens to preserve the canonical "
            "source prompt (%s tokens)",
            len(source_ids),
        )
        return [
            *source_ids,
            *self.tokenizer.encode(suffix, add_special_tokens=False),
        ]

    def _apply_with_gpt_oss_retry(
        self, messages: list[dict[str, Any]], **kwargs: Any
    ) -> list[int]:
        try:
            return self._apply(messages, **kwargs)
        except Exception as exc:
            if not self.is_gpt_oss or "channel" not in str(exc).casefold():
                raise
            cleaned = [dict(message) for message in messages]
            changed = False
            for message in cleaned:
                if message.get("role") != "assistant":
                    continue
                for field in ("thinking", "content"):
                    value = message.get(field)
                    if not isinstance(value, str):
                        continue
                    sanitized = _clean_gpt_oss_field(value)
                    changed = changed or sanitized != value
                    message[field] = sanitized
            if not changed:
                raise
            LOGGER.warning(
                "Retrying GPT-OSS chat rendering after removing Harmony channel framing"
            )
            return self._apply(cleaned, **kwargs)

    def render_messages(self, messages: list[dict[str, Any]]) -> list[int]:
        return self._apply(messages, add_generation_prompt=True)

    def decode_token(self, token_id: int) -> str:
        return self.tokenizer.decode(
            [int(token_id)], skip_special_tokens=False, clean_up_tokenization_spaces=False
        )

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def _assistant(
        self, thinking: str, answer: str, *, finished: bool
    ) -> dict[str, Any]:
        if self.is_gpt_oss:
            return {"role": "assistant", "thinking": thinking, "content": answer}
        if self.is_gemma4:
            assistant: dict[str, Any] = {"role": "assistant", "content": answer}
            if thinking:
                assistant["reasoning"] = thinking
            return assistant
        if self.uses_reasoning_content:
            return {
                "role": "assistant",
                "reasoning_content": thinking,
                "content": answer if finished else "",
            }
        # Non-Harmony tokenizers generally represent reasoning in assistant
        # content; the configured model boundary is authoritative.
        content = thinking
        if finished:
            content += self.think_end_token + answer
        return {
            "role": "assistant",
            "content": content,
        }

    def _apply(self, messages: list[dict[str, Any]], **kwargs: Any) -> list[int]:
        messages, kwargs = self._prepare_apply(messages, kwargs)
        options = {**self.template_kwargs, **kwargs, "tokenize": True}
        rendered = self.tokenizer.apply_chat_template(messages, **options)
        if hasattr(rendered, "get"):
            input_ids = rendered.get("input_ids")
            if input_ids is None:
                raise ValueError("Tokenized chat template mapping has no input_ids")
            rendered = input_ids
        if hasattr(rendered, "tolist"):
            rendered = rendered.tolist()
        if rendered and isinstance(rendered[0], list):
            rendered = rendered[0]
        return [int(token) for token in rendered]

    def _apply_text(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        messages, kwargs = self._prepare_apply(messages, kwargs)
        options = {**self.template_kwargs, **kwargs, "tokenize": False}
        rendered = self.tokenizer.apply_chat_template(messages, **options)
        if not isinstance(rendered, str):
            raise TypeError(
                "Non-tokenized chat template did not return text: "
                f"{type(rendered).__name__}"
            )
        return rendered

    def _prepare_apply(
        self,
        messages: list[dict[str, Any]],
        kwargs: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        options = dict(kwargs)
        enable_thinking = bool(options.pop("enable_thinking", True))
        if not self.is_gemma4:
            return messages, options
        prepared = prepare_gemma4_messages(
            messages,
            enable_thinking=enable_thinking,
        )
        options["enable_thinking"] = enable_thinking
        return prepared, options


def _common_prefix(left: list[int], right: list[int]) -> int:
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return min(len(left), len(right))


def _same_gpt_oss_prompt_except_current_date(left: str, right: str) -> bool:
    """Allow only Harmony's single volatile system-header date to differ."""
    left_dates = list(_GPT_OSS_CURRENT_DATE.finditer(left))
    right_dates = list(_GPT_OSS_CURRENT_DATE.finditer(right))
    if len(left_dates) != 1 or len(right_dates) != 1:
        return False
    left_system_end = left.find("<|end|>")
    right_system_end = right.find("<|end|>")
    if (
        left_system_end < 0
        or right_system_end < 0
        or left_dates[0].start() >= left_system_end
        or right_dates[0].start() >= right_system_end
    ):
        return False
    return (
        _GPT_OSS_CURRENT_DATE.sub("<CURRENT_DATE>", left, count=1)
        == _GPT_OSS_CURRENT_DATE.sub("<CURRENT_DATE>", right, count=1)
    )


def _last_subsequence_start(values: list[int], pattern: list[int]) -> int | None:
    if not pattern or len(pattern) > len(values):
        return None
    for start in range(len(values) - len(pattern), -1, -1):
        if values[start : start + len(pattern)] == pattern:
            return start
    return None
