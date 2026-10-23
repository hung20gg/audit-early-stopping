from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

from methods.response import Response
from replay_eval.utils.helper import completion_kwargs_for_phase, configured_stop_tokens
from replay_eval.v2.engine.types import copy_response


class _CachedLLM:
    def __init__(self, owner: "ReplayCompatibilityAdapter") -> None:
        self.owner = owner

    def completion(self, prompt: str, **kwargs: Any) -> Response:
        prompt_ids = tuple(
            self.owner.tokenizer.encode(prompt, add_special_tokens=False)
        )
        queue = self.owner._completion_cache.get(prompt_ids)
        if not queue:
            raise RuntimeError(
                "Verifier requested an unscheduled completion. V2 never falls back to a "
                "second synchronous model call."
            )
        return copy_response(queue.popleft())


class _ChatTemplateAdapter:
    def __init__(self, tokenizer: Any, renderer: Any) -> None:
        self.tokenizer = tokenizer
        self.renderer = renderer
        self.think_start_token = renderer.think_start_token
        self.think_end_token = renderer.think_end_token
        self.template = "gpt_oss" if renderer.is_gpt_oss else "hf"

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def render(self, value: str | list[dict[str, Any]]) -> str:
        if isinstance(value, str):
            return value
        return self.tokenizer.apply_chat_template(
            value,
            tokenize=False,
            add_generation_prompt=True,
            **self.renderer.template_kwargs,
        )


class ReplayCompatibilityAdapter:
    """Synchronous facade consumed by the unchanged v1 verifier classes."""

    def __init__(
        self,
        tokenizer: Any,
        renderer: Any,
        *,
        completion_kwargs: dict[str, Any] | None = None,
        stop_tokens: list[str] | None = None,
        code_probe_token_multiplier: int = 1,
        code_threshold_multiplier: float = 1.0,
        verify_batch_token: int = 16,
    ) -> None:
        self.tokenizer = tokenizer
        self.renderer = renderer
        self.completion_kwargs = dict(completion_kwargs or {})
        if stop_tokens is not None:
            self.completion_kwargs["stop"] = list(stop_tokens)
        self.code_probe_token_multiplier = int(code_probe_token_multiplier)
        self.code_threshold_multiplier = float(code_threshold_multiplier)
        self.verify_batch_token = int(verify_batch_token)
        self.chat_template = _ChatTemplateAdapter(tokenizer, renderer)
        self.llm = _CachedLLM(self)
        self.hidden_states: tuple[Any, Any] | None = None
        self.hidden_trace: Any | None = None
        self._temp_cache: dict[tuple[int, str, int], Response] = {}
        self._temp_reads: defaultdict[tuple[int, str], int] = defaultdict(int)
        self._force_start: defaultdict[tuple[int, str], int] = defaultdict(int)
        self._completion_cache: defaultdict[tuple[int, ...], deque[Response]] = (
            defaultdict(deque)
        )

    def reset(self) -> None:
        self._temp_cache.clear()
        self._temp_reads.clear()
        self._force_start.clear()
        self._completion_cache.clear()
        self.hidden_states = None
        self.hidden_trace = None

    @staticmethod
    def _temp_key(prefix: Response, position: int) -> tuple[int, str]:
        return int(position), prefix.get_generation_content()

    def put_temp_answer(
        self, position: int, response: Response, slot: int = 0, *, prefix: Response
    ) -> None:
        key = (*self._temp_key(prefix, position), int(slot))
        self._temp_cache[key] = copy_response(response)

    def put_completion(self, prompt_token_ids: list[int], response: Response) -> None:
        self._completion_cache[tuple(prompt_token_ids)].append(copy_response(response))

    def set_force_start(self, position: int, slot: int, *, prefix: Response) -> None:
        self._force_start[self._temp_key(prefix, position)] = int(slot)

    def generate_temp_answer(
        self,
        prefix: Response,
        position: int,
        kwargs: dict[str, Any],
        *,
        logprobs: int = 1,
        force_new: bool = False,
    ) -> Response:
        position_key = self._temp_key(prefix, position)
        slot = (
            self._force_start[position_key] + self._temp_reads[position_key]
            if force_new
            else 0
        )
        key = (*position_key, int(slot))
        response = self._temp_cache.get(key)
        if response is None:
            raise RuntimeError(
                f"Temporary answer was not asynchronously scheduled for position={position}, slot={slot}"
            )
        if force_new:
            self._temp_reads[position_key] += 1
        copied = copy_response(response)
        object.__setattr__(copied, "_from_temp_answer_cache", not force_new)
        object.__setattr__(copied, "_temp_answer_position", position)
        return copied

    def configured_stop_tokens(self, kwargs: dict[str, Any]) -> list[str]:
        return configured_stop_tokens(self.completion_kwargs, kwargs)

    def think_start_token(self) -> str:
        return self.renderer.think_start_token

    def think_end_token(self) -> str:
        return self.renderer.think_end_token

    def scale_probe_tokens(self, probe_tokens: int, kwargs: dict[str, Any]) -> int:
        multiplier = (
            self.code_probe_token_multiplier if self._is_code_task(kwargs) else 1
        )
        return int(probe_tokens) * multiplier

    def scale_threshold(self, threshold: float, kwargs: dict[str, Any]) -> float:
        multiplier = (
            self.code_threshold_multiplier if self._is_code_task(kwargs) else 1.0
        )
        return float(threshold) * multiplier

    @staticmethod
    def _is_code_task(kwargs: dict[str, Any]) -> bool:
        if kwargs.get("is_code") is not None:
            return bool(kwargs["is_code"])
        identity = " ".join(
            str(kwargs.get(key) or "")
            for key in ("dataset", "dataloader", "evaluator_name")
        ).casefold()
        return any(name in identity for name in ("code", "humaneval", "livecodebench"))

    def _completion_kwargs_for_phase(
        self, phase: str, kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        values = completion_kwargs_for_phase(phase, self.completion_kwargs, kwargs)
        if phase == "temp_answer":
            phase_limit = kwargs.get("max_temp_answer_completion_token")
        elif phase == "final":
            phase_limit = kwargs.get("max_final_answer_completion_token")
        else:
            phase_limit = None
        if phase_limit is not None:
            values.pop("max_tokens", None)
            values.pop("max_competion_token", None)
            values["max_completion_token"] = int(phase_limit)
        return values

    def _thinking_end_char(self, trajectory: Response) -> int:
        index = trajectory.content.find(self.think_end_token())
        return index if index >= 0 else len(trajectory.content)

    def temp_answer_extraction_content(
        self, response: Response, kwargs: dict[str, Any]
    ) -> str:
        if self._is_code_task(kwargs) or not response.content.startswith("{"):
            return response.content
        return self.temp_answer_suffix(kwargs) + response.content

    def temp_answer_suffix(self, kwargs: dict[str, Any]) -> str:
        if self._is_code_task(kwargs):
            return (
                f"You are out of thinking budget {self.think_end_token()} "
                "the final code snippet is \n\n"
            )
        return (
            f"You are out of thinking budget {self.think_end_token()} "
            "the final answer is \\boxed"
        )
