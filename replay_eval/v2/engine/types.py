from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any
import copy

from methods.response import Response


class RequestKind(StrEnum):
    FULL_TRACE = "full_trace"
    NORMAL_PROBE = "normal_probe"
    DTSR = "dtsr"
    FINAL = "final"
    RANDOM_EXIT = "random_exit"
    WARMUP = "warmup"


def copy_response(response: Response) -> Response:
    model_copy = getattr(response, "model_copy", None)
    return model_copy(deep=True) if callable(model_copy) else response.copy(deep=True)


def freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(sorted((str(key), freeze(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(freeze(item) for item in value)
    if isinstance(value, set):
        return tuple(sorted(freeze(item) for item in value))
    return value


@dataclass(frozen=True)
class RequestSpec:
    prompt_token_ids: tuple[int, ...]
    sampling: tuple[tuple[str, Any], ...]
    checkpoint: int
    kind: RequestKind
    example_id: str = field(compare=False, default="")
    cache_slot: int = 0
    cache_salt: str | None = None
    sampling_values: dict[str, Any] = field(
        compare=False, hash=False, repr=False, default_factory=dict
    )

    @classmethod
    def create(
        cls,
        prompt_token_ids: list[int] | tuple[int, ...],
        sampling: dict[str, Any],
        checkpoint: int,
        kind: RequestKind,
        *,
        example_id: str = "",
        cache_slot: int = 0,
        cache_salt: str | None = None,
    ) -> "RequestSpec":
        return cls(
            tuple(int(token) for token in prompt_token_ids),
            tuple(sorted((str(key), freeze(value)) for key, value in sampling.items())),
            int(checkpoint),
            kind,
            str(example_id),
            int(cache_slot),
            str(cache_salt) if cache_salt is not None else None,
            copy.deepcopy(sampling),
        )

    def sampling_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self.sampling_values)


@dataclass
class Timing:
    queue: float = 0.0
    prefill: float = 0.0
    decode: float = 0.0
    end_to_end: float = 0.0


@dataclass
class RequestResult:
    response: Response
    prompt_token_ids: list[int]
    output_token_ids: list[int]
    timing: Timing = field(default_factory=Timing)
    cached: bool = False
    cached_prompt_tokens: int = 0
    raw: Any = None

    @property
    def computed_tokens(self) -> int:
        if self.cached:
            return 0
        return (
            max(0, len(self.prompt_token_ids) - self.cached_prompt_tokens)
            + len(self.output_token_ids)
        )
