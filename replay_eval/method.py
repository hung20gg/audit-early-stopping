"""Abstractions for the (S, pi, G) triple that defines an early-exit method."""

from __future__ import annotations

import sys
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from methods.response import Response

if TYPE_CHECKING:
    from replay_eval.replay import ReplayEvaluatorAgent


@dataclass
class VerifierContext:
    """Everything a checkpoint / verifier / stop rule may look at.

    Verifiers must NOT read ``trajectory`` — it is exposed only so
    checkpoint functions and verifiers that intentionally use replay
    approximations (e.g. peeking at the next logprobs of the shared trace)
    can opt in. The proposal's clean semantics forbid future-token access;
    use ``agent`` to make fresh model calls instead.
    """

    question: str
    prefix: Response
    latest_token: str
    position: int
    agent: "ReplayEvaluatorAgent"
    kwargs: dict[str, Any]
    state: dict[str, Any]
    logs: dict[str, Any] = field(default_factory=dict)
    trajectory: Response | None = None
    method_name: str | None = None


@dataclass
class MethodResult:
    """One row of the replay-evaluation result set."""

    method_name: str
    position: int
    stopped_early: bool
    verifier_score: float | None
    reasoning_prefix: str
    final_answer: str
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    thinking_tokens: int | None = None
    answer_tokens: int | None = None
    completion_tokens: int | None = None
    verifier_prompt_tokens: int = 0
    verifier_completion_tokens: int = 0
    state: dict[str, Any] = field(default_factory=dict)
    logs: dict[str, Any] = field(default_factory=dict)

    base_method: str | None = None
    exit: bool | None = None
    exit_index: int | None = None

    def mark_continued(self, index: int | None) -> None:
        """Freeze a confirmed exit or terminal sample from continued replay."""
        self.state = deepcopy(self.state)
        self.logs = deepcopy(self.logs)
        self.base_method = self.method_name
        self.exit = False
        self.exit_index = index
        self.method_name = f"{self.base_method}:{index if index is not None else 'full'}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method_name,
            **(
                {"base_method": self.base_method, "exit": self.exit,
                 "exit_index": self.exit_index}
                if self.exit is False else {}
            ),
            "position": self.position,
            "stopped_early": self.stopped_early,
            "verifier_score": self.verifier_score,
            "final_answer": self.final_answer,
            "finish_reason": self.finish_reason,
            "prompt_tokens": self.prompt_tokens,
            "thinking_tokens": self.thinking_tokens,
            "answer_tokens": self.answer_tokens,
            "completion_tokens": self.completion_tokens,
            "verifier_prompt_tokens": self.verifier_prompt_tokens,
            "verifier_completion_tokens": self.verifier_completion_tokens,
            **({"state": self.state} if self.state else {}),
            **({"logs": self.logs} if self.logs else {}),
        }


class CheckpointFn:
    """S_i: decide whether to check at the current position."""

    def reset(self, method: "EarlyExitMethod") -> None:
        return None

    def __call__(self, ctx: VerifierContext) -> bool:
        raise NotImplementedError

    def replay_check(
        self,
        ctx: VerifierContext,
        *,
        full_content: str,
        prefix_chars: int,
    ) -> bool | None:
        """Fast replay check that can avoid materializing a full prefix.

        ``None`` asks the replay runtime to construct the normal prefix and
        call ``__call__``. Built-in checkpoints override this when position
        and a character boundary contain all the information they need.
        """
        del ctx, full_content, prefix_chars
        return None


class Verifier:
    """pi_i: estimate whether the current reasoning state is sufficient."""

    def reset(self, method: "EarlyExitMethod") -> None:
        return None

    def __call__(self, ctx: VerifierContext) -> Any:
        raise NotImplementedError


class StopRule:
    """G_i: convert verifier output into a binary stop decision."""

    def reset(self, method: "EarlyExitMethod") -> None:
        return None

    def __call__(self, verifier_output: Any, ctx: VerifierContext) -> bool:
        raise NotImplementedError


# Adapters so plain callables can be passed in without subclassing.
CheckpointLike = CheckpointFn | Callable[[VerifierContext], bool]
VerifierLike = Verifier | Callable[[VerifierContext], Any] | None
StopRuleLike = StopRule | Callable[[Any, VerifierContext], bool] | None


def _as_checkpoint(fn: CheckpointLike) -> CheckpointFn:
    if isinstance(fn, CheckpointFn):
        return fn
    return _CallableCheckpoint(fn)


def _as_verifier(fn: VerifierLike) -> Verifier:
    if fn is None:
        return _NullVerifier()
    if isinstance(fn, Verifier):
        return fn
    return _CallableVerifier(fn)


def _as_stop_rule(fn: StopRuleLike) -> StopRule:
    if fn is None:
        return _AlwaysStopRule()
    if isinstance(fn, StopRule):
        return fn
    return _CallableStopRule(fn)


class _CallableCheckpoint(CheckpointFn):
    def __init__(self, fn: Callable[[VerifierContext], bool]):
        self._fn = fn

    def __call__(self, ctx: VerifierContext) -> bool:
        return bool(self._fn(ctx))


class _CallableVerifier(Verifier):
    def __init__(self, fn: Callable[[VerifierContext], Any]):
        self._fn = fn

    def __call__(self, ctx: VerifierContext) -> Any:
        return self._fn(ctx)


class _CallableStopRule(StopRule):
    def __init__(self, fn: Callable[[Any, VerifierContext], bool]):
        self._fn = fn

    def __call__(self, verifier_output: Any, ctx: VerifierContext) -> bool:
        return bool(self._fn(verifier_output, ctx))


class _NullVerifier(Verifier):
    def __call__(self, ctx: VerifierContext) -> None:
        return None


class _AlwaysStopRule(StopRule):
    def __call__(self, verifier_output: Any, ctx: VerifierContext) -> bool:
        return True


class EarlyExitMethod:
    """A composed early-exit method i = (S_i, pi_i, G_i)."""

    def __init__(
        self,
        name: str,
        checkpoint: CheckpointLike,
        verifier: VerifierLike,
        stop_rule: StopRuleLike,
        *,
        no_thinking: bool = False,
    ):
        self.name = name
        self.checkpoint = _as_checkpoint(checkpoint)
        self.verifier = _as_verifier(verifier)
        self.stop_rule = _as_stop_rule(stop_rule)
        self.no_thinking = no_thinking
        self.state: dict[str, Any] = {}
        self.logs: dict[str, Any] = {}

    def reset(self) -> None:
        self.state = {}
        self.logs = {}
        self.checkpoint.reset(self)
        self.verifier.reset(self)
        self.stop_rule.reset(self)

    def should_check(self, ctx: VerifierContext) -> bool:
        return bool(self.checkpoint(ctx))

    def verify(self, ctx: VerifierContext) -> Any:
        return self.verifier(ctx)

    def should_stop(self, verifier_output: Any, ctx: VerifierContext) -> bool:
        return bool(self.stop_rule(verifier_output, ctx))
