"""Checkpoint functions (S_i)."""

from __future__ import annotations

from typing import Iterable, Sequence

from replay_eval.method import CheckpointFn, CheckpointLike, VerifierContext, _as_checkpoint
from replay_eval.utils.steps import separate_step_spans


class AlwaysCheckpoint(CheckpointFn):
    """S_i = 1 at every position (mostly useful for tests)."""

    def __call__(self, ctx: VerifierContext) -> bool:
        return True

    def replay_check(self, ctx, *, full_content, prefix_chars) -> bool:
        del ctx, full_content, prefix_chars
        return True


class EveryNTokens(CheckpointFn):
    """S_i = 1 every ``n`` generated tokens starting at ``offset``."""

    def __init__(self, n: int, offset: int = 0):
        if n <= 0:
            raise ValueError("EveryNTokens requires n >= 1")
        self.n = int(n)
        self.offset = int(offset)

    def __call__(self, ctx: VerifierContext) -> bool:
        if ctx.position < self.offset:
            return False
        return (ctx.position - self.offset) % self.n == 0

    def replay_check(self, ctx, *, full_content, prefix_chars) -> bool:
        del full_content, prefix_chars
        if ctx.position < self.offset:
            return False
        return (ctx.position - self.offset) % self.n == 0
    

class StopMarkerCheckpoint(CheckpointFn):
    """S_i = 1 when the prefix has just emitted one of ``markers``.

    Compares against the tail of ``ctx.prefix.content`` so the marker can
    span multiple tokens. Callers usually pass the same stop tokens the
    generator uses for chunk boundaries (e.g. "Wait", "Alternatively", ...).
    """

    def __init__(self, markers: Sequence[str] | None = None):
        self.markers = tuple(m for m in (markers or ()) if m)

    def reset(self, method):
        return None

    def _effective_markers(self, ctx: VerifierContext) -> tuple[str, ...]:
        if self.markers:
            return self.markers
        stop = ctx.agent.configured_stop_tokens(ctx.kwargs)
        return tuple(stop)

    def __call__(self, ctx: VerifierContext) -> bool:
        content = ctx.prefix.content
        if not content:
            return False
        for marker in self._effective_markers(ctx):
            if content.endswith(marker):
                return True
        return False

    def replay_check(self, ctx, *, full_content, prefix_chars) -> bool:
        if prefix_chars <= 0:
            return False
        return any(
            full_content.endswith(marker, 0, prefix_chars)
            for marker in self._effective_markers(ctx)
        )


class IntervalSignalCheckpoint(CheckpointFn):
    """S_i = 1 when a reflection signal appeared and >= ``min_interval`` tokens elapsed.

    Matches DTSR's default checkpoint: after every reflection cue (e.g.
    "wait", "but"), but no more often than every ``min_interval`` new
    thinking tokens.
    """

    def __init__(
        self,
        signals: Sequence[str] | None,
        min_interval: int = 32,
    ):
        self.signals = tuple(s for s in (signals or ()) if s)
        self.min_interval = int(min_interval)

    def reset(self, method) -> None:
        method.state["_isc_last_position"] = 0

    def _effective_signals(self, ctx: VerifierContext) -> tuple[str, ...]:
        if self.signals:
            return self.signals
        return tuple(ctx.agent.configured_stop_tokens(ctx.kwargs))

    def __call__(self, ctx: VerifierContext) -> bool:
        state = ctx.state
        last_position = int(state.get("_isc_last_position", 0))
        if ctx.position - last_position < self.min_interval:
            return False
        signals = self._effective_signals(ctx)
        if not signals:
            return False
        # Check for a signal on the newly added tail.
        tail = ctx.prefix.content[-max(len(s) for s in signals):] if signals else ""
        matched = any(ctx.prefix.content.endswith(sig) for sig in signals) or any(
            sig in tail for sig in signals
        )
        if not matched:
            return False
        state["_isc_last_position"] = ctx.position
        return True

    def replay_check(self, ctx, *, full_content, prefix_chars) -> bool:
        state = ctx.state
        last_position = int(state.get("_isc_last_position", 0))
        if ctx.position - last_position < self.min_interval:
            return False
        signals = self._effective_signals(ctx)
        if not signals:
            return False
        tail_start = max(0, prefix_chars - max(len(signal) for signal in signals))
        tail = full_content[tail_start:prefix_chars]
        matched = any(
            full_content.endswith(signal, 0, prefix_chars) for signal in signals
        ) or any(signal in tail for signal in signals)
        if not matched:
            return False
        state["_isc_last_position"] = ctx.position
        return True


class AnyCheckpoint(CheckpointFn):
    """Union of multiple checkpoint functions (S_i = OR of children)."""

    def __init__(self, children: Iterable[CheckpointLike]):
        self.children = [_as_checkpoint(child) for child in children]

    def reset(self, method) -> None:
        for child in self.children:
            child.reset(method)

    def __call__(self, ctx: VerifierContext) -> bool:
        return any(child(ctx) for child in self.children)

    def replay_check(self, ctx, *, full_content, prefix_chars) -> bool | None:
        if any(
            type(child).replay_check is CheckpointFn.replay_check
            for child in self.children
        ):
            return None
        results = []
        for child in self.children:
            result = child.replay_check(
                ctx,
                full_content=full_content,
                prefix_chars=prefix_chars,
            )
            if result is None:
                return None
            results.append(result)
            if result:
                return True
        return any(results)


class SemanticStepCheckpoint(CheckpointFn):
    """S_i = 1 at semantic reasoning-step boundaries.

    Boundaries are computed from the already-generated reasoning trajectory,
    then mapped back to replay positions. This keeps PUMA-style checks aligned
    with reasoning steps instead of scanning arbitrary words/tokens.
    """

    def __init__(
        self,
        min_step_chars: int = 200,
        max_step_chars: int = 1000,
    ):
        self.min_step_chars = int(min_step_chars)
        self.max_step_chars = int(max_step_chars)

    def reset(self, method) -> None:
        method.state["_semantic_step_boundaries"] = None
        method.state["_semantic_step_boundary_index"] = 0

    def __call__(self, ctx: VerifierContext) -> bool:
        if ctx.trajectory is None:
            return False
        boundaries = self._boundaries(ctx)
        boundary_index = int(ctx.state.get("_semantic_step_boundary_index", 0))
        if boundary_index >= len(boundaries):
            return False
        if len(ctx.prefix.content) < boundaries[boundary_index]:
            return False

        ctx.state["_semantic_step_boundary_index"] = boundary_index + 1
        ctx.state["_semantic_step_index"] = boundary_index + 1
        ctx.logs.setdefault("semantic_step_checkpoints", []).append(
            {
                "step": boundary_index + 1,
                "position": ctx.position,
                "boundary": boundaries[boundary_index],
            }
        )
        return True

    def replay_check(self, ctx, *, full_content, prefix_chars) -> bool:
        del full_content
        if ctx.trajectory is None:
            return False
        boundaries = self._boundaries(ctx)
        boundary_index = int(ctx.state.get("_semantic_step_boundary_index", 0))
        if boundary_index >= len(boundaries):
            return False
        if prefix_chars < boundaries[boundary_index]:
            return False

        ctx.state["_semantic_step_boundary_index"] = boundary_index + 1
        ctx.state["_semantic_step_index"] = boundary_index + 1
        ctx.logs.setdefault("semantic_step_checkpoints", []).append(
            {
                "step": boundary_index + 1,
                "position": ctx.position,
                "boundary": boundaries[boundary_index],
            }
        )
        return True

    def _boundaries(self, ctx: VerifierContext) -> list[int]:
        cached = ctx.state.get("_semantic_step_boundaries")
        if cached is not None:
            return list(cached)

        thinking_end = ctx.agent._thinking_end_char(ctx.trajectory)
        reasoning = ctx.trajectory.content[:thinking_end]
        spans = separate_step_spans(
            reasoning,
            min_step_chars=self.min_step_chars,
            max_step_chars=self.max_step_chars,
        )
        boundaries = [span.end for span in spans if span.end > 0]
        if not boundaries and reasoning:
            boundaries = [len(reasoning)]
        ctx.state["_semantic_step_boundaries"] = boundaries
        # ctx.logs.setdefault("semantic_steps", []).extend(
        #     {
        #         "step": idx,
        #         "start": span.start,
        #         "end": span.end,
        #         "chars": len(span.text),
        #         "text": span.text,
        #     }
        #     for idx, span in enumerate(spans, start=1)
        # )
        return boundaries
