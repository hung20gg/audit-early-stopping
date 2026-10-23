"""Replay-v2 method exposing precomputed DoLa disagreement features."""

from __future__ import annotations

import math
from typing import Any

from replay_eval.checkpoints import EveryNTokens
from replay_eval.method import EarlyExitMethod, Verifier, VerifierContext
from replay_eval.stop_rules import ThresholdGE


def _trace(ctx: VerifierContext) -> Any:
    trace = getattr(ctx.agent, "hidden_trace", None)
    if trace is None:
        raise RuntimeError(
            "DoLa disagreement is configured, but extraction did not provide "
            "an example-scoped hidden trace"
        )
    return trace


class DolaDisagreementVerifier(Verifier):
    """Replay precomputed final-vs-premature layer JSD as a causal feature."""

    uses_hidden_trace = True
    permits_token_state_release = True

    def __init__(self, *, window_size: int = 1) -> None:
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        self.window_size = int(window_size)

    def reset(self, method: EarlyExitMethod) -> None:
        del method

    @staticmethod
    def prepare_hidden_trace(trace: Any) -> None:
        if "dola_jsd.max" not in trace.model_outputs:
            raise RuntimeError(
                "DoLa verifier requires extraction.dola_layers and worker JSD scores"
            )

    def __call__(self, ctx: VerifierContext) -> float:
        trace = _trace(ctx)
        self.prepare_hidden_trace(trace)
        end = int(ctx.position)
        start = max(0, end - self.window_size)
        maximum = trace.model_outputs["dola_jsd.max"]
        window = maximum[start:end]
        score = float(sum(window) / len(window)) if window else 0.0
        previous = ctx.state.get("_dola_previous_score")
        ctx.state["_dola_previous_score"] = score
        selected = trace.model_outputs.get("dola_jsd.selected_layer", [])
        per_layer = {
            key.removeprefix("dola_jsd.layer_"): float(values[end - 1])
            for key, values in trace.model_outputs.items()
            if key.startswith("dola_jsd.layer_") and end > 0
        }
        ctx.logs.setdefault("dola_disagreement", []).append(
            {
                "position": end,
                "window_start": start + 1,
                "jsd": score,
                "delta_jsd": None if previous is None else score - float(previous),
                "selected_layer": int(selected[end - 1]) if selected and end else None,
                "layer_jsd": per_layer,
            }
        )
        return score


def build_dola_disagreement_method(
    name: str = "dola_disagreement",
    *,
    check_every: int = 32,
    window_size: int = 1,
    threshold: float = math.inf,
) -> EarlyExitMethod:
    """Build the DoLa detector; infinite threshold makes it diagnostic-only."""
    if check_every < 1:
        raise ValueError("check_every must be >= 1")
    return EarlyExitMethod(
        name=name,
        checkpoint=EveryNTokens(check_every),
        verifier=DolaDisagreementVerifier(window_size=window_size),
        stop_rule=ThresholdGE(float(threshold)),
    )


__all__ = ["DolaDisagreementVerifier", "build_dola_disagreement_method"]
