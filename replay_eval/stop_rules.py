"""Stop rules (G_i)."""

from __future__ import annotations

from typing import Any

from replay_eval.method import StopRule, VerifierContext


class ThresholdGE(StopRule):
    """G_i = 1 iff verifier output >= threshold."""

    def __init__(self, threshold: float, scale_for_code: bool = False):
        self.threshold = float(threshold)
        self.scale_for_code = bool(scale_for_code)

    def __call__(self, verifier_output: Any, ctx: VerifierContext) -> bool:
        try:
            threshold = (
                ctx.agent.scale_threshold(self.threshold, ctx.kwargs)
                if self.scale_for_code
                else self.threshold
            )
            return float(verifier_output) >= threshold
        except (TypeError, ValueError):
            return False


class ThresholdLE(StopRule):
    """G_i = 1 iff verifier output <= threshold (e.g. variance-based rules)."""

    def __init__(self, threshold: float, scale_for_code: bool = False):
        self.threshold = float(threshold)
        self.scale_for_code = bool(scale_for_code)

    def __call__(self, verifier_output: Any, ctx: VerifierContext) -> bool:
        try:
            threshold = (
                ctx.agent.scale_threshold(self.threshold, ctx.kwargs)
                if self.scale_for_code
                else self.threshold
            )
            return float(verifier_output) <= threshold
        except (TypeError, ValueError):
            return False


class ThresholdGT(StopRule):
    """G_i = 1 iff verifier output > threshold."""

    def __init__(self, threshold: float):
        self.threshold = float(threshold)

    def __call__(self, verifier_output: Any, ctx: VerifierContext) -> bool:
        try:
            return float(verifier_output) > self.threshold
        except (TypeError, ValueError):
            return False


class CodeStopRule(StopRule):
    """Stop when CodeStop confidence or degeneration criteria fired."""

    def __call__(self, verifier_output: Any, ctx: VerifierContext) -> bool:
        return bool(ctx.state.get("_codestop_should_stop"))


class PUMAStopRule(StopRule):
    """Stop when the PUMA verifier marks the current reasoning step as final."""

    def __call__(self, verifier_output: Any, ctx: VerifierContext) -> bool:
        return bool(ctx.state.get("_puma_should_stop"))


class AlwaysStop(StopRule):
    """G_i = 1 unconditionally. Useful when S itself decides the stop."""

    def __call__(self, verifier_output: Any, ctx: VerifierContext) -> bool:
        return True
