"""Backend-independent DoLa distribution operations."""

from __future__ import annotations

import math
from typing import Any


def js_divergence_from_logits(final_logits: Any, premature_logits: Any) -> Any:
    """Return Jensen-Shannon divergence along the vocabulary dimension."""
    import torch

    final = torch.log_softmax(final_logits.float(), dim=-1)
    premature = torch.log_softmax(premature_logits.float(), dim=-1)
    log_mean = torch.logaddexp(final, premature) - math.log(2.0)
    return 0.5 * (
        torch.sum(torch.exp(final) * (final - log_mean), dim=-1)
        + torch.sum(torch.exp(premature) * (premature - log_mean), dim=-1)
    )


def layer_js_divergences(final_logits: Any, premature_logits: Any) -> Any:
    """Vectorized JSD for ``[batch, layers, vocab]`` premature logits."""
    import torch

    if premature_logits.ndim != final_logits.ndim + 1:
        raise ValueError(
            "premature_logits must add one candidate-layer dimension to final_logits"
        )
    final = torch.log_softmax(final_logits.float(), dim=-1).unsqueeze(1)
    premature = torch.log_softmax(premature_logits.float(), dim=-1)
    log_mean = torch.logaddexp(final, premature) - math.log(2.0)
    return 0.5 * (
        torch.sum(torch.exp(final) * (final - log_mean), dim=-1)
        + torch.sum(torch.exp(premature) * (premature - log_mean), dim=-1)
    )


def contrast_logits(final_logits: Any, premature_logits: Any, *, alpha: float = 0.1) -> Any:
    """Compute DoLa scores with the adaptive plausibility constraint."""
    import torch

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    final = torch.log_softmax(final_logits.float(), dim=-1)
    premature = torch.log_softmax(premature_logits.float(), dim=-1)
    scores = final - premature
    if alpha > 0.0:
        cutoff = final.amax(dim=-1, keepdim=True) + math.log(alpha)
        scores = scores.masked_fill(final < cutoff, float("-inf"))
    return scores


def select_premature_layer(
    final_logits: Any, premature_logits: dict[int, Any]
) -> tuple[Any, Any]:
    """Select the candidate with maximum final-vs-layer JSD per batch item."""
    import torch

    if not premature_logits:
        raise ValueError("At least one premature layer is required")
    layer_ids = list(premature_logits)
    stacked = torch.stack([premature_logits[layer] for layer in layer_ids], dim=1)
    divergences = layer_js_divergences(final_logits, stacked)
    selected_indices = divergences.argmax(dim=-1)
    gather_index = selected_indices[:, None, None].expand(-1, 1, stacked.shape[-1])
    selected_logits = stacked.gather(1, gather_index).squeeze(1)
    layer_tensor = torch.as_tensor(layer_ids, device=selected_indices.device)
    return layer_tensor[selected_indices], selected_logits


__all__ = [
    "contrast_logits",
    "js_divergence_from_logits",
    "layer_js_divergences",
    "select_premature_layer",
]
