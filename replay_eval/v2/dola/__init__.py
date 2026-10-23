"""DoLa detection and contrastive decoding for replay-v2."""

from replay_eval.v2.dola.decoding import DoLaDecoder, DoLaGeneration
from replay_eval.v2.dola.math import (
    contrast_logits,
    js_divergence_from_logits,
    layer_js_divergences,
    select_premature_layer,
)

__all__ = [
    "DoLaDecoder",
    "DoLaGeneration",
    "contrast_logits",
    "js_divergence_from_logits",
    "layer_js_divergences",
    "select_premature_layer",
]
