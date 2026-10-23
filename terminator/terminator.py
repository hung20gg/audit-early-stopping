"""
Training-side Terminator adapter for Qwen3, Qwen3.5, Gemma4E, and GPT-OSS.

Architecture:
    backbone hidden states
        -> one standalone Hugging Face decoder layer
        -> TerminatorFFN
        -> raw per-token termination logits

This module intentionally has no vLLM dependency. It uses the Hugging Face
model config so the Terminator can be trained and checkpointed separately.

Supported families:
    - Qwen3 (including deepseek-ai/DeepSeek-R1-0528-Qwen3-8B)
    - Qwen3.5 dense text decoder
    - Gemma4E dense text decoder
    - GPT-OSS

The extra decoder layer is deliberately selected from a full-attention layer
type. This avoids recurrent linear-attention state for Qwen3.5.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from transformers import PretrainedConfig


class SimpleRMSNorm(nn.Module):
    """RMSNorm with a `weight` parameter matching the probe checkpoint."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(
            variance + self.variance_epsilon
        )
        return self.weight * hidden_states.to(input_dtype)


class TerminatorFFN(nn.Module):
    """Per-position binary classifier returning raw logits."""

    def __init__(
        self,
        hidden_size: int,
        num_hidden_layers: int = 1,
        activation: str = "gelu",
        intermediate_size: Optional[int] = None,
        dropout: float = 0.0,
        rms_norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.norm = SimpleRMSNorm(hidden_size, eps=rms_norm_eps)

        if num_hidden_layers == 1:
            self.network = nn.Linear(hidden_size, 1)
            return

        if intermediate_size is None:
            intermediate_size = hidden_size * 2

        acts = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}
        if activation not in acts:
            raise ValueError(f"Unknown activation: {activation}")

        layers: list[nn.Module] = [
            nn.Linear(hidden_size, intermediate_size),
            acts[activation](),
            nn.Dropout(dropout),
        ]
        for _ in range(num_hidden_layers - 2):
            layers.extend(
                [
                    nn.Linear(intermediate_size, intermediate_size),
                    acts[activation](),
                    nn.Dropout(dropout),
                ]
            )
        layers.append(nn.Linear(intermediate_size, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.network(self.norm(hidden_states)).squeeze(-1)


def _get_text_config(config: PretrainedConfig) -> PretrainedConfig:
    text_config = getattr(config, "text_config", None)
    if text_config is not None:
        return text_config

    get_text_config = getattr(config, "get_text_config", None)
    if callable(get_text_config):
        try:
            resolved = get_text_config()
        except TypeError:
            resolved = get_text_config(decoder=True)
        if resolved is not None:
            return resolved
    return config


def _normalize_family(name: str) -> str:
    name = name.lower().replace("-", "_").replace(".", "_")
    aliases = {
        "qwen3": "qwen3",
        "qwen_3": "qwen3",
        "qwen3_5": "qwen3_5",
        "qwen35": "qwen3_5",
        "qwen_3_5": "qwen3_5",
        "gemma4": "gemma4e",
        "gemma_4": "gemma4e",
        "gemma4e": "gemma4e",
        "gemma4_e": "gemma4e",
        "gemma_4e": "gemma4e",
        "gpt_oss": "gpt_oss",
        "gptoss": "gpt_oss",
    }
    if name not in aliases:
        raise ValueError(f"Unsupported model family: {name}")
    return aliases[name]


def _detect_family(config: PretrainedConfig) -> str:
    model_type = str(getattr(config, "model_type", "")).lower()
    if model_type == "qwen3":
        return "qwen3"
    if model_type in {"qwen3_5", "qwen3_5_text"}:
        return "qwen3_5"
    if model_type in {"qwen3_5_moe", "qwen3_5_moe_text"}:
        raise NotImplementedError(
            "This file currently supports dense Qwen3.5 only. "
            "Qwen3.5-MoE uses a separate HF decoder implementation."
        )
    if model_type == "gemma4_text":
        if bool(getattr(config, "enable_moe_block", False)):
            raise NotImplementedError(
                "This file currently supports dense Gemma4E only. "
                "Gemma4 MoE uses expert and router blocks."
            )
        return "gemma4e"
    if model_type == "gpt_oss":
        return "gpt_oss"
    raise ValueError(
        f"Could not infer Terminator family from model_type={model_type!r}."
    )


def _choose_full_attention_layer_idx(config: PretrainedConfig) -> int:
    layer_types = getattr(config, "layer_types", None)
    if layer_types:
        for idx in range(len(layer_types) - 1, -1, -1):
            if layer_types[idx] == "full_attention":
                return idx
        raise ValueError("No full_attention layer exists in config.layer_types")
    return max(int(getattr(config, "num_hidden_layers", 1)) - 1, 0)


def _validate_full_attention_idx(config: PretrainedConfig, layer_idx: int) -> None:
    layer_types = getattr(config, "layer_types", None)
    if not layer_types:
        return
    if not 0 <= layer_idx < len(layer_types):
        raise IndexError(f"layer_idx={layer_idx} is out of range")
    if layer_types[layer_idx] != "full_attention":
        raise ValueError(
            f"layer_idx={layer_idx} is {layer_types[layer_idx]!r}; "
            "the standalone Terminator uses a full_attention decoder layer."
        )


def _make_causal_mask(
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    if hidden_states.dim() != 3:
        raise ValueError(
            "hidden_states must have shape [batch, seq_len, hidden_size]"
        )

    batch_size, seq_len, _ = hidden_states.shape
    dtype = hidden_states.dtype
    device = hidden_states.device
    min_value = torch.finfo(dtype).min

    mask = torch.full(
        (seq_len, seq_len), min_value, dtype=dtype, device=device
    )
    mask = torch.triu(mask, diagonal=1)
    mask = mask[None, None, :, :].expand(batch_size, 1, seq_len, seq_len)

    if attention_mask is not None:
        if tuple(attention_mask.shape) != (batch_size, seq_len):
            raise ValueError(
                "attention_mask must have shape [batch, seq_len], got "
                f"{tuple(attention_mask.shape)}"
            )
        key_padding = attention_mask.to(device=device).eq(0)
        mask = mask.masked_fill(key_padding[:, None, None, :], min_value)

    return mask


class TerminatorProbe(nn.Module):
    """Generic HF training-side Terminator for all supported families."""

    def __init__(
        self,
        config: PretrainedConfig,
        *,
        model_family: Optional[str] = None,
        layer_idx: Optional[int] = None,
        ffn_num_hidden_layers: int = 1,
        ffn_activation: str = "gelu",
        ffn_intermediate_size: Optional[int] = None,
        ffn_dropout: float = 0.0,
    ):
        super().__init__()

        config = _get_text_config(config)
        detected = _detect_family(config)
        family = detected if model_family is None else _normalize_family(model_family)
        if family != detected:
            raise ValueError(
                f"Requested {family!r}, but config looks like {detected!r}"
            )

        if family == "qwen3_5":
            # Make a private config for the extra Terminator layer.
            # We append a new layer type and explicitly make it full attention.
            config = copy.deepcopy(config)

            layer_types = list(config.layer_types)

            if layer_idx is None:
                layer_idx = len(layer_types)

            layer_idx = int(layer_idx)

            if layer_idx == len(layer_types):
                layer_types.append("full_attention")
            elif 0 <= layer_idx < len(layer_types):
                layer_types[layer_idx] = "full_attention"
            else:
                raise IndexError(
                    f"Invalid Qwen3.5 terminator layer_idx={layer_idx}"
                )

            config.layer_types = layer_types

        elif family == "gemma4e":
            # Gemma4E's final layers may reuse KV projections produced by
            # earlier base-model layers.  The Terminator owns one standalone
            # block, so give that block private KV projections instead.
            config = copy.deepcopy(config)
            config.num_kv_shared_layers = 0

            # Per-layer embeddings are computed by Gemma4TextModel from token
            # ids and input embeddings.  The probe receives hidden states only,
            # so its additional block cannot reproduce that auxiliary input.
            config.hidden_size_per_layer_input = 0

            if layer_idx is None:
                layer_idx = _choose_full_attention_layer_idx(config)

            layer_idx = int(layer_idx)
            _validate_full_attention_idx(config, layer_idx)

        else:
            if layer_idx is None:
                layer_idx = _choose_full_attention_layer_idx(config)

            layer_idx = int(layer_idx)
            _validate_full_attention_idx(config, layer_idx)

        self.config = config
        self.model_family = family
        self.layer_idx = layer_idx
        self.ffn_num_hidden_layers = int(ffn_num_hidden_layers)
        self.ffn_activation = ffn_activation
        self.ffn_intermediate_size = ffn_intermediate_size
        self.ffn_dropout = float(ffn_dropout)

        if family == "qwen3":
            from transformers.models.qwen3.modeling_qwen3 import (
                Qwen3DecoderLayer,
                Qwen3RotaryEmbedding,
            )
            self.block = Qwen3DecoderLayer(config=config, layer_idx=layer_idx)
            self.rotary_emb = Qwen3RotaryEmbedding(config=config)

        elif family == "qwen3_5":
            from transformers.models.qwen3_5.modeling_qwen3_5 import (
                Qwen3_5DecoderLayer,
                Qwen3_5TextRotaryEmbedding,
            )
            self.block = Qwen3_5DecoderLayer(config=config, layer_idx=layer_idx)
            self.rotary_emb = Qwen3_5TextRotaryEmbedding(config=config)

        elif family == "gemma4e":
            from transformers.models.gemma4.modeling_gemma4 import (
                Gemma4TextDecoderLayer,
                Gemma4TextRotaryEmbedding,
            )
            self.block = Gemma4TextDecoderLayer(config=config, layer_idx=layer_idx)
            self.rotary_emb = Gemma4TextRotaryEmbedding(config=config)

        elif family == "gpt_oss":
            from transformers.models.gpt_oss.modeling_gpt_oss import (
                GptOssDecoderLayer,
                GptOssRotaryEmbedding,
            )
            self.block = GptOssDecoderLayer(config=config, layer_idx=layer_idx)
            self.rotary_emb = GptOssRotaryEmbedding(config=config)

        else:
            raise AssertionError(f"Unhandled family: {family}")

        self.terminator_ffn = TerminatorFFN(
            hidden_size=config.hidden_size,
            num_hidden_layers=ffn_num_hidden_layers,
            activation=ffn_activation,
            intermediate_size=ffn_intermediate_size,
            dropout=ffn_dropout,
            rms_norm_eps=getattr(config, "rms_norm_eps", 1e-6),
        )

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_dir: str | Path,
        *,
        device: str | torch.device = "cpu",
        dtype: Optional[torch.dtype] = None,
        attention_backend: Optional[str] = None,
        strict: bool = True,
    ) -> "TerminatorProbe":
        """Reconstruct a probe from its directory-level ``config.json``."""
        from transformers import AutoConfig

        checkpoint_dir = Path(checkpoint_dir)
        deployment_config = json.loads(
            (checkpoint_dir / "config.json").read_text(encoding="utf-8")
        )
        required = {
            "base_model", "model_family", "layer_idx", "checkpoint_file",
        }
        missing = sorted(required - deployment_config.keys())
        if missing:
            raise ValueError(f"Terminator config.json is missing keys: {missing}")
        base_config = AutoConfig.from_pretrained(deployment_config["base_model"])
        backend = attention_backend or deployment_config.get("attention_backend")
        if backend:
            base_config._attn_implementation = backend
        probe = cls(
            base_config,
            model_family=deployment_config["model_family"],
            layer_idx=int(deployment_config["layer_idx"]),
            ffn_num_hidden_layers=int(deployment_config.get("ffn_num_hidden_layers", 1)),
            ffn_activation=deployment_config.get("ffn_activation", "gelu"),
            ffn_intermediate_size=deployment_config.get("ffn_intermediate_size"),
            ffn_dropout=float(deployment_config.get("ffn_dropout", 0.0)),
        )
        probe.load_checkpoint(
            str(checkpoint_dir / deployment_config["checkpoint_file"]),
            map_location=device,
            strict=strict,
            assign=dtype is None,
        )
        # assign=True preserves every saved tensor's dtype, including mixed or
        # quantized checkpoints. Conversion is strictly an explicit opt-in.
        probe.to(device=device)
        if dtype is not None:
            probe.to(dtype=dtype)
        return probe.eval()

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        new_tokens: Optional[int] = None,
        output_router_logits: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, Optional[torch.Tensor]]:
        if hidden_states.dim() != 3:
            raise ValueError(
                "hidden_states must have shape [batch, seq_len, hidden_size]"
            )

        batch_size, seq_len, _ = hidden_states.shape
        if position_ids is None:
            position_ids = torch.arange(
                seq_len, device=hidden_states.device, dtype=torch.long
            ).unsqueeze(0).expand(batch_size, -1)

        attention_backend = getattr(self.config, "_attn_implementation", None)
        if attention_backend == "flash_attention_2":
            # FlashAttention applies causality internally and accepts only the
            # 2-D padding mask. Avoid materializing a [B, 1, T, T] mask.
            causal_mask = attention_mask
            if causal_mask is not None and bool(torch.all(causal_mask)):
                causal_mask = None
        elif attention_backend == "sdpa" and (
            attention_mask is None or bool(torch.all(attention_mask))
        ):
            # SDPA selects its fused causal kernel only when mask=None.
            causal_mask = None
        else:
            causal_mask = _make_causal_mask(hidden_states, attention_mask)
        if self.model_family == "gemma4e":
            position_embeddings = self.rotary_emb(
                hidden_states,
                position_ids,
                layer_type=self.config.layer_types[self.layer_idx],
            )
        else:
            position_embeddings = self.rotary_emb(hidden_states, position_ids)

        kwargs = dict(
            hidden_states=hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
        )
        router_logits = None
        router = None
        if output_router_logits:
            mlp = getattr(self.block, "mlp", None)
            router = getattr(mlp, "router", None)
            if router is None:
                router = getattr(mlp, "gate", None)

        hook = None
        if router is not None:
            def capture_router_logits(_module, _inputs, output):
                nonlocal router_logits
                # HF MoE routers return (full logits, top-k weights, indices).
                # The full logits are required by the load-balancing loss.
                router_logits = output[0] if isinstance(output, tuple) else output

            hook = router.register_forward_hook(capture_router_logits)

        try:
            if self.model_family == "qwen3_5":
                block_output = self.block(**kwargs)
            elif self.model_family == "gemma4e":
                block_output = self.block(**kwargs, shared_kv_states={})
            else:
                block_output = self.block(**kwargs, use_cache=False)
        finally:
            if hook is not None:
                hook.remove()

        # Decoder implementations normally return a tensor, but keep the
        # adapter compatible with MoE variants that expose extra outputs.
        hidden_states = (
            block_output[0] if isinstance(block_output, tuple) else block_output
        )

        if new_tokens is not None:
            if new_tokens <= 0:
                raise ValueError("new_tokens must be positive")
            hidden_states = hidden_states[:, -new_tokens:, :]

        logits = self.terminator_ffn(hidden_states)
        if output_router_logits:
            return logits, router_logits
        return logits

    @torch.no_grad()
    def predict_proba(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        return torch.sigmoid(self.forward(hidden_states, **kwargs))

    def save_checkpoint(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load_checkpoint(
        self,
        path: str,
        *,
        map_location: str | torch.device = "cpu",
        strict: bool = True,
        assign: bool = False,
    ):
        state_dict = torch.load(path, map_location=map_location, weights_only=True)
        if (
            isinstance(state_dict, dict)
            and "state_dict" in state_dict
            and isinstance(state_dict["state_dict"], dict)
        ):
            state_dict = state_dict["state_dict"]
        return self.load_state_dict(state_dict, strict=strict, assign=assign)


class Qwen3Terminator(TerminatorProbe):
    def __init__(self, config: PretrainedConfig, **kwargs):
        super().__init__(config, model_family="qwen3", **kwargs)


class DeepSeekR1Qwen3Terminator(Qwen3Terminator):
    """DeepSeek-R1-0528-Qwen3-8B probe using its native Qwen3 config.

    Keep the checkpoint's YaRN RoPE settings: this distilled model uses
    Qwen3ForCausalLM, not the DeepSeek V2/V3 decoder architecture. The
    canonical checkpoint family remains ``qwen3`` for both runtimes.
    """


class Qwen3_5Terminator(TerminatorProbe):
    def __init__(self, config: PretrainedConfig, **kwargs):
        super().__init__(config, model_family="qwen3_5", **kwargs)


class GptOssTerminator(TerminatorProbe):
    def __init__(self, config: PretrainedConfig, **kwargs):
        super().__init__(config, model_family="gpt_oss", **kwargs)


class Gemma4ETerminator(TerminatorProbe):
    def __init__(self, config: PretrainedConfig, **kwargs):
        super().__init__(config, model_family="gemma4e", **kwargs)


__all__ = [
    "SimpleRMSNorm",
    "TerminatorFFN",
    "TerminatorProbe",
    "Qwen3Terminator",
    "DeepSeekR1Qwen3Terminator",
    "Qwen3_5Terminator",
    "Gemma4ETerminator",
    "GptOssTerminator",
]
