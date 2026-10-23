"""
vLLM-side Terminator adapter for Qwen3, Qwen3.5, Gemma4E, and GPT-OSS.

This is the inference counterpart of terminator_models.py.

Architecture:
    vLLM backbone hidden/residual streams
        -> one extra vLLM decoder block
        -> reconstructed complete block output
        -> TerminatorFFN
        -> termination logit/probability

VllmConfig must be passed down from your custom vLLM model constructor. Do not
create a separate EngineArgs/VllmConfig inside this adapter.

Supported families:
    - Qwen3 (including deepseek-ai/DeepSeek-R1-0528-Qwen3-8B)
    - Qwen3.5 dense text decoder
    - Gemma4E dense text decoder
    - GPT-OSS

The adapter is unquantized by default. Set use_quant_config=True only if the
adapter checkpoint itself was prepared for the base model's quantization.
"""

from __future__ import annotations

import copy
import inspect
from collections.abc import Iterable
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.models.gpt_oss import (
    TransformerBlock as GPTOSSDecoderLayer,
)
from vllm.model_executor.models.gemma4 import Gemma4DecoderLayer
from vllm.model_executor.models.qwen3 import Qwen3DecoderLayer
from vllm.model_executor.models.qwen3_5 import Qwen3_5DecoderLayer
from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper


class SimpleRMSNorm(nn.Module):
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


# HF Qwen/Qwen3.5 checkpoint names -> packed vLLM parameter names.
QWEN_HF_TO_VLLM_MAPPER = WeightsMapper(
    orig_to_new_stacked={
        ".q_proj": (".qkv_proj", "q"),
        ".k_proj": (".qkv_proj", "k"),
        ".v_proj": (".qkv_proj", "v"),
        ".mlp.gate_proj": (".mlp.gate_up_proj", 0),
        ".mlp.up_proj": (".mlp.gate_up_proj", 1),
    }
)


def _get_text_config(vllm_config: VllmConfig):
    config = getattr(vllm_config.model_config, "hf_text_config", None)
    if config is not None:
        return config
    config = vllm_config.model_config.hf_config
    return getattr(config, "text_config", config)


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


def _detect_family(config) -> str:
    model_type = str(getattr(config, "model_type", "")).lower()
    if model_type == "qwen3":
        return "qwen3"
    if model_type in {"qwen3_5", "qwen3_5_text"}:
        return "qwen3_5"
    if model_type in {"qwen3_5_moe", "qwen3_5_moe_text"}:
        raise NotImplementedError(
            "This pair currently supports dense Qwen3.5 only."
        )
    if model_type == "gemma4_text":
        if bool(getattr(config, "enable_moe_block", False)):
            raise NotImplementedError(
                "This adapter currently supports dense Gemma4E only. "
                "Gemma4 MoE requires a separate checkpoint loader."
            )
        return "gemma4e"
    if model_type == "gpt_oss":
        return "gpt_oss"
    raise ValueError(f"Unsupported model_type={model_type!r}")


def _choose_full_attention_layer_idx(config) -> int:
    layer_types = getattr(config, "layer_types", None)
    if layer_types:
        for idx in range(len(layer_types) - 1, -1, -1):
            if layer_types[idx] == "full_attention":
                return idx
        raise ValueError("No full_attention layer exists in config.layer_types")
    return max(int(getattr(config, "num_hidden_layers", 1)) - 1, 0)


def _validate_full_attention_idx(config, layer_idx: int) -> None:
    layer_types = getattr(config, "layer_types", None)
    if not layer_types:
        return
    if not 0 <= layer_idx < len(layer_types):
        raise IndexError(f"layer_idx={layer_idx} is out of range")
    if layer_types[layer_idx] != "full_attention":
        raise ValueError(
            f"layer_idx={layer_idx} is {layer_types[layer_idx]!r}; "
            "this adapter uses a full_attention decoder layer."
        )


def _adapter_vllm_config(
    vllm_config: VllmConfig,
    use_quant_config: bool,
) -> VllmConfig:
    """Shallow-copy config so adapter quantization can be disabled locally."""
    if use_quant_config or vllm_config.quant_config is None:
        return vllm_config
    local = copy.copy(vllm_config)
    try:
        local.quant_config = None
    except (AttributeError, TypeError):
        object.__setattr__(local, "quant_config", None)
    return local


def _copy_exact(param, tensor: torch.Tensor, name: str) -> None:
    if tuple(param.shape) != tuple(tensor.shape):
        raise ValueError(
            f"Shape mismatch for {name}: target={tuple(param.shape)}, "
            f"source={tuple(tensor.shape)}"
        )
    param.data.copy_(tensor.to(device=param.device, dtype=param.dtype))


def _load_param(param, tensor: torch.Tensor, name: str, shard_id=None) -> None:
    """Call vLLM weight loaders across old/new shard-id signatures."""
    loader = getattr(param, "weight_loader", None)
    if loader is None:
        if shard_id is not None:
            raise RuntimeError(
                f"{name} requires a shard-aware weight_loader but has none"
            )
        _copy_exact(param, tensor, name)
        return

    if shard_id is None:
        loader(param, tensor)
        return

    try:
        loader_params = inspect.signature(loader).parameters
    except (TypeError, ValueError):
        loader_params = {}

    if "loaded_shard_id" in loader_params:
        loader(param, tensor, loaded_shard_id=shard_id)
    elif "shard_id" in loader_params:
        loader(param, tensor, shard_id=shard_id)
    else:
        loader(param, tensor, shard_id)


def _find_param(params: dict, *names: str) -> str:
    for name in names:
        if name in params:
            return name
    raise KeyError("Could not find vLLM parameter: " + " or ".join(names))


class VLLMTerminatorProbe(nn.Module):
    """Generic vLLM Terminator for all supported model families."""

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "terminator",
        model_family: Optional[str] = None,
        layer_idx: Optional[int] = None,
        ffn_num_hidden_layers: int = 1,
        ffn_activation: str = "gelu",
        ffn_intermediate_size: Optional[int] = None,
        ffn_dropout: float = 0.0,
        use_quant_config: bool = False,
    ):
        super().__init__()

        config = _get_text_config(vllm_config)
        detected = _detect_family(config)
        family = detected if model_family is None else _normalize_family(model_family)
        if family != detected:
            raise ValueError(
                f"Requested {family!r}, but config looks like {detected!r}"
            )

        if family == "qwen3_5":
            # Terminator is an extra decoder layer after the base model.
            if layer_idx is None:
                layer_idx = int(config.num_hidden_layers)

            layer_idx = int(layer_idx)

        elif family == "gemma4e":
            # The probe owns a standalone block and receives hidden states,
            # not the base model's shared KV cache or per-layer embeddings.
            # Match the training-side adapter by disabling both features on a
            # private config used only to construct this extra block.
            config = copy.deepcopy(config)
            config.num_kv_shared_layers = 0
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
        self.vllm_config = vllm_config
        self.model_family = family
        self.layer_idx = layer_idx
        self.use_quant_config = use_quant_config

        adapter_config = _adapter_vllm_config(vllm_config, use_quant_config)
        quant_config = adapter_config.quant_config if use_quant_config else None

        # A numeric index in the prefix is important because vLLM attention
        # implementations call extract_layer_index(prefix).
        block_prefix = f"{prefix}.layers.{layer_idx}"

        if family == "qwen3":
            self.block = Qwen3DecoderLayer(
                config=config,
                cache_config=adapter_config.cache_config,
                quant_config=quant_config,
                prefix=block_prefix,
                per_layer_sliding_window=None,
            )
        elif family == "qwen3_5":
            q35_params = inspect.signature(
                Qwen3_5DecoderLayer.__init__
            ).parameters
            if "vllm_config" in q35_params:
                self.block = Qwen3_5DecoderLayer(
                    vllm_config=adapter_config,
                    layer_type="full_attention",
                    prefix=block_prefix,
                )
            elif "config" in q35_params and "layer_idx" in q35_params:
                # Compatibility fallback for versions exposing a more HF-like
                # internal constructor.
                q35_kwargs = {
                    "config": config,
                    "layer_idx": layer_idx,
                }
                if "prefix" in q35_params:
                    q35_kwargs["prefix"] = block_prefix
                self.block = Qwen3_5DecoderLayer(**q35_kwargs)
            else:
                raise TypeError(
                    "Unsupported Qwen3_5DecoderLayer constructor in this "
                    "vLLM version: " + str(inspect.signature(
                        Qwen3_5DecoderLayer.__init__
                    ))
                )
        elif family == "gemma4e":
            self.block = Gemma4DecoderLayer(
                config=config,
                cache_config=adapter_config.cache_config,
                quant_config=quant_config,
                prefix=block_prefix,
            )
        elif family == "gpt_oss":
            gpt_params = inspect.signature(
                GPTOSSDecoderLayer.__init__
            ).parameters
            if "vllm_config" in gpt_params:
                self.block = GPTOSSDecoderLayer(
                    vllm_config=adapter_config,
                    quant_config=quant_config,
                    prefix=block_prefix,
                )
            elif "config" in gpt_params:
                # vLLM 0.10.x GPT-OSS used config=GptOssConfig directly.
                self.block = GPTOSSDecoderLayer(
                    config=config,
                    quant_config=quant_config,
                    prefix=block_prefix,
                )
            else:
                raise TypeError(
                    "Unsupported GPTOSS TransformerBlock constructor in this "
                    "vLLM version: " + str(inspect.signature(
                        GPTOSSDecoderLayer.__init__
                    ))
                )
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

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        *,
        new_tokens: Optional[int] = 1,
        return_probabilities: bool = True,
    ) -> torch.Tensor:
        forward_params = inspect.signature(
            self.block.forward
        ).parameters

        if "residual" in forward_params:
            block_out = self.block(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
            )
            if not (
                isinstance(block_out, tuple) and len(block_out) == 2
            ):
                raise RuntimeError(
                    "vLLM decoder accepted residual= but did not return "
                    "(hidden_states, residual)."
                )
            block_hidden, block_residual = block_out
            if self.model_family == "gemma4e":
                # Gemma4 reconstructs its residual stream inside the layer and
                # returns the complete output with residual=None.
                complete_output = block_hidden
            elif block_residual is None:
                raise RuntimeError("Decoder block returned residual=None")
            else:
                complete_output = block_hidden + block_residual
        else:
            # Older GPT-OSS vLLM blocks consume and return a complete hidden
            # state instead of vLLM's split hidden/residual streams.
            complete_input = (
                hidden_states
                if residual is None
                else hidden_states + residual
            )
            complete_output = self.block(
                hidden_states=complete_input,
                positions=positions,
            )
            if isinstance(complete_output, tuple):
                complete_output = complete_output[0]
        if new_tokens is not None:
            if new_tokens <= 0:
                raise ValueError("new_tokens must be positive")
            complete_output = complete_output[-new_tokens:]

        logits = self.terminator_ffn(complete_output)
        return torch.sigmoid(logits) if return_probabilities else logits

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        weights = [
            (name, tensor)
            for name, tensor in weights
            if not name.startswith("rotary_emb.")
        ]

        if self.model_family in {"qwen3", "qwen3_5"}:
            return AutoWeightsLoader(self).load_weights(
                weights,
                mapper=QWEN_HF_TO_VLLM_MAPPER,
            )

        if self.model_family == "gemma4e":
            if self.use_quant_config:
                raise NotImplementedError(
                    "The Gemma4E Terminator checkpoint loader currently "
                    "targets unquantized adapter weights only."
                )
            return self._load_gemma4e_weights(weights)

        if self.model_family == "gpt_oss":
            if self.use_quant_config:
                raise NotImplementedError(
                    "The custom HF -> vLLM GPT-OSS adapter loader currently "
                    "targets unquantized adapter weights only."
                )
            return self._load_gpt_oss_weights(weights)

        raise AssertionError(f"Unhandled family: {self.model_family}")

    def _load_gemma4e_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        """Load the dense HF Gemma4E block into vLLM's packed layout."""
        weights = list(weights)
        loaded: set[str] = set()
        remaining: list[tuple[str, torch.Tensor]] = []

        # Gemma4's layer scalar is a buffer in both implementations, while
        # AutoWeightsLoader traverses parameters. Load it explicitly.
        for name, tensor in weights:
            if name == "block.layer_scalar":
                _copy_exact(self.block.layer_scalar, tensor, name)
                loaded.add(name)
            else:
                remaining.append((name, tensor))

        loaded.update(
            AutoWeightsLoader(self).load_weights(
                remaining,
                mapper=QWEN_HF_TO_VLLM_MAPPER,
            )
        )
        return loaded

    def _load_gpt_oss_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        """
        Convert the HF GPT-OSS decoder block into vLLM's packed/FusedMoE layout.

        Supported: unquantized adapter + tensor parallelism.
        Not supported here: expert parallelism for the separately trained
        adapter, because global expert IDs must be remapped to rank-local IDs.
        """
        if getattr(
            self.vllm_config.parallel_config,
            "enable_expert_parallel",
            False,
        ):
            raise NotImplementedError(
                "GPT-OSS Terminator checkpoint conversion currently supports "
                "tensor parallelism without expert parallelism."
            )

        state = dict(weights)
        params = dict(self.named_parameters())
        handled: set[str] = set()
        loaded: set[str] = set()

        # HF self_attn.{q,k,v}_proj -> vLLM packed QKV.
        # GPT-OSS has used both attn.qkv_proj.* and attn.qkv.* as
        # Python parameter names across vLLM releases.
        for proj, shard in (("q_proj", "q"), ("k_proj", "k"), ("v_proj", "v")):
            for suffix in ("weight", "bias"):
                src = f"block.self_attn.{proj}.{suffix}"
                if src not in state:
                    continue
                dst = _find_param(
                    params,
                    f"block.attn.qkv_proj.{suffix}",
                    f"block.attn.qkv.{suffix}",
                )
                _load_param(params[dst], state[src], dst, shard_id=shard)
                handled.add(src)
                loaded.add(dst)

        # Attention output projection.
        for suffix in ("weight", "bias"):
            src = f"block.self_attn.o_proj.{suffix}"
            if src not in state:
                continue
            dst = f"block.attn.o_proj.{suffix}"
            if dst not in params:
                raise KeyError(f"Missing target parameter {dst}")
            _load_param(params[dst], state[src], dst)
            handled.add(src)
            loaded.add(dst)

        # Attention sinks are sharded by local TP heads.
        src = "block.self_attn.sinks"
        if src in state:
            dst = "block.attn.sinks"
            if dst not in params:
                raise KeyError(f"Missing target parameter {dst}")
            target = params[dst]
            source = state[src].reshape(-1)
            tp_rank = get_tensor_model_parallel_rank()
            local_heads = target.numel()
            start = tp_rank * local_heads
            stop = start + local_heads
            _copy_exact(target, source[start:stop].reshape(target.shape), dst)
            handled.add(src)
            loaded.add(dst)

        # Norm names differ between GPT-OSS vLLM releases.
        norm_mapping = {
            "block.input_layernorm.weight": (
                "block.input_layernorm.weight",
                "block.attn.norm.weight",
            ),
            "block.post_attention_layernorm.weight": (
                "block.post_attention_layernorm.weight",
                "block.mlp.norm.weight",
            ),
        }
        for src, candidates in norm_mapping.items():
            if src not in state:
                continue
            dst = _find_param(params, *candidates)
            _load_param(params[dst], state[src], dst)
            handled.add(src)
            loaded.add(dst)

        for name in (
            "block.mlp.router.weight",
            "block.mlp.router.bias",
        ):
            if name in state:
                if name not in params:
                    raise KeyError(f"Missing target parameter {name}")
                _load_param(params[name], state[name], name)
                handled.add(name)
                loaded.add(name)

        # TerminatorFFN is an ordinary PyTorch module with identical layout.
        for name, tensor in state.items():
            if not name.startswith("terminator_ffn."):
                continue
            if name not in params:
                raise KeyError(f"Missing target parameter {name}")
            _copy_exact(params[name], tensor, name)
            handled.add(name)
            loaded.add(name)

        # HF expert layout:
        #   gate_up_proj      [E, H, 2I]
        #   gate_up_proj_bias [E, 2I]
        #   down_proj         [E, I, H]
        #   down_proj_bias    [E, H]
        #
        # vLLM unquantized FusedMoE layout:
        #   w13_weight [E, 2I/TP, H]
        #   w13_bias   [E, 2I/TP]
        #   w2_weight  [E, H, I/TP]
        #   w2_bias    [E, H]
        src_w13 = "block.mlp.experts.gate_up_proj"
        src_b13 = "block.mlp.experts.gate_up_proj_bias"
        src_w2 = "block.mlp.experts.down_proj"
        src_b2 = "block.mlp.experts.down_proj_bias"

        for name in (src_w13, src_b13, src_w2, src_b2):
            if name not in state:
                raise KeyError(f"GPT-OSS checkpoint is missing {name}")

        dst_w13 = _find_param(
            params,
            "block.mlp.experts.routed_experts.w13_weight",
            "block.mlp.experts.w13_weight",
        )
        dst_b13 = _find_param(
            params,
            "block.mlp.experts.routed_experts.w13_bias",
            "block.mlp.experts.w13_bias",
        )
        dst_w2 = _find_param(
            params,
            "block.mlp.experts.routed_experts.w2_weight",
            "block.mlp.experts.w2_weight",
        )
        dst_b2 = _find_param(
            params,
            "block.mlp.experts.routed_experts.w2_bias",
            "block.mlp.experts.w2_bias",
        )

        tp_rank = get_tensor_model_parallel_rank()
        tp_world = get_tensor_model_parallel_world_size()

        full_w13 = state[src_w13].transpose(1, 2).contiguous()
        target = params[dst_w13]
        if (
            full_w13.shape[0] != target.shape[0]
            or full_w13.shape[2] != target.shape[2]
            or full_w13.shape[1] != target.shape[1] * tp_world
        ):
            raise ValueError(
                f"GPT-OSS w13 shape mismatch: HF {tuple(full_w13.shape)}, "
                f"vLLM local {tuple(target.shape)}, TP={tp_world}"
            )
        width = target.shape[1]
        _copy_exact(
            target,
            full_w13[:, tp_rank * width : (tp_rank + 1) * width, :],
            dst_w13,
        )
        handled.add(src_w13)
        loaded.add(dst_w13)

        full_b13 = state[src_b13]
        target = params[dst_b13]
        if (
            full_b13.shape[0] != target.shape[0]
            or full_b13.shape[1] != target.shape[1] * tp_world
        ):
            raise ValueError("GPT-OSS w13 bias TP shape mismatch")
        width = target.shape[1]
        _copy_exact(
            target,
            full_b13[:, tp_rank * width : (tp_rank + 1) * width],
            dst_b13,
        )
        handled.add(src_b13)
        loaded.add(dst_b13)

        full_w2 = state[src_w2].transpose(1, 2).contiguous()
        target = params[dst_w2]
        if (
            full_w2.shape[0] != target.shape[0]
            or full_w2.shape[1] != target.shape[1]
            or full_w2.shape[2] != target.shape[2] * tp_world
        ):
            raise ValueError("GPT-OSS w2 TP shape mismatch")
        width = target.shape[2]
        _copy_exact(
            target,
            full_w2[:, :, tp_rank * width : (tp_rank + 1) * width],
            dst_w2,
        )
        handled.add(src_w2)
        loaded.add(dst_w2)

        # Row-parallel down-projection bias is applied only once.
        target = params[dst_b2]
        if tp_rank == 0:
            _copy_exact(target, state[src_b2], dst_b2)
        else:
            target.data.zero_()
        handled.add(src_b2)
        loaded.add(dst_b2)

        unexpected = [name for name in state if name not in handled]
        if unexpected:
            raise KeyError(
                "Unhandled GPT-OSS checkpoint tensors: "
                + ", ".join(unexpected[:20])
            )

        return loaded

    def load_checkpoint(
        self,
        path: str | Path,
        *,
        map_location: str | torch.device = "cpu",
    ) -> set[str]:
        state_dict = torch.load(
            str(path), map_location=map_location, weights_only=True
        )
        if (
            isinstance(state_dict, dict)
            and "state_dict" in state_dict
            and isinstance(state_dict["state_dict"], dict)
        ):
            state_dict = state_dict["state_dict"]
        if not isinstance(state_dict, dict):
            raise TypeError("Expected a state_dict mapping")

        if state_dict and all(name.startswith("module.") for name in state_dict):
            state_dict = {
                name[len("module.") :]: tensor
                for name, tensor in state_dict.items()
            }
        return self.load_weights(state_dict.items())


class VLLMQwen3Terminator(VLLMTerminatorProbe):
    def __init__(self, *, vllm_config: VllmConfig, **kwargs):
        super().__init__(
            vllm_config=vllm_config, model_family="qwen3", **kwargs
        )


class VLLMDeepSeekR1Qwen3Terminator(VLLMQwen3Terminator):
    """DeepSeek-R1-0528-Qwen3-8B with Qwen3 blocks and packed weights.

    Pass the original model's VllmConfig so its YaRN RoPE settings are
    preserved. HF checkpoints retain the canonical ``qwen3`` family.
    """


class VLLMQwen3_5Terminator(VLLMTerminatorProbe):
    def __init__(self, *, vllm_config: VllmConfig, **kwargs):
        super().__init__(
            vllm_config=vllm_config, model_family="qwen3_5", **kwargs
        )


class VLLMGemma4ETerminator(VLLMTerminatorProbe):
    def __init__(self, *, vllm_config: VllmConfig, **kwargs):
        super().__init__(
            vllm_config=vllm_config, model_family="gemma4e", **kwargs
        )


class VLLMGptOssTerminator(VLLMTerminatorProbe):
    def __init__(self, *, vllm_config: VllmConfig, **kwargs):
        super().__init__(
            vllm_config=vllm_config, model_family="gpt_oss", **kwargs
        )


__all__ = [
    "SimpleRMSNorm",
    "TerminatorFFN",
    "VLLMTerminatorProbe",
    "VLLMQwen3Terminator",
    "VLLMDeepSeekR1Qwen3Terminator",
    "VLLMQwen3_5Terminator",
    "VLLMGemma4ETerminator",
    "VLLMGptOssTerminator",
]
