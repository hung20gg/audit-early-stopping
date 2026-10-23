from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

_MODEL_CACHE: dict[tuple[str, str], "LoadedModel"] = {}
_MODEL_CACHE_LOCK = threading.RLock()


def resolve_device(device: str | None = None) -> str:
    """Resolve 'auto' (default) to the best available torch device, mirroring
    the `embedding_device: auto|cuda|mps|cpu` convention used by PUMA's
    SimilarityBackend."""
    import torch

    if device and device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class LoadedModel:
    model_id: str
    device: str
    model: Any
    tokenizer: Any

    @property
    def hidden_size(self) -> int:
        return int(self.model.config.hidden_size)

    def last_decoder_layer(self):
        """Return the base model's final decoder block, to be deep-copied by
        `TerminatorProbe`. Assumes a Llama-family layout (`model.model.layers`),
        which Qwen3 follows."""
        return self.model.model.layers[-1]

    def rotary_embedding(self):
        """Return the module that computes (cos, sin) position embeddings for
        decoder layers. The copied probe block is still causal and needs the
        same rotary embeddings as the frozen base model."""
        return getattr(self.model.model, "rotary_emb", None)


def load_model_and_tokenizer(
    model_id: str,
    device: str | None = None,
    dtype: str | None = None,
    trust_remote_code: bool = False,
) -> LoadedModel:
    """Load (and cache) a causal LM + tokenizer for direct hidden-state access.

    Cached per (model_id, resolved_device) so generation, curation, dataset
    precomputation and training within one process share a single copy of the
    frozen weights instead of reloading them repeatedly.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    resolved_device = resolve_device(device)
    cache_key = (model_id, resolved_device)

    with _MODEL_CACHE_LOCK:
        cached = _MODEL_CACHE.get(cache_key)
        if cached is not None:
            return cached

        tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=trust_remote_code
        )
        torch_dtype = getattr(torch, dtype) if dtype else "auto"
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
        )
        model.to(resolved_device)
        model.eval()

        loaded = LoadedModel(
            model_id=model_id,
            device=resolved_device,
            model=model,
            tokenizer=tokenizer,
        )
        _MODEL_CACHE[cache_key] = loaded
        return loaded


def build_chat_prompt(
    tokenizer: Any,
    question: str,
    system_prompt: str | None = None,
    enable_thinking: bool = True,
) -> str:
    """Render a rendered prompt string via the tokenizer's chat template,
    matching `agent/chat_template.py`'s `type: hf` rendering path so prompts
    stay consistent with the rest of the repo."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": question})
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def clear_model_cache() -> None:
    """Drop cached models (e.g. to free GPU memory between pipeline stages)."""
    with _MODEL_CACHE_LOCK:
        _MODEL_CACHE.clear()