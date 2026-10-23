"""Interactive chat entry point for normal and DoLa Hugging Face decoding."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import logging
import sys
import time
from typing import Any

from replay_eval.v2.dola.decoding import DoLaDecoder, DoLaGeneration


LOGGER = logging.getLogger(__name__)


def parse_layer_spec(value: str, num_hidden_layers: int) -> list[int]:
    """Resolve ``auto`` or comma-separated Transformers hidden-state indices."""
    if num_hidden_layers < 2:
        raise ValueError("DoLa requires a model with at least two hidden layers")
    if value.strip().casefold() == "auto":
        # hidden_states[0] is embeddings and hidden_states[num_hidden_layers]
        # is mature. Choose four roughly evenly spaced premature states.
        candidates = {
            max(1, min(num_hidden_layers - 1, round(num_hidden_layers * fraction)))
            for fraction in (0.2, 0.4, 0.6, 0.8)
        }
        return sorted(candidates)
    try:
        layers = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError("--layers must be 'auto' or comma-separated integers") from exc
    if not layers:
        raise ValueError("--layers must contain at least one candidate")
    if len(layers) != len(set(layers)):
        raise ValueError("--layers must not contain duplicates")
    invalid = [layer for layer in layers if layer < 0 or layer >= num_hidden_layers]
    if invalid:
        raise ValueError(
            f"Invalid candidate layers {invalid}; expected hidden-state indices in "
            f"[0, {num_hidden_layers - 1}]"
        )
    return layers


def _text_config(config: Any) -> Any:
    value = getattr(config, "text_config", None)
    return value if value is not None else config


def _render_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> list[int]:
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return tokenizer.encode(rendered, add_special_tokens=False)


def _normal_generate(
    model: Any,
    tokenizer: Any,
    prompt_ids: list[int],
    args: argparse.Namespace,
) -> list[int]:
    import torch

    device = next(model.parameters()).device
    inputs = torch.tensor([prompt_ids], device=device, dtype=torch.long)
    kwargs: dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0.0,
    }
    if args.temperature > 0.0:
        kwargs.update(
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )
    if args.seed is not None:
        torch.manual_seed(args.seed)
    with torch.inference_mode():
        output = model.generate(inputs, **kwargs)
    return [int(value) for value in output[0, inputs.shape[-1] :].tolist()]


def _diagnostics(result: DoLaGeneration) -> dict[str, Any]:
    counts = Counter(result.selected_layers)
    means = {
        str(layer): sum(row[layer] for row in result.layer_jsd) / len(result.layer_jsd)
        for layer in sorted({key for row in result.layer_jsd for key in row})
    }
    return {
        "tokens": len(result.token_ids),
        "finish_reason": result.finish_reason,
        "total_seconds": result.total_seconds,
        "first_token_seconds": result.first_token_seconds,
        "tokens_per_second": result.tokens_per_second,
        "selected_layer_counts": {str(key): counts[key] for key in sorted(counts)},
        "mean_jsd_by_layer": means,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Chat with a Hugging Face causal LM using normal or DoLa decoding"
    )
    parser.add_argument("--model", required=True, help="Hugging Face model ID or path")
    parser.add_argument(
        "--decoder", choices=("dola", "normal"), default="dola"
    )
    parser.add_argument(
        "--layers",
        default="auto",
        help="DoLa hidden-state indices, e.g. 4,8,12,16, or 'auto'",
    )
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dtype", default="auto", choices=("auto", "float16", "bfloat16", "float32"))
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--system-prompt", default=None)
    parser.add_argument(
        "--prompt",
        default=None,
        help="Run one turn and exit; omit for an interactive multi-turn chat",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--show-layer-stats", action="store_true")
    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="Log DoLa progress every N generated tokens; 0 disables progress logs",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser


def _load(args: argparse.Namespace) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    started = time.perf_counter()
    LOGGER.info("Loading tokenizer and model model=%s", args.model)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=args.trust_remote_code
    )
    model_kwargs: dict[str, Any] = {
        "device_map": args.device_map,
        "trust_remote_code": args.trust_remote_code,
        "dtype": "auto" if args.dtype == "auto" else getattr(torch, args.dtype),
    }
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs).eval()
    devices = sorted({str(parameter.device) for parameter in model.parameters()})
    footprint = getattr(model, "get_memory_footprint", lambda: 0)()
    LOGGER.info(
        "Model loaded model=%s elapsed=%.2fs device_map=%s devices=%s dtype=%s "
        "memory_gib=%.2f",
        args.model,
        time.perf_counter() - started,
        args.device_map,
        devices,
        args.dtype,
        float(footprint) / (1024**3),
    )
    if not any(device.startswith("cuda") for device in devices):
        LOGGER.warning(
            "Model has no CUDA parameters; DoLa performs extra vocabulary "
            "projections and will be very slow on CPU"
        )
    return model, tokenizer


def run_chat(args: argparse.Namespace, *, model: Any, tokenizer: Any) -> None:
    num_hidden_layers = int(_text_config(model.config).num_hidden_layers)
    layers = parse_layer_spec(args.layers, num_hidden_layers)
    decoder = DoLaDecoder(model, tokenizer) if args.decoder == "dola" else None
    messages: list[dict[str, str]] = []
    if args.system_prompt:
        messages.append({"role": "system", "content": args.system_prompt})
    one_shot = args.prompt is not None

    if not one_shot:
        print(
            f"Loaded {args.model} with {args.decoder} decoding. "
            "Enter /exit or /quit to stop."
        )
        if decoder is not None:
            print(f"DoLa candidate hidden-state indices: {layers}")

    while True:
        if one_shot:
            user_text = args.prompt
        else:
            try:
                user_text = input("user> ").strip()
            except EOFError:
                print()
                break
            if user_text.casefold() in {"/exit", "/quit"}:
                break
            if not user_text:
                continue
        messages.append({"role": "user", "content": user_text})
        turn_started = time.perf_counter()
        prompt_ids = _render_prompt(tokenizer, messages)
        LOGGER.info(
            "Generation started decoder=%s prompt_tokens=%d max_new_tokens=%d",
            args.decoder,
            len(prompt_ids),
            args.max_new_tokens,
        )
        if decoder is None:
            generation_started = time.perf_counter()
            token_ids = _normal_generate(model, tokenizer, prompt_ids, args)
            result = None
            generation_seconds = time.perf_counter() - generation_started
            LOGGER.info(
                "Normal decode finished tokens=%d generation=%.2fs "
                "tokens_per_second=%.2f",
                len(token_ids),
                generation_seconds,
                len(token_ids) / generation_seconds if generation_seconds else 0.0,
            )
        else:
            result = decoder.generate(
                prompt_ids,
                dola_layers=layers,
                max_new_tokens=args.max_new_tokens,
                alpha=args.alpha,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                seed=args.seed,
                log_every=args.log_every,
            )
            token_ids = result.token_ids
        answer = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
        LOGGER.info(
            "Turn finished decoder=%s prompt_tokens=%d output_tokens=%d total=%.2fs",
            args.decoder,
            len(prompt_ids),
            len(token_ids),
            time.perf_counter() - turn_started,
        )
        print(answer if one_shot else f"assistant> {answer}")
        messages.append({"role": "assistant", "content": answer})
        if args.show_layer_stats and result is not None:
            print(json.dumps(_diagnostics(result), indent=2), file=sys.stderr)
        if one_shot:
            break


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    )
    model, tokenizer = _load(args)
    run_chat(args, model=model, tokenizer=tokenizer)


__all__ = ["build_parser", "main", "parse_layer_spec", "run_chat"]
