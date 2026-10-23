"""Adapters for early-exit models trained on ``replay_eval.hidden_states``."""

from __future__ import annotations

import json
import importlib.util
import logging
import math
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any

from replay_eval.checkpoints import EveryNTokens
from replay_eval.method import EarlyExitMethod, Verifier, VerifierContext
from replay_eval.stop_rules import ThresholdGE, ThresholdGT


LOGGER = logging.getLogger(__name__)


@lru_cache(maxsize=None)
def _thought_module(filename: str) -> Any:
    """Load a standalone Thought Calibration module without package side effects.

    The current package initializer imports an optional legacy dataset symbol;
    inference needs only the self-contained model and calibration modules.
    """
    path = Path(__file__).resolve().parents[3] / "thought_calibration" / filename
    module_name = f"_replay_v2_thought_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load Thought Calibration module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _trace(ctx: VerifierContext) -> Any:
    trace = getattr(ctx.agent, "hidden_trace", None)
    if trace is None:
        raise RuntimeError(
            "A hidden-state method is configured, but extraction did not "
            "provide an example-scoped hidden trace"
        )
    return trace


def _module_device_dtype(module: Any) -> tuple[Any, Any]:
    import torch

    tensors = list(module.parameters()) + list(module.buffers())
    return (
        (tensors[0].device, tensors[0].dtype)
        if tensors
        else (torch.device("cpu"), torch.float32)
    )


def _validate_model(trace: Any, expected_model: str | None) -> None:
    if expected_model and trace.model_id and trace.model_id != expected_model:
        raise RuntimeError(
            f"Hidden states came from {trace.model_id!r}, but the probe was "
            f"trained for {expected_model!r}"
        )


class TerminatorHiddenStateVerifier(Verifier):
    """Precompute token scores once, then replay rolling positive windows."""

    uses_hidden_trace = True
    permits_token_state_release = True

    def __init__(
        self,
        *,
        probe: Any | None,
        expected_model: str | None,
        score_key: str = "terminator",
        position_mode: str = "assistant_relative",
        replay_tokens: int = 16,
        window_size: int = 10,
        classification_threshold: float = 0.5,
        precompute_chunk_size: int = 256,
    ) -> None:
        if position_mode not in {"absolute_prompt", "assistant_relative"}:
            raise ValueError(
                "position_mode must be 'absolute_prompt' or 'assistant_relative'"
            )
        if replay_tokens < 1 or window_size < 1:
            raise ValueError("replay_tokens and window_size must be >= 1")
        if precompute_chunk_size < 1:
            raise ValueError("precompute_chunk_size must be >= 1")
        if not 0.0 <= classification_threshold <= 1.0:
            raise ValueError("classification_threshold must be in [0, 1]")
        self.probe = probe
        self.expected_model = expected_model
        self.score_key = score_key
        self.position_mode = position_mode
        self.replay_tokens = int(replay_tokens)
        self.window_size = int(window_size)
        self.classification_threshold = float(classification_threshold)

        self.precompute_chunk_size = int(precompute_chunk_size)

    def reset(self, method: EarlyExitMethod) -> None:
        del method

    def __call__(self, ctx: VerifierContext) -> float:
        trace = _trace(ctx)
        self.prepare_hidden_trace(trace)
        all_probabilities = trace.model_outputs.get(self.score_key)
        if self.score_key in trace.precomputed_score_keys:
            if self.score_key not in trace.reported_precomputed_score_keys:
                trace.reported_precomputed_score_keys.add(self.score_key)
                score_source = "hf_precomputed"
            else:
                score_source = "hf_cached"
        else:
            score_source = "worker"
        assert all_probabilities is not None

        chunk_start = max(0, ctx.position - (self.window_size + self.replay_tokens - 1))
        probabilities = all_probabilities[chunk_start : ctx.position]

        labels = [
            int(float(value) >= self.classification_threshold)
            for value in probabilities
        ]
        first_new_end = max(1, ctx.position - self.replay_tokens + 1)
        max_positive_count = 0
        for window_end in range(first_new_end, ctx.position + 1):
            window_start = window_end - self.window_size
            if window_start < 0 or window_start < chunk_start:
                continue
            local_start = window_start - chunk_start
            local_end = window_end - chunk_start
            positive_count = sum(labels[local_start:local_end])
            max_positive_count = max(max_positive_count, positive_count)
        score = float(max_positive_count)
        ctx.logs.setdefault("terminator_scans", []).append(
            {
                "position": ctx.position,
                "score_source": score_source,
                "precomputed_tokens": len(all_probabilities),
                "classified_start": chunk_start + 1,
                "precompute_chunk_size": self.precompute_chunk_size,
                "classified_tokens": len(probabilities),
                "classification_threshold": self.classification_threshold,
                "window_size": self.window_size,
                "max_positive_count": int(score),
            }
        )
        return score

    def prepare_hidden_trace(self, trace: Any) -> None:
        if trace.model_outputs.get(self.score_key) is not None:
            return
        if self.probe is None:
            raise RuntimeError(
                f"Extraction did not return worker score stream {self.score_key!r}"
            )
        trace.require_pre_final_rmsnorm()
        _validate_model(trace, self.expected_model)
        if trace.raw is None:
            raise RuntimeError("Terminator preparation requires token hidden states")
        trace.model_outputs[self.score_key] = self._precompute_probe(trace)
        trace.precomputed_score_keys.add(self.score_key)

    def _precompute_probe(self, trace: Any) -> list[float]:
        import torch

        token_count = int(trace.raw.shape[0])
        if token_count < 1:
            return []
        device, dtype = _module_device_dtype(self.probe)
        if self.position_mode == "absolute_prompt":
            if trace.prompt_tokens is None:
                raise RuntimeError(
                    "absolute Terminator positions require ExtractionResult.prompt_tokens"
                )
            position_start = int(trace.prompt_tokens)
        else:
            position_start = 0
        position_ids = torch.arange(
            position_start,
            position_start + token_count,
            device=trace.raw.device,
            dtype=torch.long,
        ).unsqueeze(0)
        LOGGER.debug(
            "Terminator precompute model_family=%s raw_shape=%s "
            "position_ids_shape=%s position_start=%d chunk_size=%d",
            getattr(self.probe, "model_family", None),
            tuple(trace.raw.shape),
            tuple(position_ids.shape),
            position_start,
            self.precompute_chunk_size,
        )
        with torch.inference_mode():
            chunked = getattr(self.probe, "predict_proba_chunked", None)
            if callable(chunked):
                probabilities = chunked(
                    trace.raw.unsqueeze(0),
                    position_ids=position_ids,
                    chunk_size=self.precompute_chunk_size,
                )
            else:
                hidden_states = trace.raw.to(device=device, dtype=dtype)
                position_ids = position_ids.to(device=device)
                attention_mask = torch.ones(
                    (1, token_count), device=device, dtype=torch.long
                )
                probabilities = self.probe.predict_proba(
                    hidden_states.unsqueeze(0),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )
            probabilities = probabilities.reshape(-1).detach().cpu().tolist()
        return [float(value) for value in probabilities]


class ThoughtCalibrationVerifier(Verifier):
    """Apply training-time segment-mean pooling and score smoothing."""

    uses_hidden_trace = True
    permits_token_state_release = True

    def __init__(
        self,
        *,
        probe: Any,
        expected_model: str | None,
        smoothing_window: int,
        segment_size: int,
    ) -> None:
        if smoothing_window < 1 or segment_size < 1:
            raise ValueError("smoothing_window and segment_size must be >= 1")
        self.probe = probe
        self.expected_model = expected_model
        self.smoothing_window = int(smoothing_window)
        self.segment_size = int(segment_size)

    def prepare_hidden_trace(self, trace: Any) -> None:
        import torch

        if self.segment_size in trace.pooled_hidden_states:
            return
        trace.require_pre_final_rmsnorm()
        _validate_model(trace, self.expected_model)
        if trace.raw is None:
            raise RuntimeError("Thought Calibration preparation requires token hidden states")
        token_count = int(trace.raw.shape[0])
        pooled = []
        for end in range(self.segment_size, token_count + 1, self.segment_size):
            start = end - self.segment_size
            pooled.append(trace.raw[start:end].mean(dim=0, keepdim=True))
        trace.pooled_hidden_states[self.segment_size] = (
            torch.cat(pooled, dim=0)
            if pooled
            else trace.raw.new_empty((0, trace.raw.shape[-1]))
        )

    def __call__(self, ctx: VerifierContext) -> float:
        import torch

        moving_average = _thought_module("calibration.py").moving_average

        trace = _trace(ctx)
        self.prepare_hidden_trace(trace)
        start = int(ctx.state.get("_thought_calibration_boundary", 0))
        end = int(ctx.position)
        pooled_segments = trace.pooled_hidden_states[self.segment_size]
        if end - start != self.segment_size or end % self.segment_size != 0:
            raise RuntimeError(
                f"Invalid Thought Calibration segment [{start}:{end}] for "
                f"segment size {self.segment_size}"
            )
        segment_index = end // self.segment_size - 1
        if segment_index >= int(pooled_segments.shape[0]):
            raise RuntimeError(f"Missing pooled Thought Calibration segment at {end}")
        device, dtype = _module_device_dtype(self.probe)
        pooled = pooled_segments[segment_index : segment_index + 1].to(
            device=device, dtype=dtype
        )
        with torch.inference_mode():
            raw_score = float(self.probe(pooled).reshape(-1)[-1].detach().cpu())
        raw_scores = ctx.state.setdefault("_thought_calibration_raw_scores", [])
        raw_scores.append(raw_score)
        ctx.state["_thought_calibration_boundary"] = end
        score = float(moving_average(raw_scores, self.smoothing_window)[-1])
        ctx.logs.setdefault("thought_calibration_scores", []).append(
            {
                "position": end,
                "segment_start": start,
                "raw_score": raw_score,
                "smoothed_score": score,
            }
        )
        return score


@lru_cache(maxsize=None)
def _load_terminator_probe(
    artifact_dir: str,
    device: str,
    dtype: str | None,
) -> Any:
    import torch
    from transformers import AutoConfig
    from terminator.terminator import TerminatorProbe

    directory = Path(artifact_dir)
    artifact_config = _load_terminator_artifact_config(directory)
    base_model = str(artifact_config["base_model"])
    resolved_device = device
    if resolved_device == "auto":
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    config = AutoConfig.from_pretrained(base_model)
    text_config = getattr(config, "text_config", config)
    artifact_hidden_size = int(artifact_config["hidden_size"])
    if int(text_config.hidden_size) != artifact_hidden_size:
        raise ValueError(
            "Terminator artifact hidden_size does not match its base model: "
            f"{artifact_hidden_size} != {text_config.hidden_size}"
        )
    text_config._attn_implementation = "sdpa"
    layer_idx = _resolve_terminator_layer_idx(artifact_config, text_config)
    probe = TerminatorProbe(
        config,
        model_family=artifact_config.get("model_family"),
        layer_idx=layer_idx,
        ffn_num_hidden_layers=int(artifact_config.get("ffn_num_hidden_layers", 1)),
        ffn_activation=str(artifact_config.get("ffn_activation", "gelu")),
        ffn_intermediate_size=artifact_config.get("ffn_intermediate_size"),
        ffn_dropout=float(artifact_config.get("ffn_dropout", 0.0)),
    )
    probe.load_checkpoint(
        directory / str(artifact_config["checkpoint_file"]),
        map_location="cpu",
    )
    resolved_dtype = (
        artifact_config.get("dtype") if dtype in {None, "auto"} else dtype
    )
    if resolved_dtype and resolved_dtype != "auto":
        probe = probe.to(dtype=getattr(torch, str(resolved_dtype)))
    return probe.to(resolved_device).eval()


def _resolve_terminator_layer_idx(
    artifact_config: dict[str, Any], text_config: Any
) -> int | None:
    value = artifact_config.get("layer_idx")
    if value is None:
        return None
    layer_idx = int(value)
    num_hidden_layers = int(getattr(text_config, "num_hidden_layers", 0))
    if num_hidden_layers and layer_idx == num_hidden_layers:
        corrected = layer_idx - 1
        layer_types = getattr(text_config, "layer_types", None)
        if layer_types and layer_types[corrected] != "full_attention":
            raise ValueError(
                "Terminator artifact uses a one-based final layer index, but "
                f"zero-based layer {corrected} is not full_attention"
            )
        warnings.warn(
            "Terminator artifact layer_idx equals num_hidden_layers; treating "
            f"legacy one-based layer_idx={layer_idx} as zero-based {corrected}",
            UserWarning,
            stacklevel=2,
        )
        return corrected
    return layer_idx


def _load_terminator_artifact_config(directory: Path) -> dict[str, Any]:
    config_path = directory / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    required = {"base_model", "hidden_size", "checkpoint_file"}
    missing = sorted(required.difference(config))
    if missing:
        raise ValueError(
            f"Terminator artifact {config_path} is missing: {', '.join(missing)}"
        )
    checkpoint = (directory / str(config["checkpoint_file"])).resolve()
    if checkpoint.parent != directory.resolve():
        raise ValueError("Terminator checkpoint_file must name a file in artifact_dir")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Terminator checkpoint does not exist: {checkpoint}")
    return config


@lru_cache(maxsize=None)
def _load_thought_probe(artifact_dir: str, device: str, dtype: str | None) -> Any:
    import torch

    ThoughtCalibrationModule = _thought_module("model.py").ThoughtCalibrationModule

    resolved_device = device
    if resolved_device == "auto":
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = None if not dtype or dtype == "auto" else getattr(torch, dtype)
    return (
        ThoughtCalibrationModule.load(artifact_dir, dtype=torch_dtype)
        .to(resolved_device)
        .eval()
    )


def build_terminator_method(
    name: str = "terminator",
    *,
    artifact_dir: str | None = None,
    base_model: str | None = None,
    threshold: float = 0.5,
    replay_tokens: int = 16,
    window_size: int = 10,
    classification_threshold: float = 0.5,
    device: str = "auto",
    dtype: str | None = None,
    worker_score_key: str = "terminator",
    position_mode: str = "assistant_relative",
    precompute_chunk_size: int = 256,
    probe: Any | None = None,
) -> EarlyExitMethod:
    """Build a one-pass Terminator whose scores are replayed every 16 tokens."""
    if artifact_dir is not None:
        directory = Path(artifact_dir).expanduser().resolve()
        artifact_config = _load_terminator_artifact_config(directory)
        artifact_base_model = str(artifact_config["base_model"])
        if base_model is not None and base_model != artifact_base_model:
            raise ValueError(
                "Terminator artifact base_model does not match the configured model: "
                f"{artifact_base_model!r} != {base_model!r}"
            )
        base_model = artifact_base_model
    if probe is None and artifact_dir is not None:
        probe = _load_terminator_probe(
            str(directory),
            device,
            dtype,
        )
    if probe is None and not worker_score_key:
        raise ValueError("Set artifact_dir or worker_score_key for Terminator")
    return EarlyExitMethod(
        name=name,
        checkpoint=EveryNTokens(replay_tokens),
        verifier=TerminatorHiddenStateVerifier(
            probe=probe,
            expected_model=base_model,
            score_key=worker_score_key,
            position_mode=position_mode,
            replay_tokens=replay_tokens,
            window_size=window_size,
            classification_threshold=classification_threshold,
            precompute_chunk_size=precompute_chunk_size,
        ),
        stop_rule=ThresholdGT(threshold),
    )


def _risk_entry(values: dict[str, Any], risk_tolerance: float) -> Any:
    for key, value in values.items():
        if math.isclose(float(key), float(risk_tolerance), rel_tol=0.0, abs_tol=1e-12):
            return value
    raise ValueError(f"No calibrated threshold for risk tolerance {risk_tolerance}")


def build_thought_calibration_method(
    name: str = "thought_calibration",
    *,
    artifact_dir: str,
    risk_tolerance: float = 0.05,
    check_every: int = 128,
    device: str = "auto",
    dtype: str | None = None,
    probe: Any | None = None,
) -> EarlyExitMethod:
    """Build Thought Calibration from its deployment artifacts."""
    directory = Path(artifact_dir).expanduser().resolve()
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    if config.get("feature_source") != "pre_final_rmsnorm":
        raise ValueError(
            "Thought Calibration artifact has an unsupported feature_source"
        )
    if config.get("pooling") != "segment_mean":
        raise ValueError(
            "Thought Calibration artifact has an unsupported pooling policy"
        )
    if config.get("threshold_comparison") != ">=":
        raise ValueError("V2 supports Thought Calibration comparison '>=' only")
    entry = _risk_entry(config.get("thresholds") or {}, risk_tolerance)
    threshold = entry.get("selected_threshold") if isinstance(entry, dict) else entry
    threshold = math.inf if threshold is None else float(threshold)
    if probe is None:
        probe = _load_thought_probe(str(directory), device, dtype)
    return EarlyExitMethod(
        name=name,
        checkpoint=EveryNTokens(check_every),
        verifier=ThoughtCalibrationVerifier(
            probe=probe,
            expected_model=config.get("base_model"),
            smoothing_window=int(config["smoothing_window"]),
            segment_size=check_every,
        ),
        stop_rule=ThresholdGE(threshold),
    )


__all__ = [
    "TerminatorHiddenStateVerifier",
    "ThoughtCalibrationVerifier",
    "build_terminator_method",
    "build_thought_calibration_method",
]
