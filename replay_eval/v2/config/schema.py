from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    id: str
    key: str | None = None
    dtype: str = "auto"
    max_model_len: int | None = None
    max_logprobs: int = 50
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    engine_kwargs: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, str] = field(default_factory=dict)


@dataclass
class RenderingConfig:
    system_prompt: str | None = None
    think_start_token: str = "<think>"
    think_end_token: str = "</think>"
    template_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class RuntimeConfig:
    max_concurrency: int = 32
    example_concurrency: int = 4
    example_timeout_seconds: float = 300.0
    warmup: bool = True
    rerun_all: bool = True
    rerun_final_answer: bool = False
    verify_batch_token: int = 16
    min_interval: int = 0
    code_probe_token_multiplier: int = 1
    code_threshold_multiplier: float = 1.0
    max_temp_answer_completion_token: int = 256
    max_final_answer_completion_token: int = 1024
    sampling: dict[str, Any] = field(
        default_factory=lambda: {"temperature": 0.0, "max_tokens": 32768}
    )

    exit: bool = True


@dataclass
class RandomExitConfig:
    enabled: bool = False
    sample_size: int = 128
    sample_seed: int = 0
    position_seed: int = 0
    spacing: int = 16
    max_positions: int = 64
    num_runs: int = 8
    min_valid_positions: int = 5

    def validate(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("random_exit.enabled must be a boolean")
        for name in ("sample_seed", "position_seed"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"random_exit.{name} must be an integer")
        for name in (
            "sample_size",
            "spacing",
            "max_positions",
            "num_runs",
            "min_valid_positions",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"random_exit.{name} must be a positive integer")
        if self.min_valid_positions > self.max_positions:
            raise ValueError(
                "random_exit.min_valid_positions cannot exceed random_exit.max_positions"
            )


@dataclass
class ExtractionConfig:
    mode: str = "native"
    callback: str | None = None
    temp_dir: str | None = None
    dola_layers: list[int] = field(default_factory=list)
    dola_chunk_size: int = 64

    def __post_init__(self) -> None:
        if self.mode not in {"native", "callback"}:
            raise ValueError("extraction.mode must be 'native' or 'callback'")
        if self.mode == "callback" and not self.callback:
            raise ValueError("extraction.callback is required in callback mode")
        if (
            not isinstance(self.dola_chunk_size, int)
            or isinstance(self.dola_chunk_size, bool)
            or self.dola_chunk_size < 1
        ):
            raise ValueError("extraction.dola_chunk_size must be a positive integer")
        if any(
            not isinstance(layer, int) or isinstance(layer, bool) or layer < 0
            for layer in self.dola_layers
        ):
            raise ValueError("extraction.dola_layers must contain non-negative integers")
        if len(self.dola_layers) != len(set(self.dola_layers)):
            raise ValueError("extraction.dola_layers must not contain duplicates")


@dataclass
class ReplayV2Config:
    model: ModelConfig
    rendering: RenderingConfig = field(default_factory=RenderingConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    random_exit: RandomExitConfig = field(default_factory=RandomExitConfig)
    methods: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        self.random_exit.validate()
        if not isinstance(self.runtime.exit, bool):
            raise ValueError("replay_eval.exit must be a boolean")
        if self.model.max_logprobs < 50:
            raise ValueError("model.max_logprobs must be >= 50 for ENTROCUT")
        if self.runtime.max_concurrency < 1 or self.runtime.example_concurrency < 1:
            raise ValueError("runtime concurrency values must be >= 1")
        if self.runtime.example_timeout_seconds <= 0:
            raise ValueError("runtime.example_timeout_seconds must be > 0")
        if (
            not isinstance(self.runtime.max_final_answer_completion_token, int)
            or isinstance(self.runtime.max_final_answer_completion_token, bool)
            or self.runtime.max_final_answer_completion_token < 1
        ):
            raise ValueError(
                "runtime.max_final_answer_completion_token must be a positive integer"
            )
        if self.runtime.code_probe_token_multiplier < 1:
            raise ValueError("code_probe_token_multiplier must be >= 1")
        if self.runtime.verify_batch_token < 1:
            raise ValueError("verify_batch_token must be >= 1")
        if self.runtime.min_interval < 0:
            raise ValueError("runtime.min_interval must be >= 0")
        if not 0 < self.runtime.code_threshold_multiplier <= 1:
            raise ValueError("code_threshold_multiplier must be in (0, 1]")
        # ENTROCUT v2 always uses the existing residual-bucket calculation with
        # the top 50 candidates.  Normalize config here rather than forking the
        # verifier implementation.
        configured_names: set[str] = set()
        for method in self.methods:
            kwargs = method.get("kwargs") or {}
            configured_name = kwargs.get("name")
            if configured_name is not None:
                name = str(configured_name)
                if name in configured_names:
                    raise ValueError(f"Duplicate replay method name: {name!r}")
                configured_names.add(name)
            llm = kwargs.get("llm")
            llm_name = kwargs.get("llm_name")
            if llm not in {None, "same"} or llm_name is not None:
                raise ValueError(
                    f"V2 method {configured_name or method.get('factory')!r} requests "
                    f"unsupported external LLM (llm={llm!r}, llm_name={llm_name!r}); "
                    "native v2 requires llm='same'"
                )
            if kwargs.get("chat_template") is not None:
                raise ValueError(
                    f"V2 method {configured_name or method.get('factory')!r} overrides "
                    "the canonical chat template"
                )
            if str(method.get("factory", "")).endswith("build_entrocut_method"):
                kwargs["probe_logprobs"] = 50
                method["kwargs"] = kwargs
            if str(method.get("factory", "")).endswith("build_terminator_method"):
                base_model = (method.get("kwargs") or {}).get("base_model")
                if base_model and str(base_model) != self.model.id:
                    raise ValueError(
                        "Terminator base_model must match the selected v2 model: "
                        f"{base_model!r} != {self.model.id!r}"
                    )

    def engine_kwargs(self) -> dict[str, Any]:
        values = {
            "dtype": self.model.dtype,
            "tensor_parallel_size": self.model.tensor_parallel_size,
            "gpu_memory_utilization": self.model.gpu_memory_utilization,
            **self.model.engine_kwargs,
        }
        if self.model.max_model_len is not None:
            values["max_model_len"] = self.model.max_model_len
        return values


def load_config(
    path: str | Path,
    *,
    model_name: str | None = None,
) -> ReplayV2Config:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError("v2 config must be a mapping")
    payload = copy.deepcopy(payload)
    replay_settings = (
        ((payload.get("agent_kwargs") or {}).get("replay_eval") or {})
        if isinstance(payload.get("agent_kwargs"), dict)
        else {}
    )
    v2_payload = (
        replay_settings.get("v2") if isinstance(replay_settings, dict) else None
    )
    if isinstance(v2_payload, dict):
        return _load_consolidated_config(payload, v2_payload, model_name=model_name)
    if "model" not in payload:
        raise ValueError(
            "Missing agent_kwargs.replay_eval.v2 in the consolidated config"
        )
    return ReplayV2Config(
        model=ModelConfig(**payload["model"]),
        rendering=RenderingConfig(**(payload.get("rendering") or {})),
        runtime=RuntimeConfig(**(payload.get("runtime") or {})),
        extraction=ExtractionConfig(**(payload.get("extraction") or {})),
        random_exit=RandomExitConfig(**(payload.get("random_exit") or {})),
        methods=list(payload.get("methods") or []),
    )


def load_random_exit_config(path: str | Path) -> RandomExitConfig:
    """Load sampling defaults without resolving or loading a model."""
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError("v2 config must be a mapping")
    replay = (payload.get("agent_kwargs") or {}).get("replay_eval") or {}
    v2 = replay.get("v2") if isinstance(replay, dict) else None
    values = v2.get("random_exit") if isinstance(v2, dict) else None
    return RandomExitConfig(**(values or {}))


def _load_consolidated_config(
    root: dict[str, Any],
    payload: dict[str, Any],
    *,
    model_name: str | None,
) -> ReplayV2Config:
    from scripts.config_loader import resolve_model_config

    if not model_name:
        raise ValueError("Replay v2 requires a model key selected with --model")
    selected_model = model_name
    resolved = resolve_model_config(root, selected_model)
    raw_model_id = str(resolved["id"])
    provider, separator, provider_model_id = raw_model_id.partition(":")
    if separator:
        if provider != "vllm":
            raise ValueError(
                f"Replay v2 requires a native vLLM model, got {raw_model_id!r}"
            )
        model_id = provider_model_id
    else:
        model_id = raw_model_id

    llm_settings = root.get("llm") or {}
    global_vllm = llm_settings.get("vllm") or {}
    if not isinstance(global_vllm, dict):
        raise ValueError("llm.vllm must be a mapping")
    shared_vllm_values = dict(global_vllm)
    shared_environment = shared_vllm_values.pop("environment", {}) or {}
    if not isinstance(shared_environment, dict):
        raise ValueError("llm.vllm.environment must be a mapping")
    shared_serving = shared_vllm_values.pop("serving", {}) or {}
    if not isinstance(shared_serving, dict):
        raise ValueError("llm.vllm.serving must be a mapping")
    shared_engine_kwargs = shared_vllm_values.pop("engine_kwargs", {}) or {}
    if not isinstance(shared_engine_kwargs, dict):
        raise ValueError("llm.vllm.engine_kwargs must be a mapping")

    model_vllm = resolved.get("vllm") or {}
    if not isinstance(model_vllm, dict):
        raise ValueError(f"llm.models.{selected_model}.vllm must be a mapping")
    model_vllm_values = dict(model_vllm)
    model_environment = model_vllm_values.pop("environment", {}) or {}
    if not isinstance(model_environment, dict):
        raise ValueError(
            f"llm.models.{selected_model}.vllm.environment must be a mapping"
        )
    model_serving = model_vllm_values.pop("serving", {}) or {}
    if not isinstance(model_serving, dict):
        raise ValueError(
            f"llm.models.{selected_model}.vllm.serving must be a mapping"
        )
    model_engine_kwargs = model_vllm_values.pop("engine_kwargs", {}) or {}
    if not isinstance(model_engine_kwargs, dict):
        raise ValueError(
            f"llm.models.{selected_model}.vllm.engine_kwargs must be a mapping"
        )
    model_values = {
        "id": model_id,
        **shared_vllm_values,
        **model_vllm_values,
        # Preserve the CLI/config key separately from the resolved checkpoint
        # ID so every replay-v2 output row can identify its run configuration.
        "key": selected_model,
        "engine_kwargs": {
            **shared_serving,
            **shared_engine_kwargs,
            **model_serving,
            **model_engine_kwargs,
        },
        "environment": {
            **{str(key): str(value) for key, value in shared_environment.items()},
            **{str(key): str(value) for key, value in model_environment.items()},
        },
    }
    chat_template = dict(resolved.get("chat_template") or {})
    rendering_values = {
        "think_start_token": chat_template.get("think_start_token") or "<think>",
        "think_end_token": chat_template.get("think_end_token") or "</think>",
        "template_kwargs": dict(chat_template.get("kwargs") or {}),
        **dict(payload.get("rendering") or {}),
    }
    completion = dict(resolved.get("completion_kwargs") or {})
    completion_extra_body = dict(completion.get("extra_body") or {})
    replay = (root.get("agent_kwargs") or {}).get("replay_eval") or {}
    runtime_overrides = dict(payload.get("runtime") or {})
    sampling_overrides = dict(runtime_overrides.pop("sampling", {}) or {})
    runtime_values = {
        "exit": replay.get("exit", True),
        "verify_batch_token": int(replay.get("verify_batch_token", 16)),
        "code_probe_token_multiplier": int(
            replay.get("code_probe_token_multiplier", 1)
        ),
        "code_threshold_multiplier": float(
            replay.get("code_threshold_multiplier", 1.0)
        ),
        "max_temp_answer_completion_token": int(
            completion.get("max_temp_answer_completion_token", 256)
        ),
        "max_final_answer_completion_token": int(
            completion.get("max_final_answer_completion_token", 1024)
        ),
        "sampling": {
            "temperature": float(completion.get("temperature", 0.0)),
            "max_tokens": int(completion.get("max_completion_token", 32768)),
            **(
                {
                    "skip_special_tokens": bool(
                        completion_extra_body["skip_special_tokens"]
                    )
                }
                if "skip_special_tokens" in completion_extra_body
                else {}
            ),
            **sampling_overrides,
        },
        **runtime_overrides,
    }
    shared_methods = list(replay.get("methods") or [])
    v2_methods = list(payload.get("methods") or [])
    return ReplayV2Config(
        model=ModelConfig(**model_values),
        rendering=RenderingConfig(**rendering_values),
        runtime=RuntimeConfig(**runtime_values),
        extraction=ExtractionConfig(**(payload.get("extraction") or {})),
        random_exit=RandomExitConfig(**(payload.get("random_exit") or {})),
        methods=[*shared_methods, *v2_methods],
    )
