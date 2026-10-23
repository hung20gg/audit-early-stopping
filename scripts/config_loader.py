from collections.abc import Mapping
import importlib
import sys
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_config(config_path: str | Path) -> dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}

    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    return config


def import_from_path(import_path: str) -> Any:
    module_name, object_name = import_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, object_name)


def get_config_entry(config: dict[str, Any], section: str, name: str) -> Any:
    entries = config.get(section, {})
    if not isinstance(entries, dict):
        raise ValueError(f"Config section '{section}' must be a mapping.")

    entry = entries.get(name)
    if not entry:
        available = ", ".join(sorted(entries)) or "<none>"
        raise ValueError(
            f"Unknown {section} '{name}'. Available {section} entries: {available}"
        )
    return entry


def get_config_entry_path(config: dict[str, Any], section: str, name: str) -> str:
    entry = get_config_entry(config, section, name)
    if isinstance(entry, str):
        return entry
    if isinstance(entry, Mapping):
        import_path = entry.get("class", entry.get("path", entry.get("import")))
        if isinstance(import_path, str) and import_path:
            return import_path
        raise ValueError(
            f"Config entry '{section}.{name}' must contain a 'class' import path."
        )
    raise ValueError(
        f"Config entry '{section}.{name}' must be a string or mapping, got {type(entry).__name__}."
    )


def import_config_entry(config: dict[str, Any], section: str, name: str) -> Any:
    return import_from_path(get_config_entry_path(config, section, name))


def read_prompt_file(prompt_path: str | Path | None) -> str | None:
    if not prompt_path:
        return None

    path = Path(prompt_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.read_text(encoding="utf-8").strip()


def get_system_prompt_path(config: dict[str, Any], prompt_type: str, name: str) -> str | None:
    prompts = config.get("system_prompt", {}) or {}
    if not isinstance(prompts, dict):
        raise ValueError("Config section 'system_prompt' must be a mapping.")

    typed_prompts = prompts.get(prompt_type)
    if isinstance(typed_prompts, dict):
        return typed_prompts.get(name)

    if prompt_type == "dataset":
        prompt_path = prompts.get(name)
        if isinstance(prompt_path, str):
            return prompt_path

    return None


def load_system_prompt(config: dict[str, Any], prompt_type: str, name: str) -> str | None:
    return read_prompt_file(get_system_prompt_path(config, prompt_type, name))


def apply_system_prompts(
    model_config: dict[str, Any],
    dataset_system_prompt: str | None = None,
    agent_system_prompt: str | None = None,
) -> dict[str, Any]:
    resolved_config = dict(model_config)
    chat_template = dict(resolved_config.get("chat_template") or {})
    if dataset_system_prompt:
        chat_template["generation_system_prompt"] = dataset_system_prompt
    resolved_config["chat_template"] = chat_template
    resolved_config["agent_system_prompt"] = agent_system_prompt
    return resolved_config


def get_model_configs(config: dict[str, Any]) -> dict[str, Any]:
    llm_config = config.get("llm", {}) or {}
    models = llm_config.get("models", {})
    if not isinstance(models, dict):
        raise ValueError("Config section 'llm.models' must be a mapping.")
    return models


def get_default_model_name(config: dict[str, Any]) -> str | None:
    llm_config = config.get("llm", {}) or {}
    default_model = llm_config.get("default")
    return str(default_model) if default_model else None


def normalize_stop_tokens(stop: Any) -> list[str]:
    if stop is None:
        stop_tokens: list[str] = []
    elif isinstance(stop, str):
        stop_tokens = [stop]
    elif isinstance(stop, Mapping):
        value = stop.get("text", stop.get("token"))
        stop_tokens = [value] if isinstance(value, str) else []
    else:
        stop_tokens = []
        for token in stop:
            if isinstance(token, str):
                stop_tokens.append(token)
            elif isinstance(token, Mapping):
                value = token.get("text", token.get("token"))
                if isinstance(value, str):
                    stop_tokens.append(value)
            elif token is not None:
                stop_tokens.append(str(token))

    if "\n\n" not in stop_tokens:
        stop_tokens.append("\n\n")
    return stop_tokens


def normalize_logprobs_count(value: Any) -> int | None:
    if value is None:
        return None
    try:
        count = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"Invalid logprobs count: {value!r}") from None

    if count < 1 or count > 50:
        raise ValueError(f"logprobs must be between 1 and 50, got {count}")
    return count


def merge_completion_kwargs(
    config: dict[str, Any],
    model_config: dict[str, Any],
) -> dict[str, Any]:
    llm_config = config.get("llm", {}) or {}
    global_kwargs = dict(llm_config.get("completion_kwargs") or {})
    model_kwargs = dict(model_config.get("completion_kwargs") or {})
    completion_kwargs = {**global_kwargs, **model_kwargs}

    global_logprobs = normalize_logprobs_count(llm_config.get("logprobs"))
    if global_logprobs is not None and "logprobs" not in completion_kwargs:
        completion_kwargs["logprobs"] = global_logprobs

    completion_kwargs["stop"] = normalize_stop_tokens(completion_kwargs.get("stop"))
    return completion_kwargs


def resolve_model_config(config: dict[str, Any], model_name: str | None = None) -> dict[str, Any]:
    models = get_model_configs(config)
    selected_model = model_name or get_default_model_name(config)
    if not selected_model:
        raise ValueError("No model selected. Set llm.default in config or pass --model.")

    model_config = models.get(selected_model)
    if not isinstance(model_config, dict):
        available = ", ".join(sorted(models)) or "<none>"
        raise ValueError(f"Unknown model '{selected_model}'. Available models: {available}")

    resolved_config = {"name": selected_model, **model_config}
    chat_template = dict(resolved_config.get("chat_template") or {})
    if "think_start_token" not in chat_template:
        chat_template["think_start_token"] = (
            resolved_config.get("think_start_token")
            or resolved_config.get("think_token")
        )
    if "think_end_token" not in chat_template:
        chat_template["think_end_token"] = resolved_config.get("think_end_token")
    resolved_config["chat_template"] = chat_template
    resolved_config["completion_kwargs"] = merge_completion_kwargs(config, resolved_config)
    return resolved_config
