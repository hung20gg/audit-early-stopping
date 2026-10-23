import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

DEFAULT_CONFIG = Path(__file__).with_name("config.yml")
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.config_loader import (
    apply_system_prompts,
    get_model_configs,
    import_config_entry,
    load_config,
    load_system_prompt,
    resolve_model_config,
)
from methods.base import Response


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke load an agent and dataloader from config.yml."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="YAML config path.")
    parser.add_argument("--method", default="simple", help="Agent key from config.yml.")
    parser.add_argument("--dataloader", default="gsm8k", help="Dataloader key from config.yml.")
    parser.add_argument("--evaluator", default="gsm8k", help="Evaluator key from config.yml.")
    parser.add_argument("--model", default=None, help="Model config key from llm.models.")
    parser.add_argument("--list-models", action="store_true", help="List configured models and exit.")
    parser.add_argument("--split", default="test", help="Dataset split passed to dataloader.")
    parser.add_argument("--limit", type=int, default=1, help="Number of examples to load.")
    parser.add_argument(
        "--llm",
        default=None,
        help="Optional LLM name. When omitted, the agent class is imported but not run.",
    )
    return parser.parse_args()


def get_question(record: dict[str, Any]) -> str:
    question = record.get("question")
    if not isinstance(question, str):
        raise ValueError("Loaded record must contain a string 'question' field.")
    return question


def serialize_response_log(response: Response) -> dict:
    return {
        "input": response.input,
        "content": response.content or response.token_content(),
        "num_token": response.token_count(),
        "finish_reason": response.finish_reason,
    }


def serialize_score_trace(scores) -> list[dict]:
    score_trace = []
    for item in scores or []:
        if isinstance(item, dict):
            position = item.get("position")
            score = item.get("score")
        else:
            try:
                position, score = item
            except (TypeError, ValueError):
                continue
        score_trace.append({"position": position, "score": score})
    return score_trace


def serialize_score_usage(response: Response) -> dict:
    score_prompt_tokens = getattr(response, "score_prompt_tokens", None)
    score_completion_tokens = getattr(response, "score_completion_tokens", None)
    if score_prompt_tokens is None and score_completion_tokens is None:
        return {}

    score_prompt_tokens = score_prompt_tokens or 0
    score_completion_tokens = score_completion_tokens or 0
    return {
        "score_prompt_tokens": score_prompt_tokens,
        "score_completion_tokens": score_completion_tokens,
        "score_tokens": score_prompt_tokens + score_completion_tokens,
    }


def serialize_agent_response(response, include_detailed_logprobs: bool = False):
    if isinstance(response, Response):
        additional_logs = {}
        probe_scores = getattr(response, "probe_scores", [])
        sufficiency_scores = getattr(response, "sufficiency_scores", [])
        score_trace = serialize_score_trace(
            getattr(response, "score_trace", []) or probe_scores or sufficiency_scores
        )
        if score_trace:
            additional_logs["score_trace"] = score_trace
        if probe_scores:
            additional_logs["probe_scores"] = list(probe_scores)
        full_generation_response = getattr(response, "full_generation_response", None)
        if isinstance(full_generation_response, Response):
            additional_logs["full_generation_response"] = serialize_response_log(
                full_generation_response
            )
        method_results_dicts = getattr(response, "method_results_dicts", None)
        if method_results_dicts:
            additional_logs["method_results"] = list(method_results_dicts)
        score_usage = serialize_score_usage(response)
        if score_usage:
            additional_logs.update(score_usage)
        token_budgets = getattr(response, "token_budgets", [])
        return {
            "output": response.content,
            "logprobs": response.logprobs_as_dicts() if include_detailed_logprobs else [],
            "num_token": response.token_count(),
            "num_loop": response.num_loop,
            "num_chunk": response.num_chunk,
            **({"token_budgets": list(token_budgets)} if token_budgets else {}),
            **({"additional_logs": additional_logs} if additional_logs else {}),
        }
    return {
        "output": str(response or ""),
        "logprobs": [],
        "num_token": None,
        "num_loop": 0,
        "num_chunk": 0,
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.list_models:
        print(json.dumps(get_model_configs(config), ensure_ascii=False, indent=2))
        return

    agent_class = import_config_entry(config, "method", args.method)
    dataloader = import_config_entry(config, "dataloader", args.dataloader)
    evaluator = import_config_entry(config, "evaluator", args.evaluator)
    model_config = resolve_model_config(config, args.model)
    if args.llm:
        model_config["id"] = args.llm
    model_config = apply_system_prompts(
        model_config,
        dataset_system_prompt=load_system_prompt(config, "dataset", args.dataloader),
        agent_system_prompt=load_system_prompt(config, "agent", args.method),
    )
    include_detailed_logprobs = bool(config.get("llm", {}).get("include_detailed_logprobs", False))

    records = dataloader(split=args.split, limit=args.limit)
    if not records:
        raise ValueError("Dataloader returned no records.")

    result = {
        "method": args.method,
        "agent_class": f"{agent_class.__module__}.{agent_class.__name__}",
        "model": model_config,
        "dataloader": args.dataloader,
        "evaluator": args.evaluator,
        "num_loaded": len(records),
        "first_record": records[0],
        "evaluator_self_check": evaluator(records[0]["answer"], records[0]["answer"]),
    }

    if args.llm:
        agent_kwargs = copy.deepcopy(
            config.get("agent_kwargs", {}).get(args.method, {})
        )
        if args.method == "replay_eval":
            agent_kwargs.pop("version", None)
            agent_kwargs.pop("v2", None)
        agent = agent_class(
            llm_name=model_config["id"],
            chat_template=model_config.get("chat_template"),
            completion_kwargs=model_config.get("completion_kwargs"),
            agent_system_prompt=model_config.get("agent_system_prompt"),
            **agent_kwargs,
        )
        first_response = serialize_agent_response(
            agent.solve(
                get_question(records[0]),
                answer=records[0].get("answer"),
            ),
            include_detailed_logprobs=include_detailed_logprobs,
        )
        result["first_output"] = first_response["output"]
        result["first_logprobs"] = first_response["logprobs"]
        result["first_num_token"] = first_response["num_token"]
        result["first_num_loop"] = first_response["num_loop"]
        result["first_num_chunk"] = first_response["num_chunk"]
        if "additional_logs" in first_response:
            result["first_additional_logs"] = first_response["additional_logs"]

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
