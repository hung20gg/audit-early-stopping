import argparse
import asyncio
import copy
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from pathlib import Path

DEFAULT_CONFIG = Path(__file__).with_name("config.yml")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOGGER = logging.getLogger("early_exist.runner")

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


def parse_bool(value):
    normalized = str(value).strip().casefold()
    if normalized in {"true", "1", "yes", "on"}:
        return True
    if normalized in {"false", "0", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run an early-exit agent over a dataset."
    )
    parser.add_argument(
        "--config", default=str(DEFAULT_CONFIG), help="Path to YAML config."
    )
    parser.add_argument(
        "--agent", default="simple", help="Agent key selected from config method."
    )
    parser.add_argument(
        "--model", default=None, help="Model config key from llm.models."
    )
    parser.add_argument(
        "--llm", default=None, help="Override actual LLM id passed to get_llm_wrapper."
    )
    parser.add_argument(
        "--list-models", action="store_true", help="List configured models and exit."
    )
    parser.add_argument(
        "--dataloader",
        default="gsm8k",
        help="Dataloader key selected from config dataloader.",
    )
    parser.add_argument(
        "--evaluator",
        default="gsm8k",
        help="Evaluator key selected from config evaluator.",
    )
    parser.add_argument(
        "--split", default="test", help="Dataset split passed to dataloader."
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Optional dataset row limit."
    )
    parser.add_argument(
        "--exit", type=parse_bool, default=None,
        help="Replay: stop after the first confirmed exit (default true).",
    )
    parser.add_argument(
        "--rerun-all",
        type=parse_bool,
        default=None,
        metavar="TRUE_OR_FALSE",
        help="V2: rerun every configured method or only methods missing from each row.",
    )
    parser.add_argument(
        "--rerun-final-answer",
        type=parse_bool,
        default=None,
        metavar="TRUE_OR_FALSE",
        help=(
            "V2: regenerate only final answers removed from existing method results; "
            "ignored when --rerun-all is true."
        ),
    )
    parser.add_argument(
        "--reverse",
        action="store_true",
        help="Process the selected dataset records in reverse order.",
    )
    parser.add_argument("--input-file", default=None, help="Optional input JSONL file.")
    parser.add_argument(
        "--output-file",
        default="outputs/gsm8k.jsonl",
        help="Output JSONL file.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume from --output-file and skip records already answered "
            "successfully in it."
        ),
    )
    parser.add_argument(
        "--resume-from",
        default=None,
        help=(
            "Optional checkpoint JSONL to resume from. Skips records whose "
            "line_number has a successful answer."
        ),
    )
    parser.add_argument("--log-file", default=None, help="Optional runner log file.")
    parser.add_argument("--log-level", default="INFO", help="Python logging level.")
    parser.add_argument(
        "--name", default=None, help="Optional custom run name stored in each result."
    )
    parser.add_argument(
        "--num-thread",
        "--num-threads",
        "--threads",
        dest="num_thread",
        type=int,
        default=1,
        help="Number of worker threads. Multithreading is only used when this is greater than 1.",
    )
    parser.add_argument(
        "--max-error",
        "--max-errors",
        dest="max_error",
        type=int,
        default=3,
        help="Interrupt the run after more than this many record errors.",
    )
    parser.add_argument(
        "--max-total-thinking-token",
        type=int,
        default=None,
        help=(
            "Override the model's total thinking budget "
            "(completion_kwargs.max_completion_token)."
        ),
    )
    parser.add_argument(
        "--max-final-answer-completion-token",
        type=int,
        default=None,
        help="Override the separate forced final-answer rollout budget.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="V2: override the generation sampling temperature.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=None,
        help="V2: override the vLLM GPU memory utilization fraction.",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=None,
        help="V2: override the vLLM maximum number of batched sequences.",
    )
    parser.add_argument(
        "--random-exit",
        type=parse_bool,
        default=None,
        metavar="TRUE_OR_FALSE",
        help="Enable deterministic random-exit replay using the configured replay version.",
    )
    parser.add_argument("--random-exit-sample-size", type=int, default=None)
    parser.add_argument("--random-exit-sample-seed", type=int, default=None)
    parser.add_argument("--random-exit-position-seed", type=int, default=None)
    parser.add_argument("--random-exit-spacing", type=int, default=None)
    parser.add_argument("--random-exit-max-positions", type=int, default=None)
    parser.add_argument("--random-exit-num-runs", type=int, default=None)
    parser.add_argument("--random-exit-min-valid-positions", type=int, default=None)
    parser.add_argument(
        "--problem-field",
        default="problem",
        help="Input field to use as the problem text. Falls back to prompt/question/input.",
    )
    return parser.parse_args()


def setup_logging(log_level, log_file=None):
    level = getattr(logging, str(log_level).upper(), None)
    if not isinstance(level, int):
        raise ValueError(f"Unknown log level: {log_level}")

    handlers = [logging.StreamHandler()]
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )


def print_model_configs(config):
    print(json.dumps(get_model_configs(config), ensure_ascii=False, indent=2))


def load_optional_hf_token(env_path: Path | None = None) -> bool:
    if os.environ.get("HF_TOKEN"):
        return True

    env_path = env_path or PROJECT_ROOT / ".env"
    if not env_path.exists():
        return False

    token = None
    with open(env_path, "r", encoding="utf-8") as env_file:
        for line in env_file:
            key, separator, value = line.strip().removeprefix("export ").partition("=")
            if separator and key in {"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"}:
                token = value.strip().strip("\"'")
                if token:
                    break
    if not token:
        return False

    os.environ["HF_TOKEN"] = token
    return True


def load_agent(config, agent_name, model_config):
    LOGGER.info("Loading agent '%s'", agent_name)
    agent_class = import_config_entry(config, "method", agent_name)
    agent_kwargs = get_agent_kwargs(config, agent_name)
    llm_kwargs = config.get("llm_kwargs", {})
    model_llm_kwargs = model_config.get("llm_kwargs", {})
    return agent_class(
        llm_name=model_config["id"],
        chat_template=model_config.get("chat_template"),
        completion_kwargs=model_config.get("completion_kwargs"),
        agent_system_prompt=model_config.get("agent_system_prompt"),
        **llm_kwargs,
        **model_llm_kwargs,
        **agent_kwargs,
    )


def get_agent_kwargs(config, agent_name):
    values = copy.deepcopy(config.get("agent_kwargs", {}).get(agent_name, {}))
    if agent_name == "replay_eval":
        values.pop("version", None)
        values.pop("v2", None)
    return values


def get_replay_version(config) -> int:
    replay = (config.get("agent_kwargs") or {}).get("replay_eval") or {}
    version = int(replay.get("version", 1))
    if version not in {1, 2}:
        raise ValueError("agent_kwargs.replay_eval.version must be 1 or 2")
    return version


def random_exit_enabled(config, cli_value=None) -> bool:
    replay = (config.get("agent_kwargs") or {}).get("replay_eval") or {}
    v2 = replay.get("v2") if isinstance(replay, dict) else {}
    random_exit = v2.get("random_exit") if isinstance(v2, dict) else {}
    configured = (
        random_exit.get("enabled", False)
        if isinstance(random_exit, dict)
        else False
    )
    return bool(configured) if cli_value is None else bool(cli_value)


def run_v2(args, config):
    if not args.input_file:
        raise ValueError("Random-exit and replay version 2 require --input-file JSONL")
    if not args.model:
        raise ValueError("Random-exit and replay version 2 require --model from llm.models")
    if args.resume and args.resume_from:
        raise ValueError("Use either --resume or --resume-from, not both")
    if args.reverse:
        raise ValueError("Replay version 2 does not currently support --reverse")
    if args.limit is not None:
        raise ValueError("Replay version 2 does not currently support --limit")

    from replay_eval.v2.config.schema import load_config as load_v2_config
    from replay_eval.v2.runtime.runner import ReplayV2Runner

    setup_logging(args.log_level, args.log_file)
    load_optional_hf_token()
    v2_config = load_v2_config(args.config, model_name=args.model)
    if getattr(args, "exit", None) is not None:
        v2_config.runtime.exit = args.exit
    if args.llm:
        provider, separator, model_id = str(args.llm).partition(":")
        if separator and provider != "vllm":
            raise ValueError("Replay version 2 --llm must use the vllm: provider")
        v2_config.model.id = model_id if separator else provider
    system_prompt = load_system_prompt(config, "dataset", args.dataloader)
    if system_prompt:
        v2_config.rendering.system_prompt = system_prompt
    if args.max_total_thinking_token is not None:
        v2_config.runtime.sampling["max_tokens"] = args.max_total_thinking_token
    if args.max_final_answer_completion_token is not None:
        v2_config.runtime.max_final_answer_completion_token = (
            args.max_final_answer_completion_token
        )
    if args.temperature is not None:
        if args.temperature < 0:
            raise ValueError("--temperature must be >= 0")
        v2_config.runtime.sampling["temperature"] = args.temperature
    if args.gpu_memory_utilization is not None:
        if not 0 < args.gpu_memory_utilization <= 1:
            raise ValueError("--gpu-memory-utilization must be in (0, 1]")
        v2_config.model.gpu_memory_utilization = args.gpu_memory_utilization
    if args.max_num_seqs is not None:
        if args.max_num_seqs < 1:
            raise ValueError("--max-num-seqs must be >= 1")
        v2_config.model.engine_kwargs["max_num_seqs"] = args.max_num_seqs
    if args.random_exit is not None:
        v2_config.random_exit.enabled = args.random_exit
    random_overrides = {
        "sample_size": args.random_exit_sample_size,
        "sample_seed": args.random_exit_sample_seed,
        "position_seed": args.random_exit_position_seed,
        "spacing": args.random_exit_spacing,
        "max_positions": args.random_exit_max_positions,
        "num_runs": args.random_exit_num_runs,
        "min_valid_positions": args.random_exit_min_valid_positions,
    }
    for field, value in random_overrides.items():
        if value is not None:
            setattr(v2_config.random_exit, field, value)
    if getattr(args, "rerun_all", None) is not None:
        v2_config.runtime.rerun_all = args.rerun_all
    if getattr(args, "rerun_final_answer", None) is not None:
        v2_config.runtime.rerun_final_answer = args.rerun_final_answer
    v2_config.validate()
    evaluator = import_config_entry(config, "evaluator", args.evaluator)
    recovery_dataloader = import_config_entry(
        config, "dataloader", args.dataloader
    )
    if args.num_thread != 1:
        LOGGER.warning(
            "Random-exit/replay v2 ignores --num-thread; use "
            "runtime.example_concurrency in config"
        )
    LOGGER.info(
        "Starting replay version %s with model=%s random_exit=%s rerun_all=%s rerun_final_answer=%s evaluator=%s",
        get_replay_version(config),
        v2_config.model.id,
        v2_config.random_exit.enabled,
        v2_config.runtime.rerun_all,
        v2_config.runtime.rerun_final_answer,
        args.evaluator,
    )
    LOGGER.info(
        "Replay v2 token budgets full=%s final=%s temporary=%s",
        v2_config.runtime.sampling.get("max_tokens"),
        v2_config.runtime.max_final_answer_completion_token,
        v2_config.runtime.max_temp_answer_completion_token,
    )
    resume_from = args.output_file if args.resume else args.resume_from
    if v2_config.random_exit.enabled and get_replay_version(config) == 1:
        from replay_eval.v1_random_exit import run_random_exit

        v1_model_config = resolve_model_config(config, args.model)
        if args.llm:
            v1_model_config["id"] = args.llm
        metrics = asyncio.run(
            run_random_exit(
                v2_config,
                input_file=args.input_file,
                output_file=args.output_file,
                resume_from=resume_from,
                model_config=v1_model_config,
                llm_kwargs={
                    **config.get("llm_kwargs", {}),
                    **v1_model_config.get("llm_kwargs", {}),
                },
            )
        )
    else:
        metrics = asyncio.run(
            ReplayV2Runner(
                v2_config,
                evaluator=evaluator,
                evaluator_name=args.evaluator,
            ).run(
                args.input_file,
                args.output_file,
                resume_from=resume_from,
                recovery_dataloader=recovery_dataloader,
                recovery_split=args.split,
                dataloader_name=args.dataloader,
            )
        )
    LOGGER.info("Replay metrics: %s", json.dumps(metrics, sort_keys=True))
    return metrics


def load_records(config, args):
    reverse = bool(getattr(args, "reverse", False))
    limit = getattr(args, "limit", None)
    if args.input_file:
        LOGGER.info("Loading records from JSONL file: %s", args.input_file)
        records = iter_jsonl(args.input_file)
        if reverse:
            records = reversed(list(records))
        if limit is not None:
            records = islice(records, limit)
        for source_line_number, record in records:
            record_id = record.get("id", record.get("line_number", source_line_number))
            yield (
                record_id if isinstance(record_id, int) else source_line_number,
                record,
            )
        return

    LOGGER.info(
        "Loading records with dataloader '%s' split='%s' limit=%s",
        args.dataloader,
        args.split,
        args.limit,
    )
    dataloader = import_config_entry(config, "dataloader", args.dataloader)
    records = dataloader(split=args.split, limit=None if reverse else limit)
    numbered_records = enumerate(records, start=1)
    if reverse:
        numbered_records = reversed(list(numbered_records))
        if limit is not None:
            numbered_records = islice(numbered_records, limit)
    for record_id, record in numbered_records:
        yield record_id, record


def iter_jsonl(path):
    with open(path, "r") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_number, json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc


def load_processed_line_numbers(path):
    processed = set()
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        LOGGER.info("Resume checkpoint does not exist yet: %s", checkpoint_path)
        return processed

    with open(checkpoint_path, "r") as checkpoint:
        checkpoint_rows = []
        for checkpoint_line_number, line in enumerate(checkpoint, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                checkpoint_rows.append((checkpoint_line_number, json.loads(line)))
            except json.JSONDecodeError as exc:
                LOGGER.warning(
                    "Ignoring invalid checkpoint JSON: %s line %s: %s",
                    checkpoint_path,
                    checkpoint_line_number,
                    exc,
                )

    for checkpoint_line_number, row in checkpoint_rows:
        line_number = row.get("line_number")
        if isinstance(line_number, int) and row.get("ok") is True and not row.get(
            "generation_failed", False
        ):
            processed.add(line_number)
        elif isinstance(line_number, int):
            LOGGER.info(
                "Retrying unsuccessful checkpoint row: %s line %s record %s",
                checkpoint_path,
                checkpoint_line_number,
                line_number,
            )
        else:
            LOGGER.warning(
                "Ignoring checkpoint row without integer line_number: %s line %s",
                checkpoint_path,
                checkpoint_line_number,
            )
    return processed


def get_problem(record, problem_field):
    for field in (problem_field, "prompt", "question", "input"):
        value = record.get(field)
        if isinstance(value, str):
            return value
    raise ValueError(
        "Input record must contain a string field named "
        f"'{problem_field}', 'prompt', 'question', or 'input'."
    )


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
        if sufficiency_scores:
            additional_logs["sufficiency_scores"] = list(sufficiency_scores)
        method_results_dicts = getattr(response, "method_results_dicts", None)
        if method_results_dicts:
            additional_logs["method_results"] = list(method_results_dicts)
        if getattr(response, "generation_failed", False):
            additional_logs["failure_reason"] = getattr(
                response, "failure_reason", None
            )
            failure_max_completion_token = getattr(
                response,
                "failure_max_completion_token",
                None,
            )
            if failure_max_completion_token is not None:
                additional_logs["failure_max_completion_token"] = (
                    failure_max_completion_token
                )
        if getattr(response, "forced_final_answer", False):
            additional_logs["forced_final_answer"] = True
            additional_logs["thinking_limit_reached"] = bool(
                getattr(response, "thinking_limit_reached", False)
            )
            additional_logs["final_answer_truncated"] = bool(
                getattr(response, "final_answer_truncated", False)
            )
            additional_logs["final_answer_prompt_tokens"] = int(
                getattr(response, "final_answer_prompt_tokens", 0) or 0
            )
            additional_logs["logical_total_tokens"] = int(
                getattr(response, "logical_total_tokens", response.token_count())
            )
        score_usage = serialize_score_usage(response)
        if score_usage:
            additional_logs.update(score_usage)
        token_budgets = getattr(response, "token_budgets", [])
        return {
            "output": response.content,
            "logprobs": response.logprobs_as_dicts()
            if include_detailed_logprobs
            else [],
            "num_token": response.token_count(),
            "prompt_tokens": response.prompt_tokens,
            "thinking_tokens": getattr(response, "thinking_tokens", None),
            "answer_tokens": getattr(response, "answer_tokens", None),
            "completion_tokens": response.completion_tokens,
            "num_loop": response.num_loop,
            "num_chunk": response.num_chunk,
            **(
                {"generation_failed": True}
                if getattr(response, "generation_failed", False)
                else {}
            ),
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


def score_method_results(method_results, evaluator, answer):
    if answer is None:
        return method_results
    scored_results = []
    for method_result in method_results:
        scored = dict(method_result)
        if "final_answer" in scored:
            scored["correct"] = evaluator(scored["final_answer"], answer)
        scored_results.append(scored)
    return scored_results


def extract_output_for_scoring(output, think_end_token=None):
    """Return only the final-answer channel when the model exposes one."""
    text = str(output or "")
    if think_end_token and think_end_token in text:
        return text.rsplit(think_end_token, 1)[-1].strip()
    return text


def process_record(
    record_id,
    record,
    *,
    config,
    args,
    model_config,
    evaluator,
    include_detailed_logprobs,
):
    run_logger = logging.getLogger(f"early_exist.runner.record.{record_id}")
    is_reuse_record = isinstance(record.get("content"), str) and isinstance(
        record.get("input"), str
    )
    is_code = args.evaluator == "code_exec" or args.dataloader in {
        "humaneval",
        "livecodebench",
    }
    result = {
        "line_number": record_id,
        "agent": args.agent,
        "model": model_config["name"],
        "llm": model_config["id"],
        "dataloader": args.dataloader
        if (not args.input_file or is_reuse_record)
        else None,
        "evaluator": args.evaluator,
        "num_token": 0,
        "num_loop": 0,
        "num_chunk": 0,
    }
    if not is_reuse_record:
        excluded_log_fields = {"answer", "private_test_cases"} if is_code else set()
        result["input"] = {
            key: value
            for key, value in record.items()
            if key not in excluded_log_fields
        }
    if args.name:
        result["run_name"] = args.name

    try:
        run_logger.info("Started")
        agent = load_agent(config, args.agent, copy.deepcopy(model_config))
        problem = get_problem(record, args.problem_field)
        solve_kwargs = {
            "answer": record.get("answer"),
            "evaluator": evaluator,
            "evaluator_name": args.evaluator,
            "dataloader": args.dataloader,
            "dataset": args.dataloader,
            "is_code": is_code,
        }
        if isinstance(record.get("content"), str):
            solve_kwargs["reuse_generation_content"] = record["content"]
            if isinstance(record.get("input"), str):
                solve_kwargs["reuse_generation_input"] = record["input"]
        response = agent.solve(
            problem,
            **solve_kwargs,
        )
        serialized_response = serialize_agent_response(
            response,
            include_detailed_logprobs=include_detailed_logprobs,
        )
        result["output"] = serialized_response["output"]
        result["logprobs"] = serialized_response["logprobs"]
        result["num_token"] = serialized_response["num_token"]
        result["num_loop"] = serialized_response["num_loop"]
        result["num_chunk"] = serialized_response["num_chunk"]
        for token_field in (
            "prompt_tokens",
            "thinking_tokens",
            "answer_tokens",
            "completion_tokens",
        ):
            if serialized_response.get(token_field) is not None:
                result[token_field] = serialized_response[token_field]
        if "token_budgets" in serialized_response:
            result["token_budgets"] = serialized_response["token_budgets"]
        if "additional_logs" in serialized_response:
            result["additional_logs"] = serialized_response["additional_logs"]
        if serialized_response.get("generation_failed"):
            result["generation_failed"] = True
        if "answer" in record:
            scored_output = extract_output_for_scoring(
                result["output"],
                model_config.get("think_end_token"),
            )
            result["correct"] = evaluator(scored_output, record["answer"])
            method_results = (
                result.get("additional_logs", {}).get("method_results")
                if isinstance(result.get("additional_logs"), dict)
                else None
            )
            if method_results:
                result["additional_logs"]["method_results"] = score_method_results(
                    method_results,
                    evaluator,
                    record["answer"],
                )
        result["ok"] = True
        run_logger.info(
            "Finished ok=%s num_loop=%s num_chunk=%s num_token=%s",
            result.get("correct", True),
            result.get("num_loop"),
            result.get("num_chunk"),
            result.get("num_token"),
        )
    except Exception as exc:
        result["ok"] = False
        result["error"] = {
            "type": exc.__class__.__name__,
            "message": str(exc),
        }
        run_logger.exception("Failed")

    return result


def run(args):
    config = load_config(args.config)
    if getattr(args, "exit", None) is not None:
        replay_config = config.setdefault("agent_kwargs", {}).setdefault("replay_eval", {})
        replay_config["exit"] = args.exit
    if args.list_models:
        print_model_configs(config)
        return
    if get_replay_version(config) == 2 or random_exit_enabled(
        config, args.random_exit
    ):
        return run_v2(args, config)
    if args.num_thread < 1:
        raise ValueError("--num-thread must be >= 1")
    if args.max_error < 0:
        raise ValueError("--max-error must be >= 0")
    if args.limit is not None and args.limit < 0:
        raise ValueError("--limit must be >= 0")
    if args.max_total_thinking_token is not None and args.max_total_thinking_token < 1:
        raise ValueError("--max-total-thinking-token must be >= 1")
    if (
        args.max_final_answer_completion_token is not None
        and args.max_final_answer_completion_token < 1
    ):
        raise ValueError("--max-final-answer-completion-token must be >= 1")

    setup_logging(args.log_level, args.log_file)
    load_optional_hf_token()
    LOGGER.info("Starting runner")
    LOGGER.info("Config: %s", args.config)
    LOGGER.info("Output file: %s", args.output_file)
    LOGGER.info("Run name: %s", args.name or "")
    LOGGER.info("Reverse: %s", args.reverse)
    LOGGER.info("Num thread: %s", args.num_thread)
    LOGGER.info("Max error: %s", args.max_error)
    if args.resume and args.resume_from:
        raise ValueError("Use either --resume or --resume-from, not both.")
    checkpoint_file = args.resume_from or (args.output_file if args.resume else None)
    include_detailed_logprobs = bool(
        config.get("llm", {}).get("include_detailed_logprobs", False)
    )

    model_config = resolve_model_config(config, args.model)
    if args.llm:
        model_config["id"] = args.llm
    model_config = apply_system_prompts(
        model_config,
        dataset_system_prompt=load_system_prompt(config, "dataset", args.dataloader),
        agent_system_prompt=load_system_prompt(config, "agent", args.agent),
    )
    completion_kwargs = dict(model_config.get("completion_kwargs") or {})
    if args.max_total_thinking_token is not None:
        completion_kwargs["max_completion_token"] = args.max_total_thinking_token
        completion_kwargs.pop("max_tokens", None)
        completion_kwargs.pop("max_competion_token", None)
    if args.max_final_answer_completion_token is not None:
        completion_kwargs["max_final_answer_completion_token"] = (
            args.max_final_answer_completion_token
        )
    model_config["completion_kwargs"] = completion_kwargs

    LOGGER.info("Using model '%s' id='%s'", model_config["name"], model_config["id"])
    LOGGER.info(
        "Token budgets: thinking=%s final_answer=%s",
        completion_kwargs.get("max_completion_token"),
        completion_kwargs.get("max_final_answer_completion_token"),
    )
    LOGGER.info("Loading evaluator '%s'", args.evaluator)
    evaluator = import_config_entry(config, "evaluator", args.evaluator)

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed_record_ids = (
        load_processed_line_numbers(checkpoint_file) if checkpoint_file else set()
    )
    if checkpoint_file:
        LOGGER.info(
            "Resume checkpoint: %s processed=%s",
            checkpoint_file,
            len(processed_record_ids),
        )

    total = 0
    succeeded = 0
    failed = 0
    skipped = 0
    interrupted = False

    def unprocessed_records():
        nonlocal skipped
        for record_id, record in load_records(config, args):
            if record_id in processed_record_ids:
                skipped += 1
                continue
            yield record_id, record

    with open(output_path, "a") as output:
        records_to_run = unprocessed_records()
        if args.num_thread == 1:
            result_iter = (
                process_record(
                    record_id,
                    record,
                    config=config,
                    args=args,
                    model_config=model_config,
                    evaluator=evaluator,
                    include_detailed_logprobs=include_detailed_logprobs,
                )
                for record_id, record in records_to_run
            )
        else:
            records = list(records_to_run)
            executor = ThreadPoolExecutor(max_workers=args.num_thread)
            result_iter = executor.map(
                lambda item: process_record(
                    item[0],
                    item[1],
                    config=config,
                    args=args,
                    model_config=model_config,
                    evaluator=evaluator,
                    include_detailed_logprobs=include_detailed_logprobs,
                ),
                records,
            )

        try:
            for result in result_iter:
                total += 1
                if result.get("ok"):
                    succeeded += 1
                else:
                    failed += 1
                output.write(json.dumps(result, ensure_ascii=False) + "\n")
                output.flush()

                if failed > args.max_error:
                    interrupted = True
                    LOGGER.error(
                        "Interrupting runner because failed=%s exceeded max_error=%s",
                        failed,
                        args.max_error,
                    )
                    break
        finally:
            if args.num_thread > 1:
                executor.shutdown(wait=not interrupted, cancel_futures=interrupted)

    LOGGER.info(
        "Finished runner total=%s skipped=%s succeeded=%s failed=%s",
        total,
        skipped,
        succeeded,
        failed,
    )
    if interrupted:
        raise RuntimeError(f"Interrupted runner after {failed} record errors.")


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
