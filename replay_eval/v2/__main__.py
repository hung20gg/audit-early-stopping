from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path

from replay_eval.v2.config.schema import load_config
from replay_eval.v2.runtime.runner import ReplayV2Runner
from scripts.config_loader import (
    import_config_entry,
    load_config as load_shared_config,
    load_system_prompt,
)


def _parse_bool(value: str) -> bool:
    normalized = value.strip().casefold()
    if normalized in {"true", "1", "yes", "on"}:
        return True
    if normalized in {"false", "0", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run asynchronous native-vLLM replay evaluation"
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[2] / "scripts" / "config.yml"),
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model key from llm.models; optional vllm values override llm.vllm.",
    )
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--evaluator", default="gsm8k")
    parser.add_argument("--dataloader", default=None)
    parser.add_argument("--exit", type=_parse_bool, default=None)
    parser.add_argument("--rerun-all", type=_parse_bool, default=None)
    parser.add_argument("--rerun-final-answer", type=_parse_bool, default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from --output-file and append only unfinished records.",
    )
    parser.add_argument(
        "--resume-from",
        default=None,
        help="Read completed records from another JSONL checkpoint.",
    )
    parser.add_argument("--max-final-answer-completion-token", type=int, default=None)
    parser.add_argument("--random-exit", type=_parse_bool, default=None)
    parser.add_argument("--random-exit-sample-size", type=int, default=None)
    parser.add_argument("--random-exit-sample-seed", type=int, default=None)
    parser.add_argument("--random-exit-position-seed", type=int, default=None)
    parser.add_argument("--random-exit-spacing", type=int, default=None)
    parser.add_argument("--random-exit-max-positions", type=int, default=None)
    parser.add_argument("--random-exit-num-runs", type=int, default=None)
    parser.add_argument("--random-exit-min-valid-positions", type=int, default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


async def _main() -> None:
    args = parse_args()
    if args.resume and args.resume_from:
        raise ValueError("Use either --resume or --resume-from, not both")
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    )
    config = load_config(args.config, model_name=args.model)
    if args.exit is not None:
        config.runtime.exit = args.exit
    if args.rerun_all is not None:
        config.runtime.rerun_all = args.rerun_all
    if args.rerun_final_answer is not None:
        config.runtime.rerun_final_answer = args.rerun_final_answer
    if args.max_final_answer_completion_token is not None:
        config.runtime.max_final_answer_completion_token = (
            args.max_final_answer_completion_token
        )
    if args.random_exit is not None:
        config.random_exit.enabled = args.random_exit
    for field in (
        "sample_size",
        "sample_seed",
        "position_seed",
        "spacing",
        "max_positions",
        "num_runs",
        "min_valid_positions",
    ):
        value = getattr(args, f"random_exit_{field}")
        if value is not None:
            setattr(config.random_exit, field, value)
    config.validate()
    shared_config = load_shared_config(args.config)
    if args.dataloader:
        system_prompt = load_system_prompt(shared_config, "dataset", args.dataloader)
        if system_prompt:
            config.rendering.system_prompt = system_prompt
    evaluator = import_config_entry(shared_config, "evaluator", args.evaluator)
    recovery_dataloader = (
        import_config_entry(shared_config, "dataloader", args.dataloader)
        if args.dataloader
        else None
    )
    metrics = await ReplayV2Runner(
        config,
        evaluator=evaluator,
        evaluator_name=args.evaluator,
    ).run(
        args.input_file,
        args.output_file,
        resume_from=(args.output_file if args.resume else args.resume_from),
        recovery_dataloader=recovery_dataloader,
        dataloader_name=args.dataloader,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    asyncio.run(_main())
