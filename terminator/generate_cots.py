from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataloader.math_train import load_math_train
from terminator.hf_runtime import build_chat_prompt, load_model_and_tokenizer

CHAT_TEMPLATE_VERSION = "hf-apply_chat_template-v1"
THINK_START = "<think>"
THINK_END = "</think>"


def split_problems(
    problems: list[dict],
    seed: int = 1337,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
) -> dict[str, list[dict]]:
    """Shuffle-and-slice by problem id, 80/10/10 by default. Splitting happens
    here, before CoT sampling, so multiple CoTs from one problem stay together."""
    shuffled = list(problems)
    random.Random(seed).shuffle(shuffled)

    n_total = len(shuffled)
    n_train = int(n_total * train_frac)
    n_val = int(n_total * val_frac)

    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }


def _think_boundary_token_ids(tokenizer) -> tuple[int | None, int | None]:
    start_id = tokenizer.convert_tokens_to_ids(THINK_START)
    end_id = tokenizer.convert_tokens_to_ids(THINK_END)
    unk_id = getattr(tokenizer, "unk_token_id", None)
    start_id = None if start_id == unk_id else start_id
    end_id = None if end_id == unk_id else end_id
    return start_id, end_id


def _split_reasoning_and_solution(full_response: str) -> tuple[str, str]:
    """Split generated continuation into thinking text and final solution.

    Qwen3's chat template opens `<think>\n` as part of the *prompt* when
    `enable_thinking=True`, so the generated continuation usually contains
    only the closing `</think>`, not the opening tag. Handle both that case
    and models that regenerate the literal opening tag themselves.
    """
    if THINK_END not in full_response:
        return full_response, ""
    reasoning_text, _, final_solution = full_response.partition(THINK_END)
    if THINK_START in reasoning_text:
        reasoning_text = reasoning_text.split(THINK_START, 1)[1]
    return reasoning_text, final_solution.strip()


def generate_one_cot(
    loaded,
    problem: dict,
    *,
    seed: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
) -> dict[str, Any]:
    import torch

    model, tokenizer = loaded.model, loaded.tokenizer
    prompt = build_chat_prompt(tokenizer, problem["question"], enable_thinking=True)
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids
    prompt_ids = prompt_ids.to(loaded.device)

    torch.manual_seed(seed)
    with torch.no_grad():
        output = model.generate(
            prompt_ids,
            attention_mask=torch.ones_like(prompt_ids),
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )

    reasoning_token_ids = output[0][prompt_ids.shape[1] :].tolist()
    full_response = tokenizer.decode(reasoning_token_ids, skip_special_tokens=False)
    reasoning_text, final_solution = _split_reasoning_and_solution(full_response)
    think_start_token, think_end_token = _think_boundary_token_ids(tokenizer)

    return {
        "problem_id": problem["problem_id"],
        "prompt": prompt,
        "model_name": loaded.model_id,
        "chat_template_version": CHAT_TEMPLATE_VERSION,
        "seed": seed,
        "temperature": temperature,
        "top_p": top_p,
        "full_response": full_response,
        "reasoning_text": reasoning_text,
        "final_solution": final_solution,
        "reasoning_token_ids": reasoning_token_ids,
        "think_start_token": think_start_token,
        "think_end_token": think_end_token,
    }


def generate_split(
    loaded,
    problems: list[dict],
    *,
    cots_per_problem: int,
    base_seed: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
) -> list[dict]:
    rows = []
    for problem in problems:
        for cot_index in range(cots_per_problem):
            rows.append(
                generate_one_cot(
                    loaded,
                    problem,
                    seed=base_seed + cot_index,
                    temperature=temperature,
                    top_p=top_p,
                    max_new_tokens=max_new_tokens,
                )
            )
    return rows


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate raw TERMINATOR CoTs.")
    parser.add_argument("--model", required=True, help="HF model id, e.g. Qwen/Qwen3-4B")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dataset-limit", type=int, default=None)
    parser.add_argument("--cots-per-problem", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("terminator/data/raw_cots/pilot"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    problems = load_math_train(limit=args.dataset_limit)
    splits = split_problems(problems, seed=args.seed)

    loaded = load_model_and_tokenizer(args.model, device=args.device)

    for split_name, split_problems_ in splits.items():
        rows = generate_split(
            loaded,
            split_problems_,
            cots_per_problem=args.cots_per_problem,
            base_seed=args.seed,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
        )
        output_path = args.output_dir / f"{split_name}.jsonl"
        write_jsonl(rows, output_path)
        print(f"{split_name}: {len(split_problems_)} problems -> {len(rows)} CoTs -> {output_path}")


if __name__ == "__main__":
    main()