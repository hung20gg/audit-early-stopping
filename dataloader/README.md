# Dataloader and Evaluation Guide

All dataloaders should return records in this shape:

```json
{
    "question": "What is the capital of France?",
    "answer": "Paris"
}
```

If the question is multiple choice, flatten the choices into the question and keep the answer as a string:

```json
{
    "question": "What is the capital of France? (A) Berlin (B) Madrid (C) Paris (D) Rome",
    "answer": "C"
}
```

## Config

Evaluation components are imported from `scripts/config.yml`:

```yaml
method:
  simple: methods.simple.SimpleAgent
  deer: methods.deer.DEERAgent

dataloader:
  gsm8k: dataloader.gsm8k.load_gsm8k

evaluator:
  gsm8k: evaluator.gsm8k.math_equal

system_prompt:
  dataset:
    gsm8k: prompt/gsm8k.md
  agent: {}

agent_kwargs:
  simple:
    max_steps: 50
  simple_full:
    max_steps: 50
  deer:
    threshold: 0.95
    max_steps: 50
    probe_tokens: 20
    probe_logprobs: 3
  deer_full:
    threshold: 0.95
    max_steps: 50
    probe_tokens: 20
    probe_logprobs: 3

llm:
  default: qwen3_32b_think
  logprobs: 5
  completion_kwargs:
    stop:
      - "\n\n"
      - "Wait"
    max_thinking_completion_token: 50
    max_completion_token: 12000
  models:
    qwen3_32b_think:
      name: qwen3_32b_think
      id: deepinfra:Qwen/Qwen3-32B
      think_start_token: "<think>"
      think_end_token: "</think>"
      chat_template:
        type: hf
        model_id: Qwen/Qwen3-32B
        add_generation_prompt: true
        kwargs:
          enable_thinking: true
      completion_kwargs:
        temperature: 0.6
```

The runner defaults to `gsm8k` for both dataloader and evaluator.

List configured models:

```bash
python scripts/runner.py --list-models
```

Each model config has:

- `name`: local alias
- `id`: actual string passed to `get_llm_wrapper`
- `think_start_token`: token that opens model thinking text, commonly `<think>`
- `think_end_token`: token that closes model thinking text, commonly `</think>`
- `chat_template`: agent-side chat template controller config
- `completion_kwargs`: merged model/global completion settings. The agent consumes internal keys such as `stop`, `max_thinking_completion_token`, and `max_completion_token` before provider calls.
- `logprobs`: global top-logprob count for all LLM completion calls. Must be an integer from `1` to `5`.
- `include_detailed_logprobs`: when `true`, the runner writes detailed token logprobs into the JSONL `logprobs` field. Default is `false`.

The agent applies the configured chat template before calling `llm.completion()`. This bypasses the wrapper's default chat-template path because the wrapper receives a rendered prompt string.

Generation is chunked locally:

- Thinking calls request `max_thinking_completion_token` tokens per step and do not pass `stop`/`stop_token` to the provider.
- Configured stop markers are scanned in the returned chunk. The chunk is kept through the first matched marker, for example `\n\n` or `Wait`.
- If a thinking chunk uses fewer than `max_thinking_completion_token` tokens, or the provider finish reason is not `length`, generation is treated as ended.
- Final answer calls use `max_completion_token` and also do not pass provider stop tokens.
- DEER writes probe trace data into JSONL `additional_logs.probe_scores` as `[(position, score)]`.
- DTSR writes sufficiency trace data into JSONL `additional_logs.sufficiency_scores` as `[(position, score)]`.
- DEER requests only `max_thinking_completion_token` on the first thinking call. If the returned text crosses a local stop marker but has fewer than `probe_tokens` after that marker, DEER makes one continuation call for only the missing probe tokens and reuses that continuation when appending a low-confidence probe.

System prompts are split by role:

- `system_prompt.dataset`: task prompt loaded as `chat_template.generation_system_prompt`. The agent builds the generation messages and then passes those messages into the chat template renderer. Example: `gsm8k -> prompt/gsm8k.md`.
- `system_prompt.agent`: optional internal prompt for agent-control calls, such as a future LLM-based `should_stop` decision. It is loaded onto `agent.agent_system_prompt` and is not inserted into the dataset chat template.

## Agents

- `simple`: generates thinking chunks until both think tokens appear, the provider signals generation ended, the chunk is shorter than `max_thinking_completion_token`, or the simple wait/but heuristic fires.
- `deer`: DEER-style early-exit agent. It generates a reasoning chunk, finds the local break marker, scores `probe_tokens` tokens after that marker when available, and compares the probe mean max token probability to `threshold`. If the initial chunk has too few post-marker tokens, DEER generates only the missing probe tokens. If the probe is below threshold, DEER appends the already-generated probe text back without injecting any control token.
- `dtsr`: Dynamic Thought Sufficiency in Reasoning. It checks a sufficiency score only after a reflection signal and `min_interval` generated tokens; if the score reaches `threshold`, it closes thinking with `</think>` and moves to final answer generation.

## GSM8K Dataloader

`dataloader/gsm8k.py` loads the Hugging Face dataset directly:

- dataset: `openai/gsm8k`
- subset: `main`
- default split: `test`
- output schema: `question`, `answer`

GSM8K answers from Hugging Face include worked solutions with a final `####` marker. The loader keeps only the final answer after `####`.

To export JSONL manually:

```bash
python dataloader/gsm8k.py \
  --output-file data/gsm8k_test.jsonl
```

For a small export:

```bash
python dataloader/gsm8k.py \
  --limit 5 \
  --output-file data/gsm8k_test_5.jsonl
```

## Smoke Check

Use the smoke runner to check that config imports, dataloader loading, and evaluator loading work. This does not require an LLM unless `--llm` is provided.

```bash
python scripts/smoke_runner.py --limit 1
```

Expected behavior:

- imports `methods.simple.SimpleAgent`
- imports `dataloader.gsm8k.load_gsm8k`
- imports `evaluator.gsm8k.math_equal`
- loads one GSM8K test example
- runs an evaluator self-check against the example answer

## HaluEval Dataloader

`dataloader/halueval.py` loads the `general` subset of `pminervini/HaluEval`:

- split: `data`
- input columns: `user_query`, `chatgpt_response`, `hallucination`
- output schema: `question`, `answer`, where `answer` is `yes` or `no`

Use it with the `halueval` evaluator and dataset prompt. The model must return
only `\boxed{yes}` or `\boxed{no}`.

```bash
python scripts/smoke_runner.py \
  --dataloader halueval \
  --evaluator halueval \
  --split data \
  --limit 1
```

## Main Runner

The main runner defaults to GSM8K:

```bash
python scripts/runner.py \
  --model qwen3_8b \
  --limit 1
```

Run with DEER:

```bash
python scripts/runner.py \
  --agent deer \
  --model qwen3_8b \
  --limit 1
```

Default values:

- `--config scripts/config.yml`
- `--agent simple`
- `--model llm.default`, currently `qwen3_32b`
- `--dataloader gsm8k`
- `--evaluator gsm8k`
- `--split test`
- `--reverse` disabled
- `--output-file outputs/gsm8k.jsonl`
- `--num-thread 1`
- `--max-error 3`

`--num-thread` controls record-level concurrency. The runner only creates a thread pool when `--num-thread` is greater than `1`; the default path remains serial. Each record gets its own agent instance.

`--max-error` controls failure interruption. The runner writes each failed record to JSONL, then interrupts once the failed-record count is greater than `--max-error` and exits non-zero.

Pass `--reverse` to process records from last to first. Reversal happens before `--limit`, while original record IDs are preserved in the output `line_number`, so forward and reverse runs can use the same resume checkpoint.

To run records concurrently:

```bash
python scripts/runner.py \
  --model qwen3_8b \
  --limit 10 \
  --num-thread 4
```

To attach a custom run name to every JSONL row:

```bash
python scripts/runner.py \
  --name exp \
  --model qwen3_8b \
  --limit 1
```

To override the output path:

```bash
python scripts/runner.py \
  --model qwen3_8b \
  --limit 1 \
  --output-file outputs/gsm8k_smoke.jsonl
```

To override the actual model id while keeping the configured chat template:

```bash
python scripts/runner.py \
  --model qwen3_8b \
  --llm vllm:Qwen/Qwen3-8B \
  --limit 1
```

To use an existing JSONL file instead of the configured dataloader:

```bash
python scripts/runner.py \
  --model qwen3_8b \
  --input-file data/gsm8k_test_5.jsonl \
  --output-file outputs/gsm8k_from_file.jsonl
```

To resume an interrupted file-based run, append to the same output file and skip
records whose ID in `line_number` already appears in that output:

```bash
python scripts/runner.py \
  --model qwen3_8b \
  --input-file data/gsm8k_test_5.jsonl \
  --output-file outputs/gsm8k_from_file.jsonl \
  --resume
```

To resume from one checkpoint file while writing remaining rows somewhere else:

```bash
python scripts/runner.py \
  --model qwen3_8b \
  --input-file data/gsm8k_test_5.jsonl \
  --resume-from outputs/gsm8k_from_file.jsonl \
  --output-file outputs/gsm8k_remaining.jsonl
```

## Evaluation Wrapper

`scripts/run_eval.sh` wraps `scripts/runner.py` and creates timestamped output/log paths:

```bash
bash scripts/run_eval.sh \
  --name exp \
  --num-thread 4 \
  --max-error 3 \
  --agent simple \
  --model qwen3_8b_think \
  --limit 10
```

When resuming through the wrapper, use `--resume` only with an explicit
`--output-file` because the wrapper normally creates a fresh timestamped path:

```bash
bash scripts/run_eval.sh \
  --model qwen3_8b \
  --input-file data/gsm8k_test_5.jsonl \
  --output-file outputs/gsm8k_from_file.jsonl \
  --resume
```

Use `--resume-from` when you want the wrapper to create a new timestamped output
file for records missing from an older checkpoint:

```bash
bash scripts/run_eval.sh \
  --model qwen3_8b \
  --input-file data/gsm8k_test_5.jsonl \
  --resume-from outputs/gsm8k_from_file.jsonl
```

Default generated paths are:

- `output/<dataset_name>/<agent>_<model_name>_<datetime>.jsonl`
- `logs/<dataset_name>/<agent>_<model_name>_<datetime>.log`

When `--name` is provided, it is prepended to the generated filenames:

- `output/<dataset_name>/<name>_<agent>_<model_name>_<datetime>.jsonl`
- `logs/<dataset_name>/<name>_<agent>_<model_name>_<datetime>.log`

The same name is forwarded to the runner and written as `run_name` in each JSONL row.

To run the wrapper multiple times sequentially, use `scripts/run_eval_many.sh`:

```bash
bash scripts/run_eval_many.sh \
  --num-run 3 \
  --name exp \
  --num-thread 4 \
  --agent simple \
  --model qwen3_8b_think \
  --limit 10
```

This runs `scripts/run_eval.sh` three times with `--name exp_1`, `--name exp_2`, and `--name exp_3`. `--runs` is accepted as a compatibility alias for `--num-run`.

`run_eval_many.sh` captures the input arguments once before the loop and reuses them for each run. Do not pass `--output-file` or `--log-file`; the wrapper generates separate paths for each numbered run.

## Logging

`py_compile` only checks syntax. It does not execute the runner, so it will not produce runner logs:

```bash
python -m py_compile scripts/runner.py
```

To see logs, execute the runner:

```bash
python scripts/runner.py \
  --model qwen3_8b \
  --limit 1 \
  --log-level INFO
```

To write logs to a file:

```bash
python scripts/runner.py \
  --model qwen3_8b \
  --limit 1 \
  --log-file logs/gsm8k.log \
  --log-level INFO
```

The runner uses a global logger named `early_exist.runner` and per-record loggers named like `early_exist.runner.record.1`.
