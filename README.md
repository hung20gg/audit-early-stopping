# Early exist

## Notices

- Run project commands through the `text2sql` conda environment on quanghung20gg's computer.
- Do not edit `llm/` unless the task specifically asks for it. It is managed separately and should normally be updated with `git submodule update --remote llm`.
- Early-exit generation must keep the original rendered prompt when appending model output. `Response.input` stores that prompt, and `Response.get_generation_content()` returns `input + content` for follow-up completion calls.
- Some providers, including DeepInfra for `/completions`, may return `logprobs=None` even when `logprobs` is requested. In that case the runner writes an empty `logprobs` list while still using the generated text.
- `llm.include_detailed_logprobs` controls whether the runner writes detailed token logprobs into JSONL output. It defaults to `false`.
- Thinking generation is chunked locally. The agent does not pass configured stop tokens to provider completion calls; it requests `max_thinking_completion_token` tokens, then scans the returned chunk for local stop markers such as `\n\n` or `Wait`.
- If a thinking chunk returns fewer tokens than `max_thinking_completion_token`, or the provider finish reason is not `length`, generation is treated as ended. Final answer generation uses `max_completion_token`.
- Temporary answers, including PUMA trial answers, are fresh model calls through `generate_temp_answer()`. The temporary-answer cache only deduplicates repeated requests for the same rendered prompt and replay position; it does not reuse the full-generation answer.

Clone llm folder before run

```bash
git clone https://github.com/hung20gg/llm.git
```

## GSM8K evaluation

Run commands through the `text2sql` conda environment:

```bash
python scripts/smoke_runner.py --limit 1
```

Run the default GSM8K evaluation:

```bash
bash scripts/run_eval.sh \
  --agent simple \
  --model qwen3_8b_think \
  --limit 1
```

Run with a custom run name and multiple record workers:

```bash
bash scripts/run_eval.sh \
  --name exp \
  --num-thread 4 \
  --max-error 3 \
  --agent deer \
  --model qwen36_35b_a3b \
  --limit 10
```

The wrapper writes detailed JSONL responses and logs to timestamped paths:

- `output/<dataset_name>/<agent>_<model_name>_<datetime>.jsonl`
- `logs/<dataset_name>/<agent>_<model_name>_<datetime>.log`
- when `--name` is provided: `output/<dataset_name>/<name>_<agent>_<model_name>_<datetime>.jsonl`
- each JSONL row also includes `run_name` when `--name` is provided

For example, GSM8K with `simple` and `qwen3_8b_think` writes under `output/gsm8k/` and `logs/gsm8k/`.

Run the wrapper multiple times sequentially with numbered run names:

```bash
bash scripts/run_eval_many.sh \
  --num-run 3 \
  --name exp \
  --num-thread 4 \
  --agent simple \
  --model qwen3_8b_think \
  --limit 10
```

This calls `scripts/run_eval.sh` three times with names `exp_1`, `exp_2`, and `exp_3`. The repeated-run wrapper captures the input arguments once before the loop and reuses them for every run. Do not pass `--output-file` or `--log-file` to `run_eval_many.sh`; it generates separate paths for each run.

## V2

There is one runtime configuration file: `scripts/config.yml`. Select replay
v1 or v2 with:

```yaml
agent_kwargs:
  replay_eval:
    version: 2  # change back to 1 for the existing synchronous replay
```

Version 2 combines every uncommented shared method under
`agent_kwargs.replay_eval.methods` with the Terminator and Thought Calibration
methods under `agent_kwargs.replay_eval.v2.methods`, in that order. The committed
configuration resolves 22 unique active methods; `--rerun-all true` reruns all 22,
while `false` runs only names missing from that same union. Runtime and extraction
settings come from the adjacent `agent_kwargs.replay_eval.v2` mapping. The required `--model` argument selects
an existing entry under `llm.models`; no second model/chat-template config is
needed.
Completion temperature and token budgets come from that shared model/LLM
configuration, replay code multipliers come from `agent_kwargs.replay_eval`,
and shared native engine options come from `llm.vllm`. A selected model's
optional `vllm` mapping holds only key-specific overrides.
One long-lived native vLLM engine serves every method and row. Identical prompt/checkpoint requests are shared, and canonical prefixes allow the enabled vLLM KV-prefix cache to be reused without recreating the engine.
Shared prefix caching, chunked prefill, and continuous-batching limits are
configured once under `llm.vllm.serving`, together with the Qwen3-Next
MTP speculative-decoding settings.

Version 2 currently runs JSONL input directly:

```bash
bash scripts/run_eval.sh \
  --name full_v2 \
  --agent replay_eval \
  --model qwen35_4b \
  --input-file INPUT.jsonl \
  --output-file OUTPUT.jsonl
```

## Note for rerun evaluation

Download the original JSONL
```bash
python -m scripts.download_trace_from_hf hung20gg/early-exit --destination output  --filename checkpoints.zip
```

Download the model weights
```bash
python -m scripts.download_trace_from_hf hung20gg/early-exit  --filename checkpoints.zip
```

Rerun evaluation with the same model and input file, but only rerun final answer generation:
```bash
bash scripts/run_eval.sh \
  --name full_v2 \
  --agent replay_eval \
  --model qwen35_4b \
  --input-file INPUT.jsonl \
  --output-file OUTPUT.jsonl \
  --rerun-all false \
  --rerun-final-answer true 
```

Replay-v2 now writes the selected model config key to every new trace row,
including rows recovered from a missing-trajectory input. For an older JSONL,
add `"model": "gpt_oss_20b_high"` without rerunning evaluation:

```bash
conda run -n text2sql python scripts/add_model_key.py INPUT.jsonl \
  --output PATCHED.jsonl
```

To patch atomically in place while retaining a backup:

```bash
conda run -n text2sql python scripts/add_model_key.py INPUT.jsonl \
  --in-place \
  --backup-suffix .bak
```

`gpt_oss_20b_high` is the default; use `--model MODEL_KEY` for another model.
Existing `model` values are replaced with the requested key.


It uses asynchronous `runtime.example_concurrency` over one shared vLLM engine instead of OS-worker `--num-thread` and does not
yet support `--resume`, `--reverse`, or `--limit`. Detailed GPU integration and
callback requirements are in [`replay_eval/v2/README.md`](replay_eval/v2/README.md).
Notice that the committed callback path is a placeholder: replace it and
provide `pre_final_rmsnorm` extraction before enabling Terminator or Thought
Calibration on the GPU.

You can still pass normal runner options through the wrapper:

```bash
bash scripts/run_eval.sh \
  --agent simple \
  --model qwen3_8b_think \
  --dataloader gsm8k \
  --split test \
  --limit 10
```

## Code evaluation

Run HumanEval:

```bash
bash scripts/run_eval.sh \
  --agent simple \
  --model qwen3_8b_think \
  --dataloader humaneval \
  --evaluator code_exec \
  --limit 5
```

Run LiveCodeBench:

```bash
bash scripts/run_eval.sh \
  --agent simple \
  --model qwen3_8b_think \
  --dataloader livecodebench \
  --evaluator code_exec \
  --limit 5
```

Code datasets use the `code_exec` evaluator instead of the default `gsm8k` evaluator. The evaluator runs generated code in a subprocess with a 30-second timeout.

### LiveCodeBench evaluator patch

Use `scripts/rerun_livecodebench_eval.py` to apply the current
`evaluator.code_exec_fix.code_exec_equal` implementation to existing LiveCodeBench
runner outputs:

```bash
python scripts/rerun_livecodebench_eval.py \
  examples/output/livecodebench \
  --num-thread 8
```

> [!WARNING]
> Runner JSONL records contain `input.public_test_cases`, but normally do not
> contain `input.private_test_cases`. Passing `record["input"]` directly to an
> evaluator therefore checks public tests only, even when the evaluator's
> `test_type` is `full`. Use one of the dataset-backed rerun scripts below.
> They load the matching LiveCodeBench release and resolve each record using
> its one-based `line_number`, making both public and private tests available.
> The `--split` and `--version-tag` values must match the original run.

To recheck every failed or missing LeetCode result in place with the fixed
functional/stdin evaluator and private tests:

```bash
conda run -n verl python scripts/rerun_leetcode_code_exec_fix.py \
  INPUT.jsonl \
  --rerun failed \
  --in-place \
  --threads 16
```

To evaluate only entries that do not have a `correct` key, writing the results
to a separate directory:

```bash
conda run -n verl python scripts/rerun_livecodebench_eval.py \
  INPUT.jsonl \
  --empty-only \
  --num-threads 16 \
  --output-dir output/livecodebench-empty-fixed
```

`--empty-only` means the `correct` key is absent. Entries containing an
explicit `correct: true`, `correct: false`, or `correct: null` are preserved.

`rerun_livecodebench_eval.py` does not modify the original JSONL files. By
default, it writes the reevaluated records, with the same filenames, to:

```text
examples/output/livecodebench-fixed/
```

Use `--output-dir <directory>` to select another destination, or `--dry-run`
to evaluate and print statistics without writing files.

The script loads the requested LiveCodeBench dataset version and maps each
output record's one-based `line_number` to the corresponding dataset row. It
rebuilds the evaluator reference from that complete row so private test cases
and either LiveCodeBench metadata spelling (`metadata` or `meta_data`) are
available. This is required for functional problems, where
`metadata.func_name` identifies the generated method to call.

For every record, the script updates:

- the top-level `correct` status using `output`;
- each direct `method_results[*].correct` status that has a `final_answer`;
- each legacy `additional_logs.method_results[*].correct` status that has a
  `final_answer`.

Records with a missing or non-string `output` are copied unchanged and counted
as skipped. Reports are printed per file and in total, with separate top-level
and method-result counts for:

- evaluated and unchanged statuses;
- `false -> true` and `true -> false` changes;
- previously missing statuses that were added;
- final true and false totals;
- skipped records.

The dataset options must match the original run. They default to `test` and
`release_v6`:

```bash
python scripts/rerun_livecodebench_eval.py \
  examples/output/livecodebench \
  --split test \
  --version-tag release_v6 \
  --output-dir examples/output/livecodebench-fixed \
  --num-thread 8
```


List configured models:

```bash
python scripts/runner.py --list-models
```

The runner defaults to `scripts/config.yml`, `agent=simple`, `model=llm.default`, `dataloader=gsm8k`, `evaluator=gsm8k`, and `split=test`.
`--num-thread` defaults to `1`, so execution stays serial unless you set a value greater than `1`.
`--max-error` defaults to `3`; the runner interrupts after more than that many record errors and exits non-zero.
For file-based runs, add `--resume` to append to `--output-file` while skipping input lines already present in that JSONL output, or use `--resume-from <checkpoint.jsonl>` to read processed line numbers from a separate checkpoint.

In replay evaluation, the full generation is reused only as the reasoning trajectory. Verifiers that need a temporary answer still call the model through `generate_temp_answer()`, with per-position caching to avoid duplicate calls when multiple checks hit the same position.

### Scalable random-exit method replay

See [the random-exit replay evaluation guide](docs/random_exit_replay_evaluation.md)
for input format, a Python example, capacity planning, artifact layout, and
interpretation of supported and unsupported estimates.

For method evaluation from saved random exits, use
`replay_eval.v2.runtime.random_exit_evaluation.RandomExitReplayEvaluation`.
It accepts prepared trace records (`data_id`, `question`, `response` with the
original rendered `input`, `output_token_ids`, and `reasoning_token_count`), a
canonical renderer, an artifact directory, optional engine and grader, and
`SamplingSettings`. Call `plan()` before `collect_missing()` to see the exact
call counts, prefill and maximum generated tokens, storage by component, pilot
GPU-hour projection, and advisory warnings. `sample_size=None` selects every
supplied trace; the existing 64-trace setting in `scripts/config.yml` applies
only to the older continuation-stability runner. `collect_missing()` saves one
artifact per exact prompt, decoder, budget, position, sample slot and seed, so
it can resume after interruption. The rendered prompt appears once per trace.

Call `prepare_features(version, extractor)` to derive versioned features from
saved answers, `replay(methods)` for generation-free checkpoint and verifier
decisions, `estimate(decisions)` for exact-position correctness and Wilson
uncertainty, and `report(estimates)` for coverage and accuracy versus compute.
`pilot_measurements()` summarizes a small server run for later capacity plans.
Optional `paired_live_rows` in `report()` compares held-out live correctness
and method ranking. Unsupported positions stay unsupported; nearby estimates
from `SampledExitEstimator` are separate and can be checked with its dense
`audit()` method. A method that changes the reasoning continuation requires a
paired live run. Trial-answer and reasoning-state retention must be explicitly
requested; engines that do not expose requested trial states fail collection
clearly. New features that need discarded states require hidden-state
extraction from the stored token IDs.

### Random-exit continuations

Random-exit mode measures continuation stability
at fixed positions of already-generated reasoning trajectories. It does not run
the ordinary replay methods, extract hidden states, or generate replacement full
trajectories.

First create the permanent dataset-first manifest and one compact replay input
per existing model run:

```bash
python -m random_exit.sample_existing_runs \
  --config scripts/config.yml \
  --dataset aime26 \
  --split test \
  --output-dir data/random_exit/aime26 \
  --run qwen35_9b=output/aime26/original-qwen35-9b.jsonl \
  --run gemma4=output/aime26/original-gemma4.jsonl
```

Before sampling, the command intersects the line numbers in every supplied
`--run`/`--input-file`; an example missing from any run is reported and skipped,
so every sampled ID is available for every model. The command then creates
`aime26.test.manifest.json` once and reuses it unchanged for later models.
Manifests created by the older, dataset-only sampler are migrated once using
this intersection. The manifest stores benchmark-scoped IDs such as
`aime26:17`, the original one-based `line_number`, dataset/config metadata,
and a content fingerprint. A dataset with fewer than `sample_size` records
uses all commonly available records, so the 30-question AIME 2026 dataset keeps
all 30 when every run contains them. A changed dataset, split, smaller sample
size, or sample seed is rejected instead of resampled. Increasing `sample_size`
extends the existing manifest in place: prior items keep their order and only
newly sampled eligible IDs are appended.

Enable random-exit replay in `agent_kwargs.replay_eval.v2.random_exit`, or
override it on the command line. `agent_kwargs.replay_eval.version` selects
the backend: version 1 sends text completions through the configured v1 model
client, while version 2 uses the native vLLM engine. Both write the same flat
continuation rows and use per-continuation resume. The current config selects
version 1:

```bash
bash scripts/run_eval.sh \
  --agent replay_eval \
  --model gpt_oss_20b \
  --dataloader livecodebench \
  --evaluator code_exec_fix \
  --input-file data/random_exit/livecodebench/livecodebench.gpt_oss_20b.jsonl \
  --output-file output/random_exit/livecodebench-gpt_oss_20b.jsonl \
  --random-exit true
```

Configuration supplies defaults, and CLI values take precedence. The available
overrides are `--random-exit`, `--random-exit-sample-size`,
`--random-exit-sample-seed`, `--random-exit-position-seed`,
`--random-exit-spacing`, `--random-exit-max-positions`,
`--random-exit-num-runs`, `--random-exit-min-valid-positions`, and
`--max-final-answer-completion-token`. Replay-v2 also accepts `--temperature`,
`--gpu-memory-utilization`, and `--max-num-seqs` when a resumed run must match
an earlier sidecar. The same options are forwarded by `run_eval_many.sh` and
`run_eval_reuse_many.sh`.

Each JSONL row is one continuation and contains only its key, dataset/model
identity, position/run coordinates, output, finish reason, token count, and
`ok` status. Run-level configuration, manifest identity, prompt provenance,
exclusions, and failures are stored once in the adjacent
`*.metadata.json` sidecar. Legacy output-only trajectories have their
canonical prompt reconstructed once at replay time; stored `Response.input`
prompts and token prefixes remain authoritative when present.

Use `--resume` with the same output path, or `--resume-from CHECKPOINT.jsonl`
when writing a new path. The v2 backend validates the sidecar. The v1 backend
can resume a v2 output: it recomputes the positions and rollout seeds, checks
each successful row against the planned `data_id`, `line_number`, model,
position, and run number, then generates only missing
`data_id-position-num_run` keys. Failed rows are retried. A truncated final
JSONL line is repaired on in-place resume; malformed middle lines and
conflicting successful duplicates are rejected. V1 writes the same flat rows
and metadata layout as v2.

The checked-in Math500 and TruthfulQA 128-sample continuations can be resumed
with their original generation and vLLM settings by running:

```bash
bash scripts/continue_random_exit_128.sh
```

The script runs only `qwen35_4b` and `gpt_oss_20b_high`, appends missing
continuations to the existing files under `output/random_exit`, and evaluates
only rows that do not already have a `correct` annotation.

The runner emits ordinary `INFO` logs (no progress-bar dependency) for engine
startup and warmup, the total and resumed continuation plan, one aggregate
generation result per input JSONL record, and the final output and metadata
paths. Each record result includes its position, resumed, generated, and failure
counts plus the overall rate and ETA.

Before running the random-exit stability notebook, add correctness to the flat
continuation rows. The source is not modified; the evaluator writes atomically
and adds only top-level `correct`:

```bash
python scripts/rerun_random_exit_eval.py \
  output/random_exit/math500.qwen35_4b.jsonl \
  --prepared-input data/random_exit/math500/math500.qwen35_4b.jsonl \
  --config scripts/config.yml \
  --evaluator math500 \
  --output output/random_exit/math500.qwen35_4b.evaluated.jsonl
```

To reevaluate an existing evaluated file without creating
`*.evaluated.evaluated.jsonl`, pass that file as the input and use the atomic
in-place mode:

```bash
python scripts/rerun_random_exit_eval.py \
  output/random_exit/math500.qwen35_4b.evaluated.jsonl \
  --prepared-input data/random_exit/math500/math500.qwen35_4b.jsonl \
  --config scripts/config.yml \
  --evaluator math500 \
  --in-place
```

`--in-place` and `--output` are mutually exclusive. The original file is not
replaced unless evaluation and the atomic write complete successfully.

Open `random_exit_continuation_stability_test.ipynb` for the analysis specified
in `random_exit_continuation_stability_test.md`. The notebook is self-contained:
it does not import analysis helpers from project Python files. Like
`EDA_hung.ipynb`, it recursively discovers JSONL files using repository-relative
directories. Benchmark and model identity come from each raw file's rows, while
the evaluated output and metadata use the raw filename as their stem. Prepared
input follows `data/random_exit/<benchmark>/<benchmark>.<model>.jsonl`, so
future benchmarks and models are included without adding hard-coded output
paths. Optional benchmark/model sets at the top of the notebook select a
smaller pilot.

The notebook can invoke the correctness command above for every discovered run,
keeps only positions with every configured rollout, excludes trajectories below
`min_valid_positions`, fits weighted binomial PAVA, and runs the monotone-noise
and question-level bootstraps separately. It also audits the resolved sidecar
against the confirmatory design (128 examples, spacing 16, 64 positions, eight
runs, and five minimum positions). A mismatch is labeled pilot-only rather than
silently treated as confirmatory. Tables are written under
`output/random_exit/analysis/stability/`.

The equivalent non-plotting command supports the same recursive discovery and
repeatable `--benchmark`/`--model` filters:

```bash
python scripts/analyze_random_exit_stability.py \
  --output-dir output/random_exit \
  --prepared-dir data/random_exit \
  --benchmark math500 \
  --model qwen35_4b \
  --table-dir output/random_exit/analysis/stability
```

See `docs/early_exit_generation.md` for the detailed generation control flow.

See `dataloader/README.md` for the full dataloader, runner, and logging guide.

## VLLM

```bash
vllm serve google/gemma-4-E4B-it  \
  --max-model-len 38000   \
  --gpu-memory-utilization 0.75 \
  # --speculative-config '{"method":"mtp", "num_speculative_tokens":2}' \
  --enable-prefix-caching \
  --async-scheduling
  
```
