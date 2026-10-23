# Scripts

## Replay version selection

`scripts/config.yml` is the only runtime YAML for both replay implementations.
Set `agent_kwargs.replay_eval.version` to `1` for the existing synchronous
runner or `2` for asynchronous native-vLLM replay. Version-2-only engine,
extraction, and method settings are nested under
`agent_kwargs.replay_eval.v2`. The required `--model` key selects
`llm.models.<key>`. Global native engine options live under `llm.vllm`, with
GPU serving options nested at `llm.vllm.serving`. A selected model's optional
`vllm` mapping overrides only the shared keys it specifies; model-specific
`vllm.engine_kwargs` and `vllm.serving` mappings are merged by key.

Shared completion temperature and token budgets come from the selected
`llm.models`/`llm.completion_kwargs` entries. Shared code multipliers remain at
`agent_kwargs.replay_eval.code_probe_token_multiplier` and
`code_threshold_multiplier`.

Version 2 currently requires `--input-file`, supports `--resume`/`--resume-from`, and does not support
`--reverse`, or `--limit`:

```bash
bash scripts/run_eval.sh \
  --agent replay_eval \
  --model qwen35_4b \
  --dataloader math500 \
  --evaluator math500 \
  # --input-file INPUT.jsonl \
  --output-file OUTPUT.jsonl
```

```bash
bash scripts/run_eval_many.sh \
  --num-run 2 \
  --name full_v2 \
  --agent replay_eval \
  --model gemma4e4 \
  --dataloader math500 \
  --evaluator math500 \
  --num-threads 4

bash scripts/run_eval_many.sh \
  --num-run 3 \
  --name full_v2 \
  --agent replay_eval \
  --model gemma4e4 \
  --dataloader aime26 \
  --evaluator aime26 \
  --num-threads 4
```

## Repair truncated replay final answers

The simplest smoke test uses the standalone wrapper below. It does **not** call
`scripts/run_eval.sh`; it invokes `python -m replay_eval.v2` directly:

```bash
bash scripts/test_clean_and_replay_final_answers.sh 3
```

Positional arguments are `LIMIT`, `SOURCE_FILE`, `OUTPUT_FILE`, and `LOG_FILE`:

```bash
bash scripts/test_clean_and_replay_final_answers.sh \
  5 \
  examples/output/math500/full_v2_3_replay_eval_qwen35_9b_20260821_005015.jsonl \
  output/math500/replay_v2_repaired_test.jsonl \
  logs/math500/replay_v2_repaired_test.log
```

The wrapper defaults to `MODEL=qwen35_9b`, final-answer budget `2048`, and
`scripts/config.yml`. Override them with environment variables when needed:

```bash
MODEL=qwen35_9b \
MAX_FINAL_ANSWER_TOKENS=4096 \
CONFIG=scripts/config.yml \
bash scripts/test_clean_and_replay_final_answers.sh 5
```

It copies the requested input slice to a temporary JSONL, removes truncated
final answers, force-deletes `puma,dynasor`, then runs replay with
`rerun_all=false` and `rerun_final_answer=true`. The temporary input is removed
after the run; the replay output and log remain at the printed paths.

### Full workflow: create a clean trace and replay it

Define the source, clean trace, and replay destinations:

```bash
SOURCE_FILE=examples/output/math500/full_v2_3_replay_eval_qwen35_9b_20260821_005015.jsonl
CLEAN_TRACE=output/math500/full_v2_3_clean_for_replay.jsonl
REPLAY_OUTPUT=output/math500/full_v2_3_repaired.jsonl
REPLAY_LOG=logs/math500/full_v2_3_repaired.log
```

Create the clean trace as a separate output file. The source remains unchanged:

```bash
python scripts/clean_truncated_final_answers.py \
  --input "${SOURCE_FILE}" \
  --output "${CLEAN_TRACE}" \
  --force-delete puma,dynasor
```

The report lists, per JSONL file, samples that hit the token limit, removed
final answers, final-only methods, force-deleted methods, and methods missing
according to `scripts/config.yml`.

To edit the source directly instead, omit `--output` and use `--in-place`.
Optionally add a backup suffix:

```bash
python scripts/clean_truncated_final_answers.py \
  --input "${SOURCE_FILE}" \
  --in-place \
  --force-delete puma,dynasor \
  --backup-suffix .bak
```

Now replay the clean trace directly, without `scripts/run_eval.sh`:

```bash
mkdir -p "$(dirname "${REPLAY_OUTPUT}")" "$(dirname "${REPLAY_LOG}")"

python -m replay_eval.v2 \
  --config scripts/config.yml \
  --model gemma4e4 \
  --dataloader mmlu_pro_subset \
  --evaluator mmlu_pro_subset \
  --input-file examples/fix/mmlu_pro_subset/replay_eval_gemma4e4_20261003_065109.jsonl \
  --output-file examples/fix2/mmlu_pro_subset/replay_eval_gemma4e4_20261003_065109.jsonl \
  --max-final-answer-completion-token 2048 \
  --rerun-all false \
  --rerun-final-answer true
```

In this replay:

- `puma` and `dynasor` were fully removed, so they are scanned again from the
  beginning.
- Configured methods missing from the old trace, including train-based methods,
  run through the normal replay path.
- Other results whose truncated `final_answer` was removed reuse their stored
  stop position and run only final-answer generation.

### Cleanup command details

The cleanup command accepts one JSONL file or one directory. Directories are
scanned recursively. For a directory input, `--output` must be a directory;
the relative JSONL tree is preserved:

```bash
python scripts/clean_truncated_final_answers.py \
  --input examples/output/math500 \
  --output output/math500/clean_traces \
  --force-delete puma,dynasor
```

The folder Bash wrapper fills `--input` and `--output` automatically:

```bash
bash scripts/clean_truncated_final_answers_folder.sh \
  --input-folder examples/output \
  --output-folder examples/new \
  --force-delete puma
```

For example, this mapping is automatic:

```text
examples/output/gpqa_diamond/run.jsonl
  -> examples/new/gpqa_diamond/run.jsonl
examples/output/math500/run.jsonl
  -> examples/new/math500/run.jsonl
```

If the active environment is not `text2sql`, the wrapper runs Python through
`conda run -n text2sql`.

Use exactly one of `--output` or `--in-place`. Results with
`logs.final_answer_truncated: true` and `answer_tokens < 2048` retain their stop
position/state but lose final-answer fields. `--force-delete` removes the whole
matching method result so replay scans that method again. Results with
`logs.generation_failed: true` are also removed completely. Top-level samples
with `generation_failed: true` are removed from the JSONL. Historical
`logs.terminator_scans[*].windows` details are stripped while aggregate scan
scores are retained. The command reports affected samples, removed answers,
final-only methods, failed methods, and missing methods for every file.

Run any prepared file with selective replay:

```bash
python -m replay_eval.v2 \
  --config scripts/config.yml \
  --model qwen35_9b \
  --dataloader math500 \
  --evaluator math500 \
  --input-file PREPARED.jsonl \
  --output-file REPAIRED.jsonl \
  --rerun-all false \
  --rerun-final-answer true
```

Missing method results are replayed normally, including train-based methods.
Existing results with a removed `final_answer` only submit a new final-answer
request at the stored stop position; extraction and stop verification are not
rerun. `scripts/test_clean_and_replay_final_answers.sh` combines both steps on a
small input slice.

Older trace files that predate the replay-v2 top-level `model` field can be
patched with the standalone utility (the default value is
`gpt_oss_20b_high`):

```bash
conda run -n text2sql python scripts/add_model_key.py TRACE.jsonl \
  --in-place \
  --backup-suffix .bak
```

Use `--output PATCHED.jsonl` instead of `--in-place` to retain the source, and
pass `--model MODEL_KEY` for another model. Existing `model` values are replaced
with the requested key.

Before running, replace the placeholder extraction callback and install the
configured Terminator and Thought Calibration artifacts. See the
[v2 runbook](../replay_eval/v2/README.md) for the input/output contract,
conservative smoke settings, and GPU validation warnings.

## Extract Full Generation

`extract_full_generation.py` converts a runner JSONL from a full-generation run into a compact JSONL that can be replayed by reuse agents.

Input rows must contain:

```text
additional_logs.full_generation_response.input
additional_logs.full_generation_response.content
```

Run:

```bash
python scripts/extract_full_generation.py \
  examples/deer_full_qwen36_35b_a3b_20260627_110525.jsonl \
  output/reuse/deer_full_qwen36_35b_a3b_reuse.jsonl
```

Output rows have this shape:

```json
{"id": 1, "input": "<rendered prompt>", "content": "<full generation>"}
```

`id` is copied from the source row's `id` or `line_number`. Reuse eval writes this value as the output `line_number`, so results stay aligned with the original input rows.

Use `--skip-missing` to ignore rows that do not contain a full-generation response:

```bash
python scripts/extract_full_generation.py \
  --skip-missing \
  input/deer_full_run.jsonl \
  output/reuse/deer_full_reuse.jsonl
```

To extract every `.jsonl` file in one folder into another folder, preserving one output file per input file:

```bash
bash scripts/extract_full_generation_many.sh \
  output/math500/full_runs \
  output/reuse/math500
```

This maps:

```text
output/math500/full_runs/a.jsonl -> output/reuse/math500/a.jsonl
output/math500/full_runs/b.jsonl -> output/reuse/math500/b.jsonl
```

The folder wrapper also accepts explicit flags:

```bash
bash scripts/extract_full_generation_many.sh \
  --input-dir output/math500/full_runs \
  --output-dir output/reuse/math500 \
  --skip-missing
```

## Replay Extracted Rows

The extracted file can be passed to `run_eval.sh` with `vanilla_reuse`:

```bash
bash scripts/run_eval.sh \
  --agent vanilla_reuse \
  --model qwen36_35b_a3b \
  --dataloader math500 \
  --input-file output/reuse/deer_full_qwen36_35b_a3b_reuse.jsonl
```

The runner uses each row's `input` as the already-rendered prompt and `content` as the stored full generation. Output still goes to the normal dataloader folder, for example `output/math500/...`. The final result omits the extracted prompt from `input`; `line_number` carries the row `id`.

For replay agents, the total thinking and forced final-answer budgets can be
overridden independently:

```bash
bash scripts/run_eval.sh \
  --agent replay_eval \
  --model qwen36_35b_a3b \
  --dataloader math500 \
  --max-total-thinking-token 32000 \
  --max-final-answer-completion-token 4096
```

When a trajectory has no `</think>`, the agent force-quits reasoning and uses
the second budget for a normal final-answer rollout. These flags are also
forwarded by `run_eval_many.sh` and `run_eval_reuse_many.sh`.

To run one eval for every extracted reuse file in a folder:

```bash
bash scripts/run_eval_reuse_many.sh \
  --reuse output/reuse/math500 \
  --model qwen36_35b_a3b \
  --dataloader math500
```

`run_eval_reuse_many.sh` ignores `--num-run`/`--runs` and runs exactly one `run_eval.sh` instance per `.jsonl` file in the reuse folder. It defaults to `--agent vanilla_reuse`; pass `--agent` only when replaying with another reuse-compatible agent. Each run writes to `output/<dataloader>/...` and includes the reuse filename in the run name.
