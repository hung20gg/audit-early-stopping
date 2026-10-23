# Replay Eval v2: GPU Handoff and Test Plan

This document describes the v2 architecture, data contracts, GPU validation
procedure, known integration risks, and remaining work. Treat the first GPU run
as an integration test, especially for hidden-state extraction and GPT-OSS token
alignment.

## Architecture

```text
JSONL records
  -> one long-lived AsyncLLM engine
  -> canonical Hugging Face chat rendering
  -> generated or supplied full trajectory
  -> full-trace logprob and hidden-state extraction
  -> token-by-token replay
       -> existing checkpoint objects
       -> asynchronously scheduled normal and DTSR probes
       -> existing verifier objects
       -> existing stop-rule objects
  -> early-final-answer requests
  -> ordered JSONL output and aggregate timing metrics
```

V2 does not copy the existing replay algorithms. It directly imports and calls:

- `replay_eval.method.EarlyExitMethod`
- `replay_eval.method.VerifierContext`
- checkpoints from `replay_eval.checkpoints`
- verifiers from `replay_eval.verifiers`
- stop rules from `replay_eval.stop_rules`
- factories from `replay_eval.methods`
- `methods.response.Response`
- the existing DEER, ENTROCUT, and other scoring utilities used by those verifiers

The compatibility adapter implements the synchronous interfaces expected by
the existing verifiers, but only reads responses that the asynchronous runtime
has already completed. It raises an error rather than falling back to a new
synchronous model call.

## File map

- `__main__.py`: command-line entry point.
- `../../scripts/config.yml`: the sole runtime YAML; v2 settings live under
  `agent_kwargs.replay_eval.v2`.
- `config/schema.py`: v2 validation and adaptation of the shared configuration.
- `engine/vllm_engine.py`: long-lived `AsyncLLM`, generation, logprob conversion, extraction, timing, warmup, and shutdown.
- `engine/cache.py`: exact completed/in-flight request deduplication.
- `engine/types.py`: request keys, request kinds, results, and timing structures.
- `engine/extraction.py`: native, callback/file, and precomputed extraction connectors.
- `rendering/hf.py`: canonical HF rendering, stored-prompt validation, GPT-OSS rendering, and prefix checks.
- `runtime/adapter.py`: compatibility API consumed by unchanged verifiers.
- `runtime/orchestrator.py`: replay state, checkpoint collection, probe scheduling, verifier calls, stop decisions, and final answers.
- `runtime/runner.py`: concurrent examples, ordered output, engine lifecycle, and JSONL I/O.
- `runtime/metrics.py`: throughput, latency, cache, token, timing, and memory metrics.
- `tests/test_v2.py`: CPU tests using fake tokenizers, fake engines, and canned responses.

## Probe and cache behavior

At every checkpoint, the orchestrator determines which existing methods fired
before invoking any verifier.

For normal temporary-answer methods, it:

1. Finds the largest requested temporary-answer token count.
2. Finds the largest requested top-logprob count.
3. Submits one shared normal probe for that checkpoint.
4. Forces existing `EntrocutVerifier` instances to top-50.
5. Creates separate rollout slots for EAT.
6. Prevents EAT `force_new=True` rollouts from aliasing the shared probe.

DTSR uses `RequestKind.DTSR` and a separately rendered scoring prompt. It cannot
alias a normal temporary answer.

Every request key includes:

- exact rendered prompt token IDs;
- sampling parameters;
- replay checkpoint;
- request kind;
- rollout/cache slot.

Therefore a temporary answer cannot be replaced with the full-generation
answer, and normal temporary answers can only be reused at the same replay
position.

## Input contract

The minimal JSONL input is:

```json
{"id": "example-1", "question": "What is 2 + 2?"}
```

The runner accepts the following question fields, in priority order:

1. `problem`
2. `question`
3. `prompt`
4. `input`

Optional runtime fields include:

```json
{
  "id": "example-1",
  "question": "Write a Python function...",
  "answer": "reference answer",
  "dataset": "humaneval",
  "is_code": true
}
```

### Reusing a stored trajectory

A record may contain an existing `Response`:

```json
{
  "id": "example-1",
  "question": "What is 2 + 2?",
  "response": {
    "input": "exact rendered model prompt",
    "content": "reasoning...</think>4",
    "logprobs": [],
    "finish_reason": "stop",
    "prompt_tokens": 20,
    "completion_tokens": 10
  },
  "output_token_ids": [123, 456]
}
```

Requirements for stored trajectories:

- `response.input` must tokenize exactly like the configured HF source prompt.
- Supply `output_token_ids` whenever possible.
- The canonical assistant rendering must reproduce the same output token IDs.
- The extraction pass can populate missing observed logprobs.

## Output contract

Each output row contains the original record plus `response`, `method_results`,
and `v2_timing`:

```json
{
  "response": {
    "input": "...",
    "content": "...",
    "logprobs": [],
    "finish_reason": "stop",
    "prompt_tokens": 100,
    "completion_tokens": 500
  },
  "method_results": [
    {
      "method": "deer",
      "position": 128,
      "stopped_early": true,
      "verifier_score": 0.97,
      "final_answer": "4",
      "thinking_tokens": 128,
      "answer_tokens": 5,
      "completion_tokens": 133,
      "verifier_prompt_tokens": 200,
      "verifier_completion_tokens": 20
    }
  ],
  "v2_timing": {
    "example_seconds": 2.4
  }
}
```

Aggregate metrics are printed to stdout after the run. They include:

- model load and warmup time;
- measured execution time;
- rendering and alignment time;
- hidden extraction, loading, normalization, and cleanup time;
- verifier time;
- queue, prefill, decode, and end-to-end time by request kind;
- examples/second;
- scan tokens/second;
- probe tokens/second;
- cached and computed token counts;
- cache-hit ratio;
- peak GPU memory;
- p50, p95, and p99 example latency.

## GPU test procedure

### 1. Record the environment

```bash
conda activate text2sql

python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import vllm; print(vllm.__version__)"
python -c "import transformers; print(transformers.__version__)"
nvidia-smi
```

Record the vLLM, Transformers, PyTorch, CUDA, and driver versions with the test
results. The vLLM generation and pooling APIs differ across releases.

### 2. Run the CPU suite

```bash
\
  python -m unittest discover -s replay_eval/v2/tests -v
```

Expected result:

```text
Ran 15 tests
OK
```

### 3. Create a one-record smoke input

Save this as `/tmp/replay-v2-smoke.jsonl`:

```json
{"id": "smoke-1", "question": "What is 17 + 25?"}
```

### 4. Start with conservative limits

Temporarily configure conservative limits in `scripts/config.yml`:

```yaml
agent_kwargs:
  replay_eval:
    version: 2
    v2:
      runtime:
        max_concurrency: 2
        example_concurrency: 1
        warmup: true
        sampling:
          temperature: 0.0
          max_tokens: 512
      extraction:
        # Keep callback mode for the full Terminator/Thought Calibration test.
        mode: callback
        callback: your_gpu_package.replay_v2.extract_trace
        temp_dir: /tmp/replay-v2-extraction

llm:
  vllm:
    serving:
      enable_prefix_caching: true
      enable_chunked_prefill: true
      max_num_seqs: 2
      max_num_batched_tokens: 2048
      speculative_config:
        method: mtp
        num_speculative_tokens: 2
```

The committed defaults use 32 sequences and 8192 batched tokens. Increase
those values only after observing headroom in peak GPU memory and latency.
Prefix caching is especially useful here because probes at nearby replay
positions share almost all prompt tokens; chunked prefill prevents a long
trace from monopolizing a scheduling iteration. The speculative mapping is
passed to `AsyncEngineArgs` as a dictionary and is equivalent to
`--speculative-config '{"method":"mtp","num_speculative_tokens":2}'`.

For an engine-only smoke test, temporarily set `v2.methods: []` and use
`extraction.mode: native`. That path validates vLLM generation, prompt
logprobs, and `token_embed`, but it does not validate Terminator or Thought
Calibration and may be unsupported by the installed vLLM/model combination.

### 5. Run the smoke test

```bash
python -m replay_eval.v2 \
  --config scripts/config.yml \
  --model qwen35_4b \
  --input-file /tmp/replay-v2-smoke.jsonl \
  --output-file /tmp/replay-v2-smoke-output.jsonl \
  --log-level INFO
```

Inspect the output:

```bash
python -m json.tool \
  /tmp/replay-v2-smoke-output.jsonl
```

Confirm that the row contains `response`, `method_results`, and `v2_timing`,
and that the process summary reports model load and warmup separately from
measured example throughput.

### 6. Check temporary-file cleanup

After successful and intentionally interrupted runs:

```bash
find /tmp/replay-v2-extraction -type f
```

There should be no `.pt` files for completed or failed examples.

## Important GPU integration risks

### P0: generation and token embedding on one engine

The default native connector calls both:

- `AsyncLLM.generate()` for generation and prompt logprobs;
- `AsyncLLM.encode()` with `PoolingParams(task="token_embed")` for hidden states.

Whether one engine supports both operations depends on the installed vLLM
version, runner configuration, and model. Expected errors include:

```text
AsyncLLM.encode() unavailable
Unsupported task: token_embed
```

If native extraction is unavailable, use callback mode:

```yaml
extraction:
  mode: callback
  callback: your_module.extract_trace
  temp_dir: /tmp/replay-v2-extraction
```

The callback must be asynchronous and accept:

```python
async def extract_trace(
    *,
    engine,
    full_token_ids,
    output_start,
    output_token_ids,
    example_id,
    cache_salt=None,
):
    ...
```

It returns either an `ExtractionResult` or a dictionary:

```python
{
    "token_ids": output_token_ids,
    "observed_logprobs": [...],
    "raw_hidden_path": "/tmp/...-raw.pt",
    "normalized_hidden_path": "/tmp/...-normalized.pt",
    "hidden_state_contract": "pre_final_rmsnorm",
    "prompt_tokens": output_start,
    "model_id": "Qwen/Qwen3.5-4B",
    "model_outputs": {"terminator": [...]},  # optional worker scores
    "extraction_seconds": 0.0,
    "normalization_seconds": 0.0,
}
```

For a full hidden-model test, all token-indexed arrays and both saved tensors
must have exactly `len(output_token_ids)` rows. The callback owns partial-file
cleanup if it fails before returning; after it returns, v2 cleans the files in
its example-level `finally` path.

### P0: validate the hidden-state definition

The generic native path requests `token_embed` with `use_activation=False` and
currently constructs its second tensor with:

```python
torch.nn.functional.normalize(hidden, dim=-1)
```

This must be checked against the intended model representation. Determine
whether `token_embed` returns:

- the output before the model's final RMSNorm/LayerNorm;
- the output after the model's final norm; or
- a pooler-transformed representation.

If the requirement is the layer-`num_hidden_layers` output before final model
norm plus the same tensor after the model's learned final norm, generic L2
normalization is not equivalent. Implement extraction inside a vLLM worker or
through callback mode and apply the actual model norm in the model's configured
FP16/BF16 dtype.

V2 now enforces this distinction for Terminator and Thought Calibration. The
native path labels its output `vllm_token_embed`; both adapters fail closed
unless callback extraction reports `hidden_state_contract:
pre_final_rmsnorm`. The callback must also return `prompt_tokens` and the
canonical training model ID. Do not relabel token embeddings just to pass the
check—compare callback states against `replay_eval.hidden_states.extract_batch`
on the same token IDs first.

### P0: validate Terminator adaptation

There are two supported paths:

1. Configure `build_terminator_method` with `checkpoint` and `base_model`. V2
   loads `terminator.terminator.TerminatorProbe`, supplies `[1, T, H]` raw
   pre-final states and an all-ones attention mask. It defaults to
   assistant-relative positions because the current training collator reads
   `positions`, while the extractor writes `position_ids`; use explicit
   `absolute_prompt` only for a checkpoint trained with the latter. Every 16
   replay tokens it classifies 25 tokens (9 overlapping and 16 new), evaluates
   16 rolling windows of length 10, and returns the largest positive count.
2. Integrate `terminator.terminator_vllm.VLLMTerminatorProbe` in the custom
   worker while its split hidden/residual streams are available. Return its
   probabilities in `model_outputs.terminator`. V2 checks that there is exactly
   one score per canonical output token and applies the same 25-token,
   16-window scan to that score stream.

GPU checks:

- [ ] Compare HF-adapter and native-vLLM Terminator probabilities for the same
  checkpoint and checkpoint weights.
- [ ] Verify the trained checkpoint used absolute prompt-offset position IDs.
  If it used assistant-relative positions, configure
  `position_mode: assistant_relative` explicitly.
- [ ] Confirm Qwen3, dense Qwen3.5, or GPT-OSS family detection matches the
  checkpoint; Qwen3.5-MoE remains unsupported by the supplied Terminator pair.
- [ ] Measure memory for the standalone extra decoder at the maximum trace
  length. The standalone path transfers only each scan's 25 hidden-state
  tokens to the probe GPU instead of retaining the full trace there;
  prefer worker scores when the probe already runs inside the model executor.
- [ ] Verify a scan after token 16 reports `classified_tokens: 25` and exactly
  16 windows, and confirm equality with the count threshold does not stop.

### P0: validate Thought Calibration adaptation

Configure `build_thought_calibration_method` with the directory containing
`config.json` and `model.joblib`. At each `EveryNTokens` checkpoint, v2 pools
only the new token segment since the preceding checkpoint, applies the frozen
StandardScaler/PCA/logistic module, smooths all raw scores with the existing
moving average, then applies the selected calibrated `>=` threshold. A null
threshold never fires and therefore preserves `full_budget` fallback.

GPU checks:

- [ ] Use the same checkpoint spacing/segmentation policy used to construct the
  calibration artifact. Changing `check_every` changes the segment features.
- [ ] Compare raw pooled features, raw probabilities, smoothed probabilities,
  and the stop index against an offline calibration trace.
- [ ] Confirm `model_id`, hidden size, `feature_source`, pooling, smoothing
  window, threshold comparison, and risk tolerance all match the artifact.

### P0: GPT-OSS content and token representation

The renderer passes GPT-OSS assistant data as:

```python
{
    "role": "assistant",
    "thinking": thinking,
    "content": answer,
}
```

Inspect what offline vLLM returns in:

- `candidate.text`;
- `candidate.token_ids`;
- `candidate.logprobs[*].decoded_token`.

Determine whether Harmony channel markers are present in token IDs, text, or
decoded logprob tokens. The default config uses `</think>` as the stored
reasoning boundary. If the engine returns raw Harmony markers instead, add an
explicit normalization step or configure the exact boundary representation.

### P1: canonical rendering alignment

For the first GPU record, verify:

```python
trace.full_token_ids[:len(trace.source_token_ids)] == trace.source_token_ids
trace.output_token_ids == full_result.output_token_ids
len(trajectory.logprobs) == len(trace.output_token_ids)
```

Common mismatch causes include:

- EOS or end-message tokens included by only one path;
- `skip_special_tokens` differences;
- `add_generation_prompt` differences;
- Transformers-version differences in `continue_final_message`;
- Harmony tokens omitted from `candidate.text` but present in token IDs.

Do not disable the checks to make the run continue. First determine which
representation is canonical and make generation, rendering, hidden states, and
logprobs use that same mapping.

### P1: observed prompt-logprob indexing

The native extraction path assumes `prompt_logprobs` aligns with
`full_token_ids` and slices from `output_start`. Verify:

```python
len(prompt_rows) == len(full_token_ids)
len(observed_logprobs) == len(output_token_ids)
```

The first source-prompt entry may be `None`, but no entry in the assistant
output slice may be missing.

### P1: prefix-cache metrics

The wrapper currently reads cached token count from:

```python
output.num_cached_tokens
```

Check whether the installed vLLM version exposes it there. If not, update
`AsyncVLLMEngine._convert()` to use the version-specific metrics field.

## Prioritized TODO list

### Must complete before benchmarking

- [ ] Pin and record `vllm`, `torch`, `transformers`, CUDA, and driver versions.
- [ ] Confirm `AsyncEngineArgs(max_logprobs=50)` is accepted.
- [ ] Confirm one engine can perform both generation and token-embedding extraction.
- [ ] If not, implement and configure the callback extraction connector.
- [ ] Verify raw hidden states are layer-`num_hidden_layers` outputs.
- [ ] Replace generic L2 normalization with the model's real final norm if required.
- [ ] Confirm hidden tensors remain in the configured FP16/BF16 dtype.
- [ ] Validate GPT-OSS `thinking`/`content` rendering against Harmony token IDs.
- [ ] Confirm generated output IDs equal canonical assistant suffix IDs.
- [ ] Confirm every assistant output token receives an observed logprob.
- [ ] Confirm prompt-logprob and hidden-state positions have identical length and order.
- [ ] Confirm extraction files disappear after success, failure, and cancellation.

### Method validation

- [ ] Run a baseline-only configuration first.
- [ ] Run DEER alone and inspect its temporary-answer prompt and stop decision.
- [ ] Run ENTROCUT alone and verify each probe requests top-50.
- [ ] Compare ENTROCUT entropy with the existing residual-bucket utility on a saved probe.
- [ ] Run DTSR alone and verify it uses a separate prompt and request kind.
- [ ] Run DEER and ENTROCUT together and confirm one shared normal probe per checkpoint.
- [ ] Run EAT with DEER and confirm forced rollouts use separate cache slots.
- [ ] Run PUMA separately because its embedding backend adds memory requirements.
- [ ] Feed identical canned GPU probe responses to v1 and v2 and compare stop positions.

### Performance validation

- [ ] Confirm warmup is excluded from measured throughput.
- [ ] Test `example_concurrency` values 1, 2, 4, and 8.
- [ ] Test `max_concurrency` independently from example concurrency.
- [ ] Confirm continuous batching is visible in request metrics.
- [ ] Confirm prefix-cache hits increase for probes sharing reasoning prefixes.
- [ ] Compare reported cached-token counts with vLLM engine logs.
- [ ] Track GPU memory during extraction, not only after requests finish.
- [ ] Confirm CPU hidden tensors do not accumulate across examples.
- [ ] Benchmark with and without ENTROCUT top-50.
- [ ] Save p50, p95, p99, scan throughput, and probe throughput.

### Reliability work after the first successful run

- [ ] Add a GPU integration test for the pinned vLLM version.
- [ ] Add a real GPT-OSS fixture containing Harmony token IDs.
- [ ] Add a Qwen reasoning-template fixture.
- [ ] Add per-record error output instead of aborting the complete run.
- [ ] Add resume/checkpoint support.
- [ ] Stream ordered output incrementally instead of retaining every row.
- [ ] Add engine/version diagnostics to aggregate metrics.
- [ ] Add a debug option that saves canonical token IDs for one example.
- [ ] Add cancellation testing while extraction is active.
- [ ] Sample peak GPU memory immediately around extraction.

The first GPU acceptance milestone is proving that canonical GPT-OSS output
IDs, observed prompt logprobs, raw hidden states, and normalized hidden states
all refer to exactly the same assistant-token positions. Benchmarking should
start only after that invariant is demonstrated.
