# DoLa implementation in replay-v2

This directory implements two separate experiments from the reasoning-collapse
proposal:

1. **Detection:** measure intermediate-versus-final layer disagreement on an
   existing trace using replay-v2 and vLLM.
2. **Intervention:** generate a new response using DoLa contrastive decoding
   through the Hugging Face chat CLI or Python API.

Detector improvement and decoder improvement are different results. The first
produces features; the second changes the generated trajectory.

## Source layout

| File | Responsibility |
| --- | --- |
| `math.py` | Jensen–Shannon divergence, vectorized candidate comparison, plausibility masking, and layer selection |
| `decoding.py` | Hugging Face generation, KV caching, batched projection, sampling, EOS handling, and timing |
| `worker.py` | vLLM intermediate-block hooks and chunked vocabulary projection |
| `replay.py` | Replay-v2 diagnostic verifier and checkpoint logging |
| `chat.py` | Interactive and one-shot normal/DoLa chat CLI |
| `__main__.py` | `python -m replay_eval.v2.dola` entry point |

The generic [vLLM hidden-state worker](../engine/vllm_hidden_states.py) only
starts, invokes, and closes `DoLaWorkerCapture`; model-specific DoLa work stays
inside this directory.

## Decoding algorithm

For one generation position, let `z_L` be mature final-layer logits and `z_l`
the logits obtained by applying the final normalization and LM head to candidate
intermediate layer `l`.

### 1. Compare candidate layers

```text
J(l) = JS(P_L || P_l)
M = 0.5 * (P_L + P_l)
JS(P_L || P_l) = 0.5 * KL(P_L || M) + 0.5 * KL(P_l || M)
```

All candidates are projected in one batched LM-head call, and all JSD values
are computed in one vectorized operation.

### 2. Dynamically select the premature layer

```text
l* = argmax_l J(l)
```

Selection runs again for every output token. It is not one fixed layer for the
whole answer. `--show-layer-stats` reports how often each layer was selected.

### 3. Contrast and filter token scores

```text
score(v) = log P_L(v) - log P_l*(v)
```

Before selection or sampling, token `v` is retained only when:

```text
P_L(v) >= alpha * max_u P_L(u)
```

This adaptive plausibility constraint prevents an implausible token from
winning merely because its premature-layer probability is even lower.
`alpha=0` disables the mask; larger values are more conservative.

The decoder then applies greedy, top-k, and/or top-p sampling, updates the KV
cache, and repeats. EOS IDs come from both `model.generation_config` and the
tokenizer so models with multiple chat stop tokens terminate correctly.

## Layer-number conventions

| Interface | Meaning of layer `n` |
| --- | --- |
| Chat CLI / `DoLaDecoder` | Index in `outputs.hidden_states`; `0` is embeddings and `1` is the first block output |
| Replay-v2 `extraction.dola_layers` | Zero-based transformer block; `0` is the first block |

The mature final state cannot be a premature candidate. `--layers auto`
chooses four roughly even premature hidden-state indices at 20%, 40%, 60%, and
80% of model depth.

## Chat CLI

### Quick latency check

Use a short cap and log every token while diagnosing performance:

```bash
python -m replay_eval.v2.dola \
  --model Qwen/Qwen3-4B \
  --prompt "hi" \
  --layers auto \
  --max-new-tokens 32 \
  --log-every 1 \
  --dtype bfloat16 \
  --show-layer-stats
```

### Interactive DoLa chat

```bash
python -m replay_eval.v2.dola \
  --model Qwen/Qwen3-4B \
  --decoder dola \
  --layers auto \
  --max-new-tokens 128
```

Enter `/exit` or `/quit` to stop. The complete message history is re-rendered
with the tokenizer's normal chat template on every turn.

### Normal-decoding comparison

```bash
python -m replay_eval.v2.dola \
  --model Qwen/Qwen3-4B \
  --decoder normal \
  --prompt "hi" \
  --max-new-tokens 32
```

Use the same model, prompt, cap, temperature, top-p, top-k, and seed when
comparing normal and DoLa decoding.

## CLI parameters

| Parameter | Default | Effect |
| --- | ---: | --- |
| `--model` | required | Hugging Face model ID or local path |
| `--decoder` | `dola` | Choose `dola` or ordinary `normal` generation |
| `--layers` | `auto` | Premature hidden-state indices, such as `4,8,12,16` |
| `--alpha` | `0.1` | Adaptive plausibility threshold in the range 0–1 |
| `--max-new-tokens` | `128` | Hard output-token cap; lower it during latency tests |
| `--temperature` | `0.0` | `0` is greedy; positive values enable sampling |
| `--top-p` | `1.0` | Nucleus probability mass when sampling |
| `--top-k` | `0` | Keep the top `k` scores; `0` disables top-k |
| `--seed` | unset | Sampling seed; irrelevant for greedy decoding |
| `--dtype` | `auto` | Checkpoint dtype, `float16`, `bfloat16`, or `float32` |
| `--device-map` | `auto` | Hugging Face/Accelerate placement policy |
| `--system-prompt` | unset | Optional system message |
| `--prompt` | unset | Execute one turn and exit; omit for interactive mode |
| `--trust-remote-code` | false | Permit checkpoint-provided Python code |
| `--show-layer-stats` | false | Print timing, selection counts, and mean JSD as JSON |
| `--log-every` | `10` | Log progress every N DoLa tokens; `1` profiles latency, `0` disables progress |
| `--log-level` | `INFO` | Python logging level |

## Timing and progress logs

The CLI reports model loading and generation separately:

```text
Model loaded ... elapsed=12.40s devices=['cuda:0'] memory_gib=7.61
Generation started decoder=dola prompt_tokens=18 max_new_tokens=32
DoLa decode progress tokens=1 elapsed=0.72s tokens_per_second=1.39 ...
DoLa decode progress tokens=10 elapsed=3.91s tokens_per_second=2.56 ...
DoLa decode finished tokens=18 finish_reason=stop total=6.34s first_token=0.72s ...
Turn finished ... total=6.35s
```

- **Model load elapsed:** checkpoint loading and placement.
- **First-token time:** prompt prefill plus the first DoLa projection.
- **Tokens/second:** end-to-end DoLa decoding throughput.
- **Turn total:** prompt rendering, generation, and output decoding.
- **`finish_reason=length`:** no EOS appeared before the cap; this also emits
  a warning.

Layer-stat JSON includes `total_seconds`, `first_token_seconds`,
`tokens_per_second`, `selected_layer_counts`, and `mean_jsd_by_layer`.

## Why `hi` can take minutes

Every DoLa output token needs the normal transformer forward plus final
normalization and a full-vocabulary projection for all candidate layers. The
implementation batches those projections and avoids duplicate JSD work, but
the vocabulary projection remains expensive.

Check logs in this order:

1. **No CUDA device in `Model loaded`:** the model is on CPU. Verify CUDA and
   `--device-map auto`.
2. **Devices include `cpu` or `disk`:** Accelerate offloaded model components.
   Use a smaller model or reduce other GPU memory use.
3. **Large first-token time:** prompt prefill or model placement is expensive.
4. **Low throughput after token 1:** candidate projection is dominant. Use
   fewer layers, a smaller model, or a shorter cap.
5. **`max_new_tokens` warning:** DoLa did not select EOS. Start with 16–32
   tokens for a smoke test and inspect EOS configuration.
6. **Many candidate layers:** projection cost grows roughly with candidate
   count. Two layers are sufficient for a performance smoke test.

Recommended minimal check:

```bash
python -m replay_eval.v2.dola \
  --model YOUR_MODEL \
  --prompt "hi" \
  --layers 4,12 \
  --max-new-tokens 16 \
  --log-every 1 \
  --dtype bfloat16
```

## Python API

```python
from replay_eval.v2.dola import DoLaDecoder

result = DoLaDecoder(model, tokenizer).generate(
    prompt_token_ids,
    dola_layers=[4, 8, 12, 16],
    max_new_tokens=128,
    alpha=0.1,
    temperature=0.0,
    top_p=1.0,
    top_k=0,
    seed=0,
    log_every=10,
)
```

An optional `eos_token_ids` iterable overrides automatic EOS discovery. The
returned `DoLaGeneration` provides `token_ids`, `text`, `selected_layers`,
per-token `layer_jsd`, `finish_reason`, `total_seconds`,
`first_token_seconds`, and `tokens_per_second`.

The decoder supports one example per call. It uses KV caching but does not
continuously batch independent chat requests.

## Replay-v2 detector configuration

The detector does not change generation. It projects selected vLLM block
states from an existing trace:

```yaml
agent_kwargs:
  replay_eval:
    v2:
      extraction:
        mode: callback
        callback: replay_eval.v2.engine.vllm_hidden_states.extract_trace
        dola_layers: [4, 8, 12, 16]
        dola_chunk_size: 64
      methods:
        - factory: replay_eval.v2.dola.replay.build_dola_disagreement_method
          kwargs:
            name: dola_disagreement
            check_every: 32
            window_size: 1
```

| Replay parameter | Default | Effect |
| --- | ---: | --- |
| `extraction.dola_layers` | `[]` | Zero-based vLLM blocks; empty disables DoLa extraction |
| `extraction.dola_chunk_size` | `64` | Stored token states projected per GPU chunk; lower values reduce peak memory |
| `check_every` | `32` | Replay checkpoint spacing in assistant tokens |
| `window_size` | `1` | Preceding token-level maximum-JSD values averaged into the score |
| `threshold` | infinity | Stop threshold; infinity keeps the method diagnostic-only |

Each checkpoint logs `jsd`, `delta_jsd`, `selected_layer`, and every candidate
value under `layer_jsd`.

Worker projection currently requires `tensor_parallel_size: 1`. Intermediate
states are stored on CPU and projected back on the model device in
`dola_chunk_size` chunks to bound GPU memory.
