# Thought Calibration training

This package trains and statistically calibrates an offline correctness probe
from the dense pre-final-RMSNorm replay embeddings produced by
`replay_eval.hidden_states`. It does not integrate the probe into live
generation; inference integration is intentionally deferred.

## Configure

Edit [`config.yml`](config.yml) before running. It is the single source of
truth for datasets, output behavior, feature construction, splitting,
preprocessing, the logistic probe, smoothing, LTT calibration, and loading.

For every dataset, set:

- `shards` to a directory containing `index.json` and its embedding
  `.npy`/metadata files (or legacy `.pt` shards).
- `curation_jsonl` to the matching replay output containing
  `additional_logs.curation_records`.

Set `base_model` to the Hugging Face model whose backbone produced the stored
states. Training loads only its `AutoConfig`, not its weights, and rejects an
embedding hidden size that differs from the model's text hidden size.

Like Terminator, relative dataset and output paths are resolved from the
repository root. Config loading is strict: unknown keys, missing paths,
invalid fractions or modes,
incompatible models/hidden sizes, insufficient PCA samples, empty required
splits, and single-class training labels fail with an explanatory error.

Choose `output_policy` as follows:

- `error`: refuse to use a non-empty output directory.
- `overwrite`: replace this command's six known artifacts while preserving
  unrelated files.
- `resume`: return existing metrics when all six artifacts are complete;
  otherwise rerun the pipeline and replace artifacts as they are produced.

## Run

From the repository root:

```bash
conda run -n verl python -m thought_calibration.train \
  --config thought_calibration/config.yml
```

No other command-line parameters are accepted. Logs include adapter counters
for discarded never-correct traces, filtered/missing records, and skipped
zero-token checkpoint boundaries.

## Outputs

The configured output directory receives:

- `model.joblib`: fitted `StandardScaler`, PCA, logistic regression probe, and
  feature/model compatibility metadata.
- `config.json`: deployment-facing model configuration containing the base
  model, hidden size, preprocessing contract, smoothing, calibrated thresholds,
  and fallback rule.
- `thresholds.json`: each risk tolerance's descending threshold sweep,
  binomial-CDF p-values, empirical risks, fixed-sequence decisions, selected
  threshold, and full-budget fallback status.
- `splits.json`: deterministic problem-level train/calibration/test membership
  and dataset counters.
- `metrics.json`: untouched-test classification and stopping metrics.
- `config.resolved.yml`: the complete resolved configuration snapshot.

When LTT cannot certify any threshold for a risk tolerance, its selected
threshold is `null`; evaluation uses the final checkpoint for every trajectory
and reports a fallback rate of 1.0.

## Tensor-only inference wrapper

Load the fitted pipeline as a frozen PyTorch module when a standard `forward`
interface is more convenient:

```python
from thought_calibration import ThoughtCalibrationModule

probe = ThoughtCalibrationModule.load("checkpoints/thought_calibration/replay")
probe = probe.to(device="cuda")

# pooled_hidden_states: [..., hidden_size]
probabilities = probe(
    pooled_hidden_states,
    attention_mask=attention_mask,
    position_ids=position_ids,
    new_tokens=new_tokens,
)
```

The optional arguments are accepted for compatibility and currently ignored.
Every leading input dimension is preserved, so `[N, H]` returns `[N]` and
`[B, N, H]` returns `[B, N]`. Inputs must already be pooled with the configured
`segment_mean` policy; the wrapper applies scaler, PCA, and logistic probe only.

## Test

```bash
conda run -n verl python -m pytest -q tests/test_thought_calibration.py
```
