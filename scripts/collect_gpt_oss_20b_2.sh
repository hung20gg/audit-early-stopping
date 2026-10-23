#!/usr/bin/env bash
set -euo pipefail
# Charts are off by default. Pass --plots to enable them.
# Override PYTHON to select the CUDA/vLLM environment. Extra arguments pass through.
# --resume is forwarded to every benchmark collector; OUTPUT_ROOT selects the run.
PYTHON="${PYTHON:-python3}"
if [[ -z "${OUTPUT_ROOT:-}" ]]; then
  args=("$@")
  trace_limit=""
  for ((i=0; i<${#args[@]}; i++)); do
    case "${args[i]}" in
      --limit-traces)
        if ((i + 1 < ${#args[@]})); then trace_limit="${args[i+1]}"; fi
        ;;
      --limit-traces=*) trace_limit="${args[i]#*=}" ;;
    esac
  done
  OUTPUT_ROOT="output/random_exit/analysis/gpt_oss_20b_2/collection"
  if [[ -n "$trace_limit" ]]; then
    if [[ ! "$trace_limit" =~ ^[1-9][0-9]*$ ]]; then
      echo "--limit-traces must be a positive integer" >&2
      exit 2
    fi
    OUTPUT_ROOT="output/random_exit/analysis/gpt_oss_20b_2/sample_verl_limit_$trace_limit"
  fi
fi
echo "Collection output root: $OUTPUT_ROOT"
for dataset in math500 truthfulqa mmlu_pro_subset; do
  "$PYTHON" scripts/collect_gpt_oss_probe_scores.py \
    "output/random_exit/$dataset.gpt_oss_20b_2.jsonl" \
    --prepared-input "data/random_exit/$dataset/$dataset.gpt_oss_20b_2.jsonl" \
    --output-dir "$OUTPUT_ROOT/$dataset" \
    --trial-budget 256 --eat-rollouts 10 --artifact-compression gzip --entropy "$@"
done
