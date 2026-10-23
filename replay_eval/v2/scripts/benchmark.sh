#!/usr/bin/env bash
set -euo pipefail

input_file="${1:?input JSONL is required}"
output_file="${2:?output JSONL is required}"

python -m replay_eval.v2 \
  --config scripts/config.yml \
  --model "${3:?llm.models key is required}" \
  --input-file "$input_file" \
  --output-file "$output_file"
