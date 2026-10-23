#!/usr/bin/env bash
set -euo pipefail

python -m replay_eval.v2 \
  --config scripts/config.yml \
  --model "${3:?llm.models key is required}" \
  --input-file "${1:?input JSONL is required}" \
  --output-file "${2:?output JSONL is required}" \
  "${@:4}"
