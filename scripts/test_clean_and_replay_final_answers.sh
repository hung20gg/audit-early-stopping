#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

LIMIT="${1:-3}"
SOURCE_FILE="${2:-examples/output/math500/full_v2_3_replay_eval_qwen35_9b_20260821_005015.jsonl}"
MODEL="${MODEL:-qwen35_9b}"
CONFIG="${CONFIG:-scripts/config.yml}"
MAX_FINAL_ANSWER_TOKENS="${MAX_FINAL_ANSWER_TOKENS:-2048}"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_FILE="${3:-output/math500/replay_v2_final_only_test_${RUN_STAMP}.jsonl}"
LOG_FILE="${4:-logs/math500/replay_v2_final_only_test_${RUN_STAMP}.log}"

if ! [[ "${LIMIT}" =~ ^[1-9][0-9]*$ ]]; then
    printf 'LIMIT must be a positive integer, got: %s\n' "${LIMIT}" >&2
    exit 2
fi
if [[ ! -f "${SOURCE_FILE}" ]]; then
    printf 'Source JSONL does not exist: %s\n' "${SOURCE_FILE}" >&2
    exit 2
fi

PREPARED_INPUT="$(mktemp /tmp/replay-v2-final-only.XXXXXX.jsonl)"
cleanup() {
    rm -f -- "${PREPARED_INPUT}"
}
trap cleanup EXIT

head -n "${LIMIT}" "${SOURCE_FILE}" >"${PREPARED_INPUT}"

python scripts/clean_truncated_final_answers.py \
    --input "${PREPARED_INPUT}" \
    --in-place \
    --force-delete puma,dynasor \
    --max-truncated-answer-tokens 2048

printf 'Prepared input: %s\n' "${PREPARED_INPUT}"
printf 'Replay output:  %s\n' "${OUTPUT_FILE}"
printf 'Replay log:     %s\n' "${LOG_FILE}"

mkdir -p "$(dirname "${OUTPUT_FILE}")" "$(dirname "${LOG_FILE}")"

python -m replay_eval.v2 \
    --config "${CONFIG}" \
    --model "${MODEL}" \
    --dataloader math500 \
    --evaluator math500 \
    --input-file "${PREPARED_INPUT}" \
    --output-file "${OUTPUT_FILE}" \
    --max-final-answer-completion-token "${MAX_FINAL_ANSWER_TOKENS}" \
    --rerun-all false \
    --rerun-final-answer true \
    >"${LOG_FILE}" 2>&1

printf 'Replay finished. Output: %s\n' "${OUTPUT_FILE}"
printf 'Replay log: %s\n' "${LOG_FILE}"
