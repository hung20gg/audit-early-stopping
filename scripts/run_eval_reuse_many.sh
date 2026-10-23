#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

REUSE_DIR=""
BASE_NAME="reuse"
NUM_RUN=""
NUM_THREAD="1"
MODEL_KEY=""
AGENT="vanilla_reuse"
DATALOADER="gsm8k"
REVERSE=false
INPUT_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --reuse)
            REUSE_DIR="$2"
            shift 2
            ;;
        --reuse=*)
            REUSE_DIR="${1#*=}"
            shift
            ;;
        --num-run|--runs)
            NUM_RUN="$2"
            shift 2
            ;;
        --num-run=*|--runs=*)
            NUM_RUN="${1#*=}"
            shift
            ;;
        --name)
            BASE_NAME="$2"
            shift 2
            ;;
        --name=*)
            BASE_NAME="${1#*=}"
            shift
            ;;
        --model)
            MODEL_KEY="$2"
            INPUT_ARGS+=("$1" "$2")
            shift 2
            ;;
        --model=*)
            MODEL_KEY="${1#*=}"
            INPUT_ARGS+=("$1")
            shift
            ;;
        --agent)
            AGENT="$2"
            INPUT_ARGS+=("$1" "$2")
            shift 2
            ;;
        --agent=*)
            AGENT="${1#*=}"
            INPUT_ARGS+=("$1")
            shift
            ;;
        --dataloader)
            DATALOADER="$2"
            INPUT_ARGS+=("$1" "$2")
            shift 2
            ;;
        --dataloader=*)
            DATALOADER="${1#*=}"
            INPUT_ARGS+=("$1")
            shift
            ;;
        --num-thread|--num-threads|--threads)
            NUM_THREAD="$2"
            INPUT_ARGS+=("$1" "$2")
            shift 2
            ;;
        --num-thread=*|--num-threads=*|--threads=*)
            NUM_THREAD="${1#*=}"
            INPUT_ARGS+=("$1")
            shift
            ;;
        --input-file)
            printf 'run_eval_reuse_many.sh sets --input-file from --reuse; do not pass --input-file.\n' >&2
            exit 2
            ;;
        --input-file=*)
            printf 'run_eval_reuse_many.sh sets --input-file from --reuse; do not pass --input-file.\n' >&2
            exit 2
            ;;
        --output-file|--log-file)
            printf 'run_eval_reuse_many.sh generates per-reuse paths; do not pass %s.\n' "$1" >&2
            exit 2
            ;;
        --output-file=*|--log-file=*)
            printf 'run_eval_reuse_many.sh generates per-reuse paths; do not pass %s.\n' "${1%%=*}" >&2
            exit 2
            ;;
        --resume)
            printf 'run_eval_reuse_many.sh generates per-reuse output paths; use --resume-from per direct run if needed.\n' >&2
            exit 2
            ;;
        --resume-from)
            INPUT_ARGS+=("$1" "$2")
            shift 2
            ;;
        --resume-from=*)
            INPUT_ARGS+=("$1")
            shift
            ;;
        --reverse)
            REVERSE=true
            INPUT_ARGS+=("$1")
            shift
            ;;
        --max-total-thinking-token|--max-final-answer-completion-token|--temperature|--gpu-memory-utilization|--max-num-seqs|--exit|--random-exit|--random-exit-sample-size|--random-exit-sample-seed|--random-exit-position-seed|--random-exit-spacing|--random-exit-max-positions|--random-exit-num-runs|--random-exit-min-valid-positions)
            INPUT_ARGS+=("$1" "$2")
            shift 2
            ;;
        --max-total-thinking-token=*|--max-final-answer-completion-token=*|--temperature=*|--gpu-memory-utilization=*|--max-num-seqs=*|--exit=*|--random-exit=*|--random-exit-sample-size=*|--random-exit-sample-seed=*|--random-exit-position-seed=*|--random-exit-spacing=*|--random-exit-max-positions=*|--random-exit-num-runs=*|--random-exit-min-valid-positions=*)
            INPUT_ARGS+=("$1")
            shift
            ;;
        --)
            shift
            while [[ $# -gt 0 ]]; do
                INPUT_ARGS+=("$1")
                shift
            done
            ;;
        *)
            INPUT_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ -z "${REUSE_DIR}" ]]; then
    printf 'Usage: %s --reuse REUSE_DIR [run_eval.sh args]\n' "$0" >&2
    exit 2
fi
if [[ ! -d "${REUSE_DIR}" ]]; then
    printf 'Reuse folder does not exist: %s\n' "${REUSE_DIR}" >&2
    exit 2
fi
if [[ -n "${NUM_RUN}" ]]; then
    printf '[run_eval_reuse_many] ignoring --num-run=%s because --reuse runs once per file\n' "${NUM_RUN}"
fi

shopt -s nullglob
REUSE_FILES=("${REUSE_DIR}"/*.jsonl)
shopt -u nullglob

if [[ "${#REUSE_FILES[@]}" -eq 0 ]]; then
    printf 'No .jsonl files found in reuse folder: %s\n' "${REUSE_DIR}" >&2
    exit 2
fi

sanitize_name() {
    local value="$1"
    value="${value//\//_}"
    value="${value//:/_}"
    value="${value// /_}"
    printf '%s' "${value}" | tr -cd '[:alnum:]_.-'
}

for reuse_file in "${REUSE_FILES[@]}"; do
    reuse_name="$(sanitize_name "$(basename "${reuse_file}" .jsonl)")"
    run_name="${BASE_NAME}_${reuse_name}"
    printf '[run_eval_reuse_many] reuse=%s name=%s agent=%s model=%s dataloader=%s num_thread=%s reverse=%s\n' \
        "${reuse_file}" "${run_name}" "${AGENT}" "${MODEL_KEY:-<default>}" "${DATALOADER}" "${NUM_THREAD}" "${REVERSE}"
    if [[ "${#INPUT_ARGS[@]}" -gt 0 ]]; then
        bash "${SCRIPT_DIR}/run_eval.sh" \
            --name "${run_name}" \
            --agent "${AGENT}" \
            --input-file "${reuse_file}" \
            "${INPUT_ARGS[@]}"
    else
        bash "${SCRIPT_DIR}/run_eval.sh" \
            --name "${run_name}" \
            --agent "${AGENT}" \
            --input-file "${reuse_file}"
    fi
done
