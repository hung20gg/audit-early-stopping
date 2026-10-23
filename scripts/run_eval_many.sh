#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

NUM_RUN=1
BASE_NAME="run"
NUM_THREAD="1"
MODEL_KEY=""
AGENT="simple"
DATALOADER="gsm8k"
EVALUATOR="gsm8k"
REVERSE=false
INPUT_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
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
        --evaluator)
            EVALUATOR="$2"
            INPUT_ARGS+=("$1" "$2")
            shift 2
            ;;
        --evaluator=*)
            EVALUATOR="${1#*=}"
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
        --output-file|--log-file)
            printf 'run_eval_many.sh generates per-run paths; do not pass %s.\n' "$1" >&2
            exit 2
            ;;
        --output-file=*|--log-file=*)
            printf 'run_eval_many.sh generates per-run paths; do not pass %s.\n' "${1%%=*}" >&2
            exit 2
            ;;
        --resume)
            printf 'run_eval_many.sh generates per-run output paths; use --resume-from instead of --resume.\n' >&2
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
        --max-total-thinking-token|--max-final-answer-completion-token|--temperature|--gpu-memory-utilization|--max-num-seqs|--rerun-all|--rerun-final-answer|--exit|--random-exit|--random-exit-sample-size|--random-exit-sample-seed|--random-exit-position-seed|--random-exit-spacing|--random-exit-max-positions|--random-exit-num-runs|--random-exit-min-valid-positions)
            INPUT_ARGS+=("$1" "$2")
            shift 2
            ;;
        --max-total-thinking-token=*|--max-final-answer-completion-token=*|--temperature=*|--gpu-memory-utilization=*|--max-num-seqs=*|--rerun-all=*|--rerun-final-answer=*|--exit=*|--random-exit=*|--random-exit-sample-size=*|--random-exit-sample-seed=*|--random-exit-position-seed=*|--random-exit-spacing=*|--random-exit-max-positions=*|--random-exit-num-runs=*|--random-exit-min-valid-positions=*)
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

if ! [[ "${NUM_RUN}" =~ ^[0-9]+$ ]] || [[ "${NUM_RUN}" -lt 1 ]]; then
    printf 'Expected --num-run to be a positive integer, got: %s\n' "${NUM_RUN}" >&2
    exit 2
fi

for ((i = 1; i <= NUM_RUN; i++)); do
    RUN_NAME="${BASE_NAME}_${i}"
    printf '[run_eval_many] run=%s/%s name=%s agent=%s model=%s dataloader=%s evaluator=%s num_thread=%s reverse=%s\n' \
        "${i}" "${NUM_RUN}" "${RUN_NAME}" "${AGENT}" "${MODEL_KEY:-<default>}" "${DATALOADER}" "${EVALUATOR}" "${NUM_THREAD}" "${REVERSE}"
    bash "${SCRIPT_DIR}/run_eval.sh" --name "${RUN_NAME}" "${INPUT_ARGS[@]}"
done
