#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# CUDA 13 pip wheels keep NVRTC in a namespaced site-packages directory that
# is not always on the dynamic loader path. vLLM can otherwise run but emits
# DeepGEMM and worker-shutdown import errors.
NVIDIA_CU13_LIB="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"] + "/nvidia/cu13/lib")')"
if [[ -d "${NVIDIA_CU13_LIB}" ]]; then
    export LD_LIBRARY_PATH="${NVIDIA_CU13_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

CONFIG="scripts/config.yml"
AGENT="simple"
DATALOADER="gsm8k"
EVALUATOR="gsm8k"
MODEL_KEY=""
LLM_ID=""
RUN_NAME=""
OUTPUT_FILE=""
OUTPUT_FILE_EXPLICIT=false
LOG_FILE=""
NUM_THREAD="1"
LIST_MODELS=false
INPUT_FILE=""
RERUN_ALL=""
RERUN_FINAL_ANSWER=""
RESUME=false
RESUME_FROM=""
REVERSE=false
MAX_TOTAL_THINKING_TOKEN=""
MAX_FINAL_ANSWER_COMPLETION_TOKEN=""
TEMPERATURE=""
GPU_MEMORY_UTILIZATION=""
MAX_NUM_SEQS=""
RANDOM_EXIT=""
RANDOM_EXIT_SAMPLE_SIZE=""
RANDOM_EXIT_SAMPLE_SEED=""
RANDOM_EXIT_POSITION_SEED=""
RANDOM_EXIT_SPACING=""
RANDOM_EXIT_MAX_POSITIONS=""
RANDOM_EXIT_NUM_RUNS=""
RANDOM_EXIT_MIN_VALID_POSITIONS=""
RUNNER_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            CONFIG="$2"
            RUNNER_ARGS+=("$1" "$2")
            shift 2
            ;;
        --config=*)
            CONFIG="${1#*=}"
            RUNNER_ARGS+=("$1")
            shift
            ;;
        --dataloader)
            DATALOADER="$2"
            RUNNER_ARGS+=("$1" "$2")
            shift 2
            ;;
        --dataloader=*)
            DATALOADER="${1#*=}"
            RUNNER_ARGS+=("$1")
            shift
            ;;
        --evaluator)
            EVALUATOR="$2"
            RUNNER_ARGS+=("$1" "$2")
            shift 2
            ;;
        --evaluator=*)
            EVALUATOR="${1#*=}"
            RUNNER_ARGS+=("$1")
            shift
            ;;
        --agent)
            AGENT="$2"
            RUNNER_ARGS+=("$1" "$2")
            shift 2
            ;;
        --agent=*)
            AGENT="${1#*=}"
            RUNNER_ARGS+=("$1")
            shift
            ;;
        --model)
            MODEL_KEY="$2"
            RUNNER_ARGS+=("$1" "$2")
            shift 2
            ;;
        --model=*)
            MODEL_KEY="${1#*=}"
            RUNNER_ARGS+=("$1")
            shift
            ;;
        --llm)
            LLM_ID="$2"
            RUNNER_ARGS+=("$1" "$2")
            shift 2
            ;;
        --llm=*)
            LLM_ID="${1#*=}"
            RUNNER_ARGS+=("$1")
            shift
            ;;
        --name)
            RUN_NAME="$2"
            RUNNER_ARGS+=("$1" "$2")
            shift 2
            ;;
        --name=*)
            RUN_NAME="${1#*=}"
            RUNNER_ARGS+=("$1")
            shift
            ;;
        --output-file)
            OUTPUT_FILE="$2"
            OUTPUT_FILE_EXPLICIT=true
            shift 2
            ;;
        --output-file=*)
            OUTPUT_FILE="${1#*=}"
            OUTPUT_FILE_EXPLICIT=true
            shift
            ;;
        --log-file)
            LOG_FILE="$2"
            shift 2
            ;;
        --log-file=*)
            LOG_FILE="${1#*=}"
            shift
            ;;
        --num-thread|--num-threads|--threads)
            NUM_THREAD="$2"
            RUNNER_ARGS+=("$1" "$2")
            shift 2
            ;;
        --num-thread=*|--num-threads=*|--threads=*)
            NUM_THREAD="${1#*=}"
            RUNNER_ARGS+=("$1")
            shift
            ;;
        --input-file)
            INPUT_FILE="$2"
            RUNNER_ARGS+=("$1" "$2")
            shift 2
            ;;
        --input-file=*)
            INPUT_FILE="${1#*=}"
            RUNNER_ARGS+=("$1")
            shift
            ;;
        --rerun-all)
            RERUN_ALL="$2"
            RUNNER_ARGS+=("$1" "$2")
            shift 2
            ;;
        --rerun-all=*)
            RERUN_ALL="${1#*=}"
            RUNNER_ARGS+=("$1")
            shift
            ;;
        --rerun-final-answer)
            RERUN_FINAL_ANSWER="$2"
            RUNNER_ARGS+=("$1" "$2")
            shift 2
            ;;
        --rerun-final-answer=*)
            RERUN_FINAL_ANSWER="${1#*=}"
            RUNNER_ARGS+=("$1")
            shift
            ;;
        --resume)
            RESUME=true
            RUNNER_ARGS+=("$1")
            shift
            ;;
        --resume-from)
            RESUME_FROM="$2"
            RUNNER_ARGS+=("$1" "$2")
            shift 2
            ;;
        --resume-from=*)
            RESUME_FROM="${1#*=}"
            RUNNER_ARGS+=("$1")
            shift
            ;;
        --reverse)
            REVERSE=true
            RUNNER_ARGS+=("$1")
            shift
            ;;
        --max-total-thinking-token)
            MAX_TOTAL_THINKING_TOKEN="$2"
            RUNNER_ARGS+=("$1" "$2")
            shift 2
            ;;
        --max-total-thinking-token=*)
            MAX_TOTAL_THINKING_TOKEN="${1#*=}"
            RUNNER_ARGS+=("$1")
            shift
            ;;
        --max-final-answer-completion-token)
            MAX_FINAL_ANSWER_COMPLETION_TOKEN="$2"
            RUNNER_ARGS+=("$1" "$2")
            shift 2
            ;;
        --max-final-answer-completion-token=*)
            MAX_FINAL_ANSWER_COMPLETION_TOKEN="${1#*=}"
            RUNNER_ARGS+=("$1")
            shift
            ;;
        --temperature)
            TEMPERATURE="$2"
            RUNNER_ARGS+=("$1" "$2")
            shift 2
            ;;
        --temperature=*)
            TEMPERATURE="${1#*=}"
            RUNNER_ARGS+=("$1")
            shift
            ;;
        --gpu-memory-utilization)
            GPU_MEMORY_UTILIZATION="$2"
            RUNNER_ARGS+=("$1" "$2")
            shift 2
            ;;
        --gpu-memory-utilization=*)
            GPU_MEMORY_UTILIZATION="${1#*=}"
            RUNNER_ARGS+=("$1")
            shift
            ;;
        --max-num-seqs)
            MAX_NUM_SEQS="$2"
            RUNNER_ARGS+=("$1" "$2")
            shift 2
            ;;
        --max-num-seqs=*)
            MAX_NUM_SEQS="${1#*=}"
            RUNNER_ARGS+=("$1")
            shift
            ;;
        --exit)
            RUNNER_ARGS+=("$1" "$2")
            shift 2
            ;;
        --exit=*)
            RUNNER_ARGS+=("$1")
            shift
            ;;
        --random-exit)
            RANDOM_EXIT="$2"
            RUNNER_ARGS+=("$1" "$2")
            shift 2
            ;;
        --random-exit=*)
            RANDOM_EXIT="${1#*=}"
            RUNNER_ARGS+=("$1")
            shift
            ;;
        --random-exit-sample-size|--random-exit-sample-seed|--random-exit-position-seed|--random-exit-spacing|--random-exit-max-positions|--random-exit-num-runs|--random-exit-min-valid-positions)
            option="${1#--random-exit-}"
            case "${option}" in
                sample-size) RANDOM_EXIT_SAMPLE_SIZE="$2" ;;
                sample-seed) RANDOM_EXIT_SAMPLE_SEED="$2" ;;
                position-seed) RANDOM_EXIT_POSITION_SEED="$2" ;;
                spacing) RANDOM_EXIT_SPACING="$2" ;;
                max-positions) RANDOM_EXIT_MAX_POSITIONS="$2" ;;
                num-runs) RANDOM_EXIT_NUM_RUNS="$2" ;;
                min-valid-positions) RANDOM_EXIT_MIN_VALID_POSITIONS="$2" ;;
            esac
            RUNNER_ARGS+=("$1" "$2")
            shift 2
            ;;
        --random-exit-sample-size=*|--random-exit-sample-seed=*|--random-exit-position-seed=*|--random-exit-spacing=*|--random-exit-max-positions=*|--random-exit-num-runs=*|--random-exit-min-valid-positions=*)
            RUNNER_ARGS+=("$1")
            shift
            ;;
        --list-models)
            LIST_MODELS=true
            RUNNER_ARGS+=("$1")
            shift
            ;;
        --)
            RUNNER_ARGS+=("$1")
            shift
            while [[ $# -gt 0 ]]; do
                RUNNER_ARGS+=("$1")
                shift
            done
            ;;
        *)
            RUNNER_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ "${RESUME}" == true && "${OUTPUT_FILE_EXPLICIT}" != true ]]; then
    printf 'run_eval.sh requires --output-file when using --resume because default output paths are timestamped. Use --resume-from to read an old checkpoint while writing a new output.\n' >&2
    exit 2
fi

sanitize_name() {
    local value="$1"
    value="${value//\//_}"
    value="${value//:/_}"
    value="${value// /_}"
    printf '%s' "${value}" | tr -cd '[:alnum:]_.-'
}

if [[ "${LIST_MODELS}" == true ]]; then
    MODEL_NAME="${MODEL_KEY:-models}"
elif [[ -n "${LLM_ID}" ]]; then
    MODEL_NAME="${LLM_ID}"
else
    MODEL_NAME="$(
        python - "${CONFIG}" "${MODEL_KEY}" "${RANDOM_EXIT}" <<'PY'
import sys

from scripts.config_loader import load_config, resolve_model_config

config_path, model_key, cli_random_exit = sys.argv[1:4]
config = load_config(config_path)
replay = (config.get("agent_kwargs") or {}).get("replay_eval") or {}
v2 = replay.get("v2") if isinstance(replay, dict) else {}
random_exit = v2.get("random_exit") if isinstance(v2, dict) else {}
configured_random_exit = (
    random_exit.get("enabled", False) if isinstance(random_exit, dict) else False
)
resolved_random_exit = (
    cli_random_exit.strip().casefold() in {"true", "1", "yes", "on"}
    if cli_random_exit
    else bool(configured_random_exit)
)
if (int(replay.get("version", 1)) == 2 or resolved_random_exit) and not model_key:
    raise SystemExit("Replay version 2 requires --model from llm.models")
model_config = resolve_model_config(config, model_key or None)
print(model_config.get("name") or model_key or model_config["id"])
PY
    )"
fi

DATETIME="$(date +"%Y%m%d_%H%M%S")"
DATASET_NAME="$(sanitize_name "${DATALOADER}")"
AGENT_NAME="$(sanitize_name "${AGENT}")"
MODEL_NAME="$(sanitize_name "${MODEL_NAME}")"
RUN_NAME="$(sanitize_name "${RUN_NAME}")"

FILENAME_PREFIX="${AGENT_NAME}_${MODEL_NAME}"
if [[ -n "${RUN_NAME}" ]]; then
    FILENAME_PREFIX="${RUN_NAME}_${FILENAME_PREFIX}"
fi

if [[ -z "${OUTPUT_FILE}" ]]; then
    OUTPUT_FILE="output/${DATASET_NAME}/${FILENAME_PREFIX}_${DATETIME}.jsonl"
fi

if [[ -z "${LOG_FILE}" ]]; then
    LOG_FILE="logs/${DATASET_NAME}/${FILENAME_PREFIX}_${DATETIME}.log"
fi

mkdir -p "$(dirname "${OUTPUT_FILE}")" "$(dirname "${LOG_FILE}")"

{
    printf '[run_eval] output_file=%s\n' "${OUTPUT_FILE}"
    printf '[run_eval] log_file=%s\n' "${LOG_FILE}"
    printf '[run_eval] num_thread=%s\n' "${NUM_THREAD}"
    printf '[run_eval] evaluator=%s\n' "${EVALUATOR}"
    printf '[run_eval] reverse=%s\n' "${REVERSE}"
    if [[ -n "${MAX_TOTAL_THINKING_TOKEN}" ]]; then
        printf '[run_eval] max_total_thinking_token=%s\n' "${MAX_TOTAL_THINKING_TOKEN}"
    fi
    if [[ -n "${MAX_FINAL_ANSWER_COMPLETION_TOKEN}" ]]; then
        printf '[run_eval] max_final_answer_completion_token=%s\n' "${MAX_FINAL_ANSWER_COMPLETION_TOKEN}"
    fi
    if [[ -n "${TEMPERATURE}" ]]; then
        printf '[run_eval] temperature=%s\n' "${TEMPERATURE}"
    fi
    if [[ -n "${GPU_MEMORY_UTILIZATION}" ]]; then
        printf '[run_eval] gpu_memory_utilization=%s\n' "${GPU_MEMORY_UTILIZATION}"
    fi
    if [[ -n "${MAX_NUM_SEQS}" ]]; then
        printf '[run_eval] max_num_seqs=%s\n' "${MAX_NUM_SEQS}"
    fi
    if [[ -n "${RANDOM_EXIT}" ]]; then
        printf '[run_eval] random_exit=%s\n' "${RANDOM_EXIT}"
    fi
    for random_setting in \
        "sample_size=${RANDOM_EXIT_SAMPLE_SIZE}" \
        "sample_seed=${RANDOM_EXIT_SAMPLE_SEED}" \
        "position_seed=${RANDOM_EXIT_POSITION_SEED}" \
        "spacing=${RANDOM_EXIT_SPACING}" \
        "max_positions=${RANDOM_EXIT_MAX_POSITIONS}" \
        "num_runs=${RANDOM_EXIT_NUM_RUNS}" \
        "min_valid_positions=${RANDOM_EXIT_MIN_VALID_POSITIONS}"; do
        if [[ -n "${random_setting#*=}" ]]; then
            printf '[run_eval] random_exit_%s\n' "${random_setting}"
        fi
    done
    if [[ -n "${INPUT_FILE}" ]]; then
        printf '[run_eval] input_file=%s\n' "${INPUT_FILE}"
    fi
    if [[ -n "${RERUN_ALL}" ]]; then
        printf '[run_eval] rerun_all=%s\n' "${RERUN_ALL}"
    fi
    if [[ -n "${RERUN_FINAL_ANSWER}" ]]; then
        printf '[run_eval] rerun_final_answer=%s\n' "${RERUN_FINAL_ANSWER}"
    fi
    if [[ "${RESUME}" == true ]]; then
        printf '[run_eval] resume=true\n'
    fi
    if [[ -n "${RESUME_FROM}" ]]; then
        printf '[run_eval] resume_from=%s\n' "${RESUME_FROM}"
    fi
    python scripts/runner.py "${RUNNER_ARGS[@]}" --output-file "${OUTPUT_FILE}"
} >"${LOG_FILE}" 2>&1

if [[ -f "${OUTPUT_FILE}" ]]; then
    printf 'Output written to %s\n' "${OUTPUT_FILE}"
else
    printf 'Output path: %s\n' "${OUTPUT_FILE}"
fi
printf 'Logs written to %s\n' "${LOG_FILE}"
