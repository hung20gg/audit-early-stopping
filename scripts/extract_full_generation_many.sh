#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

INPUT_DIR=""
OUTPUT_DIR=""
SKIP_MISSING=false
POSITIONAL=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --input-dir|--input-folder)
            INPUT_DIR="$2"
            shift 2
            ;;
        --input-dir=*|--input-folder=*)
            INPUT_DIR="${1#*=}"
            shift
            ;;
        --output-dir|--output-folder)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --output-dir=*|--output-folder=*)
            OUTPUT_DIR="${1#*=}"
            shift
            ;;
        --skip-missing)
            SKIP_MISSING=true
            shift
            ;;
        --)
            shift
            while [[ $# -gt 0 ]]; do
                POSITIONAL+=("$1")
                shift
            done
            ;;
        *)
            POSITIONAL+=("$1")
            shift
            ;;
    esac
done

if [[ -z "${INPUT_DIR}" && "${#POSITIONAL[@]}" -ge 1 ]]; then
    INPUT_DIR="${POSITIONAL[0]}"
fi
if [[ -z "${OUTPUT_DIR}" && "${#POSITIONAL[@]}" -ge 2 ]]; then
    OUTPUT_DIR="${POSITIONAL[1]}"
fi
if [[ -z "${INPUT_DIR}" || -z "${OUTPUT_DIR}" ]]; then
    printf 'Usage: %s INPUT_DIR OUTPUT_DIR [--skip-missing]\n' "$0" >&2
    printf '   or: %s --input-dir INPUT_DIR --output-dir OUTPUT_DIR [--skip-missing]\n' "$0" >&2
    exit 2
fi
if [[ ! -d "${INPUT_DIR}" ]]; then
    printf 'Input folder does not exist: %s\n' "${INPUT_DIR}" >&2
    exit 2
fi

mkdir -p "${OUTPUT_DIR}"
shopt -s nullglob
INPUT_FILES=("${INPUT_DIR}"/*.jsonl)
shopt -u nullglob

if [[ "${#INPUT_FILES[@]}" -eq 0 ]]; then
    printf 'No .jsonl files found in %s\n' "${INPUT_DIR}" >&2
    exit 2
fi

for input_file in "${INPUT_FILES[@]}"; do
    output_file="${OUTPUT_DIR}/$(basename "${input_file}")"
    printf '[extract_full_generation_many] %s -> %s\n' "${input_file}" "${output_file}"
    if [[ "${SKIP_MISSING}" == true ]]; then
        python scripts/extract_full_generation.py \
            --skip-missing \
            "${input_file}" \
            "${output_file}"
    else
        python scripts/extract_full_generation.py \
            "${input_file}" \
            "${output_file}"
    fi
done
