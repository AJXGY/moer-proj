#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH="/home/o_mabin/moerxiancheng-clj-xyj-proj/clj-proj/model/Meta-Llama-3.1-8B"
PROMPTS_FILE="${ROOT_DIR}/prompts.jsonl"
DEVICE_TYPE="auto"
SINGLE_DEVICE_IDS="0"
DUAL_DEVICE_IDS="0,1"
DRY_RUN="false"
SINGLE_ONLY="false"
MAX_NEW_TOKENS="8"
EXTRA_LD_PATHS=()

for candidate in \
  "/home/o_mabin/.local/gfortran/usr/lib/x86_64-linux-gnu" \
  "/home/o_mabin/.local/openblas/usr/lib/x86_64-linux-gnu/openblas-pthread" \
  "/home/o_mabin/.local/musa_toolkits/musa_toolkits_4.2.0/lib" \
  "/home/o_mabin/.local/mudnn/mudnn/lib" \
  "/usr/local/musa/lib"
do
  if [[ -d "${candidate}" ]]; then
    EXTRA_LD_PATHS+=("${candidate}")
  fi
done

if [[ ${#EXTRA_LD_PATHS[@]} -gt 0 ]]; then
  EXTRA_JOINED="$(IFS=:; echo "${EXTRA_LD_PATHS[*]}")"
  export LD_LIBRARY_PATH="${EXTRA_JOINED}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-path)
      MODEL_PATH="$2"
      shift 2
      ;;
    --prompts-file)
      PROMPTS_FILE="$2"
      shift 2
      ;;
    --device-type)
      DEVICE_TYPE="$2"
      shift 2
      ;;
    --single-device-ids)
      SINGLE_DEVICE_IDS="$2"
      shift 2
      ;;
    --dual-device-ids)
      DUAL_DEVICE_IDS="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN="true"
      shift
      ;;
    --single-only)
      SINGLE_ONLY="true"
      shift
      ;;
    --max-new-tokens)
      MAX_NEW_TOKENS="$2"
      shift 2
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 1
      ;;
  esac
done

STAMP="$(date -u +"%Y%m%dT%H%M%SZ")"
ARTIFACT_DIR="${ROOT_DIR}/artifacts/${STAMP}"
mkdir -p "${ARTIFACT_DIR}/preflight" "${ARTIFACT_DIR}/single"
if [[ "${SINGLE_ONLY}" != "true" ]]; then
  mkdir -p "${ARTIFACT_DIR}/dual"
fi

python3 "${ROOT_DIR}/preflight_check.py" \
  --output "${ARTIFACT_DIR}/preflight/preflight.json"

SINGLE_ARGS=()
DUAL_ARGS=()
if [[ "${DRY_RUN}" == "true" ]]; then
  SINGLE_ARGS+=(--dry-run)
  DUAL_ARGS+=(--dry-run)
fi

python3 "${ROOT_DIR}/infer_runner.py" \
  --model-path "${MODEL_PATH}" \
  --prompts-file "${PROMPTS_FILE}" \
  --output-dir "${ARTIFACT_DIR}/single" \
  --mode-name single \
  --num-devices 1 \
  --device-type "${DEVICE_TYPE}" \
  --device-ids "${SINGLE_DEVICE_IDS}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  "${SINGLE_ARGS[@]}"

if [[ "${SINGLE_ONLY}" != "true" ]]; then
  python3 "${ROOT_DIR}/infer_runner.py" \
    --model-path "${MODEL_PATH}" \
    --prompts-file "${PROMPTS_FILE}" \
    --output-dir "${ARTIFACT_DIR}/dual" \
    --mode-name dual \
    --num-devices 2 \
    --device-type "${DEVICE_TYPE}" \
    --device-ids "${DUAL_DEVICE_IDS}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    "${DUAL_ARGS[@]}"
fi

SUMMARY_ARGS=()
if [[ "${SINGLE_ONLY}" == "true" ]]; then
  SUMMARY_ARGS+=(--single-only)
fi

python3 "${ROOT_DIR}/summarize_results.py" \
  --artifacts-dir "${ARTIFACT_DIR}" \
  --output "${ARTIFACT_DIR}/5.1.5任务进展.md" \
  "${SUMMARY_ARGS[@]}"

cp "${ARTIFACT_DIR}/5.1.5任务进展.md" "${ROOT_DIR}/5.1.5任务进展.md"

echo "${ARTIFACT_DIR}"
