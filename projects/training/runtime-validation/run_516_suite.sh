#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
REPO_ROOT="$(cd "../../.." && pwd)"
PYTHON_WITH_ENV="${REPO_ROOT}/projects/shared/train-infer-estimation/tools/python_with_env.sh"
MODEL_PATH="${REPO_ROOT}/clj-proj/model/Meta-Llama-3.1-8B"

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

STAMP="$(date -u +"%Y%m%dT%H%M%SZ")"
ARTIFACT_DIR="$(pwd)/artifacts/${STAMP}"
mkdir -p "${ARTIFACT_DIR}/single" "${ARTIFACT_DIR}/dual" "${ARTIFACT_DIR}/tp"

"${PYTHON_WITH_ENV}" - "${ARTIFACT_DIR}/preflight.json" "${MODEL_PATH}" <<'PY'
import json
import os
import sys
from pathlib import Path

output = Path(sys.argv[1])
model_path = sys.argv[2]
payload = {
    "backend": "cpu",
    "device_count": 0,
    "device_names": [],
    "model_path": model_path,
}
try:
    import torch_musa  # noqa: F401
    import torch
    if hasattr(torch, "musa") and torch.musa.is_available():
        payload["backend"] = "musa"
        payload["device_count"] = int(torch.musa.device_count())
        payload["device_names"] = [torch.musa.get_device_name(i) for i in range(torch.musa.device_count())]
except Exception:
    pass
output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print(output)
PY

"${PYTHON_WITH_ENV}" \
  run_train_task.py \
  --mode single \
  --output-dir "${ARTIFACT_DIR}/single"

"${PYTHON_WITH_ENV}" \
  run_train_task.py \
  --mode dual \
  --output-dir "${ARTIFACT_DIR}/dual"

"${PYTHON_WITH_ENV}" \
  run_train_task.py \
  --mode tp \
  --output-dir "${ARTIFACT_DIR}/tp"

python3 summarize_results.py \
  --artifacts-dir "${ARTIFACT_DIR}" \
  --output "${ARTIFACT_DIR}/5.1.6任务进展.md"

cp "${ARTIFACT_DIR}/5.1.6任务进展.md" "$(pwd)/5.1.6任务进展.md"
echo "${ARTIFACT_DIR}" > "$(pwd)/latest_artifact.txt"
echo "${ARTIFACT_DIR}"
