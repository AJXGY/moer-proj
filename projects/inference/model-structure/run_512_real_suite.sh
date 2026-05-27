#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${ROOT_DIR}/../../.." && pwd)"
RUN_515="${REPO_ROOT}/projects/inference/runtime-validation/run_515_suite.sh"
DEFAULT_MODEL_PATH="${REPO_ROOT}/clj-proj/model/Meta-Llama-3.1-8B"

MODEL_PATH="${DEFAULT_MODEL_PATH}"
DEVICE_TYPE="musa"
SINGLE_DEVICE_IDS="0"
DUAL_DEVICE_IDS="0,1"
DRY_RUN="false"
SKIP_MODEL_BUILD="false"
SINGLE_ONLY="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-path)
      MODEL_PATH="$2"
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
    --skip-model-build)
      SKIP_MODEL_BUILD="true"
      shift
      ;;
    --single-only)
      SINGLE_ONLY="true"
      shift
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 1
      ;;
  esac
done

if [[ ! -x "${RUN_515}" ]]; then
  chmod +x "${RUN_515}"
fi

if [[ "${SKIP_MODEL_BUILD}" != "true" ]]; then
  echo "[1/4] 生成 5.1.12 建模与图表产物..."
  bash "${ROOT_DIR}/run_512_suite.sh"
else
  echo "[1/4] 跳过建模生成（--skip-model-build）"
fi

echo "[2/4] 执行 5.1.5 单卡/双卡推理实测..."
LOG_FILE="$(mktemp)"
RUN_515_ARGS=(
  --model-path "${MODEL_PATH}"
  --device-type "${DEVICE_TYPE}"
  --single-device-ids "${SINGLE_DEVICE_IDS}"
  --dual-device-ids "${DUAL_DEVICE_IDS}"
)
if [[ "${DRY_RUN}" == "true" ]]; then
  RUN_515_ARGS+=(--dry-run)
fi
if [[ "${SINGLE_ONLY}" == "true" ]]; then
  RUN_515_ARGS+=(--single-only)
fi

bash "${RUN_515}" "${RUN_515_ARGS[@]}" | tee "${LOG_FILE}"
ARTIFACT_515="$(tail -n1 "${LOG_FILE}")"
rm -f "${LOG_FILE}"

if [[ ! -d "${ARTIFACT_515}" ]]; then
  echo "无法识别 5.1.5 实测产物目录: ${ARTIFACT_515}" >&2
  exit 2
fi

STAMP="$(date -u +"%Y%m%dT%H%M%SZ")"
TARGET_ARTIFACT="${ROOT_DIR}/artifacts/${STAMP}"
mkdir -p "${TARGET_ARTIFACT}"

echo "[3/4] 收集实测证据到 5.1.12..."
cp -f "${ROOT_DIR}/inference_execution_model.json" "${TARGET_ARTIFACT}/inference_execution_model.json"
cp -f "${ROOT_DIR}/5.1.12任务进展.md" "${TARGET_ARTIFACT}/5.1.12任务进展.md"
cp -f "${ROOT_DIR}/5.1.12图表汇总.md" "${TARGET_ARTIFACT}/5.1.12图表汇总.md"
cp -rf "${ROOT_DIR}/charts" "${TARGET_ARTIFACT}/charts"
cp -rf "${ARTIFACT_515}/preflight" "${TARGET_ARTIFACT}/runtime_preflight"
cp -rf "${ARTIFACT_515}/single" "${TARGET_ARTIFACT}/runtime_single"
if [[ "${SINGLE_ONLY}" != "true" ]]; then
  cp -rf "${ARTIFACT_515}/dual" "${TARGET_ARTIFACT}/runtime_dual"
fi

python3 - <<'PY' "${TARGET_ARTIFACT}" "${ARTIFACT_515}" "${MODEL_PATH}" "${DEVICE_TYPE}" "${SINGLE_DEVICE_IDS}" "${DUAL_DEVICE_IDS}" "${DRY_RUN}" "${SINGLE_ONLY}"
import json
import os
import sys
from datetime import datetime, timezone

out_dir = sys.argv[1]
artifact_515 = sys.argv[2]
model_path = sys.argv[3]
device_type = sys.argv[4]
single_ids = sys.argv[5]
dual_ids = sys.argv[6]
dry_run = sys.argv[7] == "true"
single_only = sys.argv[8] == "true"

with open(os.path.join(artifact_515, "single", "summary.json"), "r", encoding="utf-8") as f:
    single = json.load(f)
dual = None
if not single_only:
  with open(os.path.join(artifact_515, "dual", "summary.json"), "r", encoding="utf-8") as f:
    dual = json.load(f)

runtime_summary = {
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "task_id": "MTT-INFER-MODEL-STRUCT-TEST",
    "validation_type": "runtime_consistency_check",
    "source_515_artifact": artifact_515,
    "model_path": model_path,
    "device_type": device_type,
    "single_device_ids": single_ids,
    "dual_device_ids": dual_ids,
    "dry_run": dry_run,
    "single_only": single_only,
    "single_success": bool(single.get("success", False)),
    "dual_success": bool(dual.get("success", False)) if dual else None,
    "single_outputs_count": int(single.get("outputs_count", 0)),
    "dual_outputs_count": int(dual.get("outputs_count", 0)) if dual else 0,
    "overall_pass": bool(single.get("success", False) if single_only else single.get("success", False) and dual.get("success", False)),
}

summary_path = os.path.join(out_dir, "runtime_validation_summary.json")
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(runtime_summary, f, ensure_ascii=False, indent=2)

status = "通过" if runtime_summary["overall_pass"] else "未通过"
if single_only:
  status = "通过" if runtime_summary["single_success"] else "未通过"
md = f"""# 5.1.12 实测结果

- 生成时间：{runtime_summary['generated_at']}
- 任务标识：MTT-INFER-MODEL-STRUCT-TEST
- 实测来源：{artifact_515}
- 模型路径：{model_path}
- 设备类型：{device_type}
- 单卡设备：{single_ids}
- 双卡设备：{dual_ids}
- dry-run：{str(dry_run).lower()}
- single-only：{str(single_only).lower()}

## 实测判定

- 单卡成功：{runtime_summary['single_success']}
- 双卡成功：{runtime_summary['dual_success']}
- 单卡输出条数：{runtime_summary['single_outputs_count']}
- 双卡输出条数：{runtime_summary['dual_outputs_count']}
- 综合判定：{status}

## 证据文件

- runtime_validation_summary.json
- runtime_preflight/preflight.json
- runtime_single/summary.json
- runtime_single/outputs.jsonl
- runtime_dual/summary.json
- runtime_dual/outputs.jsonl
- inference_execution_model.json
- 5.1.12任务进展.md
- 5.1.12图表汇总.md
"""
if single_only:
  md = md.replace("- runtime_dual/summary.json\n- runtime_dual/outputs.jsonl\n", "")

report_path = os.path.join(os.path.dirname(out_dir), "5.1.12实测结果.md")
with open(report_path, "w", encoding="utf-8") as f:
    f.write(md)
PY

echo "[4/4] 完成"
echo "5.1.12 实测产物目录: ${TARGET_ARTIFACT}"
echo "5.1.12 实测报告: ${ROOT_DIR}/artifacts/5.1.12实测结果.md"
