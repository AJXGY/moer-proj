#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${ROOT_DIR}/../../.." && pwd)"
RUN_512_REAL="${REPO_ROOT}/projects/inference/model-structure/run_512_real_suite.sh"

DEFAULT_MODEL_PATH="/home/o_mabin/moerxiancheng-clj-xyj-proj/clj-proj/model/Meta-Llama-3.1-8B"
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

if [[ ! -x "${RUN_512_REAL}" ]]; then
  chmod +x "${RUN_512_REAL}"
fi

echo "[1/4] 调用 5.1.12 一键实测脚本..."
LOG_FILE="$(mktemp)"
RUN_512_ARGS=(
  --model-path "${MODEL_PATH}"
  --device-type "${DEVICE_TYPE}"
  --single-device-ids "${SINGLE_DEVICE_IDS}"
  --dual-device-ids "${DUAL_DEVICE_IDS}"
)
if [[ "${DRY_RUN}" == "true" ]]; then
  RUN_512_ARGS+=(--dry-run)
fi
if [[ "${SKIP_MODEL_BUILD}" == "true" ]]; then
  RUN_512_ARGS+=(--skip-model-build)
fi
if [[ "${SINGLE_ONLY}" == "true" ]]; then
  RUN_512_ARGS+=(--single-only)
fi

bash "${RUN_512_REAL}" "${RUN_512_ARGS[@]}" | tee "${LOG_FILE}"

SOURCE_ARTIFACT="$(grep -E '^5\.1\.12 实测产物目录:' "${LOG_FILE}" | tail -n1 | sed 's/^5\.1\.12 实测产物目录: //')"
SOURCE_REPORT="$(grep -E '^5\.1\.12 实测报告:' "${LOG_FILE}" | tail -n1 | sed 's/^5\.1\.12 实测报告: //')"
rm -f "${LOG_FILE}"

if [[ -z "${SOURCE_ARTIFACT}" || ! -d "${SOURCE_ARTIFACT}" ]]; then
  echo "无法识别 5.1.12 实测产物目录" >&2
  exit 2
fi
if [[ -z "${SOURCE_REPORT}" || ! -f "${SOURCE_REPORT}" ]]; then
  echo "无法识别 5.1.12 实测报告" >&2
  exit 3
fi

echo "[2/4] 归档到 5.2.15..."
STAMP="$(date -u +"%Y%m%dT%H%M%SZ")"
TARGET_ARTIFACT="${ROOT_DIR}/artifacts/${STAMP}"
mkdir -p "${TARGET_ARTIFACT}"
cp -rf "${SOURCE_ARTIFACT}" "${TARGET_ARTIFACT}/runtime_bundle"
cp -f "${SOURCE_REPORT}" "${TARGET_ARTIFACT}/5.1.12实测结果.md"

echo "[3/4] 生成 5.2.15 摘要报告..."
python3 - <<'PY' "${TARGET_ARTIFACT}" "${SOURCE_ARTIFACT}" "${SOURCE_REPORT}" "${MODEL_PATH}" "${DEVICE_TYPE}" "${SINGLE_DEVICE_IDS}" "${DUAL_DEVICE_IDS}" "${DRY_RUN}" "${SINGLE_ONLY}"
import json
import os
import sys
from datetime import datetime, timezone

target_artifact = sys.argv[1]
source_artifact = sys.argv[2]
source_report = sys.argv[3]
model_path = sys.argv[4]
device_type = sys.argv[5]
single_ids = sys.argv[6]
dual_ids = sys.argv[7]
dry_run = sys.argv[8] == "true"
single_only = sys.argv[9] == "true"

runtime_summary_path = os.path.join(source_artifact, "runtime_validation_summary.json")
overall_pass = False
single_success = False
dual_success = False

if os.path.exists(runtime_summary_path):
    with open(runtime_summary_path, "r", encoding="utf-8") as f:
        rs = json.load(f)
    overall_pass = bool(rs.get("overall_pass", False))
    single_success = bool(rs.get("single_success", False))
  dual_success = bool(rs.get("dual_success", False)) if not single_only else None

payload = {
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "task_id": "MTT-PARALLEL-INFER-TIME-TEST",
    "archive_id": os.path.basename(target_artifact),
    "source_artifact_512": source_artifact,
    "source_report_512": source_report,
    "model_path": model_path,
    "device_type": device_type,
    "single_device_ids": single_ids,
    "dual_device_ids": dual_ids,
    "dry_run": dry_run,
    "single_only": single_only,
    "single_success": single_success,
    "dual_success": dual_success,
    "overall_pass": overall_pass if not single_only else single_success,
}

summary_json = os.path.join(target_artifact, "runtime_validation_summary.json")
with open(summary_json, "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)

status = "通过" if overall_pass else "未通过"
if single_only:
  status = "通过" if single_success else "未通过"
report_md = f"""# 5.2.15 实测结果

- 生成时间：{payload['generated_at']}
- 任务标识：MTT-PARALLEL-INFER-TIME-TEST
- 数据来源：{source_artifact}
- 模型路径：{model_path}
- 设备类型：{device_type}
- 单卡设备：{single_ids}
- 双卡设备：{dual_ids}
- dry-run：{str(dry_run).lower()}
- single-only：{str(single_only).lower()}

## 判定结果

- 单卡成功：{single_success}
- 双卡成功：{dual_success}
- 综合判定：{status}

## 产物清单

- runtime_validation_summary.json
- runtime_bundle/
- runtime_bundle/runtime_validation_summary.json
- runtime_bundle/runtime_preflight/preflight.json
- runtime_bundle/runtime_single/summary.json
- runtime_bundle/runtime_dual/summary.json
- runtime_bundle/5.1.12实测结果.md
"""
if single_only:
  report_md = report_md.replace("- runtime_bundle/runtime_dual/summary.json\n", "")

report_out = os.path.join(target_artifact, "5.2.15实测结果.md")
with open(report_out, "w", encoding="utf-8") as f:
    f.write(report_md)
PY

mkdir -p "${ROOT_DIR}/artifacts"
cp -f "${TARGET_ARTIFACT}/5.2.15实测结果.md" "${ROOT_DIR}/artifacts/5.2.15实测结果.md"

if [[ -f "${TARGET_ARTIFACT}/runtime_validation_summary.json" ]]; then
  cp -f "${TARGET_ARTIFACT}/runtime_validation_summary.json" "${ROOT_DIR}/artifacts/latest_runtime_validation_summary.json"
fi

echo "[4/4] 完成"
echo "5.2.15 实测产物目录: ${TARGET_ARTIFACT}"
echo "5.2.15 实测报告: ${ROOT_DIR}/artifacts/5.2.15实测结果.md"
