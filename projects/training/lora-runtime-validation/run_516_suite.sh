#!/usr/bin/env bash
# 5.1.6 统一执行入口脚本 - 摩尔线程架构上训练任务运行测试
# MTT-TRAIN-RUN-TEST Suite Runner

set -euo pipefail

# ============================================================================
# 配置与初始化
# ============================================================================

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH="${MODEL_PATH:=/home/o_mabin/moerxiancheng-clj-xyj-proj/clj-proj/model/Meta-Llama-3.1-8B}"
CONFIG_FILE="${ROOT_DIR}/train_config.json"
DEVICE_TYPE="auto"
SINGLE_DEVICE_IDS="0"
DUAL_DEVICE_IDS="0,1"
TP_DEVICE_IDS="0,1"
DRY_RUN="false"
EXTRA_LD_PATHS=()

# 尝试找到MUSA库路径
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

# 配置LD_LIBRARY_PATH
if [[ ${#EXTRA_LD_PATHS[@]} -gt 0 ]]; then
  EXTRA_JOINED="$(IFS=:; echo "${EXTRA_LD_PATHS[*]}")"
  export LD_LIBRARY_PATH="${EXTRA_JOINED}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

# ============================================================================
# 参数解析
# ============================================================================

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-path)
      MODEL_PATH="$2"
      shift 2
      ;;
    --config-file)
      CONFIG_FILE="$2"
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
    --tp-device-ids)
      TP_DEVICE_IDS="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN="true"
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

# ============================================================================
# 时间戳和目录创建
# ============================================================================

STAMP="$(date -u +"%Y%m%dT%H%M%SZ")"
ARTIFACT_DIR="${ROOT_DIR}/artifacts/${STAMP}"
mkdir -p "${ARTIFACT_DIR}/preflight" "${ARTIFACT_DIR}/single" "${ARTIFACT_DIR}/dual" "${ARTIFACT_DIR}/tp"

echo "=========================================="
echo "5.1.6 训练任务运行测试 (MTT-TRAIN-RUN-TEST)"
echo "=========================================="
echo "开始时间：$(date)"
echo "时间戳：${STAMP}"
echo "模型路径：${MODEL_PATH}"
echo "配置文件：${CONFIG_FILE}"
echo "Dry-run模式：${DRY_RUN}"
echo "输出目录：${ARTIFACT_DIR}"
echo "=========================================="
echo ""

# ============================================================================
# Step A & B: 前置检查
# ============================================================================

echo "[执行] 步骤A-B: 环境预检查..."
PREFLIGHT_OUTPUT="${ARTIFACT_DIR}/preflight/preflight.json"

set +e
python3 "${ROOT_DIR}/preflight_check.py" \
  --output "${PREFLIGHT_OUTPUT}" \
  --model_path "${MODEL_PATH}"
PREFLIGHT_RC=$?
set -e

echo "✓ 前置检查完成，结果：${PREFLIGHT_OUTPUT}"
echo ""

# ============================================================================
# Step C-E: 单卡训练
# ============================================================================

echo "[执行] 步骤C-E: 单卡训练任务..."
SINGLE_ARGS=()
SINGLE_OUTPUT_DIR="${ARTIFACT_DIR}/single"

if [[ "${DRY_RUN}" == "true" ]]; then
  SINGLE_ARGS+=(--dry-run)
  echo "  (干运行模式)"
fi

SINGLE_LOG="${SINGLE_OUTPUT_DIR}/training.log"

# 记录单卡训练开始时间
SINGLE_START_TIME=$(date +%s)

set +e
python3 "${ROOT_DIR}/train_runner.py" \
  --model_path "${MODEL_PATH}" \
  --config_file "${CONFIG_FILE}" \
  --output_dir "${SINGLE_OUTPUT_DIR}" \
  --num_gpus 1 \
  --task_type "lora_training" \
  --device_ids "${SINGLE_DEVICE_IDS}" \
  "${SINGLE_ARGS[@]}" \
  2>&1 | tee "${SINGLE_LOG}"
SINGLE_RC=${PIPESTATUS[0]}
set -e

# 记录单卡训练结束时间
SINGLE_END_TIME=$(date +%s)
SINGLE_DURATION=$((SINGLE_END_TIME - SINGLE_START_TIME))

# 如 train_runner 已写入更完整 summary，则只补充 execution_time_seconds；
# 否则回退到兼容格式，避免覆盖主 runtime 产物。
if [[ -f "${SINGLE_OUTPUT_DIR}/summary.json" ]]; then
  python3 - "${SINGLE_OUTPUT_DIR}/summary.json" "${SINGLE_DURATION}" <<'PY'
import json, sys
path = sys.argv[1]
duration = int(sys.argv[2])
with open(path, "r", encoding="utf-8") as f:
    payload = json.load(f)
payload["execution_time_seconds"] = duration
with open(path, "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)
PY
else
  if [[ ${SINGLE_RC} -eq 0 ]]; then
    SINGLE_SUCCESS=true
    SINGLE_ERRORS='[]'
  else
    SINGLE_SUCCESS=false
    SINGLE_ERRORS='["train_runner_exit_code:'"${SINGLE_RC}"'"]'
  fi
  cat > "${SINGLE_OUTPUT_DIR}/summary.json" <<EOF
{
  "test_id": "MTT-TRAIN-RUN-TEST-SINGLE",
  "model_path": "${MODEL_PATH}",
  "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "dry_run": ${DRY_RUN},
  "num_gpus": 1,
  "success": ${SINGLE_SUCCESS},
  "execution_time_seconds": ${SINGLE_DURATION},
  "outputs": ["${SINGLE_LOG}"],
  "errors": ${SINGLE_ERRORS}
}
EOF
fi

echo "✓ 单卡训练完成，结果：${SINGLE_OUTPUT_DIR}"
echo ""

# ============================================================================
# Step C-E: 双卡训练
# ============================================================================

echo "[执行] 步骤C-E: 单机双卡训练任务..."
DUAL_ARGS=()
DUAL_OUTPUT_DIR="${ARTIFACT_DIR}/dual"

if [[ "${DRY_RUN}" == "true" ]]; then
  DUAL_ARGS+=(--dry-run)
  echo "  (干运行模式)"
fi

DUAL_LOG="${DUAL_OUTPUT_DIR}/training.log"

# 记录双卡训练开始时间
DUAL_START_TIME=$(date +%s)

set +e
python3 "${ROOT_DIR}/train_runner.py" \
  --model_path "${MODEL_PATH}" \
  --config_file "${CONFIG_FILE}" \
  --output_dir "${DUAL_OUTPUT_DIR}" \
  --num_gpus 2 \
  --task_type "lora_training" \
  --device_ids "${DUAL_DEVICE_IDS}" \
  --distributed \
  "${DUAL_ARGS[@]}" \
  2>&1 | tee "${DUAL_LOG}"
DUAL_RC=${PIPESTATUS[0]}
set -e

# 记录双卡训练结束时间
DUAL_END_TIME=$(date +%s)
DUAL_DURATION=$((DUAL_END_TIME - DUAL_START_TIME))

if [[ -f "${DUAL_OUTPUT_DIR}/summary.json" ]]; then
  python3 - "${DUAL_OUTPUT_DIR}/summary.json" "${DUAL_DURATION}" <<'PY'
import json, sys
path = sys.argv[1]
duration = int(sys.argv[2])
with open(path, "r", encoding="utf-8") as f:
    payload = json.load(f)
payload["execution_time_seconds"] = duration
with open(path, "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)
PY
else
  if [[ ${DUAL_RC} -eq 0 ]]; then
    DUAL_SUCCESS=true
    DUAL_ERRORS='[]'
  else
    DUAL_SUCCESS=false
    DUAL_ERRORS='["train_runner_exit_code:'"${DUAL_RC}"'"]'
  fi
  cat > "${DUAL_OUTPUT_DIR}/summary.json" <<EOF
{
  "test_id": "MTT-TRAIN-RUN-TEST-DUAL",
  "model_path": "${MODEL_PATH}",
  "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "dry_run": ${DRY_RUN},
  "num_gpus": 2,
  "success": ${DUAL_SUCCESS},
  "execution_time_seconds": ${DUAL_DURATION},
  "outputs": ["${DUAL_LOG}"],
  "errors": ${DUAL_ERRORS}
}
EOF
fi

echo "✓ 双卡训练完成，结果：${DUAL_OUTPUT_DIR}"
echo ""

# ============================================================================
# Step C-E: 双卡张量并行训练
# ============================================================================

echo "[执行] 步骤C-E: 双卡张量并行训练任务..."
TP_ARGS=()
TP_OUTPUT_DIR="${ARTIFACT_DIR}/tp"

if [[ "${DRY_RUN}" == "true" ]]; then
  TP_ARGS+=(--dry-run)
  echo "  (干运行模式)"
fi

TP_LOG="${TP_OUTPUT_DIR}/training.log"
TP_START_TIME=$(date +%s)

set +e
python3 "${ROOT_DIR}/train_runner.py" \
  --model_path "${MODEL_PATH}" \
  --config_file "${CONFIG_FILE}" \
  --output_dir "${TP_OUTPUT_DIR}" \
  --num_gpus 2 \
  --task_type "lora_training" \
  --device_ids "${TP_DEVICE_IDS}" \
  --parallel_mode tp \
  --distributed \
  "${TP_ARGS[@]}" \
  2>&1 | tee "${TP_LOG}"
TP_RC=${PIPESTATUS[0]}
set -e

TP_END_TIME=$(date +%s)
TP_DURATION=$((TP_END_TIME - TP_START_TIME))

if [[ -f "${TP_OUTPUT_DIR}/summary.json" ]]; then
  python3 - "${TP_OUTPUT_DIR}/summary.json" "${TP_DURATION}" <<'PY'
import json, sys
path = sys.argv[1]
duration = int(sys.argv[2])
with open(path, "r", encoding="utf-8") as f:
    payload = json.load(f)
payload["execution_time_seconds"] = duration
with open(path, "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)
PY
else
  if [[ ${TP_RC} -eq 0 ]]; then
    TP_SUCCESS=true
    TP_ERRORS='[]'
  else
    TP_SUCCESS=false
    TP_ERRORS='["train_runner_exit_code:'"${TP_RC}"'"]'
  fi
  cat > "${TP_OUTPUT_DIR}/summary.json" <<EOF
{
  "test_id": "MTT-TRAIN-RUN-TEST-TP",
  "model_path": "${MODEL_PATH}",
  "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "dry_run": ${DRY_RUN},
  "num_gpus": 2,
  "parallel_mode": "tp",
  "success": ${TP_SUCCESS},
  "execution_time_seconds": ${TP_DURATION},
  "outputs": ["${TP_LOG}"],
  "errors": ${TP_ERRORS}
}
EOF
fi

echo "✓ 张量并行训练完成，结果：${TP_OUTPUT_DIR}"
echo ""

# ============================================================================
# Step F: 结果汇总与判定
# ============================================================================

echo "[执行] 步骤F: 结果汇总与判定..."
SUMMARY_OUTPUT="${ARTIFACT_DIR}/5.1.6任务完成总结.md"

python3 "${ROOT_DIR}/train_summarize.py" \
  --output "${SUMMARY_OUTPUT}" \
  --preflight "${PREFLIGHT_OUTPUT}" \
  --single "${SINGLE_OUTPUT_DIR}/summary.json" \
  --dual "${DUAL_OUTPUT_DIR}/summary.json" \
  --tp "${TP_OUTPUT_DIR}/summary.json"

echo "✓ 结果汇总完成，报告：${SUMMARY_OUTPUT}"
echo ""

# ============================================================================
# 完成总结
# ============================================================================

echo "=========================================="
echo "5.1.6 训练任务运行测试 - 执行完成"
echo "=========================================="
echo "完成时间：$(date)"
echo "输出目录：${ARTIFACT_DIR}"
echo ""
echo "生成的输出文件："
echo "  - 前置检查：${PREFLIGHT_OUTPUT}"
echo "  - 单卡训练：${SINGLE_OUTPUT_DIR}/"
echo "  - 双卡训练：${DUAL_OUTPUT_DIR}/"
echo "  - TP训练：${TP_OUTPUT_DIR}/"
echo "  - 完成总结：${SUMMARY_OUTPUT}"
echo "=========================================="
echo ""
OVERALL_RC=0
if [[ ${PREFLIGHT_RC} -ne 0 && "${DRY_RUN}" != "true" ]]; then
  OVERALL_RC=1
fi
if [[ ${SINGLE_RC} -ne 0 || ${DUAL_RC} -ne 0 || ${TP_RC} -ne 0 ]]; then
  OVERALL_RC=1
fi

if [[ ${OVERALL_RC} -ne 0 ]]; then
  echo "[警告] 测试存在失败项：preflight=${PREFLIGHT_RC}, single=${SINGLE_RC}, dual=${DUAL_RC}, tp=${TP_RC}" >&2
elif [[ ${PREFLIGHT_RC} -ne 0 ]]; then
  echo "[提示] dry-run 流程已通过；当前环境未验证到 MUSA 设备，真实训练需在可见设备上复测。" >&2
fi

exit ${OVERALL_RC}
