#!/bin/bash

# 5.1.6 单卡训练脚本 - 摩尔线程架构上训练任务运行测试
# MTT-TRAIN-RUN-TEST Single GPU Training Script

set -e

# 配置参数
MODEL_PATH="${MODEL_PATH:=/path/to/Meta-Llama-3.1-8B}"
CONFIG_FILE="train_config.json"
OUTPUT_DIR="./checkpoints/single_gpu"
LOG_DIR="./logs/single_gpu"
TIMESTAMP=$(date +%Y%m%dT%H%M%SZ)
LOG_FILE="${LOG_DIR}/${TIMESTAMP}.log"

# 创建输出目录
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${LOG_DIR}"

echo "========================================" | tee -a "${LOG_FILE}"
echo "5.1.6 单卡训练测试" | tee -a "${LOG_FILE}"
echo "测试ID: MTT-TRAIN-RUN-TEST-SINGLE-GPU" | tee -a "${LOG_FILE}"
echo "时间: ${TIMESTAMP}" | tee -a "${LOG_FILE}"
echo "========================================" | tee -a "${LOG_FILE}"

# 步骤A：环境检查
echo "" | tee -a "${LOG_FILE}"
echo "[步骤A] 环境配置检查..." | tee -a "${LOG_FILE}"
echo "========================================" | tee -a "${LOG_FILE}"

# 检查CUDA/MUSA环境
if command -v nvidia-smi &> /dev/null; then
    echo "GPU信息 (NVIDIA):" | tee -a "${LOG_FILE}"
    nvidia-smi -i 0 | tee -a "${LOG_FILE}"
elif command -v musa-version &> /dev/null; then
    echo "GPU信息 (MUSA):" | tee -a "${LOG_FILE}"
    musa-version | tee -a "${LOG_FILE}"
else
    echo "警告: 未检测到GPU驱动" | tee -a "${LOG_FILE}"
fi

# 检查PyTorch
python -c "import torch; print(f'PyTorch版本: {torch.__version__}')" | tee -a "${LOG_FILE}"
python -c "import torch; print(f'GPU可用: {torch.cuda.is_available()}')" | tee -a "${LOG_FILE}"

# 步骤B：模型和训练脚本准备
echo "" | tee -a "${LOG_FILE}"
echo "[步骤B] 模型和训练脚本准备..." | tee -a "${LOG_FILE}"
echo "========================================" | tee -a "${LOG_FILE}"

if [ -d "${MODEL_PATH}" ]; then
    echo "✓ 模型路径存在: ${MODEL_PATH}" | tee -a "${LOG_FILE}"
else
    echo "✗ 错误: 模型路径不存在: ${MODEL_PATH}" | tee -a "${LOG_FILE}"
    exit 1
fi

if [ -f "${CONFIG_FILE}" ]; then
    echo "✓ 配置文件存在: ${CONFIG_FILE}" | tee -a "${LOG_FILE}"
else
    echo "✗ 错误: 配置文件不存在: ${CONFIG_FILE}" | tee -a "${LOG_FILE}"
    exit 1
fi

echo "训练配置:" | tee -a "${LOG_FILE}"
cat "${CONFIG_FILE}" | jq . | tee -a "${LOG_FILE}"

# 步骤C：启动单卡训练任务
echo "" | tee -a "${LOG_FILE}"
echo "[步骤C] 启动单卡训练任务..." | tee -a "${LOG_FILE}"
echo "========================================" | tee -a "${LOG_FILE}"

TRAIN_START_TIME=$(date +%s)
echo "训练开始时间: $(date)" | tee -a "${LOG_FILE}"

# 执行训练脚本 (这里使用占位符，实际需要替换为真实的训练命令)
python train_runner.py \
    --model_path "${MODEL_PATH}" \
    --config_file "${CONFIG_FILE}" \
    --output_dir "${OUTPUT_DIR}" \
    --num_gpus 1 \
    --task_type "full_training" \
    2>&1 | tee -a "${LOG_FILE}"

TRAIN_END_TIME=$(date +%s)
TRAIN_DURATION=$((TRAIN_END_TIME - TRAIN_START_TIME))
echo "训练结束时间: $(date)" | tee -a "${LOG_FILE}"
echo "总训练时间: ${TRAIN_DURATION} 秒" | tee -a "${LOG_FILE}"

# 步骤D：监控和日志分析
echo "" | tee -a "${LOG_FILE}"
echo "[步骤D] 监控数据和日志分析..." | tee -a "${LOG_FILE}"
echo "========================================" | tee -a "${LOG_FILE}"

# 检查输出目录
if [ -d "${OUTPUT_DIR}/checkpoints" ]; then
    echo "✓ checkpoints目录存在" | tee -a "${LOG_FILE}"
    CHECKPOINT_COUNT=$(find "${OUTPUT_DIR}/checkpoints" -type d -name "checkpoint-*" | wc -l)
    echo "✓ 保存的checkpoint数量: ${CHECKPOINT_COUNT}" | tee -a "${LOG_FILE}"
else
    echo "✗ checkpoints目录不存在" | tee -a "${LOG_FILE}"
fi

# 检查日志文件
if [ -f "${OUTPUT_DIR}/training_log.json" ]; then
    echo "✓ 训练日志文件存在" | tee -a "${LOG_FILE}"
else
    echo "⚠ 训练日志文件不存在" | tee -a "${LOG_FILE}"
fi

# 步骤E：验证训练完成和结果
echo "" | tee -a "${LOG_FILE}"
echo "[步骤E] 验证训练结果..." | tee -a "${LOG_FILE}"
echo "========================================" | tee -a "${LOG_FILE}"

# 检查是否存在异常
ERROR_PATTERNS=("RuntimeError" "CUDA Error" "OutOfMemory" "core dumped" "Segmentation fault")
ERRORS_FOUND=0

for PATTERN in "${ERROR_PATTERNS[@]}"; do
    if grep -q "${PATTERN}" "${LOG_FILE}"; then
        echo "✗ 发现异常: ${PATTERN}" | tee -a "${LOG_FILE}"
        ERRORS_FOUND=$((ERRORS_FOUND + 1))
    fi
done

if [ $ERRORS_FOUND -eq 0 ]; then
    echo "✓ 未发现异常错误" | tee -a "${LOG_FILE}"
else
    echo "✗ 发现 ${ERRORS_FOUND} 个错误" | tee -a "${LOG_FILE}"
fi

# 步骤F：测试判定
echo "" | tee -a "${LOG_FILE}"
echo "[步骤F] 测试判定..." | tee -a "${LOG_FILE}"
echo "========================================" | tee -a "${LOG_FILE}"

TEST_RESULT="通过"
if [ -d "${OUTPUT_DIR}/checkpoints" ] && [ $ERRORS_FOUND -eq 0 ]; then
    echo "✓ 测试状态: 通过" | tee -a "${LOG_FILE}"
    TEST_RESULT="通过"
else
    echo "✗ 测试状态: 失败" | tee -a "${LOG_FILE}"
    TEST_RESULT="失败"
fi

echo "" | tee -a "${LOG_FILE}"
echo "========================================" | tee -a "${LOG_FILE}"
echo "单卡训练测试完成" | tee -a "${LOG_FILE}"
echo "最终判定: ${TEST_RESULT}" | tee -a "${LOG_FILE}"
echo "日志文件: ${LOG_FILE}" | tee -a "${LOG_FILE}"
echo "========================================" | tee -a "${LOG_FILE}"

[ "${TEST_RESULT}" = "通过" ] && exit 0 || exit 1
