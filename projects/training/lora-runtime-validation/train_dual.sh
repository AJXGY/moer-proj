#!/bin/bash

# 5.1.6 单机双卡训练脚本 - 摩尔线程架构上训练任务运行测试
# MTT-TRAIN-RUN-TEST Dual GPU Training Script

set -e

# 配置参数
MODEL_PATH="${MODEL_PATH:=/path/to/Meta-Llama-3.1-8B}"
CONFIG_FILE="train_config.json"
OUTPUT_DIR="./checkpoints/dual_gpu"
LOG_DIR="./logs/dual_gpu"
TIMESTAMP=$(date +%Y%m%dT%H%M%SZ)
LOG_FILE="${LOG_DIR}/${TIMESTAMP}.log"

# 创建输出目录
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${LOG_DIR}"

echo "========================================" | tee -a "${LOG_FILE}"
echo "5.1.6 单机双卡训练测试" | tee -a "${LOG_FILE}"
echo "测试ID: MTT-TRAIN-RUN-TEST-DUAL-GPU" | tee -a "${LOG_FILE}"
echo "时间: ${TIMESTAMP}" | tee -a "${LOG_FILE}"
echo "========================================" | tee -a "${LOG_FILE}"

# 步骤A：环境检查
echo "" | tee -a "${LOG_FILE}"
echo "[步骤A] 环境配置检查..." | tee -a "${LOG_FILE}"
echo "========================================" | tee -a "${LOG_FILE}"

# 检查GPU数量
echo "GPU信息:" | tee -a "${LOG_FILE}"
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    echo "GPU数量 (NVIDIA): ${GPU_COUNT}" | tee -a "${LOG_FILE}"
    nvidia-smi | tee -a "${LOG_FILE}"
elif command -v musa-version &> /dev/null; then
    echo "GPU信息 (MUSA):" | tee -a "${LOG_FILE}"
    musa-version | tee -a "${LOG_FILE}"
    # 对于MUSA，获取设备数
    GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")
    echo "MUSA设备数: ${GPU_COUNT}" | tee -a "${LOG_FILE}"
fi

if [ "${GPU_COUNT}" -lt 2 ]; then
    echo "✗ 错误: 需要至少2个GPU，当前检测到 ${GPU_COUNT} 个" | tee -a "${LOG_FILE}"
    exit 1
fi

echo "✓ 检测到足够的GPU资源" | tee -a "${LOG_FILE}"

# 检查PyTorch和分布式支持
python -c "import torch; print(f'PyTorch版本: {torch.__version__}')" | tee -a "${LOG_FILE}"
python -c "import torch.distributed; print('✓ 分布式训练支持: 可用')" | tee -a "${LOG_FILE}"

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

# 步骤C：启动单机双卡训练任务
echo "" | tee -a "${LOG_FILE}"
echo "[步骤C] 启动单机双卡训练任务..." | tee -a "${LOG_FILE}"
echo "========================================" | tee -a "${LOG_FILE}"

TRAIN_START_TIME=$(date +%s)
echo "训练开始时间: $(date)" | tee -a "${LOG_FILE}"

# 使用分布式启动
export CUDA_VISIBLE_DEVICES=0,1
export NCCL_DEBUG=INFO

python -m torch.distributed.launch \
    --nproc_per_node=2 \
    train_runner.py \
    --model_path "${MODEL_PATH}" \
    --config_file "${CONFIG_FILE}" \
    --output_dir "${OUTPUT_DIR}" \
    --num_gpus 2 \
    --task_type "full_training" \
    --distributed 2>&1 | tee -a "${LOG_FILE}"

TRAIN_END_TIME=$(date +%s)
TRAIN_DURATION=$((TRAIN_END_TIME - TRAIN_START_TIME))
echo "训练结束时间: $(date)" | tee -a "${LOG_FILE}"
echo "总训练时间: ${TRAIN_DURATION} 秒" | tee -a "${LOG_FILE}"

# 步骤D：监控数据和日志分析
echo "" | tee -a "${LOG_FILE}"
echo "[步骤D] 监控数据和日志分析..." | tee -a "${LOG_FILE}"
echo "========================================" | tee -a "${LOG_FILE}"

# 检查显存占用
echo "显存占用情况:" | tee -a "${LOG_FILE}"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader | tee -a "${LOG_FILE}"
fi

# 检查输出目录
if [ -d "${OUTPUT_DIR}/checkpoints" ]; then
    echo "✓ checkpoints目录存在" | tee -a "${LOG_FILE}"
    CHECKPOINT_COUNT=$(find "${OUTPUT_DIR}/checkpoints" -type d -name "checkpoint-*" | wc -l)
    echo "✓ 保存的checkpoint数量: ${CHECKPOINT_COUNT}" | tee -a "${LOG_FILE}"
else
    echo "✗ checkpoints目录不存在" | tee -a "${LOG_FILE}"
fi

# 检查梯度计算和参数更新
if grep -q "gradient" "${LOG_FILE}"; then
    echo "✓ 梯度计算日志存在" | tee -a "${LOG_FILE}"
fi

# 检查卡间通信
if grep -qi "allreduce\|nccl\|mccl" "${LOG_FILE}"; then
    echo "✓ 卡间通信日志存在" | tee -a "${LOG_FILE}"
fi

# 检查同步一致性
SYNC_ERRORS=$(grep -i "mismatch\|inconsistency" "${LOG_FILE}" | wc -l)
if [ ${SYNC_ERRORS} -eq 0 ]; then
    echo "✓ 数据同步: 一致" | tee -a "${LOG_FILE}"
else
    echo "✗ 数据同步: 发现 ${SYNC_ERRORS} 个不一致" | tee -a "${LOG_FILE}"
fi

# 步骤E：验证训练完成和结果
echo "" | tee -a "${LOG_FILE}"
echo "[步骤E] 验证训练结果..." | tee -a "${LOG_FILE}"
echo "========================================" | tee -a "${LOG_FILE}"

# 检查是否存在异常
ERROR_PATTERNS=("RuntimeError" "CUDA Error" "OutOfMemory" "core dumped" "Segmentation fault" "Communication error")
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

# 检查损失函数收敛
if grep -q "final loss" "${LOG_FILE}"; then
    echo "✓ 损失函数收敛数据存在" | tee -a "${LOG_FILE}"
fi

# 步骤F：测试判定
echo "" | tee -a "${LOG_FILE}"
echo "[步骤F] 测试判定..." | tee -a "${LOG_FILE}"
echo "========================================" | tee -a "${LOG_FILE}"

TEST_RESULT="通过"
if [ -d "${OUTPUT_DIR}/checkpoints" ] && [ $ERRORS_FOUND -eq 0 ] && [ ${SYNC_ERRORS} -eq 0 ]; then
    echo "✓ 测试状态: 通过" | tee -a "${LOG_FILE}"
    TEST_RESULT="通过"
else
    echo "✗ 测试状态: 失败" | tee -a "${LOG_FILE}"
    TEST_RESULT="失败"
fi

echo "" | tee -a "${LOG_FILE}"
echo "========================================" | tee -a "${LOG_FILE}"
echo "单机双卡训练测试完成" | tee -a "${LOG_FILE}"
echo "最终判定: ${TEST_RESULT}" | tee -a "${LOG_FILE}"
echo "日志文件: ${LOG_FILE}" | tee -a "${LOG_FILE}"
echo "========================================" | tee -a "${LOG_FILE}"

[ "${TEST_RESULT}" = "通过" ] && exit 0 || exit 1
