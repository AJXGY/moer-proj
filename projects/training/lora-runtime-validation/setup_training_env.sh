#!/usr/bin/env bash
# 5.1.6 训练环境配置脚本
# MTT-TRAIN-RUN-TEST Setup MUSA Environment

set -euo pipefail

echo "=========================================="
echo "5.1.6 训练环境配置脚本"
echo "=========================================="
echo ""

# 检测MUSA/CUDA环境
echo "[检查] 加速器环境..."

if command -v nvidia-smi &> /dev/null; then
    echo "✓ 检测到 NVIDIA CUDA"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
    echo ""
elif command -v musa-version &> /dev/null; then
    echo "✓ 检测到 MUSA"
    musa-version
    echo ""
else
    echo "⚠ 未检测到 GPU 加速器"
fi

# 检查Python环境
echo "[检查] Python环境..."

PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python 版本：${PYTHON_VERSION}"

if python3 -c "import torch" 2>/dev/null; then
    TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)")
    echo "✓ PyTorch 版本：${TORCH_VERSION}"
else
    echo "✗ PyTorch 未安装"
    exit 1
fi

if python3 -c "import torch_musa" 2>/dev/null; then
    echo "✓ torch-musa 已安装"
elif python3 -c "import torch.cuda" 2>/dev/null; then
    echo "✓ CUDA 支持已启用"
else
    echo "⚠ 无加速器支持"
fi

echo ""

# 配置必要的库路径
echo "[配置] 库路径..."

EXTRA_LD_PATHS=()

# MUSA 相关路径
for candidate in \
  "/home/o_mabin/.local/musa_toolkits/musa_toolkits_4.2.0/lib" \
  "/home/o_mabin/.local/mudnn/mudnn/lib" \
  "/usr/local/musa/lib" \
  "/usr/lib/musa"
do
  if [[ -d "${candidate}" ]]; then
    EXTRA_LD_PATHS+=("${candidate}")
    echo "  + Found: ${candidate}"
  fi
done

# OpenBLAS 路径
for candidate in \
  "/home/o_mabin/.local/openblas/usr/lib/x86_64-linux-gnu/openblas-pthread" \
  "/usr/lib/x86_64-linux-gnu/atlas" \
  "/usr/lib/openblas"
do
  if [[ -d "${candidate}" ]]; then
    EXTRA_LD_PATHS+=("${candidate}")
    echo "  + Found: ${candidate}"
  fi
done

if [[ ${#EXTRA_LD_PATHS[@]} -gt 0 ]]; then
  EXTRA_JOINED="$(IFS=:; echo "${EXTRA_LD_PATHS[*]}")"
  export LD_LIBRARY_PATH="${EXTRA_JOINED}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  echo "✓ LD_LIBRARY_PATH 已更新"
fi

echo ""

# 检查分布式训练支持
echo "[检查] 分布式训练支持..."

if python3 -c "import torch.distributed" 2>/dev/null; then
    echo "✓ torch.distributed 可用"
else
    echo "✗ torch.distributed 不可用"
    exit 1
fi

if python3 -c "import torch_musa" 2>/dev/null; then
    if python3 -c "import torch_musa.distributed" 2>/dev/null; then
        echo "✓ torch-MUSA 分布式支持可用"
    fi
elif python3 -c "import torch.cuda" 2>/dev/null; then
    echo "✓ CUDA 分布式支持可用"
fi

echo ""

# 检查依赖库
echo "[检查] Python依赖..."

REQUIRED_PACKAGES=(
    "torch"
    "transformers"
    "numpy"
    "pyyaml"
    "tqdm"
)

MISSING=()
for package in "${REQUIRED_PACKAGES[@]}"; do
    if python3 -c "import ${package}" 2>/dev/null; then
        VERSION=$(python3 -c "import ${package}; print(getattr(${package}, '__version__', 'unknown'))" 2>/dev/null || echo "unknown")
        echo "  ✓ ${package} (${VERSION})"
    else
        echo "  ✗ ${package}"
        MISSING+=("${package}")
    fi
done

if [[ ${#MISSING[@]} -gt 0 ]]; then
    echo ""
    echo "缺失的依赖：${MISSING[@]}"
    echo "请运行：pip install -r requirements.txt"
    exit 1
fi

echo ""

# 显示环境总结
echo "=========================================="
echo "环境配置完成！"
echo "=========================================="
echo ""
echo "环境变量："
echo "  LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-未设置}"
echo ""
echo "检查无误后，可以执行："
echo "  bash run_516_suite.sh [选项]"
echo ""
echo "可用选项："
echo "  --dry-run              执行干运行（冒烟测试）"
echo "  --model-path PATH      指定模型路径"
echo "  --single-device-ids IDS 指定单卡GPU ID"
echo "  --dual-device-ids IDS  指定双卡GPU ID"
echo ""
