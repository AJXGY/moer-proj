# 5.1.6 摩尔线程架构上训练任务运行测试 - 使用指南

## 项目概述

本项目用于验证摩尔线程GPU在英特尔CPU + 摩尔线程GPU异构计算架构上的训练任务运行能力。

- **测试项名称**：摩尔线程架构上训练任务运行测试
- **测试项标识**：MTT-TRAIN-RUN-TEST
- **目标指标**：1.2 异构计算架构适配

## 项目结构

```
5.1.6/
├── README.md                      # 项目简介
├── README_CN.md                   # 中文使用指南（本文件）
├── 5.1.6任务进展.md              # 任务进展跟踪表
├── train_config.json              # 训练配置文件
├── train_runner.py                # 训练运行器（主脚本）
├── train_single.sh                # 单卡训练脚本
├── train_dual.sh                  # 双卡训练脚本
├── checklist.md                   # 详细测试检查清单
├── test_results_summary.md        # 测试结果汇总模板
├── requirements.txt               # Python依赖列表
├── logs/                          # 训练日志目录（运行时创建）
└── checkpoints/                   # 模型检查点目录（运行时创建）
    ├── single_gpu/                # 单卡训练检查点
    └── dual_gpu/                  # 双卡训练检查点
```

## 前置条件

### 硬件要求

1. **CPU**：英特尔处理器
   - 推荐：Intel Xeon系列
   - 最低：Intel Core i7/i9

2. **GPU**：摩尔线程GPU
   - 数量要求：
     - 单卡测试：≥1张
     - 双卡测试：≥2张
   - 显存要求：≥40GB（单卡），≥80GB（双卡）

3. **内存**：≥256GB物理内存

4. **存储**：≥500GB可用存储空间

5. **网络**：千兆以太网（用于双卡通信）

### 软件要求

1. **操作系统**：Linux (推荐：Ubuntu 18.04/20.04 或 CentOS 8+)

2. **Python环境**：
   ```bash
   Python >= 3.7
   pip >= 20.0
   ```

3. **驱动程序**：
   - 摩尔线程GPU驱动（最新版本）
   - MCCL（摩尔线程集合通信库）

4. **深度学习框架**：
   - PyTorch >= 1.9.0（支持MUSA）
   - 或其他支持摩尔线程的框架

5. **其他依赖**：
   - numpy >= 1.19.0
   - pyyaml >= 5.3
   - tqdm >= 4.50.0

## 快速开始

### 1. 环境准备

#### 安装Python依赖
```bash
pip install -r requirements.txt
```

#### 验证环境
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'GPU available: {torch.cuda.is_available()}')"
```

### 2. 配置修改

编辑 `train_config.json`，根据实际情况调整以下参数：

```json
{
  "model_config": {
    "model_path": "/path/to/Model-8B"  // 修改为实际模型路径
  },
  "data_config": {
    "batch_size": 32                    // 根据显存调整
  },
  "hardware_config": {
    "device_type": "musa"               // MUSA或cuda
  }
}
```

### 3. 单卡测试

#### 方式一：直接运行Python脚本
```bash
python train_runner.py \
    --model_path /path/to/Model-8B \
    --config_file train_config.json \
    --output_dir ./checkpoints \
    --num_gpus 1 \
    --task_type full_training
```

#### 方式二：运行Bash脚本（推荐）
```bash
bash train_single.sh
```

**预期输出**：
- 训练日志：`logs/single_gpu/YYYYMMDDTHHMMSSZ.log`
- 模型检查点：`checkpoints/single_gpu/checkpoints/`
- 训练曲线数据

### 4. 双卡测试

```bash
bash train_dual.sh
```

**关键参数自动配置**：
- 设置 `CUDA_VISIBLE_DEVICES=0,1`
- 使用 `torch.distributed.launch` 启动
- 自动启用NCCL/MCCL通信库

**预期输出**：
- 训练日志（每个进程一个）：`logs/dual_gpu/YYYYMMDDTHHMMSSZ.log`
- 模型检查点：`checkpoints/dual_gpu/checkpoints/`
- 通信性能指标

## 详细使用流程

### 第一阶段：环境搭建

```bash
# 1. 检查硬件
nvidia-smi  # 或 musa-version

# 2. 安装环境
pip install -r requirements.txt

# 3. 验证GPU识别
python -c "import torch; print(torch.cuda.device_count())"

# 4. 检查分布式支持
python -c "import torch.distributed; print('OK')"
```

**验证清单**：
- ✓ GPU驱动正常
- ✓ PyTorch可用
- ✓ CUDA/MUSA支持
- ✓ 分布式库可用

### 第二阶段：配置准备

```bash
# 1. 检查模型文件
ls -lh /path/to/Model-8B/

# 2. 验证模型完整性
python -c "import torch; torch.load('/path/to/Model-8B/model.pt')"

# 3. 验证数据集
ls -lh data/

# 4. 修改配置
vim train_config.json
```

### 第三阶段：单卡测试

```bash
# 1. 执行训练
bash train_single.sh

# 2. 实时监控日志
tail -f logs/single_gpu/*.log

# 3. 监控GPU使用
watch -n 1 nvidia-smi

# 4. 等待完成后检查结果
ls -lh checkpoints/single_gpu/checkpoints/
```

**预期现象**：
- GPU显存逐渐占用
- Loss值逐步下降
- 无错误或警告信息（正常错误除外）

### 第四阶段：双卡测试

```bash
# 1. 执行训练
bash train_dual.sh

# 2. 监控所有进程
ps aux | grep train_runner

# 3. 监控GPU通信
nvidia-smi dmon  # 或对应的MUSA命令

# 4. 查看详细日志
tail -f logs/dual_gpu/*.log
```

**预期现象**：
- 两个GPU几乎同时启动
- 梯度同步正常
- Loss曲线与单卡基本一致
- 吞吐量约为单卡的2倍（考虑通信开销）

### 第五阶段：结果分析

```bash
# 1. 收集日志
cp logs/single_gpu/*.log results/
cp logs/dual_gpu/*.log results/

# 2. 提取性能指标
grep "training time" results/*.log

# 3. 对比单卡/双卡
diff <(grep "Loss" results/single_*.log) \
     <(grep "Loss" results/dual_*.log)

# 4. 填写测试报告
vim test_results_summary.md
```

### 第六阶段：问题诊断

如遇到问题，按以下步骤诊断：

#### 显存不足
```bash
# 减小batch_size
sed -i 's/"batch_size": 32/"batch_size": 16/g' train_config.json

# 启用梯度检查点
sed -i 's/"use_gradient_checkpointing": false/"use_gradient_checkpointing": true/g' train_config.json
```

#### 通信错误（双卡）
```bash
# 检查MCCL/NCCL环境
python -c "import torch.distributed; print(torch.version.cuda)"

# 启用通信调试
export NCCL_DEBUG=INFO
bash train_dual.sh
```

#### 模型加载失败
```bash
# 验证模型格式
file /path/to/Model-8B/model.safetensors

# 检查模型完整性
python -c "import safetensors.torch; safetensors.torch.load_file(...)"
```

## 检查清单

### 测试前检查清单

- [ ] GPU驱动已安装
- [ ] PyTorch正确安装
- [ ] 分布式库可用
- [ ] 模型文件完整
- [ ] 数据集准备就绪
- [ ] 配置文件正确
- [ ] 存储空间充足（>500GB）

### 测试中检查清单

- [ ] 硬件识别正确
- [ ] GPU显存占用正常
- [ ] Loss逐步下降
- [ ] 无异常错误信息
- [ ] 双卡梯度同步正常
- [ ] 通信延迟可接受

### 测试后检查清单

- [ ] checkpoint已保存
- [ ] 日志文件完整
- [ ] Loss收敛
- [ ] 模型结构完整
- [ ] 结果填入报告表格
- [ ] 异常现象已记录

## 性能基准

### 预期性能指标

| 配置 | 吞吐量 | 显存 | 训练时间 |
|------|-------|------|--------|
| 单卡 | ~50 samples/s | ~35GB | ~6小时 |
| 双卡 | ~90 samples/s | ~70GB | ~3.5小时 |

（实际数据需根据硬件配置和批次大小调整）

## 故障排除

### 常见问题

#### Q: 运行时报"CUDA out of memory"
**A**: 
1. 减小batch_size
2. 启用gradient_checkpointing
3. 检查是否有其他进程占用GPU

#### Q: 双卡无法通信
**A**:
1. 检查MCCL库是否正确安装
2. 运行 `ncclTester` 进行通信测试
3. 检查GPU连接方式（PCIe/NVLink）

#### Q: 训练崩溃无日志
**A**:
1. 启用coredump：`ulimit -c unlimited`
2. 启用调试模式：`CUDA_LAUNCH_BLOCKING=1`
3. 检查系统日志：`dmesg | tail`

### 联系支持

遇到无法解决的问题，请提供：
1. 完整的训练日志
2. GPU信息（nvidia-smi输出）
3. 系统信息（lsb_release -a）
4. 错误信息的完整堆栈跟踪

## 扩展功能

### LoRA微调

编辑 `train_config.json` 启用LoRA：
```json
{
  "training_config": {
    "task_type": "lora_training"
  },
  "lora_config": {
    "lora_rank": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.1
  }
}
```

然后运行：
```bash
bash train_single.sh  # 或 train_dual.sh
```

### 使用不同优化器

在 `train_config.json` 中修改：
```json
{
  "optimization_strategy": {
    "optimizer": "AdamW",
    "learning_rate": 1e-4,
    "weight_decay": 0.01
  }
}
```

## 参考资源

- [摩尔线程官方文档](https://mttt.com)
- [PyTorch分布式训练](https://pytorch.org/docs/stable/distributed.html)
- [LLaMA模型文档](https://huggingface.co/meta-llama)

## 许可证

本项目遵循项目主体的许可证规定。

## 贡献

欢迎提出改进建议和bug报告。

---

**最后更新**：2026-04-15
