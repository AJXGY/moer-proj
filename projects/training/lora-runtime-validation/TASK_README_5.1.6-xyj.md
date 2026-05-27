# 5.1.6 摩尔线程架构上 LoRA 训练任务运行测试

这个目录提供一套完整可复现的测试工程，用于完成 `MTT-TRAIN-RUN-TEST`：

- 环境预检查：识别 CPU、网络、GPU/NPU 可见性、Python 依赖
- 训练执行：支持单卡、单机双卡、`dry-run` 冒烟验证，默认执行 LoRA 风格低秩适配器训练
- 日志归档：输出结构化 JSON、JSONL、Markdown 报告
- 结果判定：按任务 A-F 指标自动汇总完成情况

## 目录说明

- `preflight_check.py` - 环境与硬件可见性检查
- `train_runner.py` - 训练执行器，支持真实训练、`dry-run`、双卡 `PP` / `TP`
- `train_summarize.py` - 汇总结果并生成 Markdown 报告
- `generate_training_charts.py` - 性能分析与可视化
- `run_516_suite.sh` - 一键执行入口（推荐使用）
- `setup_training_env.sh` - 环境配置脚本
- `train_config.json` - 训练配置文件
- `train_data.jsonl` - 示例训练数据
- `requirements.txt` - Python 依赖
- `docker/` - 容器化示例
- `checklist.md` - 详细测试检查清单

## 推荐目录结构

默认假设模型权重位于：

```text
/home/o_mabin/moerxiancheng-clj-xyj-proj/clj-proj/model/Meta-Llama-3.1-8B
```

也可以通过命令行参数改成其他路径。

## 快速开始

### 前置条件
```bash
bash setup_training_env.sh
pip install -r requirements.txt
```

### 冒烟验证（干运行）
```bash
bash run_516_suite.sh \
  --model-path /home/o_mabin/moerxiancheng-clj-xyj-proj/clj-proj/model/Meta-Llama-3.1-8B \
  --dry-run
```

这会验证：

- 流程可执行
- 报告可生成
- 单卡 / 双卡 `PP` / 双卡 `TP` 任务编排逻辑正常

### 实机 LoRA 训练

本任务在摩尔线程实机环境上的完整执行：

```bash
bash run_516_suite.sh \
  --model-path /home/o_mabin/moerxiancheng-clj-xyj-proj/clj-proj/model/Meta-Llama-3.1-8B
```

默认使用 `lora_training`，冻结 8B backbone，仅更新低秩适配器参数，避免全量微调导致运行时间过长或显存压力过大。

## 预期输出

执行后生成的输出文件：
- `artifacts/TIMESTAMP/preflight/preflight.json` - 环境检查结果
- `artifacts/TIMESTAMP/single/summary.json` - 单卡训练结果
- `artifacts/TIMESTAMP/dual/summary.json` - 双卡训练结果
- `artifacts/TIMESTAMP/5.1.6任务完成总结.md` - 最终报告
- 训练日志及性能指标
- 损失函数收敛曲线
- 模型检查点(checkpoints)
