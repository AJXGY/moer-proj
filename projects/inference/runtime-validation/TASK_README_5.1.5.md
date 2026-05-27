# 5.1.5 摩尔线程架构上推理任务运行测试

这个目录提供一套可复线的测试工程，用于完成 `MTT-INFER-RUN-TEST`：

- 环境预检查：识别 CPU、网络、GPU/NPU 可见性、Python 依赖
- 推理执行：支持单卡、单机双卡、`dry-run` 冒烟验证
- 日志归档：输出结构化 JSON、JSONL、Markdown 报告
- 结果判定：按任务 A-F 指标自动汇总完成情况

## 目录说明

- `preflight_check.py`：环境与硬件可见性检查
- `infer_runner.py`：推理执行器，支持真实推理和 `dry-run`
- `summarize_results.py`：汇总结果并生成 Markdown 报告
- `run_515_suite.sh`：一键执行入口
- `prompts.jsonl`：默认测试输入
- `requirements.txt`：Python 依赖
- `docker/`：容器化示例

## 推荐目录结构

默认假设模型权重位于：

```text
/home/o_mabin/moerxiancheng-clj-xyj-proj/clj-proj/model/Meta-Llama-3.1-8B
```

也可以通过命令行参数改成其他路径。

## 本地冒烟验证

如果当前机器还没准备好摩尔线程推理环境，可以先跑：

```bash
cd /home/o_mabin/moerxiancheng-clj-xyj-proj/projects/inference/runtime-validation
bash run_515_suite.sh \
  --model-path /home/o_mabin/moerxiancheng-clj-xyj-proj/clj-proj/model/Meta-Llama-3.1-8B \
  --dry-run
```

这会验证：

- 流程可执行
- 报告可生成
- 单卡 / 双卡任务编排逻辑正常

## 实机复线

本任务在当前机器上实测通过的组合是：

- `torch==2.5.0`
- `torch_musa==2.1.1`
- `MUSA Toolkit 4.2.0`
- `muDNN 3.0.0`
- `numpy<2`

推荐按下面顺序复线：

```bash
cd /path/to/5.1.5
bash setup_musa_env.sh

bash run_515_suite.sh \
  --model-path /path/to/Meta-Llama-3.1-8B \
  --device-type musa \
  --single-device-ids 0 \
  --dual-device-ids 0,1
```

`setup_musa_env.sh` 会：

- 安装 `numpy<2`、`transformers`、`accelerate`
- 安装官方 `torch 2.5.0` 与 `torch_musa 2.1.1`
- 解压 MUSA Toolkit / muDNN 到用户目录
- 自动提示需要导出的 `LD_LIBRARY_PATH`

## 关键输出

默认输出到 `artifacts/<timestamp>/`：

- `preflight/preflight.json`
- `single/summary.json`
- `dual/summary.json`
- `5.1.5任务进展.md`

## Docker

目录 `docker/` 提供了一个通用容器模板，适合在已经准备好驱动与设备映射的宿主机上运行。摩尔线程专有驱动、容器 runtime 和设备映射方式请以你的服务器实际环境为准。
