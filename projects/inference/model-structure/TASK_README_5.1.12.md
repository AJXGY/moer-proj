# 5.1.12 摩尔线程架构上人工智能推理任务处理模型输出测试

本目录用于完成 `MTT-INFER-MODEL-STRUCT-TEST`。

工程能力包括：

- 读取 Llama3.1-8B 推理任务配置与资源映射
- 构建推理任务处理模型 JSON
- 输出 CPU/GPU 任务分配、并行方式、连续 Batch/Microbatch 执行逻辑
- 生成 DAG 图、资源拓扑图、阶段划分图和完成情况图表
- 自动汇总 A-H 指标，输出 `5.1.12任务进展.md`

## 快速开始

```bash
cd /home/o_mabin/moerxiancheng-clj-xyj-proj/projects/inference/model-structure
python3 build_inference_model.py
python3 generate_charts.py
python3 summarize_results.py
```

也可以一键执行：

```bash
cd /home/o_mabin/moerxiancheng-clj-xyj-proj/projects/inference/model-structure
bash run_512_suite.sh
```

## 一键实测（推荐）

如果需要把真实推理执行结果一并回填到 5.1.12 目录，可直接运行：

```bash
cd /home/o_mabin/moerxiancheng-clj-xyj-proj/projects/inference/model-structure
bash run_512_real_suite.sh \
	--model-path /home/o_mabin/moerxiancheng-clj-xyj-proj/clj-proj/model/Meta-Llama-3.1-8B \
	--device-type musa \
	--single-device-ids 0 \
	--dual-device-ids 0,1
```

可选参数：

- `--dry-run`：仅冒烟验证流程，不加载真实模型
- `--skip-model-build`：跳过 `run_512_suite.sh`，只做实测采集与归档

运行结束后将输出：

- `artifacts/<timestamp>/`：完整证据归档（建模结果 + 5.1.5 实测结果）
- `artifacts/5.1.12实测结果.md`：实测结论与证据索引

## 主要文件

- `inference_task_config.json`：推理任务描述
- `resource_mapping.json`：CPU + 摩尔线程 GPU 资源映射
- `build_inference_model.py`：推理处理模型构建器
- `generate_charts.py`：生成 SVG 图表
- `summarize_results.py`：生成 `5.1.12任务进展.md`
- `run_512_real_suite.sh`：一键实测并汇总到 5.1.12
- `charts/`：可视化图表输出目录
- `artifacts/`：最终产物目录
