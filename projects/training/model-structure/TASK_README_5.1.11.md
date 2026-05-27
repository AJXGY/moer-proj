# 5.1.11 摩尔线程架构上训练任务处理模型输出测试

本目录用于完成 `MTT-TRAIN-MODEL-STRUCT-TEST`。

工程能力包括：

- 读取 Llama3.1-8B 训练任务配置与资源映射
- 构建训练任务处理模型 JSON
- 对生成的执行模型做结构一致性核验并输出 `validation_report.json`
- 输出 CPU/GPU 任务分配、并行方式、microbatch 执行逻辑
- 生成 DAG 图、资源拓扑图、阶段划分图和完成情况图表
- 自动汇总 A-H 指标，输出 `5.1.11任务进展.md`

口径说明：

- 本目录验证的是“训练任务处理模型输出”，不是“真实训练任务运行”。
- `resource_mapping.json` 中的摩尔线程 GPU 只用于建模输入；本目录本身不校验 `torch_musa` 导入、显卡可见性或真实训练执行。
- 当前配置中 `tensor_parallel_size=1`，表示张量并行未启用，实际展示重点是数据并行 + 流水线并行。
- 当前目录额外补了 1 次基于 `projects/shared/train-infer-estimation/mvp_llama_train_runtime.py` 的真实双卡训练观测，用来辅助核对结构模型没有明显偏离真实训练链路。
- 如需补充查看 `TP=2` 的建模表达能力，可执行 `python3 build_tp_supplement.py`，产物会写入 `tp_supplement/`。

## 快速开始

```bash
cd /home/o_mabin/moerxiancheng-clj-xyj-proj/projects/training/model-structure
python3 build_training_model.py
python3 generate_charts.py
python3 summarize_results.py
```

也可以一键执行：

```bash
cd /home/o_mabin/moerxiancheng-clj-xyj-proj/projects/training/model-structure
bash run_5111_suite.sh
```

## 主要文件

- `training_task_config.json`：训练任务描述
- `resource_mapping.json`：CPU + 摩尔线程 GPU 资源映射
- `build_training_model.py`：训练处理模型构建器
- `verify_training_model.py`：按任务配置与资源映射核验执行模型结构
- `generate_charts.py`：生成 SVG 图表
- `summarize_results.py`：生成 `5.1.11任务进展.md`
- `build_tp_supplement.py`：额外生成 `TP=2` 建模补充产物
- `charts/`：可视化图表输出目录
- `artifacts/`：最终产物目录
- `tp_supplement/`：张量并行补充建模目录
