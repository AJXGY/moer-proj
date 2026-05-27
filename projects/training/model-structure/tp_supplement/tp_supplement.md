# 5.1.11 TP 补充建模说明

- 生成时间：2026-04-21T14:11:17.548490+00:00
- 说明：这是在不改变 5.1.11 主线 `PP` 测试结论的前提下，额外补充的一份 `TP=2` 训练任务处理模型样例。

## 补充结论

该补充产物用于证明当前任务处理模型不仅能表达流水线阶段划分，也能表达张量切片、跨卡 AllReduce 通信和 TP 场景下的 Microbatch 调度逻辑。它属于建模输出补充，不替代 5.1.11 原始题面主记录。

## 关键摘要

- 模型：Meta-Llama-3.1-8B
- 数据并行：2
- 流水线并行：1
- 张量并行：2
- TP 通信节点：allreduce_hidden_states, allreduce_gradients
- 结构核验：8/8 项通过

## 关键文件

- 任务配置：[training_task_config_tp2.json](/home/o_mabin/moerxiancheng-clj-xyj-proj/projects/training/model-structure/tp_supplement/training_task_config_tp2.json)
- 执行模型：[training_execution_model_tp2.json](/home/o_mabin/moerxiancheng-clj-xyj-proj/projects/training/model-structure/tp_supplement/training_execution_model_tp2.json)
- 核验报告：[validation_report_tp2.json](/home/o_mabin/moerxiancheng-clj-xyj-proj/projects/training/model-structure/tp_supplement/validation_report_tp2.json)
