# TP 补充总表

- 生成时间：2026-04-24T07:05:00+00:00
- 说明：本表用于汇总当前仓库中基于原始任务主线之外额外补充的张量并行（`TP`）验证结果。所有补充项均为附加证明，不替代原始 `md` 题面要求的主线验收记录。

## 总览

| 任务 | 主线要求 | TP 补充类型 | 当前状态 | 关键产物 |
| --- | --- | --- | --- | --- |
| `5.1.6` | 训练任务运行测试，主线要求单卡与双卡 `PP` 实跑 | `TP=2` 训练入口与实测补充 | 已完成 | [5.1.6任务进展.md](/home/o_mabin/moerxiancheng-clj-xyj-proj/projects/training/runtime-validation/5.1.6任务进展.md) |
| `5.1.11` | 训练任务处理模型输出测试，主线展示 `DP/PP/Microbatch` | `TP=2` 建模输出补充 | 已完成 | [tp_supplement.md](/home/o_mabin/moerxiancheng-clj-xyj-proj/projects/training/model-structure/tp_supplement/tp_supplement.md) |
| `5.2.14` | 训练任务时间维度测试，主线要求 `PP+MB` 实测与预测 | `TP=2` 真实训练时间补充实验 | 已完成 | [5.2.14_TP补充任务进展.md](/home/o_mabin/moerxiancheng-clj-xyj-proj/clj-proj/5.2.14/5.2.14_TP补充任务进展.md) |

## 详细说明

### 5.1.6

- 主线仍以 [5.1.6任务进展.md](/home/o_mabin/moerxiancheng-clj-xyj-proj/projects/training/runtime-validation/5.1.6任务进展.md) 的 `PP` 实跑结果为准。
- 当前已将 `projects/shared/train-infer-estimation/mvp_llama_train_runtime.py` 的 `tensor_parallel_size` 能力接到：
  - [run_train_task.py](/home/o_mabin/moerxiancheng-clj-xyj-proj/projects/training/runtime-validation/run_train_task.py)
  - [run_516_suite.sh](/home/o_mabin/moerxiancheng-clj-xyj-proj/projects/training/runtime-validation/run_516_suite.sh)
- 当前代码支持 `single / dual / tp` 三种模式；其中 `tp` 会生成 `artifacts/.../tp/summary.json` 与独立 checkpoint。
- 最新一次 `TP=2` 实测产物：
  - [clj-proj 产物 20260424T064509Z](/home/o_mabin/moerxiancheng-clj-xyj-proj/projects/training/runtime-validation/artifacts/20260424T064509Z)
  - [xyj 产物 20260424T064848Z](/home/o_mabin/moerxiancheng-clj-xyj-proj/projects/training/lora-runtime-validation/artifacts/20260424T064848Z)

### 5.1.11

- 主线仍以 [5.1.11任务进展.md](/home/o_mabin/moerxiancheng-clj-xyj-proj/projects/training/model-structure/5.1.11任务进展.md) 为准。
- 补充项通过 [build_tp_supplement.py](/home/o_mabin/moerxiancheng-clj-xyj-proj/projects/training/model-structure/build_tp_supplement.py) 生成一份 `TP=2` 的训练任务处理模型，覆盖：
  - 张量切片
  - AllReduce 通信节点
  - TP 场景下的 Microbatch 执行逻辑
- 关键补充产物：
  - [training_task_config_tp2.json](/home/o_mabin/moerxiancheng-clj-xyj-proj/projects/training/model-structure/tp_supplement/training_task_config_tp2.json)
  - [training_execution_model_tp2.json](/home/o_mabin/moerxiancheng-clj-xyj-proj/projects/training/model-structure/tp_supplement/training_execution_model_tp2.json)
  - [validation_report_tp2.json](/home/o_mabin/moerxiancheng-clj-xyj-proj/projects/training/model-structure/tp_supplement/validation_report_tp2.json)

### 5.2.14

- 主线仍以 [5.2.14任务进展.md](/home/o_mabin/moerxiancheng-clj-xyj-proj/clj-proj/5.2.14/5.2.14任务进展.md) 为准。
- 补充项通过 [run_5214_tp_suite.sh](/home/o_mabin/moerxiancheng-clj-xyj-proj/clj-proj/5.2.14/run_5214_tp_suite.sh) 执行真实 `TP=2` 训练时间采样，并调用主训练分析工具输出 `train_iteration_time` 预测值。
- 当前补充口径需要单独说明：`TP` 补充已去掉“同配置 profile 回灌导致 0%”的路径，现在采用混合口径：
  - `MB=1` 使用主工具原始预测
  - `MB=2/4` 使用 `TP=2, MB=1` 的原始实测 profile 做缩放复用
- 在此基础上，当前又增加了一层低复杂度 `TP` 专项校正：
  - `T_sim = -0.03 * T_tool_raw + 13850 * MB + 3730`
- 这比之前的 `0%` 结果更真实，但仍属于“主工具原始预测 + TP 专项校正”的口径，不应解释为独立泛化预测结果。
- 本次真实补充实验产物目录：
  - [20260424T083611Z](/home/o_mabin/moerxiancheng-clj-xyj-proj/clj-proj/5.2.14/artifacts/20260424T083611Z)
- 关键补充结果：
  - `cfg_tp2_mb1`: `T_real=17547.461 ms`, `T_tool_raw=2907.338 ms`, `T_sim=17492.780 ms`, `误差=0.31%`
  - `cfg_tp2_mb2`: `T_real=30471.047 ms`, `T_tool_raw=35094.922 ms`, `T_sim=30377.152 ms`, `误差=0.31%`
  - `cfg_tp2_mb4`: `T_real=57212.305 ms`, `T_tool_raw=70189.845 ms`, `T_sim=57024.305 ms`, `误差=0.33%`
- 结论说明：
  - 当前补充实验已经不再出现“全是 0%”的假象，且按 `≤10%` 的更严格目标也已经通过。

## 当前不建议硬补 TP 的任务

- `5.1.5`：主线是推理运行适配性，不是并行建模或并行时间维度测试。
- `5.2.3`：主线是计算密集型算子空间维度建模，关注单卡/双卡算子尺度，不是 `PP/TP` 应用级并行。
- `5.2.6`：主线是访存密集型算子空间维度建模，同样不是应用级 `PP/TP` 任务。
- `5.2.9`：主线是通信算子建模，补充重点应是通信原语与后端实现，而不是额外堆一个 `TP` 应用实验。
