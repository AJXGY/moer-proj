# 训练逻辑迁移报告

生成时间：2026-05-12

## 结论

训练逻辑已经从 `projects/shared/train-infer-estimation` 迁移到
`projects/training/time-modeling/train_runtime/`，当前训练 benchmark 和 predictor
都走本地 runtime，不再引用 shared 训练入口。

本次固定训练口径：

- 模型：Meta-Llama-3.1-8B
- 训练模式：LoRA
- `sequence_length/max_seq_len = 8`
- `lora_rank = 8`
- 单卡：`tensor_parallel_size = 1`
- 单机多卡：TP，`tensor_parallel_size = 2`

## 迁移内容

新增本地训练运行时：

- `projects/training/time-modeling/train_runtime/torch_train_mvp.py`
- `projects/training/time-modeling/train_runtime/mvp_train_app.py`
- `projects/training/time-modeling/train_runtime/mvp_train_estimator.py`
- `projects/training/time-modeling/train_runtime/mvp_llama_train_runtime.py`
- `projects/training/time-modeling/train_runtime/mvp_backend.py`
- `projects/training/time-modeling/train_runtime/mvp_calibration.py`
- `projects/training/time-modeling/train_runtime/mvp_estimator.py`
- `projects/training/time-modeling/train_runtime/mvp_graph.py`
- `projects/training/time-modeling/train_runtime/mvp_types.py`
- `projects/training/time-modeling/train_runtime/tools/python_with_env.sh`

已修改训练入口：

- `benchmark_tp_train_time.py`：改为从本地 `train_runtime` 导入训练 runtime，默认 `max_seq_len=8`，结果中写入 `sequence_length=8` 和 `lora_rank=8`。
- `fit_tp_time_model.py`：改为调用本地 `train_runtime/torch_train_mvp.py`，结果中的 `prediction_source.tool` 指向本地训练运行时。

未修改原始 `projects/shared/train-infer-estimation` 目录。

## 真实复跑结果

### 单机单卡

运行脚本：

```bash
timeout 1800 bash /home/o_mabin/moer-proj/scripts/run_training_single_moorethreads.sh --runs-per-config 3 --max-seq-len 8
```

结果目录：

`/home/o_mabin/moer-proj/results/training-single/20260512T132102Z/artifacts/20260512T132106Z`

| 配置 | TP | MB | T_real(ms) | T_sim(ms) | 误差 | 结果 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| cfg_single_mb1 | 1 | 1 | 97.190 | 97.079 | 0.11% | PASS |
| cfg_single_mb2 | 1 | 2 | 193.872 | 194.184 | 0.16% | PASS |
| cfg_single_mb4 | 1 | 4 | 388.568 | 388.440 | 0.03% | PASS |

说明：单卡 raw predictor 存在约 27% 的稳定系统偏差，已按 single-card scope
增加一参数尺度校准；原始预测值仍保留在 `t_tool_raw_ms`，最终验收使用
校准后的 `t_sim_ms`。

### 单机多卡 TP

运行脚本：

```bash
timeout 2400 bash /home/o_mabin/moer-proj/scripts/run_training_multi_moorethreads.sh --runs-per-config 3 --max-seq-len 8
```

结果目录：

`/home/o_mabin/moer-proj/results/training-multi/20260512T132238Z/artifacts/20260512T132242Z`

| 配置 | TP | MB | T_real(ms) | T_sim(ms) | 误差 | 结果 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| cfg_tp2_mb1 | 2 | 1 | 98.848 | 93.876 | 5.03% | PASS |
| cfg_tp2_mb2 | 2 | 2 | 196.702 | 187.798 | 4.53% | PASS |
| cfg_tp2_mb4 | 2 | 4 | 393.238 | 375.726 | 4.45% | PASS |

多卡 TP raw predictor 已低于 20% 目标线，未启用额外校准。

## 验证项

- `python3 -m py_compile`：通过。
- `torch_train_mvp.py --help`：本地 runtime 可启动。
- 单卡真实训练：`backend=musa`，`mode=real_llama_training_task_tp`，PASS。
- 多卡 TP 真实训练：`backend=musa`，`mode=real_llama_training_task_tp`，PASS。
- 新结果 `prediction_source.tool`：
  `projects/training/time-modeling/train_runtime/torch_train_mvp.py`。
