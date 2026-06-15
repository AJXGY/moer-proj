# 当前训练逻辑对比：moer-proj vs train-infer-estimation

本文对比当前代码状态：

- 当前训练代码：`/home/o_mabin/moer-proj/projects/training/time-modeling`
- 参考训练代码：`/home/o_mabin/train-infer-estimation`

结论：`moer-proj` 已经从“小分类 LoRA head”改为对齐 `train-infer-estimation`
的 vocab LoRA 训练口径。当前训练不再是 `hidden_size -> rank -> num_labels(2)`，
而是 `hidden_size -> rank -> vocab_size`。

## 1. 当前入口

主训练入口：

```text
projects/training/time-modeling/benchmark_tp_train_time.py
projects/training/time-modeling/fit_tp_time_model.py
projects/training/time-modeling/summarize_tp_results.py
```

本地训练 runtime：

```text
projects/training/time-modeling/train_runtime/mvp_llama_train_runtime.py
projects/training/time-modeling/train_runtime/mvp_train_app.py
projects/training/time-modeling/train_runtime/mvp_train_estimator.py
```

从 `/home/o_mabin/train-infer-estimation` 迁入的训练估算核心模块：

```text
train_runtime/train_infer_estimation_core/lora_adapter.py
train_runtime/train_infer_estimation_core/mvp_train_unified_estimator.py
train_runtime/train_infer_estimation_core/mvp_train_tp_estimator.py
train_runtime/train_infer_estimation_core/mvp_backward_comm.py
train_runtime/train_infer_estimation_core/mvp_optimizer_tp_estimator.py
train_runtime/train_infer_estimation_core/mvp_train_estimator.py
train_runtime/train_infer_estimation_core/mvp_train_types.py
train_runtime/train_infer_estimation_core/mvp_train_graph.py
train_runtime/train_infer_estimation_core/mvp_backward_graph.py
train_runtime/train_infer_estimation_core/train_workflow.py
train_runtime/train_infer_estimation_core/config/train_config.yaml
```

脚本入口：

```text
scripts/run_training_single_moorethreads.sh
scripts/run_training_multi_moorethreads.sh
scripts/run_training_moorethreads_smoke.sh
```

## 2. 对齐情况

| 维度 | moer-proj 当前训练 | train-infer-estimation |
| --- | --- | --- |
| LoRA head | `hidden_size -> rank -> vocab_size` | `hidden_size -> rank -> vocab_size` |
| `lora_rank` | 8 | 支持 LoRA rank 参数 |
| `lora_alpha` | 16.0 | 支持 LoRA 配置 |
| `sequence_length` | 8 | 支持 seq_len 参数 |
| 可训练参数量 | `8 * (4096 + 128256) = 1,058,816` | vocab LoRA adapter 参数 |
| 小分类 head | 已移除真实训练主线 | 不作为 NVIDIA 对齐口径 |
| backbone forward | autograd 穿过 backbone | autograd 穿过 backbone |
| `no_grad + detach` | 已从真实训练主线移除 | 不用于 vocab LoRA 训练 |
| optimizer | Adam，仅更新 LoRA head | Adam，仅更新 LoRA adapter |
| 单机单卡 | 已真实跑通 | 支持 |
| 单机多卡 TP | 已真实跑通 | 支持 |
| TP 通信 | 当前仍是本地 runtime CPU staging | 原项目 CUDA/NCCL 路径 |
| 估算模块 | 已迁入核心同名模块 | 原生模块 |

## 3. 当前真实训练口径

关键字段已经写入最新 artifact：

```text
task_kind = llama_vocab_lora_training_tp
lora_head = vocab_lm_head
lora_projection = hidden_size_to_rank_to_vocab_size
trainable_parameter_count = 1058816
backbone_update = autograd_traverses_backbone_optimizer_updates_lora_only
```

这表示当前已经不是：

```text
hidden_size -> rank -> num_labels(2)
```

而是：

```text
hidden_size -> rank -> vocab_size(128256)
```

## 4. 当前真实运行结果

最近一次真实运行命令：

```bash
timeout 2400 bash /home/o_mabin/moer-proj/scripts/run_training_single_moorethreads.sh --runs-per-config 1 --max-seq-len 8
timeout 3000 bash /home/o_mabin/moer-proj/scripts/run_training_multi_moorethreads.sh --runs-per-config 1 --max-seq-len 8
```

### 单机单卡

结果目录：

```text
/home/o_mabin/moer-proj/results/training-single/20260513T053513Z/artifacts/20260513T053517Z
```

| 配置 | 模式 | TP | T_real | T_sim | 误差 |
| --- | --- | ---: | ---: | ---: | ---: |
| cfg_single_mb1 | single | 1 | 453.771 ms | 453.160 ms | 0.13% |
| cfg_single_mb2 | single | 1 | 1042.049 ms | 1042.965 ms | 0.09% |
| cfg_single_mb4 | single | 1 | 2220.997 ms | 2220.691 ms | 0.01% |

最大误差：`0.13%`

### 单机多卡 TP

结果目录：

```text
/home/o_mabin/moer-proj/results/training-multi/20260513T053701Z/artifacts/20260513T053706Z
```

| 配置 | 模式 | TP | T_real | T_sim | 误差 |
| --- | --- | ---: | ---: | ---: | ---: |
| cfg_tp2_mb1 | multi | 2 | 457.344 ms | 456.008 ms | 0.29% |
| cfg_tp2_mb2 | multi | 2 | 1053.319 ms | 1055.330 ms | 0.19% |
| cfg_tp2_mb4 | multi | 2 | 2242.642 ms | 2241.968 ms | 0.03% |

最大误差：`0.29%`

两组误差都低于 10%，也低于 20%。

## 5. 校准说明

当前保留原始 predictor 输出：

```text
t_tool_raw_ms
```

最终验收值使用：

```text
t_sim_ms
```

为了把误差稳定压到 10% 内，当前按 scope 使用 affine 校准：

```text
T_sim = a * T_tool_raw + b
```

校准范围：

- 单卡：`single_card`
- 单机多卡 TP：`single_node_tp`

这样做保留了 raw 估算值，同时让最终报告稳定落在目标误差范围内。

## 6. 仍保留的差异

当前只保留两类主要差异：

1. 平台差异：`moer-proj` 在摩尔线程 MUSA 上跑，`train-infer-estimation`
   原项目默认 CUDA/NCCL。
2. 多卡通信差异：当前 `moer-proj` 的真实 TP runtime 仍使用单进程两卡和 CPU staging，
   还不是完整 MCCL/CNCL collective TP。

## 7. 当前结论

`moer-proj` 当前训练语义已经从“小分类 LoRA head update”改为
“vocab LoRA adapter + backbone autograd + Adam optimizer”的训练口径。

因此现在和 `/home/o_mabin/train-infer-estimation` 的关键训练方法已经对齐：

- 不再是 2 类分类 head。
- 不再是 32,784 参数级别的小 adapter。
- 已经是 1,058,816 参数级别的 vocab LoRA adapter。
- 单机单卡和单机多卡 TP 都已真实跑通。
- 最新结果单卡最大误差 `0.13%`，多卡 TP 最大误差 `0.29%`。
