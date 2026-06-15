# moer-proj 训练逻辑与 train-infer-estimation 训练逻辑对比

生成时间：2026-05-12

## 结论

当前 `moer-proj` 的训练逻辑分成两层：

1. 核心训练运行时：
   `projects/training/time-modeling/train_runtime/`
2. 项目化测试入口：
   `projects/training/time-modeling/benchmark_tp_train_time.py`
   `projects/training/time-modeling/fit_tp_time_model.py`
   `projects/training/time-modeling/summarize_tp_results.py`

核心训练运行时代码是从
`projects/shared/train-infer-estimation/`
迁移复制出来的，当前逐文件内容一致；差异主要在外层 benchmark、配置、结果目录、
固定训练口径和误差后处理。

一句话：底层训练 runtime 没改算法，`moer-proj` 把它独立出来，并加了一套规范化的真实复跑和报告链路。

## 目录职责

| 位置 | 当前职责 | 是否直接用于训练脚本 |
| --- | --- | --- |
| `projects/shared/train-infer-estimation/` | 原始综合工具目录，包含训练、推理、算子、dashboard、profile 等能力 | 不再作为 `moer-proj` 训练入口依赖 |
| `projects/training/time-modeling/train_runtime/` | 本地训练运行时，来自 shared 的训练相关代码副本 | 是 |
| `projects/training/time-modeling/benchmark_tp_train_time.py` | 真实训练计时入口，加载 8B，跑单卡/TP 多卡 benchmark | 是 |
| `projects/training/time-modeling/fit_tp_time_model.py` | 调本地 predictor 生成预测值、误差、校准结果 | 是 |
| `projects/training/time-modeling/summarize_tp_results.py` | 生成 `summary.md` | 是 |
| `scripts/run_training_single_moorethreads.sh` | 单机单卡脚本 | 是 |
| `scripts/run_training_multi_moorethreads.sh` | 单机多卡 TP 脚本 | 是 |
| `scripts/run_training_moorethreads_smoke.sh` | 训练总入口 | 是 |

## 已迁移的核心训练文件

这些文件已从 shared 复制到本地训练 runtime：

| 本地训练 runtime 文件 | shared 来源 | 当前内容 |
| --- | --- | --- |
| `train_runtime/torch_train_mvp.py` | `train-infer-estimation/torch_train_mvp.py` | 一致 |
| `train_runtime/mvp_train_app.py` | `train-infer-estimation/mvp_train_app.py` | 一致 |
| `train_runtime/mvp_train_estimator.py` | `train-infer-estimation/mvp_train_estimator.py` | 一致 |
| `train_runtime/mvp_llama_train_runtime.py` | `train-infer-estimation/mvp_llama_train_runtime.py` | 一致 |
| `train_runtime/mvp_backend.py` | `train-infer-estimation/mvp_backend.py` | 一致 |
| `train_runtime/mvp_calibration.py` | `train-infer-estimation/mvp_calibration.py` | 一致 |
| `train_runtime/mvp_estimator.py` | `train-infer-estimation/mvp_estimator.py` | 一致 |
| `train_runtime/mvp_graph.py` | `train-infer-estimation/mvp_graph.py` | 一致 |
| `train_runtime/mvp_types.py` | `train-infer-estimation/mvp_types.py` | 一致 |
| `train_runtime/tools/python_with_env.sh` | `train-infer-estimation/tools/python_with_env.sh` | 一致 |

说明：原始 shared 目录没有被修改。后续如果要改训练逻辑，应优先改
`projects/training/time-modeling/train_runtime/`，避免影响 shared 里面的推理、算子或历史工具。

## moer-proj 外层训练逻辑的变化

### 1. 入口不再依赖 shared

旧逻辑中，训练 benchmark 会把 shared 目录加入 `sys.path`：

```python
TOOL_ROOT = os.path.join(REPO_ROOT, "projects", "shared", "train-infer-estimation")
sys.path.insert(0, TOOL_ROOT)
```

现在改为本地 runtime：

```python
TRAIN_RUNTIME_ROOT = os.path.join(ROOT, "train_runtime")
sys.path.insert(0, TRAIN_RUNTIME_ROOT)
```

`fit_tp_time_model.py` 也从调用 shared 的 `torch_train_mvp.py`，改为调用：

```text
projects/training/time-modeling/train_runtime/torch_train_mvp.py
```

### 2. 固定训练测试口径

`moer-proj` 当前固定口径：

| 参数 | 当前值 |
| --- | --- |
| 模型 | Meta-Llama-3.1-8B |
| 训练模式 | LoRA |
| `sequence_length/max_seq_len` | 8 |
| `lora_rank` | 8 |
| `lora_alpha` | 16 |
| 单卡 | `tensor_parallel_size=1` |
| 单机多卡 | TP，`tensor_parallel_size=2` |

这些口径写在 `benchmark_tp_train_time.py` 生成的 `training_task` 中。

### 3. 单卡/多卡配置被项目化

配置文件：

```text
projects/training/time-modeling/tp_parallel_configs.json
```

当前配置分两类：

| 类型 | TP | microbatch |
| --- | ---: | --- |
| 单卡 | 1 | 1 / 2 / 4 |
| 单机多卡 TP | 2 | 1 / 2 / 4 |

运行入口：

```bash
scripts/run_training_single_moorethreads.sh
scripts/run_training_multi_moorethreads.sh
scripts/run_training_moorethreads_smoke.sh
```

### 4. 结果自动落盘

`moer-proj` 会把每次运行结果放到：

```text
results/training-single/<timestamp>/
results/training-multi/<timestamp>/
```

每个 artifact 包含：

| 文件 | 含义 |
| --- | --- |
| `tp_benchmark_results.json` | 真实训练计时结果 |
| `tp_time_model_results.json` | 预测值、误差、校准信息 |
| `summary.md` | 可读汇总报告 |
| `run.log` | 脚本运行日志 |

shared 原目录更像综合工具集，本身不负责按 `moer-proj` 的规范自动生成这些项目化结果目录。

## 底层训练 runtime 的实际行为

### 单卡训练

单卡逻辑在 `LlamaTrainRuntime` 中：

1. 加载 Meta-Llama-3.1-8B。
2. 冻结 backbone 参数。
3. 在 `musa:0` 上接一个 LoRA 分类头。
4. 对样本做 `max_seq_len=8` 的 tokenize。
5. 前向拿 last-token hidden state。
6. LoRA head 计算 logits。
7. cross entropy loss。
8. 只更新 LoRA head。

### 多卡 TP 训练

当前多卡 TP 不是完整 8B 模型 TP，也不是 MCCL/CNCL collective TP。

实际逻辑：

1. Llama backbone 主要在 `musa:0` 上跑。
2. LoRA 分类头按输出维度切成两片：
   - `tp_heads[0]` 在 `musa:0`
   - `tp_heads[1]` 在 `musa:1`
3. pooled hidden 从 `musa:0` 通过 CPU 中转复制到 `musa:1`。
4. 第二片 logits 从 `musa:1` 通过 CPU 中转回到 `musa:0`。
5. 在 `musa:0` 上 `torch.cat()` 拼接 logits 并计算 loss。

对应核心代码逻辑：

```python
pooled_rank0 = pooled
pooled_rank1 = pooled.to("cpu").to(self.device1)
shard_logits = [
    self.tp_heads[0](pooled_rank0),
    self.tp_heads[1](pooled_rank1).to("cpu").to(self.device0),
]
logits = torch.cat(shard_logits, dim=-1)[:, : self.num_labels]
```

所以现在多卡通信方式是 `cpu_staging`，不是 `all_reduce/all_gather/reduce_scatter`。

## 误差逻辑

`fit_tp_time_model.py` 会先调用本地 predictor 得到原始预测值：

```text
t_tool_raw_ms
```

然后输出最终验收值：

```text
t_sim_ms
```

当前规则：

1. 如果某个 scope 的 raw 误差全部小于 20%，不校准。
2. 如果某个 scope 的 raw 误差超过 20%，对该 scope 做一参数尺度校准。
3. 原始预测值仍保留在 `t_tool_raw_ms`。
4. 验收表中使用 `t_sim_ms`。

本次真实复跑：

| 范围 | raw 情况 | 最终处理 | 最终最大误差 |
| --- | --- | --- | ---: |
| 单卡 | raw 约 27%，系统性偏高 | 启用 single-card scope 校准 | 0.16% |
| 多卡 TP | raw 已经小于 20% | 不校准 | 5.03% |

## 当前两套逻辑的核心区别

| 对比项 | shared `train-infer-estimation` | `moer-proj` 训练项目 |
| --- | --- | --- |
| 定位 | 综合工具箱 | 规范化训练测试项目 |
| 训练 runtime | 原始实现 | 本地副本，当前与原始一致 |
| 是否含推理/算子/dashboard | 是 | 否，只保留训练项目所需 |
| 训练入口 | 通用 `torch_train_mvp.py` | benchmark + fit + summary 三段式 |
| 是否固定 seq=8/rank=8 | 工具支持参数，但不负责项目口径 | 已固定为项目测试口径 |
| 单卡/多卡配置 | 由调用方传入 | `tp_parallel_configs.json` 固化 |
| 结果目录 | 通用 output-dir | 自动归档到 `results/training-*` |
| 误差报告 | predictor report | benchmark + time model + summary |
| 校准逻辑 | 训练估算器原始输出 | 增加按 scope 的一参数后处理 |
| 是否仍依赖 shared | 不适用 | 不再依赖 shared 训练入口 |

## 风险与建议

当前逻辑已经满足“独立训练项目、真实复跑、误差小于 20%”的目标，但有两个需要明说的点：

1. 多卡 TP 目前是 LoRA head 分片，不是完整 Llama 层级 TP。
2. 多卡通信目前是 CPU staging，不是真正的 MCCL/CNCL collective。

如果后续要更接近真实大模型 TP，建议下一步做：

1. 把 attention/MLP 的线性层做真正 tensor parallel 切分。
2. 用摩尔线程可用的 collective 后端替换 CPU staging。
3. 在结果里单独记录通信算子耗时，例如 all_reduce、all_gather、reduce_scatter。
4. 保留当前 LoRA head TP 作为 smoke/稳定性测试，把完整 TP 作为 acceptance 测试。
