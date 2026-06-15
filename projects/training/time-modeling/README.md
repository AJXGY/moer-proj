# Training Time Modeling

摩尔线程训练耗时建模项目。

本目录已经内置训练运行时 `train_runtime/`，脚本不再依赖
`projects/shared/train-infer-estimation` 下的训练逻辑。保留 shared 目录只是历史来源，
后续训练测试、预测、汇总都从本目录入口执行。

固定训练口径：

- 模型：Meta-Llama-3.1-8B
- 训练模式：LoRA
- `sequence_length/max_seq_len`: 8
- `lora_rank`: 8

## Card Modes

- Single-card: `tensor_parallel_size=1`
- Multi-card: 单机多卡 TP，`tensor_parallel_size=2`

## Run

```bash
../../../scripts/run_training_moorethreads_smoke.sh
```

也可以分开跑：

```bash
../../../scripts/run_training_single_moorethreads.sh
../../../scripts/run_training_multi_moorethreads.sh
```
