# 5.1.6 摩尔线程架构上训练任务运行测试

本目录把 `train-infer-estimation-release-2026-04-11` 中的训练 runtime 直接接入 `clj-proj` 任务目录，用于完成 `MTT-TRAIN-RUN-TEST`。

当前训练口径：

- 模型：`Meta-Llama-3.1-8B`
- 训练方式：真实 backbone 前向 + LoRA 风格适配头参数更新
- 单卡：`PP=1`
- 双卡流水线并行：`PP=2`，按 16/16 层切分到 `musa:0` / `musa:1`
- 双卡张量并行补充：`TP=2`，通过共享 runtime 的 `tp_heads` 分片执行
- 输出：运行日志、每 step 耗时、参数范数轨迹、adapter checkpoint

执行：

```bash
cd /home/o_mabin/moerxiancheng-clj-xyj-proj/projects/training/runtime-validation
bash run_516_suite.sh
```
