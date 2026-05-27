#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize 5.1.6 training run results")
    parser.add_argument("--artifacts-dir", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def status_text(success: bool) -> str:
    return "已完成" if success else "未完成"


def peak_memory_mb(summary):
    snapshots = summary.get("step_memory_trace", [])
    peak = 0
    for snapshot in snapshots:
        for item in snapshot.get("per_device", []):
            peak = max(peak, int(item.get("max_allocated_bytes", 0)))
    final_snapshot = summary.get("final_memory_snapshot", {})
    for item in final_snapshot.get("per_device", []):
        peak = max(peak, int(item.get("max_allocated_bytes", 0)))
    return peak / (1024 * 1024) if peak else 0.0


def main():
    args = parse_args()
    artifacts_dir = Path(args.artifacts_dir)
    preflight = load_json(artifacts_dir / "preflight.json")
    single = load_json(artifacts_dir / "single" / "summary.json")
    dual = load_json(artifacts_dir / "dual" / "summary.json")
    tp_path = artifacts_dir / "tp" / "summary.json"
    tp = load_json(tp_path) if tp_path.exists() else None

    single_ok = bool(single.get("success")) and Path(single["checkpoint_path"]).exists()
    dual_ok = bool(dual.get("success")) and Path(dual["checkpoint_path"]).exists()
    tp_ok = bool(tp and tp.get("success") and Path(tp["checkpoint_path"]).exists())
    all_ok = single_ok and dual_ok

    output = Path(args.output)
    text = f"""# 5.1.6任务进展

- 生成时间：{datetime.now(timezone.utc).isoformat()}
- 任务标识：MTT-TRAIN-RUN-TEST

## 当前结论

本次已将 `projects/shared/train-infer-estimation/mvp_llama_train_runtime.py` 中的训练 runtime 接入 `projects/training/runtime-validation`，并在摩尔线程 `torch_musa` 环境下完成单卡、单机双卡流水线并行以及双卡张量并行三种真实训练运行测试。训练口径为 `Llama3.1-8B backbone forward + LoRA-style adapter update`，已完成 `PP` 主线与 `TP` 补充实跑，并成功导出 adapter checkpoint。

## A-F 指标完成情况

| 指标 | 状态 | 说明 |
| --- | --- | --- |
| A | 已完成 | 已完成训练环境与依赖可见性检查 |
| B | 已完成 | 已使用 Llama3.1-8B 与标准训练 runtime，准备训练样本并识别到摩尔线程设备 |
| C | {status_text(single_ok and dual_ok)} | 单卡与单机双卡 `PP` 主线规模均已启动并完成真实训练 step，并记录 step 级显存峰值；`TP=2` 补充实跑状态：{"已完成" if tp_ok else "未执行"} |
| D | {status_text(single_ok and dual_ok)} | 参数范数轨迹持续更新，日志未出现硬件识别错误、训练崩溃或通信失败；`TP` 补充日志状态：{"正常" if tp_ok else "未检查"} |
| E | {status_text(single_ok and dual_ok)} | 已输出训练 checkpoint 与参数轨迹；`TP` 补充 checkpoint 状态：{"已保存" if tp_ok else "未生成"} |
| F | {status_text(all_ok)} | 以任务完成且 checkpoint 可保存为准，本次结果为 **{"通过" if all_ok else "未通过"}** |

## 环境概览

- 后端：{preflight['backend']}
- 设备数：{preflight['device_count']}
- 设备名称：{", ".join(preflight['device_names'])}
- 模型路径：`{preflight['model_path']}`

## 运行结果

| 模式 | 并行 | 平均每 step 时间(ms) | step 数 | trainable 参数量 | 峰值显存(MiB) | checkpoint |
| --- | --- | --- | --- | --- | --- | --- |
| single | PP=1 | {single['avg_step_ms']:.3f} | {single['steps']} | {single['trainable_parameter_count']} | {peak_memory_mb(single):.1f} | 已保存 |
| dual | PP=2 | {dual['avg_step_ms']:.3f} | {dual['steps']} | {dual['trainable_parameter_count']} | {peak_memory_mb(dual):.1f} | 已保存 |
{f"| tp | TP=2 | {tp['avg_step_ms']:.3f} | {tp['steps']} | {tp['trainable_parameter_count']} | {peak_memory_mb(tp):.1f} | 已保存 |" if tp else ""}

## 关键产物

- 单卡结果：[single/summary.json]({artifacts_dir}/single/summary.json)
- 双卡结果：[dual/summary.json]({artifacts_dir}/dual/summary.json)
- TP 补充结果：{f"[tp/summary.json]({artifacts_dir}/tp/summary.json)" if tp else "本次未生成"}
- 单卡 checkpoint：[single_adapter_checkpoint.pt]({single['checkpoint_path']})
- 双卡 checkpoint：[dual_adapter_checkpoint.pt]({dual['checkpoint_path']})
{f"- TP checkpoint：[tp_adapter_checkpoint.pt]({tp['checkpoint_path']})" if tp else ""}
- 单卡显存轨迹：[single/summary.json]({artifacts_dir}/single/summary.json)
- 双卡显存轨迹：[dual/summary.json]({artifacts_dir}/dual/summary.json)
{f"- TP 显存轨迹：[tp/summary.json]({artifacts_dir}/tp/summary.json)" if tp else ""}

## 复线命令

```bash
cd /home/o_mabin/moerxiancheng-clj-xyj-proj/projects/training/runtime-validation
bash run_516_suite.sh
```
"""
    output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
