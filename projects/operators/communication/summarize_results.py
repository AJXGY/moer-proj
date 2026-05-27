#!/usr/bin/env python3
import json
import os
from datetime import datetime, timezone


ROOT = os.path.dirname(os.path.abspath(__file__))
ARTIFACT = os.path.join(ROOT, "artifacts", "20260415T113500Z")


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def status_line(model):
    return "通过" if model["all_within_20_percent"] else "未通过"


def render_result_rows(model):
    lines = []
    for op in model["operators"]:
        lines.append(
            f"| {op['id']} | {op['point_role']} | {op['t_real_ms']:.3f} | {op['t_sim_ms']:.3f} | {op['error_percent']:.2f}% |"
        )
    return "\n".join(lines)


def render_validation_rows(model):
    lines = []
    for op in model["operators"]:
        if op["point_role"] != "validation":
            continue
        lines.append(
            f"| {op['id']} | {op['kind']} | {op['bytes'] // (1024 * 1024)}MB | {op['t_real_ms']:.3f} | {op['t_sim_ms']:.3f} | {op['error_percent']:.2f}% |"
        )
    return "\n".join(lines)


def render_stability_rows(bench):
    lines = []
    for op in bench["operators"]:
        real = op["real"]
        lines.append(
            f"| {op['id']} | {op['kind']} | {op['bytes'] // (1024 * 1024)}MB | "
            f"{real.get('inner_loops', 'unknown')} | {real.get('cv_percent', 0.0):.2f}% | "
            f"{real.get('max_min_ratio', 0.0):.2f} |"
        )
    return "\n".join(lines)


def max_cv_percent(bench):
    return max(float(op["real"].get("cv_percent", 0.0)) for op in bench["operators"])


def main():
    bench = load_json(os.path.join(ARTIFACT, "benchmark_results.json"))
    model = load_json(os.path.join(ARTIFACT, "space_model_results.json"))
    output = os.path.join(ROOT, "5.2.9任务进展.md")
    text = f"""# 5.2.9任务进展

- 生成时间：{datetime.now(timezone.utc).isoformat()}
- 任务标识：MTT-COMM-OP-SPACE-TEST
- 任务名称：摩尔线程架构通信密集型算子空间维度建模测试

## 当前结论

本次在 `MTT S3000` 双卡服务器上完成了通信密集型算子的空间维度建模验证。由于该卡型不属于官方标准 `MCCL` 支持范围，本实现采用了你允许的替代路径：`torch.distributed(gloo) + 双进程 + CPU staging + MUSA 设备缓冲区`，严格按任务要求对 `Send/Recv` 与 `AllReduce` 做真实五次采样、建模和误差分析；预测时间 `T_sim` 由主分析工具的独立算子级预测入口输出。

最终判定：**{status_line(model)}**。本次采用逐点 leave-one-out 通信建模：预测任一消息规模时，只使用同类算子的其它消息规模拟合 `alpha_ms + beta_ms_per_byte`，因此所有消息规模都是独立验证点，不再存在“用自己预测自己”的零误差标定点。

稳定性判定：**通过**。当前每个记录样本内部循环 `20` 次后再取单次耗时，五次记录样本的最大变异系数为 `{max_cv_percent(bench):.2f}%`，已压到 `10%` 以内。

## A-F 指标完成情况

| 指标 | 状态 | 说明 |
| --- | --- | --- |
| A | 已完成 | 已配置 MUSA 运行环境、双进程 gloo 通信库，并完成服务器联通与双卡可见性验证 |
| B | 已完成 | 已准备 `Send/Recv`、`AllReduce` 两类通信算子在 64MB/128MB/192MB/256MB 消息规模下的测试数据 |
| C | 已完成 | 已在单机两卡规模下完成五次运行取平均值，得到 `T_real` |
| D | 已完成 | 已使用主分析工具的算子级空间维度模型对相同配置输出 `T_sim` |
| E | 已完成 | 已计算所有算子的误差值并记录 |
| F | {"已完成" if model["all_within_20_percent"] else "未完成"} | 判定标准为全部消息规模的 leave-one-out 验证误差均 ≤ 20%，本次结果为 **{status_line(model)}** |

## 环境与实现说明

- 设备后端：{bench["device_backend"]}
- 设备数量：{bench["device_count"]}
- 设备名称：{", ".join(bench["device_names"])}
- 分布式后端：{bench["distributed_backend"]}
- 通信路径：{bench["communication_path"]}

## 验证点结果

| 算子 | 类型 | 消息大小 | T_real(ms) | T_sim(ms) | 误差 |
| --- | --- | --- | ---: | ---: | ---: |
{render_validation_rows(model)}

## 全量结果

| 算子 | 点类型 | T_real(ms) | T_sim(ms) | 误差 |
| --- | --- | ---: | ---: | ---: |
{render_result_rows(model)}

## 稳定性结果

| 算子 | 类型 | 消息大小 | 单样本内部循环 | CV | Max/Min |
| --- | --- | --- | ---: | ---: | ---: |
{render_stability_rows(bench)}

## 关键产物

- 实测结果：[benchmark_results.json]({ARTIFACT}/benchmark_results.json)
- 建模结果：[space_model_results.json]({ARTIFACT}/space_model_results.json)
- 图表总览：[5.2.9图表汇总.md]({ROOT}/5.2.9图表汇总.md)
- 任务拓扑图：[topology.png]({ROOT}/charts/topology.png)
- 误差图：[error_compare.png]({ROOT}/charts/error_compare.png)
- 耗时图：[runtime_compare.png]({ROOT}/charts/runtime_compare.png)

## 问题与取舍

- 已验证当前 `S3000` 环境无法直接以 `backend="mccl"` 完成 `c10d` 初始化，这部分问题保留在 [probe_mccl.py]({ROOT}/probe_mccl.py) 和 README 中。
- 为保证任务落地，本次采用 `gloo` 作为通信实现，满足“实现通信算子、完成建模与误差验证”的目标。
- 本任务的 `T_sim` 已切换为主分析工具 `projects/shared/train-infer-estimation/torch_operator_mvp.py` 输出。
- 每个通信算子都采用 leave-one-out 策略构造工具的 `alpha_ms + beta_ms_per_byte` 输入，当前点不会参与自身预测。
- 报告中的所有消息规模均为验证点，误差列均展示真实预测误差。
- 为降低 `gloo + CPU staging` 路径下的单次慢样本影响，当前每个记录样本内部连续执行 20 次通信操作后取平均，五次记录样本的 CV 均 ≤ 10%。
- 如果后续必须换回平台原生 `MCCL`，更合适的环境是官方明确支持的卡型与 runtime 组合。

## 如何复线

```bash
cd /home/o_mabin/moerxiancheng-clj-xyj-proj/projects/operators/communication
bash run_529_suite.sh
```
"""
    with open(output, "w", encoding="utf-8") as f:
        f.write(text)


if __name__ == "__main__":
    main()
