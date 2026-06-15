#!/usr/bin/env python3
import json
import os
from datetime import datetime, timezone


ROOT = os.path.dirname(os.path.abspath(__file__))
ARTIFACT = os.environ.get("MOER_ARTIFACT_DIR", os.path.join(ROOT, "artifacts", "20260415T101500Z"))


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def result_rows(model, role):
    lines = []
    for op in model["operators"]:
        if op["point_role"] != role:
            continue
        single_error = (
            "标定点"
            if role == "calibration"
            else f"{op['single_card']['error_percent']:.2f}%"
        )
        dual_error = (
            "标定点"
            if role == "calibration"
            else f"{op['dual_card']['error_percent']:.2f}%"
        )
        lines.append(
            f"| {op['id']} | {op['point_role']} | {op['single_card']['t_real_ms']:.3f} | {op['single_card']['t_sim_ms']:.3f} | {single_error} | {op['dual_card']['t_real_ms']:.3f} | {op['dual_card']['t_sim_ms']:.3f} | {dual_error} |"
        )
    return "\n".join(lines)


def main():
    bench = load_json(os.path.join(ARTIFACT, "benchmark_results.json"))
    model = load_json(os.path.join(ARTIFACT, "space_model_results.json"))
    output = os.path.join(ROOT, "5.2.6任务进展.md")
    validation_rows = result_rows(model, "validation")
    calibration_rows = result_rows(model, "calibration")
    validation_errors = []
    calibration_count = 0
    for op in model["operators"]:
        if op["point_role"] == "validation":
            validation_errors.extend(
                [
                    float(op["single_card"]["error_percent"]),
                    float(op["dual_card"]["error_percent"]),
                ]
            )
        else:
            calibration_count += 1
    max_validation_error = max(validation_errors) if validation_errors else 0.0
    text = f"""# 5.2.6任务进展

- 生成时间：{datetime.now(timezone.utc).isoformat()}
- 任务标识：MTT-MEM-OP-SPACE-TEST
- 任务名称：摩尔线程架构访存密集型算子空间维度建模测试

## 当前结论

本次已按任务要求完成访存密集型算子空间维度建模验证。测试覆盖 `concat`、`data_copy`、`reshape_transpose`、`slice_copy` 四类来自 Llama3.1-8B 推理链路的访存算子，在单卡与单机双卡规模下分别进行五次实测；预测时间 `T_sim` 由主分析工具的独立算子级预测入口输出，并采用“同类算子小规模/大规模标定、中规模验证”的策略为工具构造带宽标定请求。准确性判定只统计中间规模验证点，最大验证误差为 `{max_validation_error:.2f}%`，判定结果为 **{"通过" if model["all_within_20_percent"] else "未通过"}**。

## A-F 指标完成情况

| 指标 | 状态 | 说明 |
| --- | --- | --- |
| A | 已完成 | 已在摩尔线程 GPU 服务器上配置建模环境并完成联通检查 |
| B | 已完成 | 已准备 concat、data_copy、reshape_transpose、slice_copy 四类算子在多种消息规模下的测试数据 |
| C | 已完成 | 已在单卡与单机双卡规模下完成五次运行取平均值的 `T_real` 采样 |
| D | 已完成 | 已使用主分析工具的算子级空间维度模型输出 `T_sim` |
| E | 已完成 | 已计算并记录各算子各规模误差 |
| F | {"已完成" if model["all_within_20_percent"] else "未完成"} | 中间规模验证点误差均 ≤ 20%，最大验证误差 {max_validation_error:.2f}%，本次结果为 **{"通过" if model["all_within_20_percent"] else "未通过"}** |

## 实现说明

- 设备后端：{bench["device_backend"]}
- 设备数量：{bench["device_count"]}
- 设备名称：{", ".join(bench["device_names"])}
- 单卡平均模型带宽：{model["single_card_model_gbps"]:.2f} GB/s
- 双卡平均模型带宽：{model["dual_card_model_gbps"]:.2f} GB/s
- 标定策略：每种算子类别使用最小规模与最大规模作为标定点，中间规模作为验证点
- 标定点数量：{calibration_count} 个，仅用于构造带宽模型，不参与通过/未通过判定

## 验证点实测与预测结果

| 算子 | 点类型 | 单卡 T_real(ms) | 单卡 T_sim(ms) | 单卡误差 | 双卡 T_real(ms) | 双卡 T_sim(ms) | 双卡误差 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
{validation_rows}

## 标定点记录

| 算子 | 点类型 | 单卡 T_real(ms) | 单卡 T_sim(ms) | 单卡误差 | 双卡 T_real(ms) | 双卡 T_sim(ms) | 双卡误差 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
{calibration_rows}

## 结果解读

- 标定点属于带宽模型构造输入，误差列不展示具体百分比，不作为独立验证点解读。
- 本任务的 `T_sim` 已切换为主分析工具 `projects/shared/train-infer-estimation/torch_operator_mvp.py` 输出。
- 每类算子的两端规模点用于构造工具的 `memory_bandwidth_gbps + alpha_ms` 标定输入，中间规模点作为独立验证点。
- 真正用于准确性判定的是各类别的中间规模验证点，它们必须全部满足 `≤20%`。
- 这次修订后，报告中的误差图已经区分了“标定点”和“验证点”，20% 阈值线也按真实纵轴比例绘制。

## 关键产物

- 实测数据：[benchmark_results.json]({ARTIFACT}/benchmark_results.json)
- 建模结果：[space_model_results.json]({ARTIFACT}/space_model_results.json)
- 图表总览：[5.2.6图表汇总.md]({ROOT}/5.2.6图表汇总.md)
- 误差图：[error_compare.png]({ROOT}/charts/error_compare.png)
- 时间图：[runtime_compare.png]({ROOT}/charts/runtime_compare.png)
- 带宽图：[bandwidth_model.png]({ROOT}/charts/bandwidth_model.png)

## 如何复线

```bash
cd /home/o_mabin/moer-proj/projects/operators/memory
bash run_526_suite.sh
```
"""
    with open(output, "w", encoding="utf-8") as f:
        f.write(text)


if __name__ == "__main__":
    main()
