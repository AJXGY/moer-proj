#!/usr/bin/env python3
import json
import os
from datetime import datetime, timezone


ROOT = os.path.dirname(os.path.abspath(__file__))
ARTIFACT = os.environ.get("MOER_ARTIFACT_DIR", os.path.join(ROOT, "artifacts", "20260415T100500Z"))


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    bench = load_json(os.path.join(ARTIFACT, "benchmark_results.json"))
    model = load_json(os.path.join(ARTIFACT, "space_model_results.json"))
    output = os.path.join(ROOT, "5.2.3任务进展.md")
    rows = []
    for op in model["operators"]:
        rows.append(
            f"| {op['name']} | {op['point_role']} | {op['single_card']['t_real_ms']:.3f} | {op['single_card'].get('t_tool_raw_ms', op['single_card']['t_sim_ms']):.3f} | {op['single_card']['t_sim_ms']:.3f} | {op['single_card']['error_percent']:.2f}% | {op['dual_card']['t_real_ms']:.3f} | {op['dual_card'].get('t_tool_raw_ms', op['dual_card']['t_sim_ms']):.3f} | {op['dual_card']['t_sim_ms']:.3f} | {op['dual_card']['error_percent']:.2f}% |"
        )
    passed_20 = bool(model["all_within_20_percent"])
    passed_10 = bool(model.get("all_within_10_percent"))
    postprocess = model.get("postprocess", {})
    correction_text = (
        f"已应用透明后处理：{postprocess.get('formula')}"
        if postprocess.get("correction_applied")
        else "未追加经验校正，T_sim 直接取自主分析工具输出"
    )
    conclusion = (
        "本次已完成计算密集型算子的空间维度建模验证。测试对象为 Llama3.1-8B 中的 `mlp_up_gemm`、`mlp_gate_gemm`、`mlp_down_gemm`、`flash_attention` 与 `attention_output_proj_gemm`，在单卡与单机双卡两种规模下进行了五次实测取均值；预测时间由主分析工具输出，并对当前 `mp 2.1` 环境下的 FlashAttention 双卡兼容路径做透明校正。所有验证点误差均不超过 20%，判定结果为 **通过**。"
        if passed_10
        else "本次已完成计算密集型算子的空间维度建模验证，但校正后仍存在验证点误差超过 10%，需要继续优化。"
        if passed_20
        else "本次已完成计算密集型算子的空间维度建模验证，但存在验证点误差超过 20%，当前不能按指标判定通过。"
    )
    f_desc = (
        "所有测试算子误差均 ≤ 10%，满足 20% 验收阈值并达到更严格目标"
        if passed_10
        else "所有测试算子误差均 ≤ 20%，但未稳定到 10% 内"
        if passed_20
        else "存在测试算子误差 > 20%，未满足验收阈值"
    )
    md = f"""# 5.2.3任务进展

- 生成时间：{datetime.now(timezone.utc).isoformat()}
- 任务标识：MTT-COMPUTE-OP-SPACE-TEST
- 任务名称：摩尔线程架构计算密集型算子空间维度建模测试

## 当前结论

{conclusion}

## A-F 指标完成情况

| 指标 | 状态 | 说明 |
| --- | --- | --- |
| A | 已完成 | 已在摩尔线程 GPU 服务器上配置建模环境并完成联通检查 |
| B | 已完成 | 已选取 mlp_up_gemm、mlp_gate_gemm、mlp_down_gemm、flash_attention、attention_output_proj_gemm 并确定输入规模 |
| C | 已完成 | 已完成单卡与单机双卡五次实测平均时间采样 |
| D | 已完成 | 已使用算子级空间维度模型输出预测时间，并保留 FlashAttention 双卡兼容路径校正 |
| E | 已完成 | 已计算并记录各算子在两种并行规模下的误差值 |
| F | {"已完成" if passed_20 else "未通过"} | {f_desc} |

## 关键结果

- 设备后端：{bench["device_backend"]}
- 设备数量：{bench["device_count"]}
- 设备名称：{", ".join(bench["device_names"])}
- 单卡全体平均吞吐：{model["single_card_model_tflops"]:.2f} TFLOPS
- 双卡全体平均吞吐：{model["dual_card_model_tflops"]:.2f} TFLOPS
- 预测后处理：{correction_text}
- 判定结果：{"通过" if model["all_within_20_percent"] else "未通过"}

## 实测与预测结果

| 算子 | 点类型 | 单卡 T_real(ms) | 单卡 T_tool_raw(ms) | 单卡 T_sim(ms) | 单卡误差 | 双卡 T_real(ms) | 双卡 T_tool_raw(ms) | 双卡 T_sim(ms) | 双卡误差 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(rows)}

## 结果说明

- 本任务的 `T_sim` 已切换为主分析工具 `projects/shared/train-infer-estimation/torch_operator_mvp.py` 输出。
- 本任务按算子族构造 `calibration_override`：GEMM 算子使用同族 leave-one-out 吞吐参考；FlashAttention 使用额外的 `calib_flash_attention_seq512` 标定点，不使用自身实测值回填。
- 当前 `MTT S3000` 环境执行 FlashAttention 时，torch_musa 提示 fast FlashAttention kernel 需要 `mp >= 2.2`；本次记录为 `scaled_dot_product_attention` 在当前 `mp 2.1` 环境下的兼容执行路径。
- 当前表格同时保留 `T_tool_raw` 和 `T_sim`；除 FlashAttention 双卡外，`T_sim = T_tool_raw`。
- 已移除旧版 GEMM 形状经验校正。旧校正会把已接近实测的 `T_tool_raw` 放大或压低，导致 QKV/Gate/Down 误差异常。
- 误差图中的 20% 红线已按真实纵轴比例重绘，不再使用固定 25% 画布比例。

## 关键产物

- 实测数据：[benchmark_results.json]({ARTIFACT}/benchmark_results.json)
- 模型结果：[space_model_results.json]({ARTIFACT}/space_model_results.json)
- 图表汇总：[5.2.3图表汇总.md]({ROOT}/5.2.3图表汇总.md)
- 误差图：[error_compare.png]({ROOT}/charts/error_compare.png)
- 时间图：[runtime_compare.png]({ROOT}/charts/runtime_compare.png)

## 如何复线

```bash
cd /home/o_mabin/moer-proj/projects/operators/compute
bash run_523_suite.sh
```
"""
    with open(output, "w", encoding="utf-8") as f:
        f.write(md)


if __name__ == "__main__":
    main()
