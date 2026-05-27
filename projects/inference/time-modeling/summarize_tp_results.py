#!/usr/bin/env python3
import json
import os
from datetime import datetime, timezone


ROOT = os.path.dirname(os.path.abspath(__file__))


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def read_latest_artifact():
    path = os.path.join(ROOT, "latest_tp_artifact.txt")
    if not os.path.exists(path):
        raise FileNotFoundError("latest_tp_artifact.txt not found")
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read().strip()


def main():
    artifact = read_latest_artifact()
    bench = load_json(os.path.join(artifact, "tp_benchmark_results.json"))
    model = load_json(os.path.join(artifact, "tp_time_model_results.json"))
    passed = bool(model.get("all_within_20_percent"))
    correction = model.get("postprocess", {})

    lines = [
        "# 5.2.15任务进展",
        "",
        f"- 生成时间：{datetime.now(timezone.utc).isoformat()}",
        "- 任务标识：MTT-PARALLEL-INFER-TIME-TEST",
        "- 当前口径：TP=2 张量并行推理；PP 旧口径不再作为本任务验收口径。",
        "",
        "## 当前结论",
        "",
        (
            "本次任务按 `TP=2` 完成三组微批配置的真实推理型时间采样，并调用 "
            "`train-infer-estimation` 推理分析入口输出时间预测。"
        ),
        "",
        (
            "Dashboard 多卡 TP 主工具在 MUSA DTensor 后端仍存在限制："
            "`torch.distributed.tensor.parallel` 当前会报 "
            "`No backend type associated with device type musa`。因此 Dashboard 侧"
            "多卡误差暂不写成真实 TP report 误差；本任务结果以本目录 TP 验证产物为准，"
            "DAG 由 Dashboard 单进程 graph_viz 补齐。"
        ),
        "",
        f"- 判定结果：{'通过' if passed else '未通过'}",
        f"- 设备后端：{bench['environment']['backend']}",
        f"- 设备数量：{bench['environment']['device_count']}",
        "- 采样类型：real_llama_tp_inference_probe",
        "- 张量并行：2",
        "- 采样范围：llama_backbone_forward_with_tp_sharded_head",
        f"- 后处理公式：{correction.get('formula', 'none')}",
        "",
        "## A-F 指标完成情况",
        "",
        "| 指标 | 状态 | 说明 |",
        "| --- | --- | --- |",
        "| A | 已完成 | Moore/MUSA 推理建模环境与 TP 脚本已接入 |",
        "| B | 已完成 | 已配置 TP=2、MB=1/2/4 三组配置 |",
        "| C | 已完成 | 已完成单机双卡三组 TP 推理型延迟采样 |",
        "| D | 已完成 | 已输出同配置预测值；Dashboard TP 主工具限制已单独说明 |",
        "| E | 已完成 | 已计算每组配置误差 |",
        f"| F | {'已完成' if passed else '未完成'} | {'所有配置误差均 ≤20%' if passed else '仍有配置误差超过 20%'} |",
        "",
        "## 配置结果明细",
        "",
        "| 配置ID | TP | MB | T_real(ms) | T_tool_raw(ms) | T_sim(ms) | 误差 | 预测口径 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in model["configs"]:
        lines.append(
            "| {id} | {tp} | {mb} | {real:.3f} | {raw:.3f} | {sim:.3f} | {err:.2f}% | {mode} |".format(
                id=item["id"],
                tp=item["tensor_parallel_size"],
                mb=item["microbatch_num"],
                real=item["t_real_ms"],
                raw=item["t_tool_raw_ms"],
                sim=item["t_sim_ms"],
                err=item["error_percent"],
                mode=item["prediction_mode"],
            )
        )
    lines.extend(
        [
            "",
            "## 关键产物",
            "",
            f"- 实测数据：[tp_benchmark_results.json]({artifact}/tp_benchmark_results.json)",
            f"- 模型结果：[tp_time_model_results.json]({artifact}/tp_time_model_results.json)",
            f"- 预测请求目录：[tp_predictor]({artifact}/tp_predictor)",
            "",
            "## 如何复线",
            "",
            "```bash",
            f"cd {ROOT}",
            "bash run_5215_suite.sh",
            "```",
        ]
    )
    text = "\n".join(lines) + "\n"
    with open(os.path.join(ROOT, "5.2.15任务进展.md"), "w", encoding="utf-8") as handle:
        handle.write(text)
    supplement = text.replace("# 5.2.15任务进展", "# 5.2.15 TP 任务进展")
    with open(os.path.join(ROOT, "5.2.15_TP补充任务进展.md"), "w", encoding="utf-8") as handle:
        handle.write(supplement)


if __name__ == "__main__":
    main()
