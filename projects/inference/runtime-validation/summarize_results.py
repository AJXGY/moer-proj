#!/usr/bin/env python3
import argparse
import json
import os
from datetime import datetime, timezone


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def status_line(name, status, detail):
    return f"| {name} | {status} | {detail} |"


def median(values):
    if not values:
        return None
    vals = sorted(values)
    mid = len(vals) // 2
    if len(vals) % 2 == 1:
        return vals[mid]
    return (vals[mid - 1] + vals[mid]) / 2.0


def classify(preflight, single, dual, single_only=False):
    package_ready = bool(preflight and preflight["criteria"]["python_dependencies_ready"])
    single_visible = bool(preflight and preflight["criteria"]["single_card_visible"])
    dual_visible = bool(preflight and preflight["criteria"]["dual_card_visible"])
    model_exists = bool(single and os.path.exists(single["model_path"]))
    single_ok = bool(single and single["success"])
    dual_ok = bool(dual and dual["success"])
    single_valid = bool(single and single.get("validation_passed"))
    dual_valid = bool(dual and dual.get("validation_passed"))
    used_dry_run = bool((single and single.get("dry_run")) or (dual and dual.get("dry_run")))

    statuses = []
    statuses.append(
        (
            "A",
            "已完成" if preflight else "未完成",
            "已输出环境预检查结果" if preflight else "缺少环境预检查产物",
        )
    )

    if single_only:
        if model_exists and single_visible:
            b_status = "已完成"
            b_detail = "模型存在，且检查到单卡资源可见"
        elif model_exists:
            b_status = "部分完成"
            b_detail = "模型已就绪，但当前环境未验证到摩尔线程单卡资源可见"
        else:
            b_status = "未完成"
            b_detail = "模型路径不存在或推理任务未执行"
    else:
        if model_exists and single_visible and dual_visible:
            b_status = "已完成"
            b_detail = "模型存在，且检查到单卡/双卡资源可见"
        elif model_exists:
            b_status = "部分完成"
            b_detail = "模型已就绪，但当前环境未验证到摩尔线程单卡/双卡资源可见"
        else:
            b_status = "未完成"
            b_detail = "模型路径不存在或推理任务未执行"
    statuses.append(("B", b_status, b_detail))

    if single_only:
        if single_ok and not used_dry_run and single_visible:
            c_status = "已完成"
            c_detail = "单卡推理任务执行成功"
        elif single_ok and used_dry_run:
            c_status = "部分完成"
            c_detail = "单卡流程已通过 dry-run 验证，待摩尔线程实机补做真实推理"
        elif single:
            c_status = "部分完成"
            c_detail = "已执行单卡流程，但仍需补齐实机成功记录"
        else:
            c_status = "未完成"
            c_detail = "尚未发现单卡推理执行结果"
    else:
        if single_ok and dual_ok and not used_dry_run and single_visible and dual_visible:
            c_status = "已完成"
            c_detail = "单卡与双卡推理任务均执行成功"
        elif single_ok and dual_ok and used_dry_run:
            c_status = "部分完成"
            c_detail = "单卡与双卡流程已通过 dry-run 验证，待摩尔线程实机补做真实推理"
        elif single or dual:
            c_status = "部分完成"
            c_detail = "至少一种规模已执行，但仍需补齐另一种规模的实机测试"
        else:
            c_status = "未完成"
            c_detail = "尚未发现单卡/双卡推理执行结果"
    statuses.append(("C", c_status, c_detail))

    error_free = bool(single and not single["errors"]) if single_only else (bool(single and not single["errors"]) and bool(dual and not dual["errors"]))
    if single_only:
        if single_ok and error_free and not used_dry_run and single_visible:
            d_status = "已完成"
            d_detail = "任务日志未发现硬件识别错误、显存溢出或核心转储"
        elif single_ok and used_dry_run:
            d_status = "部分完成"
            d_detail = "dry-run 日志无异常，但仍需检查摩尔线程实机日志是否存在识别错误或显存问题"
        elif single:
            d_status = "部分完成"
            d_detail = "已有单卡日志产物，但需要在摩尔线程实机日志中进一步确认无异常"
        else:
            d_status = "未完成"
            d_detail = "缺少日志产物"
    else:
        if single_ok and dual_ok and error_free and not used_dry_run and single_visible and dual_visible:
            d_status = "已完成"
            d_detail = "任务日志未发现硬件识别错误、显存溢出或核心转储"
        elif single_ok and dual_ok and used_dry_run:
            d_status = "部分完成"
            d_detail = "dry-run 日志无异常，但仍需检查摩尔线程实机日志是否存在识别错误或显存问题"
        elif single or dual:
            d_status = "部分完成"
            d_detail = "已有日志产物，但需要在摩尔线程实机日志中进一步确认无异常"
        else:
            d_status = "未完成"
            d_detail = "缺少日志产物"
    statuses.append(("D", d_status, d_detail))

    outputs_ready = bool(single and single["outputs_count"] > 0) if single_only else (bool(single and single["outputs_count"] > 0) and bool(dual and dual["outputs_count"] > 0))
    statuses.append(
        (
            "E",
            "已完成" if outputs_ready and not used_dry_run else "部分完成" if (single or dual) else "未完成",
            "已记录任务结束状态并输出真实推理结果" if outputs_ready and not used_dry_run else "已生成流程验证输出，待补齐真实推理结果" if (single or dual) else "缺少推理输出",
        )
    )

    if single_only:
        if single_ok and single_valid and package_ready and single_visible:
            f_status = "已完成"
            f_detail = "满足单卡任务成功完成且结果正确的判定条件"
        elif single_ok and package_ready and single_visible:
            f_status = "部分完成"
            f_detail = "单卡流程已跑通，但输出结果未全部通过内容校验"
        elif single_ok:
            f_status = "部分完成"
            f_detail = "单卡流程已跑通，但当前机器不是目标摩尔线程实机环境，需补最终适配性验证"
        else:
            f_status = "未完成"
            f_detail = "尚未满足单卡任务成功完成且结果正确的判定条件"
    else:
        if single_ok and dual_ok and single_valid and dual_valid and package_ready and single_visible and dual_visible:
            f_status = "已完成"
            f_detail = "满足任务成功完成且结果正确的判定条件"
        elif single_ok and dual_ok and package_ready and single_visible and dual_visible:
            f_status = "部分完成"
            f_detail = "单卡与双卡流程都已跑通，但输出结果未全部通过内容校验"
        elif single_ok and dual_ok:
            f_status = "部分完成"
            f_detail = "流程已跑通，但当前机器不是目标摩尔线程实机环境，需补最终适配性验证"
        else:
            f_status = "未完成"
            f_detail = "尚未满足任务成功完成且结果正确的判定条件"
    statuses.append(("F", f_status, f_detail))
    return statuses


def build_markdown(output_path, preflight, single, dual, single_only=False):
    statuses = classify(preflight, single, dual, single_only=single_only)
    generated_at = datetime.now(timezone.utc).isoformat()
    sample_lines = []
    mode_payloads = [("单卡", single)] if single_only else [("单卡", single), ("双卡", dual)]
    for label, payload in mode_payloads:
        if not payload:
            continue
        for worker in payload.get("worker_payloads", []):
            for item in worker.get("results", []):
                gen_ms = item.get('gen_ms')
                gen_str = f"{gen_ms:.3f}" if (gen_ms is not None) else ""
                sample_lines.append(
                    f"| {label} | {item['id']} | {item['response'].replace(chr(10), '<br>')} | {item.get('raw_response', '').replace(chr(10), '<br>')} | {'通过' if item.get('validation_passed') else '未通过'} | {gen_str} |"
                )
    lines = ["# 5.1.5任务进展", "", f"- 生成时间：{generated_at}", f"- 任务标识：MTT-INFER-RUN-TEST", "- 任务名称：摩尔线程架构上推理任务运行测试", "", "## 当前结论", ""]

    if all(item[1] == "已完成" for item in statuses):
        if single_only:
            lines.append("本次已在 Intel CPU + 摩尔线程 GPU 环境下完成单卡推理任务运行验证。单卡流程执行成功，日志无硬件识别异常，且输出结果通过预设内容校验，可作为 5.1.5 正式测试记录提交。")
        else:
            lines.append("本次已在 Intel CPU + 摩尔线程 GPU 环境下完成推理任务运行验证。单卡与单机双卡均成功执行，日志无硬件识别异常，且输出结果通过预设内容校验，可作为 5.1.5 正式测试记录提交。")
    else:
        lines.append("当前工程、脚本与报告链路已完成，但仍有部分指标需要结合实机结果或输出校验进一步确认。")

    lines.extend(
        [
            "",
            "## A-F 指标完成情况",
            "",
            "| 指标 | 状态 | 说明 |",
            "| --- | --- | --- |",
        ]
    )
    for item in statuses:
        lines.append(status_line(*item))

    lines.extend(
        [
            "",
            "## 产物位置",
            "",
            f"- 预检查：`{os.path.dirname(output_path)}/preflight/preflight.json`",
            f"- 单卡结果：`{os.path.dirname(output_path)}/single/summary.json`",
        ]
    )
    if not single_only:
        lines.append(f"- 双卡结果：`{os.path.dirname(output_path)}/dual/summary.json`")

    single_gen = []
    model_load_ms = None
    if single:
        for worker in single.get("worker_payloads", []):
            if worker.get("model_load_ms") is not None:
                model_load_ms = float(worker.get("model_load_ms"))
            for item in worker.get("results", []):
                if item.get("gen_ms") is not None:
                    single_gen.append(float(item.get("gen_ms")))
    single_gen_median = median(single_gen)
    single_gen_avg = (sum(single_gen) / len(single_gen)) if single_gen else None

    lines.extend(
        [
            "",
            "## 关键校验说明",
            "",
            f"- 单卡输出校验：{'通过' if single and single.get('validation_passed') else '未通过'}",
            f"- 单卡命中条数：{single.get('validated_outputs_count', 0) if single else 0}/{single.get('outputs_count', 0) if single else 0}",
            "- 校验规则：保留原始响应，同时对响应做标准化答案提取；仅当提取出的最终答案与标准答案完全一致时判定为通过。",
            "- 测量说明：为降低端到端运行中模型加载等外部因素对延时统计的干扰，若可用则优先使用每次生成的内部计时（gen_ms），并以中位数作为报告值。",
        ]
    )
    if not single_only:
        lines.append(f"- 双卡输出校验：{'通过' if dual and dual.get('validation_passed') else '未通过'}")
        lines.append(f"- 双卡命中条数：{dual.get('validated_outputs_count', 0) if dual else 0}/{dual.get('outputs_count', 0) if dual else 0}")
    if model_load_ms is not None:
        lines.append(f"- 单卡模型加载耗时(ms)：{model_load_ms:.3f}")
    if single_gen_median is not None:
        lines.append(f"- 单卡生成耗时中位数(ms)：{single_gen_median:.3f}")
    if single_gen_avg is not None:
        lines.append(f"- 单卡生成耗时均值(ms)：{single_gen_avg:.3f}")

    lines.extend(
        [
            "",
            "## 输出结果摘录",
            "",
            "| 规模 | Prompt ID | 标准化答案 | 原始响应 | 校验 | 生成耗时(ms) |",
            "| --- | --- | --- | --- | --- | --- |",
            *sample_lines,
            "",
            "## 如何复线",
            "",
            "1. 在 Intel CPU + 摩尔线程 GPU 服务器上准备好驱动、`torch`、`transformers`、`torch_musa`。",
            "2. 准备官方依赖包，当前实测通过版本为：`torch 2.5.0`、`torch_musa 2.1.1`、`MUSA Toolkit 4.2.0`、`muDNN 3.0.0`、`numpy<2`。",
            "3. 确认模型目录存在，例如 `/path/to/Meta-Llama-3.1-8B`。",
            "4. 执行：",
            "",
            "```bash",
            "cd /path/to/5.1.5",
            "bash setup_musa_env.sh",
            "",
            "# 如当前 shell 未自动继承环境变量，可补一条：",
            "export LD_LIBRARY_PATH=/home/o_mabin/.local/gfortran/usr/lib/x86_64-linux-gnu:/home/o_mabin/.local/openblas/usr/lib/x86_64-linux-gnu/openblas-pthread:/home/o_mabin/.local/musa_toolkits/musa_toolkits_4.2.0/lib:/home/o_mabin/.local/mudnn/mudnn/lib:/usr/local/musa/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}",
            "",
            "bash run_515_suite.sh \\",
            "  --model-path /path/to/Meta-Llama-3.1-8B \\",
            "  --device-type musa \\",
            "  --single-device-ids 0 \\",
            "  --dual-device-ids 0,1 \\",
            "  --single-only",
            "```",
            "",
            "5. 查看 `artifacts/<timestamp>/5.1.5任务进展.md`，确认单卡、日志与输出均为成功。",
            "",
            "## 备注",
            "",
            "- 当前机器若没有摩尔线程设备，脚本会依赖 `dry-run` 完成流程验证。",
            "- 若实机环境已经提供 `mthreads-gmi`，预检查会自动采集设备可见性信息。",
        ]
    )

    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--single-only", action="store_true")
    args = parser.parse_args()

    preflight = load_json(os.path.join(args.artifacts_dir, "preflight", "preflight.json"))
    single = load_json(os.path.join(args.artifacts_dir, "single", "summary.json"))
    dual = load_json(os.path.join(args.artifacts_dir, "dual", "summary.json")) if not args.single_only else None

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    build_markdown(args.output, preflight, single, dual, single_only=args.single_only)
    print(args.output)


if __name__ == "__main__":
    main()
