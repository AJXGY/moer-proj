#!/usr/bin/env python3
"""
5.1.6 结果汇总模块 - 训练结果分析与报告生成
MTT-TRAIN-RUN-TEST Result Summarizer

按照测试步骤 A-F 进行合规性判定
"""

import argparse
import json
import os
from datetime import datetime, timezone


def load_json(path):
    """加载JSON文件"""
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Failed to load {path}: {e}")
        return None


def status_line(name, status, detail):
    """生成状态行"""
    return f"| {name} | {status} | {detail} |"


def classify(preflight, single, dual, tp):
    """
    按照测试步骤 A-F 进行合规性分类
    返回 (step, status, detail) 元组列表
    """
    package_ready = bool(preflight and preflight.get("criteria", {}).get("python_dependencies_ready"))
    single_visible = bool(preflight and preflight.get("criteria", {}).get("single_card_visible"))
    dual_visible = bool(preflight and preflight.get("criteria", {}).get("dual_card_visible"))
    model_exists = bool(preflight and preflight.get("criteria", {}).get("model_ready"))
    
    single_ok = bool(single and single.get("success"))
    dual_ok = bool(dual and dual.get("success"))
    tp_ok = bool(tp and tp.get("success"))
    used_dry_run = bool((single and single.get("dry_run")) or (dual and dual.get("dry_run")))
    
    single_errors = single.get("errors", []) if single else []
    dual_errors = dual.get("errors", []) if dual else []
    
    statuses = []
    
    # Step A: 环境配置检查
    statuses.append((
        "A",
        "已完成" if preflight else "未完成",
        "已输出环境预检查结果" if preflight else "缺少环境预检查产物"
    ))
    
    # Step B: 模型和配置准备
    if model_exists and single_visible:
        b_status = "已完成"
        b_detail = "模型存在，且检查到单卡资源可见；训练执行方法统一切换为 train-infer-estimation runtime"
    elif model_exists:
        b_status = "部分完成"
        b_detail = "模型已就绪，但当前环境未验证到摩尔线程单卡资源可见"
    else:
        b_status = "未完成"
        b_detail = "模型路径不存在或训练任务未执行"
    statuses.append(("B", b_status, b_detail))
    
    # Step C: 训练任务启动与执行
    if single_ok and dual_ok and not used_dry_run and single_visible and dual_visible:
        c_status = "已完成"
        c_detail = "单卡与双卡训练任务均执行成功，且均由 train-infer-estimation 训练 runtime 驱动；TP 补充状态：" + ("已完成" if tp_ok else "未执行")
    elif single_ok and dual_ok and used_dry_run:
        c_status = "部分完成"
        c_detail = "单卡与双卡流程已通过 dry-run 验证，待摩尔线程实机补做真实训练"
    elif single or dual:
        c_status = "部分完成"
        c_detail = "至少一种规模已执行，但仍需补齐另一种规模的实机测试"
    else:
        c_status = "未完成"
        c_detail = "尚未发现单卡/双卡训练执行结果"
    statuses.append(("C", c_status, c_detail))
    
    # Step D: 监控和日志分析
    error_free_single = bool(single and not single_errors)
    error_free_dual = bool(dual and not dual_errors)
    
    if single_ok and dual_ok and error_free_single and error_free_dual and not used_dry_run and single_visible and dual_visible:
        d_status = "已完成"
        d_detail = "任务日志未发现硬件识别错误、显存溢出或核心转储；TP 补充日志状态：" + ("正常" if tp_ok else "未检查")
    elif single_ok and dual_ok and used_dry_run:
        d_status = "部分完成"
        d_detail = "dry-run 日志无异常，但仍需检查摩尔线程实机日志是否存在识别错误或显存问题"
    elif (single or dual) and (error_free_single or error_free_dual):
        d_status = "部分完成"
        d_detail = "已有日志产物，但需要在摩尔线程实机日志中进一步确认无异常"
    else:
        d_status = "未完成"
        d_detail = "缺少日志产物或存在错误"
    statuses.append(("D", d_status, d_detail))
    
    # Step E: 结果验证与输出
    single_outputs = len(single.get("outputs", [])) if single else 0
    dual_outputs = len(dual.get("outputs", [])) if dual else 0
    outputs_ready = bool(single_outputs > 0) and bool(dual_outputs > 0)
    
    statuses.append((
        "E",
        "已完成" if outputs_ready and not used_dry_run else "部分完成" if (single or dual) else "未完成",
        "已记录任务结束状态并输出真实训练结果与 checkpoint；TP 补充状态：" + ("已生成" if tp_ok else "未生成") if outputs_ready and not used_dry_run else "已生成流程验证输出，待补齐真实训练结果" if (single or dual) else "缺少训练输出"
    ))
    
    # Step F: 测试判定
    if single_ok and dual_ok and package_ready and single_visible and dual_visible and error_free_single and error_free_dual:
        f_status = "已完成"
        f_detail = "满足任务成功完成且结果正确的判定条件，当前口径为 train-infer-estimation runtime 实跑"
    elif single_ok and dual_ok and not used_dry_run:
        f_status = "部分完成"
        f_detail = "流程已跑通，但当前机器不是目标摩尔线程实机环境，需补最终适配性验证"
    else:
        f_status = "未完成"
        f_detail = "尚未满足任务成功完成且结果正确的判定条件"
    statuses.append(("F", f_status, f_detail))
    
    return statuses


def build_markdown(output_path, preflight, single, dual, tp):
    """生成Markdown报告"""
    statuses = classify(preflight, single, dual, tp)
    
    md = []
    md.append("# 5.1.6 训练任务运行测试 - 结果汇总报告\n")
    md.append(f"**生成时间**：{datetime.now(timezone.utc).isoformat()}\n\n")
    
    md.append("## 测试概况\n")
    md.append("| 步骤 | 状态 | 详情 |\n")
    md.append("|------|------|------|\n")
    for step, status, detail in statuses:
        md.append(f"| {step} | {status} | {detail} |\n")
    
    md.append("\n## 环境信息\n\n")
    if preflight:
        system_info = preflight.get("details", {}).get("system", {})
        md.append(f"- **操作系统**：{system_info.get('os')}\n")
        md.append(f"- **Python版本**：{system_info.get('python_version')}\n")
        md.append(f"- **处理器**：{system_info.get('processor')}\n")
        
        deps = preflight.get("details", {}).get("dependencies", {})
        md.append(f"\n**Python依赖状态**：\n")
        md.append(f"- 已安装：{', '.join(deps.get('available', []))}\n")
        if deps.get("missing"):
            md.append(f"- 缺失：{', '.join(deps.get('missing', []))}\n")
        
        accelerator = preflight.get("details", {}).get("accelerator", {})
        md.append(f"\n**加速器信息**：\n")
        md.append(f"- 后端类型：{accelerator.get('backend')}\n")
        md.append(f"- 设备数量：{accelerator.get('device_count')}\n")
        if accelerator.get("devices"):
            md.append("- 设备列表：\n")
            for device in accelerator.get("devices"):
                md.append(f"  - GPU{device.get('id')}: {device.get('name')} ({device.get('memory_gb')}GB)\n")
        
        model = preflight.get("details", {}).get("model", {})
        md.append(f"\n**模型信息**：\n")
        md.append(f"- 路径：{model.get('path')}\n")
        md.append(f"- 存在：{model.get('exists')}\n")
        md.append(f"- 大小：{model.get('size_mb')}MB\n")
        md.append(f"- 完整：{model.get('complete')}\n")
    
    md.append("\n## 单卡训练结果\n\n")
    if single:
        md.append(f"- **执行状态**：{'成功' if single.get('success') else '失败'}\n")
        md.append(f"- **Dry-run模式**：{single.get('dry_run', False)}\n")
        md.append(f"- **执行时间**：{single.get('execution_time_seconds', 'N/A')}秒\n")
        if single.get("avg_step_ms") is not None:
            md.append(f"- **平均每 step 时间**：{single.get('avg_step_ms'):.3f} ms\n")
        if single.get("runtime_source"):
            md.append(f"- **训练方法来源**：`{single.get('runtime_source')}`\n")
        md.append(f"- **输出数量**：{len(single.get('outputs', []))}\n")
        if single.get("errors"):
            md.append(f"- **错误信息**：\n")
            for error in single.get("errors"):
                md.append(f"  - {error}\n")
    else:
        md.append("- 尚未执行\n")
    
    md.append("\n## 双卡训练结果\n\n")
    if dual:
        md.append(f"- **执行状态**：{'成功' if dual.get('success') else '失败'}\n")
        md.append(f"- **Dry-run模式**：{dual.get('dry_run', False)}\n")
        md.append(f"- **执行时间**：{dual.get('execution_time_seconds', 'N/A')}秒\n")
        if dual.get("avg_step_ms") is not None:
            md.append(f"- **平均每 step 时间**：{dual.get('avg_step_ms'):.3f} ms\n")
        if dual.get("runtime_source"):
            md.append(f"- **训练方法来源**：`{dual.get('runtime_source')}`\n")
        md.append(f"- **输出数量**：{len(dual.get('outputs', []))}\n")
        if dual.get("errors"):
            md.append(f"- **错误信息**：\n")
            for error in dual.get("errors"):
                md.append(f"  - {error}\n")
    else:
        md.append("- 尚未执行\n")

    md.append("\n## 张量并行训练结果（补充）\n\n")
    if tp:
        md.append(f"- **执行状态**：{'成功' if tp.get('success') else '失败'}\n")
        md.append(f"- **Dry-run模式**：{tp.get('dry_run', False)}\n")
        md.append(f"- **执行时间**：{tp.get('execution_time_seconds', 'N/A')}秒\n")
        if tp.get("avg_step_ms") is not None:
            md.append(f"- **平均每 step 时间**：{tp.get('avg_step_ms'):.3f} ms\n")
        if tp.get("runtime_source"):
            md.append(f"- **训练方法来源**：`{tp.get('runtime_source')}`\n")
        if tp.get("parallel_mode"):
            md.append(f"- **并行模式**：{tp.get('parallel_mode')}\n")
        md.append(f"- **输出数量**：{len(tp.get('outputs', []))}\n")
        if tp.get("errors"):
            md.append(f"- **错误信息**：\n")
            for error in tp.get("errors"):
                md.append(f"  - {error}\n")
    else:
        md.append("- 本次未执行 TP 补充训练\n")
    
    md.append("\n## 最终判定\n\n")
    overall_status = "已完成" if all(s[1] == "已完成" for s in statuses) else "部分完成" if any(s[1] in ["已完成", "部分完成"] for s in statuses) else "未完成"
    md.append(f"**总体状态**：{overall_status}\n\n")
    md.append(f"**满足指标1.2要求**：{overall_status == '已完成'}\n")
    
    # 写入文件
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(''.join(md))
    
    print(''.join(md))


def main():
    parser = argparse.ArgumentParser(description='5.1.6 Training Result Summarizer')
    parser.add_argument('--output', type=str, required=True, help='Output Markdown file')
    parser.add_argument('--preflight', type=str, help='Preflight check JSON')
    parser.add_argument('--single', type=str, help='Single card result JSON')
    parser.add_argument('--dual', type=str, help='Dual card result JSON')
    parser.add_argument('--tp', type=str, help='Tensor parallel result JSON')
    args = parser.parse_args()
    
    preflight = load_json(args.preflight) if args.preflight else None
    single = load_json(args.single) if args.single else None
    dual = load_json(args.dual) if args.dual else None
    tp = load_json(args.tp) if args.tp else None

    build_markdown(args.output, preflight, single, dual, tp)
    return 0


if __name__ == '__main__':
    exit(main())
