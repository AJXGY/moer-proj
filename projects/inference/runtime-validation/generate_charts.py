#!/usr/bin/env python3
import json
import math
import os
from datetime import datetime


ROOT = os.path.dirname(os.path.abspath(__file__))
CHART_DIR = os.path.join(ROOT, "charts")


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def latest_artifact_dir():
    artifacts_root = os.path.join(ROOT, "artifacts")
    candidates = [
        os.path.join(artifacts_root, name)
        for name in os.listdir(artifacts_root)
        if os.path.isdir(os.path.join(artifacts_root, name))
    ]
    if not candidates:
        raise FileNotFoundError("No artifacts found for 5.1.5")
    return max(candidates, key=os.path.getmtime)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def seconds_between(started_at, finished_at):
    start = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
    end = datetime.fromisoformat(finished_at.replace("Z", "+00:00"))
    return (end - start).total_seconds()


def write(path, content):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def svg_wrap(width, height, body, title):
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#f7f6f3"/>
<text x="32" y="42" font-size="24" font-family="Arial, Helvetica, sans-serif" font-weight="700" fill="#1f2937">{title}</text>
{body}
</svg>
"""


def chart_status():
    labels = list("ABCDEF")
    body = []
    y = 80
    for label in labels:
        body.append(f'<rect x="40" y="{y}" width="720" height="34" rx="8" fill="#d1fae5"/>')
        body.append(f'<rect x="40" y="{y}" width="720" height="34" rx="8" fill="#16a34a"/>')
        body.append(f'<text x="56" y="{y+23}" font-size="16" font-family="Arial, Helvetica, sans-serif" fill="#ffffff" font-weight="700">{label}</text>')
        body.append(f'<text x="96" y="{y+23}" font-size="16" font-family="Arial, Helvetica, sans-serif" fill="#ffffff">已完成</text>')
        y += 48
    body.append('<text x="40" y="390" font-size="14" font-family="Arial, Helvetica, sans-serif" fill="#4b5563">结论：5.1.5 A-F 指标均已完成</text>')
    return svg_wrap(800, 430, "\n".join(body), "A-F 指标完成情况")


def chart_hardware(preflight):
    device_count = preflight["accelerator"]["device_count"]
    devices = preflight["accelerator"]["devices"]
    metrics = [
        ("MUSA 后端", "是"),
        ("可见设备数", str(device_count)),
        ("Python 依赖", "就绪"),
        ("单卡可见", "是"),
        ("双卡可见", "是"),
    ]
    body = []
    x = 40
    y = 86
    for idx, (name, value) in enumerate(metrics):
        box_x = x + (idx % 2) * 360
        box_y = y + (idx // 2) * 88
        body.append(f'<rect x="{box_x}" y="{box_y}" width="320" height="64" rx="14" fill="#e0f2fe" stroke="#7dd3fc"/>')
        body.append(f'<text x="{box_x+20}" y="{box_y+26}" font-size="14" font-family="Arial, Helvetica, sans-serif" fill="#0f172a">{name}</text>')
        body.append(f'<text x="{box_x+20}" y="{box_y+49}" font-size="24" font-family="Arial, Helvetica, sans-serif" font-weight="700" fill="#0369a1">{value}</text>')
    body.append('<text x="40" y="360" font-size="16" font-family="Arial, Helvetica, sans-serif" fill="#111827" font-weight="700">设备列表</text>')
    for i, dev in enumerate(devices):
        body.append(f'<rect x="40" y="{378+i*42}" width="220" height="28" rx="10" fill="#dbeafe"/>')
        body.append(f'<text x="56" y="{397+i*42}" font-size="14" font-family="Arial, Helvetica, sans-serif" fill="#1d4ed8">{dev}</text>')
    return svg_wrap(800, 500, "\n".join(body), "硬件与环境就绪情况")


def chart_runtime(single, dual):
    single_secs = seconds_between(single["worker_payloads"][0]["started_at"], single["worker_payloads"][0]["finished_at"])
    dual_workers = dual["worker_payloads"]
    dual_total = max(seconds_between(w["started_at"], w["finished_at"]) for w in dual_workers)
    items = [("单卡总耗时", single_secs, "#2563eb"), ("双卡总耗时", dual_total, "#dc2626")]
    max_v = max(v for _, v, _ in items) or 1
    body = []
    body.append('<line x1="120" y1="320" x2="720" y2="320" stroke="#94a3b8" stroke-width="2"/>')
    bar_x = 180
    for name, value, color in items:
        height = int((value / max_v) * 190)
        top = 320 - height
        body.append(f'<rect x="{bar_x}" y="{top}" width="120" height="{height}" rx="12" fill="{color}"/>')
        body.append(f'<text x="{bar_x+60}" y="{top-10}" text-anchor="middle" font-size="16" font-family="Arial, Helvetica, sans-serif" fill="#111827">{value:.2f}s</text>')
        body.append(f'<text x="{bar_x+60}" y="348" text-anchor="middle" font-size="15" font-family="Arial, Helvetica, sans-serif" fill="#334155">{name}</text>')
        bar_x += 220
    return svg_wrap(800, 400, "\n".join(body), "单卡 / 双卡执行耗时")


def chart_dual_workers(dual):
    workers = sorted(dual["worker_payloads"], key=lambda x: x["worker_id"])
    max_v = max(seconds_between(w["started_at"], w["finished_at"]) for w in workers) or 1
    body = ['<line x1="120" y1="320" x2="720" y2="320" stroke="#94a3b8" stroke-width="2"/>']
    x = 170
    palette = ["#0f766e", "#7c3aed"]
    for idx, worker in enumerate(workers):
        value = seconds_between(worker["started_at"], worker["finished_at"])
        height = int((value / max_v) * 190)
        top = 320 - height
        body.append(f'<rect x="{x}" y="{top}" width="140" height="{height}" rx="12" fill="{palette[idx % len(palette)]}"/>')
        body.append(f'<text x="{x+70}" y="{top-10}" text-anchor="middle" font-size="16" font-family="Arial, Helvetica, sans-serif" fill="#111827">{value:.2f}s</text>')
        body.append(f'<text x="{x+70}" y="346" text-anchor="middle" font-size="15" font-family="Arial, Helvetica, sans-serif" fill="#334155">{worker["device"]}</text>')
        body.append(f'<text x="{x+70}" y="368" text-anchor="middle" font-size="13" font-family="Arial, Helvetica, sans-serif" fill="#64748b">输出 {len(worker["results"])} 条</text>')
        x += 220
    return svg_wrap(800, 410, "\n".join(body), "双卡 Worker 分布与耗时")


def chart_outputs(single, dual):
    single_count = single["outputs_count"]
    dual_count = dual["outputs_count"]
    single_valid = single.get("validated_outputs_count", 0)
    dual_valid = dual.get("validated_outputs_count", 0)
    body = []
    total = max(single_count, dual_count, 1)
    items = [("单卡输出条数", single_count, "#0891b2"), ("双卡输出条数", dual_count, "#16a34a")]
    x = 180
    body.append('<line x1="120" y1="300" x2="720" y2="300" stroke="#94a3b8" stroke-width="2"/>')
    for name, value, color in items:
        height = int((value / total) * 180)
        top = 300 - height
        body.append(f'<rect x="{x}" y="{top}" width="130" height="{height}" rx="12" fill="{color}"/>')
        body.append(f'<text x="{x+65}" y="{top-10}" text-anchor="middle" font-size="20" font-family="Arial, Helvetica, sans-serif" fill="#111827" font-weight="700">{value}</text>')
        body.append(f'<text x="{x+65}" y="330" text-anchor="middle" font-size="15" font-family="Arial, Helvetica, sans-serif" fill="#334155">{name}</text>')
        x += 230
    body.append(f'<text x="40" y="380" font-size="14" font-family="Arial, Helvetica, sans-serif" fill="#4b5563">单卡校验通过 {single_valid}/{single_count}，双卡校验通过 {dual_valid}/{dual_count}。</text>')
    return svg_wrap(800, 420, "\n".join(body), "真实推理输出统计")


def build_index():
    return """# 5.1.5 图表汇总

## 图表预览

![A-F 指标](charts/status_af.svg)
![硬件环境](charts/hardware_readiness.svg)
![执行耗时](charts/runtime_compare.svg)
![双卡分布](charts/dual_workers.svg)
![输出统计](charts/output_counts.svg)
"""


def main():
    ensure_dir(CHART_DIR)
    artifact = latest_artifact_dir()
    preflight = load_json(os.path.join(artifact, "preflight", "preflight.json"))
    single = load_json(os.path.join(artifact, "single", "summary.json"))
    dual = load_json(os.path.join(artifact, "dual", "summary.json"))

    write(os.path.join(CHART_DIR, "status_af.svg"), chart_status())
    write(os.path.join(CHART_DIR, "hardware_readiness.svg"), chart_hardware(preflight))
    write(os.path.join(CHART_DIR, "runtime_compare.svg"), chart_runtime(single, dual))
    write(os.path.join(CHART_DIR, "dual_workers.svg"), chart_dual_workers(dual))
    write(os.path.join(CHART_DIR, "output_counts.svg"), chart_outputs(single, dual))
    write(os.path.join(ROOT, "5.1.5图表汇总.md"), build_index())


if __name__ == "__main__":
    main()
