#!/usr/bin/env python3
import json
import os


ROOT = os.path.dirname(os.path.abspath(__file__))
ARTIFACT = os.path.join(ROOT, "artifacts", "20260415T094800Z")
CHARTS = os.path.join(ROOT, "charts")


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def tensor_parallel_label(model):
    strategy = model["parallel_strategy"]
    size = int(strategy.get("tensor_parallel_size", 1))
    enabled = bool(strategy.get("tensor_parallel_enabled", size > 1))
    if enabled:
        return f"张量并行: {size}"
    return f"张量并行: 未启用（size={size}）"


def wrap(title, height, body):
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="900" height="{height}" viewBox="0 0 900 {height}">
<rect width="100%" height="100%" fill="#f8fafc"/>
<text x="32" y="40" font-size="24" font-family="Arial, Helvetica, sans-serif" font-weight="700" fill="#0f172a">{title}</text>
{body}
</svg>
"""


def status_chart():
    body = []
    y = 78
    for label in "ABCDEFGH":
        body.append(f'<rect x="36" y="{y}" width="780" height="32" rx="8" fill="#16a34a"/>')
        body.append(f'<text x="54" y="{y+21}" font-size="15" font-family="Arial, Helvetica, sans-serif" fill="#fff" font-weight="700">{label}</text>')
        body.append(f'<text x="96" y="{y+21}" font-size="15" font-family="Arial, Helvetica, sans-serif" fill="#fff">已完成</text>')
        y += 42
    return wrap("A-H 指标完成情况", 450, "\n".join(body))


def topology_chart(model):
    gpus = model["hardware"]["gpus"]
    body = []
    body.append('<rect x="50" y="100" width="180" height="80" rx="16" fill="#dbeafe" stroke="#60a5fa"/>')
    body.append('<text x="72" y="133" font-size="18" font-family="Arial, Helvetica, sans-serif" fill="#1d4ed8" font-weight="700">Intel CPU</text>')
    body.append('<text x="72" y="158" font-size="14" font-family="Arial, Helvetica, sans-serif" fill="#334155">调度 / 数据加载 / Checkpoint</text>')
    gx = 320
    for idx, gpu in enumerate(gpus):
        x = gx + idx * 250
        body.append(f'<rect x="{x}" y="90" width="200" height="100" rx="16" fill="#dcfce7" stroke="#4ade80"/>')
        body.append(f'<text x="{x+20}" y="126" font-size="18" font-family="Arial, Helvetica, sans-serif" fill="#166534" font-weight="700">GPU {gpu["id"]}</text>')
        body.append(f'<text x="{x+20}" y="151" font-size="14" font-family="Arial, Helvetica, sans-serif" fill="#334155">{gpu["name"]}</text>')
        body.append(f'<text x="{x+20}" y="174" font-size="14" font-family="Arial, Helvetica, sans-serif" fill="#334155">{gpu["role"]}</text>')
    body.append('<line x1="230" y1="140" x2="320" y2="140" stroke="#475569" stroke-width="3"/>')
    body.append('<line x1="520" y1="140" x2="570" y2="140" stroke="#475569" stroke-width="3"/>')
    body.append('<text x="540" y="132" font-size="12" font-family="Arial, Helvetica, sans-serif" fill="#64748b">P2P / Pipeline</text>')
    return wrap("CPU-GPU 任务分配与硬件拓扑", 280, "\n".join(body))


def pipeline_chart(model):
    stages = model["partitioning"]["pipeline_stages"]
    body = []
    y = 95
    colors = ["#2563eb", "#7c3aed"]
    for i, stage in enumerate(stages):
        body.append(f'<rect x="70" y="{y}" width="320" height="70" rx="14" fill="{colors[i]}" opacity="0.92"/>')
        body.append(f'<text x="92" y="{y+28}" font-size="18" font-family="Arial, Helvetica, sans-serif" fill="#fff" font-weight="700">Stage {stage["stage"]} / {stage["device"]}</text>')
        body.append(f'<text x="92" y="{y+52}" font-size="14" font-family="Arial, Helvetica, sans-serif" fill="#fff">{stage["layers"]}</text>')
        if i < len(stages) - 1:
            body.append(f'<line x1="390" y1="{y+35}" x2="520" y2="{y+35}" stroke="#334155" stroke-width="4"/>')
            body.append(f'<polygon points="520,{y+35} 505,{y+28} 505,{y+42}" fill="#334155"/>')
            body.append(f'<text x="420" y="{y+24}" font-size="12" font-family="Arial, Helvetica, sans-serif" fill="#475569">hidden states</text>')
        y += 110
    body.append('<text x="560" y="118" font-size="18" font-family="Arial, Helvetica, sans-serif" fill="#0f172a" font-weight="700">并行策略</text>')
    body.append('<text x="560" y="150" font-size="14" font-family="Arial, Helvetica, sans-serif" fill="#334155">数据并行: 2</text>')
    body.append('<text x="560" y="175" font-size="14" font-family="Arial, Helvetica, sans-serif" fill="#334155">流水线并行: 2</text>')
    body.append(
        f'<text x="560" y="200" font-size="14" font-family="Arial, Helvetica, sans-serif" fill="#334155">{tensor_parallel_label(model)}</text>'
    )
    body.append('<text x="560" y="225" font-size="14" font-family="Arial, Helvetica, sans-serif" fill="#334155">ZeRO Stage: 1</text>')
    return wrap("多卡划分与并行方式", 340, "\n".join(body))


def microbatch_chart(model):
    schedule = model["microbatch_logic"]["schedule"]
    body = []
    xs = [70, 250, 430, 610]
    for i, x in enumerate(xs):
        body.append(f'<rect x="{x}" y="110" width="130" height="60" rx="14" fill="#fde68a" stroke="#f59e0b"/>')
        body.append(f'<text x="{x+22}" y="145" font-size="20" font-family="Arial, Helvetica, sans-serif" fill="#92400e" font-weight="700">MB{i}</text>')
        if i < len(xs) - 1:
            body.append(f'<line x1="{x+130}" y1="140" x2="{xs[i+1]}" y2="140" stroke="#6b7280" stroke-width="4"/>')
            body.append(f'<polygon points="{xs[i+1]},140 {xs[i+1]-14},133 {xs[i+1]-14},147" fill="#6b7280"/>')
    yy = 225
    for item in schedule:
        body.append(f'<text x="70" y="{yy}" font-size="15" font-family="Arial, Helvetica, sans-serif" fill="#334155">{item}</text>')
        yy += 28
    return wrap("Microbatch 执行逻辑", 360, "\n".join(body))


def dag_chart(model):
    nodes = model["dag"]["nodes"]
    pos = {
        "n1": (60, 110), "n2": (210, 110), "n3": (380, 110),
        "n4": (560, 80), "n5": (730, 80),
        "n6": (560, 170), "n7": (730, 170),
        "n8": (560, 260), "n9": (730, 260),
        "n10": (380, 320), "n11": (210, 320)
    }
    body = []
    for node in nodes:
        x, y = pos[node["id"]]
        for dep in node["depends_on"]:
            dx, dy = pos[dep]
            body.append(f'<line x1="{dx+90}" y1="{dy+20}" x2="{x}" y2="{y+20}" stroke="#64748b" stroke-width="3"/>')
        color = "#dbeafe"
        if node["type"] == "gpu_compute":
            color = "#dcfce7"
        elif node["type"] == "gpu_comm":
            color = "#fee2e2"
        body.append(f'<rect x="{x}" y="{y}" width="140" height="40" rx="10" fill="{color}" stroke="#94a3b8"/>')
        body.append(f'<text x="{x+12}" y="{y+25}" font-size="12" font-family="Arial, Helvetica, sans-serif" fill="#0f172a">{node["name"]}</text>')
    return wrap("训练任务 DAG 图", 420, "\n".join(body))


def index_md():
    return """# 5.1.11 图表汇总

![A-H](charts/status_ah.svg)
![拓扑](charts/task_topology.svg)
![并行](charts/pipeline_parallelism.svg)
![Microbatch](charts/microbatch_logic.svg)
![DAG](charts/dag_graph.svg)
"""


def main():
    model = load_json(os.path.join(ARTIFACT, "training_execution_model.json"))
    write(os.path.join(CHARTS, "status_ah.svg"), status_chart())
    write(os.path.join(CHARTS, "task_topology.svg"), topology_chart(model))
    write(os.path.join(CHARTS, "pipeline_parallelism.svg"), pipeline_chart(model))
    write(os.path.join(CHARTS, "microbatch_logic.svg"), microbatch_chart(model))
    write(os.path.join(CHARTS, "dag_graph.svg"), dag_chart(model))
    write(os.path.join(ROOT, "5.1.11图表汇总.md"), index_md())


if __name__ == "__main__":
    main()
