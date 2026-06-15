#!/usr/bin/env python3
import json
import math
import os
from PIL import Image, ImageDraw, ImageFont


ROOT = os.path.dirname(os.path.abspath(__file__))
ARTIFACT = os.environ.get("MOER_ARTIFACT_DIR", os.path.join(ROOT, "artifacts", "20260415T113500Z"))
CHART_DIR = os.path.join(ROOT, "charts")
W, H = 2200, 1400
DPI = (600, 600)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def font(size, bold=False):
    candidates = [
        "/home/o_mabin/.local/fonts/noto/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
        if bold else "/home/o_mabin/.local/fonts/noto/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def canvas(title):
    img = Image.new("RGB", (W, H), "#f8fafc")
    draw = ImageDraw.Draw(img)
    draw.text((80, 60), title, fill="#0f172a", font=font(58, True))
    return img, draw


def save(img, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path, dpi=DPI)


def draw_status(model):
    img, draw = canvas("5.2.9 A-F 完成情况")
    statuses = [
        ("A", "已完成", "已配置 MUSA 环境、双进程 gloo 通信与服务器联通"),
        ("B", "已完成", "已准备 7 类通信算子与 26 个消息规模测试点"),
        ("C", "已完成", "已在单机两卡上五次运行取均值"),
        ("D", "已完成", "已建立算子级空间维度模型并输出 T_sim"),
        ("E", "已完成", "已计算误差并写入结果文件"),
        ("F", "已完成" if model["all_within_20_percent"] else "未完成", "所有验证点误差均不超过 20%"),
    ]
    y = 170
    for item, state, detail in statuses:
        color = "#16a34a" if state == "已完成" else "#dc2626"
        draw.rounded_rectangle((90, y, 2070, y + 120), radius=28, fill=color)
        draw.text((130, y + 22), item, fill="white", font=font(36, True))
        draw.text((250, y + 20), state, fill="white", font=font(34, True))
        draw.text((460, y + 24), detail, fill="white", font=font(30))
        y += 150
    save(img, os.path.join(CHART_DIR, "status_af.png"))


def draw_runtime(model):
    img, draw = canvas("通信算子 T_real / T_sim 对比")
    ops = [op for op in model["operators"] if op["point_role"] == "validation"]
    left, top, row_h = 140, 210, 220
    max_ms = max(max(op["t_real_ms"], op["t_sim_ms"]) for op in ops)
    for idx, op in enumerate(ops):
        y = top + idx * row_h
        draw.text((left, y), op["id"], fill="#0f172a", font=font(24, True))
        draw.text((left, y + 36), f"{op['kind']} / {op['bytes'] / (1024 * 1024):.0f} MB / validation", fill="#475569", font=font(22))
        real_w = int((op["t_real_ms"] / max_ms) * 1200)
        sim_w = int((op["t_sim_ms"] / max_ms) * 1200)
        draw.rounded_rectangle((560, y + 8, 560 + real_w, y + 38), radius=12, fill="#2563eb")
        draw.rounded_rectangle((560, y + 56, 560 + sim_w, y + 86), radius=12, fill="#93c5fd")
        draw.text((440, y + 4), "T_real", fill="#1e3a8a", font=font(22, True))
        draw.text((440, y + 52), "T_sim", fill="#1e3a8a", font=font(22, True))
        draw.text((580 + real_w, y + 2), f"{op['t_real_ms']:.3f} ms", fill="#111827", font=font(22))
        draw.text((580 + sim_w, y + 50), f"{op['t_sim_ms']:.3f} ms", fill="#111827", font=font(22))
    save(img, os.path.join(CHART_DIR, "runtime_compare.png"))


def draw_error(model):
    img, draw = canvas("通信验证点误差分布")
    ops = [op for op in model["operators"] if op["point_role"] == "validation"]
    left, bottom, width, height = 170, 1180, 1680, 760
    ymax = max(20.0, math.ceil((max(op["error_percent"] for op in ops) + 2.0) / 5.0) * 5.0)
    draw.line((left, 250, left, bottom), fill="#94a3b8", width=5)
    draw.line((left, bottom, left + width, bottom), fill="#94a3b8", width=5)
    threshold_y = bottom - int((20.0 / ymax) * height)
    draw.line((left, threshold_y, left + width, threshold_y), fill="#ef4444", width=4)
    draw.text((left + width + 20, threshold_y - 18), "20%", fill="#ef4444", font=font(26, True))
    draw.text((left, 210), f"纵轴上限: {ymax:.0f}%", fill="#475569", font=font(24))
    x = left + 70
    for op in ops:
        value = op["error_percent"]
        bar_h = int((value / ymax) * height)
        fill = "#0ea5e9"
        draw.rounded_rectangle((x, bottom - bar_h, x + 170, bottom), radius=18, fill=fill)
        draw.text((x + 18, bottom - bar_h - 44), f"{value:.2f}%", fill="#0f172a", font=font(22, True))
        draw.text((x - 8, bottom + 22), op["id"], fill="#334155", font=font(18))
        draw.text((x + 18, bottom + 48), f"{op['bytes'] / (1024 * 1024):.0f}MB", fill="#64748b", font=font(18))
        x += 280
    save(img, os.path.join(CHART_DIR, "error_compare.png"))


def draw_topology(bench):
    img, draw = canvas("双卡通信测试拓扑")
    draw.rounded_rectangle((140, 250, 860, 1120), radius=30, fill="#dbeafe", outline="#60a5fa", width=5)
    draw.rounded_rectangle((1340, 250, 2060, 1120), radius=30, fill="#dcfce7", outline="#4ade80", width=5)
    draw.text((220, 320), "Rank 0 / musa:0", fill="#1d4ed8", font=font(42, True))
    draw.text((1420, 320), "Rank 1 / musa:1", fill="#166534", font=font(42, True))
    draw.text((220, 430), "GPU Buffer", fill="#0f172a", font=font(34, True))
    draw.text((220, 500), "CPU Staging Buffer", fill="#0f172a", font=font(34, True))
    draw.text((1420, 430), "GPU Buffer", fill="#0f172a", font=font(34, True))
    draw.text((1420, 500), "CPU Staging Buffer", fill="#0f172a", font=font(34, True))
    draw.line((860, 520, 1340, 520), fill="#f97316", width=18)
    draw.line((1340, 700, 860, 700), fill="#fb7185", width=18)
    draw.text((910, 470), "gloo collectives + p2p", fill="#9a3412", font=font(30, True))
    draw.text((900, 740), "cpu staging + musa sync", fill="#9f1239", font=font(30, True))
    draw.text((180, 930), f"通信实现: {bench['communication_path']}", fill="#334155", font=font(28))
    save(img, os.path.join(CHART_DIR, "topology.png"))


def main():
    bench = load_json(os.path.join(ARTIFACT, "benchmark_results.json"))
    model = load_json(os.path.join(ARTIFACT, "space_model_results.json"))
    draw_status(model)
    draw_runtime(model)
    draw_error(model)
    draw_topology(bench)
    with open(os.path.join(ROOT, "5.2.9图表汇总.md"), "w", encoding="utf-8") as f:
        f.write(
            "# 5.2.9 图表汇总\n\n"
            "![状态](charts/status_af.png)\n"
            "![耗时](charts/runtime_compare.png)\n"
            "![误差](charts/error_compare.png)\n"
            "![拓扑](charts/topology.png)\n"
        )


if __name__ == "__main__":
    main()
