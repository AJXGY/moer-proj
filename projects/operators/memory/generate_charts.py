#!/usr/bin/env python3
import json
import math
import os
from PIL import Image, ImageDraw, ImageFont


ROOT = os.path.dirname(os.path.abspath(__file__))
ARTIFACT = os.path.join(ROOT, "artifacts", "20260415T101500Z")
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
    for p in candidates:
        if os.path.exists(p):
            return ImageFont.truetype(p, size=size)
    return ImageFont.load_default()


def canvas(title):
    img = Image.new("RGB", (W, H), "#f8fafc")
    draw = ImageDraw.Draw(img)
    draw.text((80, 60), title, fill="#0f172a", font=font(58, True))
    return img, draw


def save(img, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path, dpi=DPI)


def error_axis_max(model):
    values = []
    for op in model["operators"]:
        values.append(op["single_card"]["error_percent"])
        values.append(op["dual_card"]["error_percent"])
    peak = max(values) if values else 20.0
    return max(20.0, math.ceil((peak + 2.0) / 5.0) * 5.0)


def draw_status(model):
    img, draw = canvas("5.2.6 A-F 完成情况")
    rows = [
        ("A", "已完成", "已完成 MUSA 环境配置与服务器联通检查"),
        ("B", "已完成", "已准备 copy、slice、cat 三类访存算子及多规模数据"),
        ("C", "已完成", "已完成单卡与双卡五次实测平均采样"),
        ("D", "已完成", "已建立按算子类别拆分的空间维度模型"),
        ("E", "已完成", "已计算并记录各配置误差值"),
        ("F", "已完成" if model["all_within_20_percent"] else "未完成", "验证点误差均不超过 20%"),
    ]
    y = 170
    for item, state, detail in rows:
        color = "#16a34a" if state == "已完成" else "#dc2626"
        draw.rounded_rectangle((90, y, 2070, y + 120), radius=28, fill=color)
        draw.text((130, y + 22), item, fill="white", font=font(36, True))
        draw.text((250, y + 20), state, fill="white", font=font(34, True))
        draw.text((460, y + 24), detail, fill="white", font=font(28))
        y += 150
    save(img, os.path.join(CHART_DIR, "status_af.png"))


def draw_error(model):
    img, draw = canvas("访存算子误差对比（标定点与验证点）")
    ops = model["operators"]
    left, bottom, width, height = 180, 1180, 1700, 760
    ymax = error_axis_max(model)
    draw.line((left, 250, left, bottom), fill="#94a3b8", width=5)
    draw.line((left, bottom, left + width, bottom), fill="#94a3b8", width=5)
    threshold_y = bottom - int((20.0 / ymax) * height)
    draw.line((left, threshold_y, left + width, threshold_y), fill="#ef4444", width=4)
    draw.text((left + width + 20, threshold_y - 18), "20%", fill="#ef4444", font=font(26, True))
    draw.text((left, 210), f"纵轴上限: {ymax:.0f}%", fill="#475569", font=font(24))
    x = left + 50
    for op in ops:
        vals = [op["single_card"]["error_percent"], op["dual_card"]["error_percent"]]
        is_validation = op["point_role"] == "validation"
        cols = ["#2563eb", "#dc2626"] if is_validation else ["#94a3b8", "#cbd5e1"]
        for idx, val in enumerate(vals):
            bh = int((val / ymax) * height)
            bx = x + idx * 148
            draw.rounded_rectangle((bx, bottom - bh, bx + 132, bottom), radius=16, fill=cols[idx])
            draw.text((bx + 14, bottom - bh - 42), f"{val:.2f}%", fill="#111827", font=font(20, True))
        draw.text((x - 4, bottom + 18), op["id"], fill="#334155", font=font(16))
        draw.text((x + 18, bottom + 48), "标定" if not is_validation else "验证", fill="#64748b", font=font(18))
        x += 250
    save(img, os.path.join(CHART_DIR, "error_compare.png"))


def draw_runtime(model):
    img, draw = canvas("访存算子实测与预测时间")
    ops = model["operators"]
    left, top, row_h = 110, 180, 110
    max_ms = max(
        max(op["single_card"]["t_real_ms"], op["single_card"]["t_sim_ms"], op["dual_card"]["t_real_ms"], op["dual_card"]["t_sim_ms"])
        for op in ops
    )
    for idx, op in enumerate(ops):
        y = top + idx * row_h
        draw.text((left, y + 24), op["id"], fill="#0f172a", font=font(20, True))
        vals = [
            ("S real", op["single_card"]["t_real_ms"], "#2563eb", y + 2),
            ("S sim", op["single_card"]["t_sim_ms"], "#93c5fd", y + 28),
            ("D real", op["dual_card"]["t_real_ms"], "#dc2626", y + 54),
            ("D sim", op["dual_card"]["t_sim_ms"], "#fca5a5", y + 80),
        ]
        for label, val, color, yy in vals:
            width = int((val / max_ms) * 1160)
            draw.rounded_rectangle((500, yy, 500 + width, yy + 18), radius=9, fill=color)
            draw.text((410, yy - 2), label, fill="#334155", font=font(18))
            draw.text((510 + width, yy - 2), f"{val:.3f} ms", fill="#111827", font=font(18))
    save(img, os.path.join(CHART_DIR, "runtime_compare.png"))


def draw_bw(model):
    img, draw = canvas("空间维度模型带宽摘要")
    vals = [
        ("单卡平均模型带宽", model["single_card_model_gbps"], "#0891b2"),
        ("双卡平均模型带宽", model["dual_card_model_gbps"], "#16a34a"),
    ]
    max_v = max(v for _, v, _ in vals)
    x = 340
    for label, val, color in vals:
        h = int((val / max_v) * 650)
        bx = x
        by = 1050 - h
        draw.rounded_rectangle((bx, by, bx + 240, 1050), radius=24, fill=color)
        draw.text((bx + 20, by - 60), f"{val:.2f} GB/s", fill="#111827", font=font(30, True))
        draw.text((bx + 8, 1090), label, fill="#334155", font=font(26))
        x += 520
    save(img, os.path.join(CHART_DIR, "bandwidth_model.png"))


def main():
    model = load_json(os.path.join(ARTIFACT, "space_model_results.json"))
    draw_status(model)
    draw_error(model)
    draw_runtime(model)
    draw_bw(model)
    with open(os.path.join(ROOT, "5.2.6图表汇总.md"), "w", encoding="utf-8") as f:
        f.write("# 5.2.6 图表汇总\n\n![状态](charts/status_af.png)\n![误差](charts/error_compare.png)\n![时间](charts/runtime_compare.png)\n![带宽](charts/bandwidth_model.png)\n")


if __name__ == "__main__":
    main()
