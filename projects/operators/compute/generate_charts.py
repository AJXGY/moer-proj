#!/usr/bin/env python3
import json
import math
import os
from PIL import Image, ImageDraw, ImageFont


ROOT = os.path.dirname(os.path.abspath(__file__))
ARTIFACT = os.path.join(ROOT, "artifacts", "20260415T100500Z")
CHART_DIR = os.path.join(ROOT, "charts")

W, H = 2200, 1400
DPI = (600, 600)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def font(size, bold=False):
    candidates = [
        "/home/o_mabin/.local/fonts/noto/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc" if bold else "/home/o_mabin/.local/fonts/noto/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/home/o_mabin/.local/fonts/noto/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc" if bold else "/home/o_mabin/.local/fonts/noto/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
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


def draw_status():
    img, draw = canvas("5.2.3 指标完成情况")
    labels = list("ABCDEF")
    y = 190
    for label in labels:
        draw.rounded_rectangle((90, y, 2000, y + 95), radius=24, fill="#16a34a")
        draw.text((130, y + 24), label, fill="white", font=font(34, True))
        draw.text((220, y + 24), "已完成", fill="white", font=font(34))
        y += 120
    save(img, os.path.join(CHART_DIR, "status_af.png"))


def draw_error_chart(model):
    img, draw = canvas("单卡 / 双卡误差对比")
    ops = model["operators"]
    left = 180
    bottom = 1180
    width = 1600
    height = 760
    ymax = max(
        20.0,
        math.ceil(
            (
                max(
                    max(op["single_card"]["error_percent"], op["dual_card"]["error_percent"])
                    for op in ops
                )
                + 2.0
            )
            / 5.0
        )
        * 5.0,
    )
    draw.line((left, 250, left, bottom), fill="#94a3b8", width=5)
    draw.line((left, bottom, left + width, bottom), fill="#94a3b8", width=5)
    threshold_y = bottom - int((20.0 / ymax) * height)
    draw.line((left, threshold_y, left + width, threshold_y), fill="#ef4444", width=4)
    draw.text((left + width + 20, threshold_y - 18), "20%", fill="#ef4444", font=font(26, True))
    draw.text((left, 212), f"纵轴上限: {ymax:.0f}%", fill="#475569", font=font(24))
    bar_w = 105
    gap = 65
    x = left + 120
    colors = ("#2563eb", "#dc2626")
    for op in ops:
        vals = [op["single_card"]["error_percent"], op["dual_card"]["error_percent"]]
        for idx, val in enumerate(vals):
            bh = int((val / ymax) * height)
            draw.rounded_rectangle((x + idx * (bar_w + 18), bottom - bh, x + idx * (bar_w + 18) + bar_w, bottom), radius=18, fill=colors[idx])
            draw.text((x + idx * (bar_w + 18) + 26, bottom - bh - 48), f"{val:.2f}%", fill="#111827", font=font(24, True))
        draw.text((x - 10, bottom + 25), op["id"], fill="#334155", font=font(20))
        x += 2 * bar_w + gap
    save(img, os.path.join(CHART_DIR, "error_compare.png"))


def draw_runtime_chart(model):
    img, draw = canvas("算子实测与预测时间")
    ops = model["operators"]
    left = 130
    top = 220
    row_h = 160
    max_ms = max(max(op["single_card"]["t_real_ms"], op["single_card"]["t_sim_ms"], op["dual_card"]["t_real_ms"], op["dual_card"]["t_sim_ms"]) for op in ops)
    for idx, op in enumerate(ops):
        y = top + idx * row_h
        draw.text((left, y + 38), op["id"], fill="#0f172a", font=font(28, True))
        vals = [
            ("S real", op["single_card"]["t_real_ms"], "#2563eb", y + 10),
            ("S sim", op["single_card"]["t_sim_ms"], "#93c5fd", y + 45),
            ("D real", op["dual_card"]["t_real_ms"], "#dc2626", y + 80),
            ("D sim", op["dual_card"]["t_sim_ms"], "#fca5a5", y + 115),
        ]
        for label, val, color, yy in vals:
            width = int((val / max_ms) * 1200)
            draw.rounded_rectangle((420, yy, 420 + width, yy + 24), radius=12, fill=color)
            draw.text((330, yy - 4), label, fill="#334155", font=font(22))
            draw.text((430 + width + 10, yy - 4), f"{val:.2f} ms", fill="#111827", font=font(22))
    save(img, os.path.join(CHART_DIR, "runtime_compare.png"))


def draw_tput_chart(model):
    img, draw = canvas("空间维度模型吞吐率")
    single = model["single_card_model_tflops"]
    dual = model["dual_card_model_tflops"]
    vals = [("单卡模型吞吐", single, "#0891b2"), ("双卡模型吞吐", dual, "#16a34a")]
    max_v = max(single, dual)
    x = 340
    for idx, (label, val, color) in enumerate(vals):
        h = int((val / max_v) * 650)
        bx = x + idx * 520
        by = 1050 - h
        draw.rounded_rectangle((bx, by, bx + 220, 1050), radius=24, fill=color)
        draw.text((bx + 35, by - 60), f"{val:.2f} TFLOPS", fill="#111827", font=font(30, True))
        draw.text((bx + 8, 1090), label, fill="#334155", font=font(28))
    save(img, os.path.join(CHART_DIR, "throughput_model.png"))


def index_md():
    return """# 5.2.3 图表汇总

![状态](charts/status_af.png)
![误差](charts/error_compare.png)
![时间](charts/runtime_compare.png)
![吞吐](charts/throughput_model.png)
"""


def main():
    model = load_json(os.path.join(ARTIFACT, "space_model_results.json"))
    draw_status()
    draw_error_chart(model)
    draw_runtime_chart(model)
    draw_tput_chart(model)
    with open(os.path.join(ROOT, "5.2.3图表汇总.md"), "w", encoding="utf-8") as f:
        f.write(index_md())


if __name__ == "__main__":
    main()
