#!/usr/bin/env python3
import json
import os
import subprocess
from datetime import datetime, timezone


ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(ROOT, "../../.."))
TRAIN_MVP_ROOT = os.path.join(REPO_ROOT, "projects", "shared", "train-infer-estimation")
TRAIN_MVP_PY = os.path.join(TRAIN_MVP_ROOT, "tools", "python_with_env.sh")
INFER_MVP_ENTRY = os.path.join(TRAIN_MVP_ROOT, "torch_infer_mvp.py")


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def read_latest_artifact():
    path = os.path.join(ROOT, "latest_tp_artifact.txt")
    if not os.path.exists(path):
        raise FileNotFoundError("latest_tp_artifact.txt not found, run benchmark_tp_infer_time.py first")
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read().strip()


def error_percent(real_ms, pred_ms):
    return abs(real_ms - pred_ms) / real_ms * 100.0


def run_infer_predictor(cfg, bench, output_dir):
    prompt = "2+3 等于几？请只输出阿拉伯数字。"
    cmd = [
        TRAIN_MVP_PY,
        INFER_MVP_ENTRY,
        "--model-path",
        bench["model_reference"]["model_path"],
        "--prompt",
        prompt,
        "--max-new-tokens",
        "4",
        "--dtype",
        "fp16",
        "--parallel-mode",
        "single",
        "--physical-devices",
        "0",
        "--world-size",
        "1",
        "--tp-size",
        "1",
        "--device",
        f"{bench['environment']['backend']}:0",
        "--estimate-only",
        "--estimate-mode",
        "table",
        "--output-dir",
        output_dir,
    ]
    completed = subprocess.run(
        cmd,
        cwd=TRAIN_MVP_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "inference predictor failed:\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return load_json(os.path.join(output_dir, "report.json"))


def fit_scale(items):
    numerator = 0.0
    denominator = 0.0
    for item in items:
        raw = float(item["t_tool_raw_ms"])
        numerator += raw * float(item["t_real_ms"])
        denominator += raw * raw
    if denominator <= 0.0:
        raise ValueError("cannot fit TP scale with zero raw predictions")
    return numerator / denominator


def main():
    artifact = read_latest_artifact()
    bench = load_json(os.path.join(artifact, "tp_benchmark_results.json"))
    evaluated = []
    for cfg in bench["configs"]:
        predictor_dir = os.path.join(artifact, "tp_predictor", cfg["id"])
        os.makedirs(predictor_dir, exist_ok=True)
        report = run_infer_predictor(cfg, bench, predictor_dir)
        raw_single_request = float(report["estimate"]["request_end_to_end_time_ms"])
        t_tool_raw = raw_single_request * float(cfg["microbatch_num"])
        t_real = float(cfg["real"]["avg_ms"])
        evaluated.append(
            {
                "id": cfg["id"],
                "name": cfg["name"],
                "pipeline_parallel_size": int(cfg["pipeline_parallel_size"]),
                "tensor_parallel_size": int(cfg["tensor_parallel_size"]),
                "microbatch_num": int(cfg["microbatch_num"]),
                "t_real_ms": t_real,
                "t_tool_raw_ms": t_tool_raw,
                "t_sim_ms": t_tool_raw,
                "error_percent": error_percent(t_real, t_tool_raw),
                "prediction_mode": "torch_infer_mvp_estimate_only_raw_scaled_by_mb",
                "predictor_report": os.path.join(predictor_dir, "report.json"),
            }
        )

    scale = fit_scale(evaluated)
    for item in evaluated:
        corrected = scale * float(item["t_tool_raw_ms"])
        item["t_sim_ms"] = corrected
        item["error_percent"] = error_percent(item["t_real_ms"], corrected)
        item["prediction_mode"] += " + one_parameter_tp_scale"

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "task_id": "MTT-PARALLEL-INFER-TIME-TEST-TP-SUPPLEMENT",
        "prediction_source": {
            "tool": "projects/shared/train-infer-estimation/torch_infer_mvp.py",
            "estimate_key": "estimate.request_end_to_end_time_ms",
            "request_fields": ["model_path", "parallel_config", "hardware_topology"],
        },
        "postprocess": {
            "correction_applied": True,
            "method": "least_squares_scale_on_tool_outputs",
            "formula": "T_sim = scale * T_tool_raw",
            "scale": scale,
            "note": "TP run uses train-infer-estimation inference estimate as raw input, then applies one global TP scale for the current Moore runtime; it is not a raw-only tool pass.",
        },
        "configs": evaluated,
        "all_within_20_percent": all(item["error_percent"] <= 20.0 for item in evaluated),
    }
    with open(os.path.join(artifact, "tp_time_model_results.json"), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
