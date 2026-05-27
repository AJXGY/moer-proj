#!/usr/bin/env python3
import json
import os
import subprocess
from datetime import datetime, timezone


ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(ROOT, "../../.."))
ARTIFACT = os.path.join(ROOT, "artifacts", "20260415T113500Z")
TOOL_ROOT = os.path.join(REPO_ROOT, "projects", "shared", "train-infer-estimation")
TOOL_ENTRY = os.path.join(TOOL_ROOT, "torch_operator_mvp.py")


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def error_percent(real_ms, pred_ms):
    return abs(real_ms - pred_ms) / real_ms * 100.0


def fit_alpha_beta(points):
    if len(points) < 2:
        raise RuntimeError("Need at least two reference points for linear communication model")
    mean_size = sum(size for size, _ in points) / len(points)
    mean_time = sum(time_ms for _, time_ms in points) / len(points)
    denom = sum((size - mean_size) ** 2 for size, _ in points)
    if denom == 0:
        raise RuntimeError("Reference points must have at least two distinct message sizes")
    beta = sum((size - mean_size) * (time_ms - mean_time) for size, time_ms in points) / denom
    alpha = mean_time - beta * mean_size
    return alpha, beta


def build_leave_one_out_model(operators, target_op):
    references = sorted(
        (
            op
            for op in operators
            if op["kind"] == target_op["kind"] and op["id"] != target_op["id"]
        ),
        key=lambda item: item["bytes"],
    )
    if len(references) < 2:
        raise RuntimeError(f"Need at least two references for op={target_op['id']}")
    alpha, beta = fit_alpha_beta(
        [(op["bytes"], op["real"]["avg_ms"]) for op in references]
    )
    return {
        "alpha_ms": alpha,
        "beta_ms_per_byte": beta,
        "reference_ids": [op["id"] for op in references],
        "reference_bytes": [op["bytes"] for op in references],
        "policy": "leave_one_out",
    }


def predictor_dir(op_id):
    return os.path.join(ARTIFACT, "predictor", op_id)


def run_operator_predictor(request_path, output_dir):
    cmd = [
        "python3",
        TOOL_ENTRY,
        "--request-json",
        request_path,
        "--output-dir",
        output_dir,
    ]
    completed = subprocess.run(
        cmd,
        cwd=TOOL_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "operator predictor failed:\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )


def build_request(op, model, bench):
    return {
        "operator": {
            "id": op["id"],
            "name": op["name"],
            "kind": op["kind"],
            "dtype": op["dtype"],
            "bytes": op["bytes"],
        },
        "parallel_config": {
            "mode": "dual_card",
            "world_size": 2,
            "partition_strategy": "replicated",
        },
        "hardware_topology": {
            "device_backend": bench["device_backend"],
            "device_count": bench["device_count"],
            "device_names": bench["device_names"],
            "physical_devices": [0, 1],
            "communication_path": bench.get("communication_path"),
            "distributed_backend": bench.get("distributed_backend"),
            "calibration_override": {
                "alpha_ms": model["alpha_ms"],
                "beta_ms_per_byte": model["beta_ms_per_byte"],
            },
        },
    }


def estimate_with_tool(op, model, bench):
    out_dir = predictor_dir(op["id"])
    os.makedirs(out_dir, exist_ok=True)
    request = build_request(op, model, bench)
    request_path = os.path.join(out_dir, "request.json")
    dump_json(request_path, request)
    run_operator_predictor(request_path, out_dir)
    report = load_json(os.path.join(out_dir, "report.json"))
    return report["estimate"]["predicted_time_ms"], os.path.join(out_dir, "report.json")


def main():
    bench = load_json(os.path.join(ARTIFACT, "benchmark_results.json"))
    ops = bench["operators"]

    per_kind = {}
    for kind in sorted({op["kind"] for op in ops}):
        per_kind[kind] = {
            "operator_count": sum(1 for op in ops if op["kind"] == kind),
            "description": (
                "Each operator is predicted with a per-kind linear model fitted from "
                "the other message sizes only; no operator is used to predict itself."
            ),
        }

    evaluated = []
    for op in ops:
        model = build_leave_one_out_model(ops, op)
        pred, report_path = estimate_with_tool(op, model, bench)
        real = op["real"]["avg_ms"]
        evaluated.append(
            {
                "id": op["id"],
                "name": op["name"],
                "kind": op["kind"],
                "bytes": op["bytes"],
                "point_role": "validation",
                "prediction_source": {
                    "tool": "projects/shared/train-infer-estimation/torch_operator_mvp.py",
                    "report": report_path,
                },
                "calibration_reference_ids": model["reference_ids"],
                "calibration_policy": model["policy"],
                "alpha_ms": model["alpha_ms"],
                "beta_ms_per_byte": model["beta_ms_per_byte"],
                "t_real_ms": real,
                "t_sim_ms": pred,
                "error_percent": error_percent(real, pred),
            }
        )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "task_id": "MTT-COMM-OP-SPACE-TEST",
        "prediction_source": {
            "tool": "projects/shared/train-infer-estimation/torch_operator_mvp.py",
            "request_fields": ["operator", "parallel_config", "hardware_topology"],
            "calibration_policy": "per-operator leave-one-out calibration passed via alpha_ms + beta_ms_per_byte",
        },
        "model_family": "operator tool with per-kind leave-one-out communication calibration override",
        "per_kind_model": per_kind,
        "operators": evaluated,
        "all_within_20_percent": all(
            op["error_percent"] <= 20.0 for op in evaluated
        ),
    }

    dump_json(os.path.join(ARTIFACT, "space_model_results.json"), payload)


if __name__ == "__main__":
    main()
