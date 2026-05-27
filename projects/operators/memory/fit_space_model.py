#!/usr/bin/env python3
import json
import os
import subprocess
from datetime import datetime, timezone


ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(ROOT, "../../.."))
ARTIFACT = os.path.join(ROOT, "artifacts", "20260415T101500Z")
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


def solve_alpha_beta(points):
    (s1, t1), (s2, t2) = points
    beta = (t2 - t1) / (s2 - s1)
    alpha = t1 - beta * s1
    return alpha, beta


def build_kind_model(ops, key):
    kind_ops = sorted((op for op in ops if op["kind"] == key), key=lambda item: item["bytes"])
    if len(kind_ops) < 3:
        raise RuntimeError(f"Need at least three points for kind={key}")
    single_alpha, single_beta = solve_alpha_beta(
        [
            (kind_ops[0]["bytes"], kind_ops[0]["single_card"]["avg_ms"]),
            (kind_ops[-1]["bytes"], kind_ops[-1]["single_card"]["avg_ms"]),
        ]
    )
    dual_alpha, dual_beta = solve_alpha_beta(
        [
            (kind_ops[0]["bytes"], kind_ops[0]["dual_card"]["effective_avg_ms"]),
            (kind_ops[-1]["bytes"], kind_ops[-1]["dual_card"]["effective_avg_ms"]),
        ]
    )
    calibration_ids = {kind_ops[0]["id"], kind_ops[-1]["id"]}
    return {
        "single_card": {
            "alpha_ms": single_alpha,
            "beta_ms_per_byte": single_beta,
            "memory_bandwidth_gbps": 1.0 / single_beta / 1e6,
        },
        "dual_card": {
            "alpha_ms": dual_alpha,
            "beta_ms_per_byte": dual_beta,
            "memory_bandwidth_gbps": 1.0 / dual_beta / 1e6,
        },
        "calibration_ids": sorted(calibration_ids),
    }


def predictor_dir(op_id, scale):
    return os.path.join(ARTIFACT, "predictor", scale, op_id)


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


def build_request(op, scale, model, bench):
    world_size = 1 if scale == "single_card" else 2
    return {
        "operator": {
            "id": op["id"],
            "name": op["name"],
            "kind": op["kind"],
            "llama_component": op["llama_component"],
            "dtype": op["dtype"],
            "bytes": op["bytes"],
        },
        "parallel_config": {
            "mode": scale,
            "world_size": world_size,
            "partition_strategy": "replicated",
        },
        "hardware_topology": {
            "device_backend": bench["device_backend"],
            "device_count": bench["device_count"],
            "device_names": bench["device_names"],
            "physical_devices": list(range(world_size)),
            "calibration_override": {
                "memory_bandwidth_gbps": model[scale]["memory_bandwidth_gbps"],
                "alpha_ms": model[scale]["alpha_ms"],
            },
        },
    }


def estimate_with_tool(op, scale, model, bench):
    out_dir = predictor_dir(op["id"], scale)
    os.makedirs(out_dir, exist_ok=True)
    request = build_request(op, scale, model, bench)
    request_path = os.path.join(out_dir, "request.json")
    dump_json(request_path, request)
    run_operator_predictor(request_path, out_dir)
    report = load_json(os.path.join(out_dir, "report.json"))
    return report["estimate"]["predicted_time_ms"], os.path.join(out_dir, "report.json")


def main():
    bench = load_json(os.path.join(ARTIFACT, "benchmark_results.json"))
    ops = bench["operators"]
    kinds = sorted({op["kind"] for op in ops})
    per_kind = {kind: build_kind_model(ops, kind) for kind in kinds}

    evaluated = []
    for op in sorted(ops, key=lambda item: (item["kind"], item["bytes"])):
        model = per_kind[op["kind"]]
        point_role = "calibration" if op["id"] in model["calibration_ids"] else "validation"
        single_pred, single_report = estimate_with_tool(op, "single_card", model, bench)
        dual_pred, dual_report = estimate_with_tool(op, "dual_card", model, bench)
        single_real = op["single_card"]["avg_ms"]
        dual_real = op["dual_card"]["effective_avg_ms"]
        evaluated.append(
            {
                "id": op["id"],
                "name": op["name"],
                "kind": op["kind"],
                "bytes": op["bytes"],
                "point_role": point_role,
                "prediction_source": {
                    "tool": "projects/shared/train-infer-estimation/torch_operator_mvp.py",
                    "single_card_report": single_report,
                    "dual_card_report": dual_report,
                },
                "single_card": {
                    "t_real_ms": single_real,
                    "t_sim_ms": single_pred,
                    "error_percent": error_percent(single_real, single_pred),
                },
                "dual_card": {
                    "t_real_ms": dual_real,
                    "t_sim_ms": dual_pred,
                    "error_percent": error_percent(dual_real, dual_pred),
                },
            }
        )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "task_id": "MTT-MEM-OP-SPACE-TEST",
        "prediction_source": {
            "tool": "projects/shared/train-infer-estimation/torch_operator_mvp.py",
            "request_fields": ["operator", "parallel_config", "hardware_topology"],
            "calibration_policy": "per-kind two-point calibration passed via memory_bandwidth_gbps + alpha_ms",
        },
        "model_family": "operator tool with per-kind bandwidth calibration override",
        "single_card_model_gbps": sum(
            per_kind[k]["single_card"]["memory_bandwidth_gbps"] for k in kinds
        ) / len(kinds),
        "dual_card_model_gbps": sum(
            per_kind[k]["dual_card"]["memory_bandwidth_gbps"] for k in kinds
        ) / len(kinds),
        "per_kind_model": per_kind,
        "operators": evaluated,
        "all_within_20_percent": all(
            op["single_card"]["error_percent"] <= 20.0 and op["dual_card"]["error_percent"] <= 20.0
            for op in evaluated
            if op["point_role"] == "validation"
        ),
    }
    dump_json(os.path.join(ARTIFACT, "space_model_results.json"), payload)


if __name__ == "__main__":
    main()
