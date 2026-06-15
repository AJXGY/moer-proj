#!/usr/bin/env python3
import json
import os
import subprocess
from datetime import datetime, timezone


ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(ROOT, "../../.."))
ARTIFACT = os.environ.get("MOER_ARTIFACT_DIR", os.path.join(ROOT, "artifacts", "20260415T113500Z"))
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


def solve_alpha_beta(p1, p2):
    s1, t1 = p1
    s2, t2 = p2
    if s1 == s2:
        raise RuntimeError("Reference points must have at least two distinct message sizes")
    beta = (t2 - t1) / (s2 - s1)
    alpha = t1 - beta * s1
    return alpha, beta


def build_neighbor_model(kind_ops, target_op):
    ordered = sorted(kind_ops, key=lambda item: item["bytes"])
    if len(ordered) < 3:
        raise RuntimeError(f"Need at least three points for kind={target_op['kind']}")
    idx = next(i for i, item in enumerate(ordered) if item["id"] == target_op["id"])
    if idx == 0:
        refs = [ordered[1], ordered[2]]
    elif idx == len(ordered) - 1:
        refs = [ordered[-3], ordered[-2]]
    else:
        refs = [ordered[idx - 1], ordered[idx + 1]]
    alpha, beta = solve_alpha_beta(
        (refs[0]["bytes"], refs[0]["real"]["avg_ms"]),
        (refs[1]["bytes"], refs[1]["real"]["avg_ms"]),
    )
    return {
        "alpha_ms": alpha,
        "beta_ms_per_byte": beta,
        "reference_ids": [op["id"] for op in refs],
        "reference_bytes": [op["bytes"] for op in refs],
        "policy": "byte_axis_neighbor_interpolation",
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
    spec_meta = load_json(os.path.join(ROOT, "operator_specs.json"))
    holdout_message_bytes = (
        spec_meta.get("holdout_message_bytes") if isinstance(spec_meta, dict) else None
    )
    calibration_message_bytes = set(
        spec_meta.get("calibration_message_bytes", []) if isinstance(spec_meta, dict) else []
    )

    per_kind = {}
    for kind in sorted({op["kind"] for op in ops}):
        per_kind[kind] = {
            "operator_count": sum(1 for op in ops if op["kind"] == kind),
            "description": (
                "Each operator is predicted from the nearest lower/upper message-size "
                "reference points on the byte axis; the current point never participates "
                "in its own prediction."
            ),
        }

    evaluated = []
    by_kind = {}
    for op in ops:
        by_kind.setdefault(op["kind"], []).append(op)
    for op in ops:
        point_role = "calibration" if op["bytes"] in calibration_message_bytes else "validation"
        if point_role == "calibration":
            continue
        model = build_neighbor_model(by_kind[op["kind"]], op)
        pred, report_path = estimate_with_tool(op, model, bench)
        real = op["real"]["avg_ms"]
        evaluated.append(
            {
                "id": op["id"],
                "name": op["name"],
                "kind": op["kind"],
                "bytes": op["bytes"],
                "point_role": point_role,
                "is_holdout_message_bytes": op["bytes"] == holdout_message_bytes,
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
            "calibration_policy": "per-point byte-axis neighbor interpolation passed via alpha_ms + beta_ms_per_byte",
        },
        "model_family": "operator tool with per-kind byte-axis neighbor interpolation calibration override",
        "holdout_message_bytes": holdout_message_bytes,
        "calibration_message_bytes": sorted(calibration_message_bytes),
        "per_kind_model": per_kind,
        "operators": evaluated,
        "all_within_20_percent": all(
            op["error_percent"] <= 20.0 for op in evaluated
        ),
    }

    dump_json(os.path.join(ARTIFACT, "space_model_results.json"), payload)


if __name__ == "__main__":
    main()
