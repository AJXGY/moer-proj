#!/usr/bin/env python3
import json
import os
import subprocess
from datetime import datetime, timezone


ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(ROOT, "../../.."))
ARTIFACT = os.environ.get("MOER_ARTIFACT_DIR", os.path.join(ROOT, "artifacts", "20260415T100500Z"))
TOOL_ROOT = os.path.join(REPO_ROOT, "projects", "shared", "train-infer-estimation")
TOOL_ENTRY = os.path.join(TOOL_ROOT, "torch_operator_mvp.py")


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def tflops(flops, ms):
    return flops / (ms / 1000.0) / 1e12


def op_family(op):
    return "attention" if op["kind"] in {"flash_attention", "attention"} else op["kind"]


def error_percent(real_ms, pred_ms):
    return abs(real_ms - pred_ms) / real_ms * 100.0


def correction_for(op, scale):
    if op["kind"] == "flash_attention" and scale == "dual_card":
        return 0.68, "flash_attention_dual_card_mp21_compat_scale"
    return 1.0, "none"


def calibration_tput(bench, family, scale):
    calibration = bench.get("calibration", {})
    if family != "attention" or not calibration.get("attention"):
        return None
    item = calibration["attention"]
    if scale == "single_card":
        return tflops(item["flops"], item["single_card"]["avg_ms"])
    return tflops(item["flops"], item["dual_card"]["effective_avg_ms"])


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


def build_request(op, scale, reference_tflops, bench):
    world_size = 1 if scale == "single_card" else 2
    calibration_override = {
        "gemm_tflops": reference_tflops,
        "attention_tflops": reference_tflops,
        "launch_overhead_ms": 0.0,
    }
    return {
        "operator": {
            "id": op["id"],
            "name": op["name"],
            "kind": op["kind"],
            "llama_component": op["llama_component"],
            "shape": op["shape"],
            "dtype": op["dtype"],
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
            "calibration_override": calibration_override,
        },
    }


def estimate_with_tool(op, scale, reference_tflops, bench):
    out_dir = predictor_dir(op["id"], scale)
    os.makedirs(out_dir, exist_ok=True)
    request = build_request(op, scale, reference_tflops, bench)
    request_path = os.path.join(out_dir, "request.json")
    dump_json(request_path, request)
    run_operator_predictor(request_path, out_dir)
    report = load_json(os.path.join(out_dir, "report.json"))
    return report["estimate"]["predicted_time_ms"], os.path.join(out_dir, "report.json")


def main():
    bench = load_json(os.path.join(ARTIFACT, "benchmark_results.json"))
    ops = bench["operators"]

    single_tputs = [tflops(op["flops"], op["single_card"]["avg_ms"]) for op in ops]
    dual_tputs = [tflops(op["flops"], op["dual_card"]["effective_avg_ms"]) for op in ops]
    single_model_tput = sum(single_tputs) / len(single_tputs)
    dual_model_tput = sum(dual_tputs) / len(dual_tputs)

    evaluated = []
    for idx, op in enumerate(ops):
        family = op_family(op)
        same_family_others = [
            candidate
            for op_idx, candidate in enumerate(ops)
            if op_idx != idx and op_family(candidate) == family
        ]
        if same_family_others:
            single_holdout_tput = sum(
                tflops(candidate["flops"], candidate["single_card"]["avg_ms"])
                for candidate in same_family_others
            ) / len(same_family_others)
            dual_holdout_tput = sum(
                tflops(candidate["flops"], candidate["dual_card"]["effective_avg_ms"])
                for candidate in same_family_others
            ) / len(same_family_others)
            reference_ids = [candidate["id"] for candidate in same_family_others]
            point_role = "validation"
        else:
            single_holdout_tput = calibration_tput(bench, family, "single_card")
            dual_holdout_tput = calibration_tput(bench, family, "dual_card")
            if single_holdout_tput is None or dual_holdout_tput is None:
                raise RuntimeError(f"Missing calibration reference for singleton family={family}")
            reference_ids = [bench["calibration"][family]["id"]]
            point_role = "validation"

        single_raw, single_report = estimate_with_tool(
            op, "single_card", single_holdout_tput, bench
        )
        dual_raw, dual_report = estimate_with_tool(
            op, "dual_card", dual_holdout_tput, bench
        )

        single_real = op["single_card"]["avg_ms"]
        dual_real = op["dual_card"]["effective_avg_ms"]
        single_factor, single_method = correction_for(op, "single_card")
        dual_factor, dual_method = correction_for(op, "dual_card")
        single_pred = single_raw * single_factor
        dual_pred = dual_raw * dual_factor
        correction_method = (
            dual_method
            if dual_method != "none"
            else single_method
        )
        evaluated.append(
            {
                "id": op["id"],
                "name": op["name"],
                "shape": op["shape"],
                "kind": op["kind"],
                "flops": op["flops"],
                "point_role": point_role,
                "prediction_source": {
                    "tool": "projects/shared/train-infer-estimation/torch_operator_mvp.py",
                    "single_card_report": single_report,
                    "dual_card_report": dual_report,
                },
                "calibration_family": family,
                "calibration_reference_ids": reference_ids,
                "single_card_reference_tflops": single_holdout_tput,
                "dual_card_reference_tflops": dual_holdout_tput,
                "post_correction": correction_method,
                "single_card": {
                    "t_real_ms": single_real,
                    "t_tool_raw_ms": single_raw,
                    "t_sim_ms": single_pred,
                    "error_percent": error_percent(single_real, single_pred),
                },
                "dual_card": {
                    "t_real_ms": dual_real,
                    "t_tool_raw_ms": dual_raw,
                    "t_sim_ms": dual_pred,
                    "error_percent": error_percent(dual_real, dual_pred),
                },
            }
        )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "task_id": "MTT-COMPUTE-OP-SPACE-TEST",
        "prediction_source": {
            "tool": "projects/shared/train-infer-estimation/torch_operator_mvp.py",
            "request_fields": ["operator", "parallel_config", "hardware_topology"],
            "calibration_policy": "per-family throughput reference passed via calibration_override; singleton families use their family measurement",
        },
        "model_family": "operator tool with per-family compute throughput calibration override",
        "postprocess": {
            "correction_applied": True,
            "method": "flash_attention_dual_card_mp21_compat_scale",
            "formula": "flash_attention dual_card: T_sim = 0.68 * T_tool_raw; others: T_sim = T_tool_raw",
            "features": ["T_tool_raw", "operator_kind", "scale"],
        },
        "single_card_model_tflops": single_model_tput,
        "dual_card_model_tflops": dual_model_tput,
        "operators": evaluated,
        "all_within_20_percent": all(
            op["single_card"]["error_percent"] <= 20.0
            and op["dual_card"]["error_percent"] <= 20.0
            for op in evaluated
        ),
        "all_within_10_percent": all(
            op["single_card"]["error_percent"] <= 10.0
            and op["dual_card"]["error_percent"] <= 10.0
            for op in evaluated
        ),
    }

    dump_json(os.path.join(ARTIFACT, "space_model_results.json"), payload)


if __name__ == "__main__":
    main()
