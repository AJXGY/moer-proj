from __future__ import annotations

import argparse
import json
import socket
import time
from pathlib import Path

import torch

from mvp_backend import default_device_string
from mvp_calibration import build_calibration
from mvp_train_estimator import benchmark_train_microbatch_ms, estimate_train_iteration
from mvp_types import ExecutionConfig, HardwareCalibration, RankPlacement


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Torch-based training prediction MVP")
    parser.add_argument("--request-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default=default_device_string())
    return parser.parse_args()


def _iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _load_request(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _execution_from_topology(
    hardware_topology: dict, device_backend: str, physical_devices: list[int]
) -> ExecutionConfig:
    host_name = socket.gethostname()
    placements = [
        RankPlacement(
            rank=index,
            host=host_name,
            node_rank=0,
            local_rank=index,
            physical_device=device_id,
        )
        for index, device_id in enumerate(physical_devices)
    ]
    local_device = physical_devices[0] if physical_devices else 0
    world_size = int(hardware_topology.get("world_size", max(len(physical_devices), 1)))
    tp_size = int(hardware_topology.get("tp_size", 1))
    nnodes = int(hardware_topology.get("nnodes", 1))
    topology = str(hardware_topology.get("topology", "local"))
    interconnect = str(hardware_topology.get("interconnect", "local"))
    return ExecutionConfig(
        device_backend=device_backend,
        parallel_mode="single" if world_size <= 1 else "tp",
        physical_devices=physical_devices,
        visible_devices=",".join(str(device_id) for device_id in physical_devices),
        world_size=world_size,
        tp_size=tp_size,
        topology=topology,
        local_topology=topology,
        interconnect=interconnect,
        nnodes=nnodes,
        nproc_per_node=max(len(physical_devices), 1),
        host_name=host_name,
        master_addr=str(hardware_topology.get("master_addr", "127.0.0.1")),
        master_port=int(hardware_topology.get("master_port", 29500)),
        local_device=local_device,
        placements=placements or [
            RankPlacement(
                rank=0,
                host=host_name,
                node_rank=0,
                local_rank=0,
                physical_device=0,
            )
        ],
        collective_bandwidth_gbps=hardware_topology.get("collective_bandwidth_gbps"),
        collective_latency_ms=hardware_topology.get("collective_latency_ms"),
        rank=0,
        local_rank=0,
        node_rank=0,
    )


def write_report(output_dir: Path, report: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    estimate = report["estimate"]
    transfer_ms = estimate["stage_breakdown_ms"]["activation_transfer_ms"]
    transfer_text = "NA" if transfer_ms is None else f"{transfer_ms:.4f}"
    lines = [
        "# Torch Training MVP Report",
        "",
        f"- model: `{report['model']['name']}`",
        f"- device: `{report['calibration']['device_name']}`",
        f"- backend: `{report['execution']['device_backend']}`",
        f"- pipeline parallel size: {estimate['parallel_config']['pipeline_parallel_size']}",
        f"- microbatch num: {estimate['parallel_config']['microbatch_num']}",
        "",
        "## Estimate",
        "",
        f"- microbatch_time_ms: {estimate['microbatch_time_ms']:.4f}",
        f"- train_iteration_time_ms: {estimate['train_iteration_time_ms']:.4f}",
        f"- activation_transfer_ms: {transfer_text}",
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    request = _load_request(args.request_json)
    model_desc = dict(request["model"])
    parallel_cfg = dict(request["parallel_config"])
    hardware_topology = dict(request["hardware_topology"])

    device_text = str(args.device or hardware_topology.get("device", default_device_string()))
    device_backend = device_text.split(":", 1)[0] if ":" in device_text else device_text
    physical_devices = list(hardware_topology.get("physical_devices", [0]))
    runtime_index = int(device_text.split(":", 1)[1]) if ":" in device_text else int(physical_devices[0] if physical_devices else 0)
    device = torch.device(device_backend, runtime_index)
    dtype_name = str(parallel_cfg.get("dtype") or model_desc.get("dtype") or "float16").lower()
    dtype = torch.bfloat16 if dtype_name in {"bf16", "bfloat16"} else torch.float16

    disable_runtime_probe = bool(request.get("disable_runtime_probe"))
    direct_profile_mode = (
        request.get("runtime_profile") is None
        and model_desc.get("train_workload") == "llama_backbone_probe"
        and not disable_runtime_probe
    )
    if request.get("skip_calibration") or direct_profile_mode:
        calibration = HardwareCalibration(
            device_name=f"{device_backend}:profile_reuse",
            device_index=runtime_index,
            gemm_tflops=1.0,
            attention_tflops=1.0,
            memory_bandwidth_gbps=1.0,
            launch_overhead_ms=0.0,
        )
    else:
        calibration = build_calibration(dtype=dtype, device=device)
    execution = _execution_from_topology(
        hardware_topology=hardware_topology,
        device_backend=device_backend,
        physical_devices=physical_devices,
    )
    runtime_profile = request.get("runtime_profile")
    profile_source = "request_runtime_profile" if runtime_profile is not None else "analytical_only" if disable_runtime_probe else "online_runtime_probe"
    if runtime_profile is None:
        if not disable_runtime_probe:
            profile_runs = 5 if model_desc.get("train_workload") == "llama_backbone_probe" else 3
            runtime_profile = benchmark_train_microbatch_ms(
                model_desc=model_desc,
                parallel_cfg=parallel_cfg,
                device_backend=device_backend,
                runs=profile_runs,
            )
    estimate = estimate_train_iteration(
        model_desc=model_desc,
        parallel_cfg=parallel_cfg,
        hardware_topology=hardware_topology,
        calibration=calibration,
        execution=execution,
        runtime_profile=runtime_profile,
    )
    report = {
        "created_at": _iso_now(),
        "request": request,
        "model": model_desc,
        "execution": {
            "device_backend": execution.device_backend,
            "world_size": execution.world_size,
            "tp_size": execution.tp_size,
            "topology": execution.topology,
            "physical_devices": execution.physical_devices,
        },
        "calibration": {
            "device_name": calibration.device_name,
            "device_index": calibration.device_index,
            "gemm_tflops": calibration.gemm_tflops,
            "attention_tflops": calibration.attention_tflops,
            "memory_bandwidth_gbps": calibration.memory_bandwidth_gbps,
            "launch_overhead_ms": calibration.launch_overhead_ms,
        },
        "estimate": estimate,
        "profile_source": profile_source,
    }
    write_report(Path(args.output_dir), report)


if __name__ == "__main__":
    main()
