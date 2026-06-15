from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import yaml

from config.config_loader import load_config
from mvp_backward_comm import estimate_backward_comm_simple
from mvp_measurement import (
    aggregate_sample_stats,
    benchmark_allreduce_ms,
    cuda_wall_time_ms_phases,
    cuda_wall_time_ms_phases_tp,
)
from mvp_runtime import prepare_inputs_from_shape
from mvp_train_estimator import (
    build_gradient_bytes_mapping,
    estimate_backward_from_graph_nodes,
    estimate_gradient_communication_bytes,
    estimate_model_architecture,
    estimate_optimizer_flops,
)
from mvp_train_graph import extract_training_graphs
from mvp_train_types import TrainCalibration, TrainConfig
from mvp_train_unified_estimator import estimate_train_step_with_tp


DEFAULT_TRAIN_CONFIG_PATH = Path(__file__).resolve().parent / "config" / "train_config.yaml"


def load_train_config_data(config_path: str | None = None) -> dict[str, Any]:
    path = config_path or str(DEFAULT_TRAIN_CONFIG_PATH)
    return load_config(path)


def resolve_train_config_path(config_path: str | None = None) -> Path:
    raw_path = Path(config_path) if config_path else DEFAULT_TRAIN_CONFIG_PATH
    if raw_path.exists():
        return raw_path.resolve()
    return raw_path.resolve()


def save_train_config_data(config_data: dict[str, Any], config_path: str | None = None) -> Path:
    path = resolve_train_config_path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config_data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    return path


def runtime_defaults(config: dict[str, Any]) -> dict[str, Any]:
    defaults = config.get("runtime_defaults", {})
    output_dir = defaults.get("output_dir", {})
    calibration_defaults = config.get("calibration_defaults", {})
    return {
        "model_path": defaults.get("model_path"),
        "prompt": defaults.get("prompt"),
        "mode": defaults.get("mode", "inference"),
        "parallel_mode": defaults.get("parallel_mode", "single"),
        "dtype": defaults.get("dtype", "bf16"),
        "device": defaults.get("device", "cuda:0"),
        "output_dir_inference": output_dir.get("inference", "reports/torch_mvp"),
        "output_dir_train": output_dir.get("train", "reports/torch_train_mvp"),
        "max_new_tokens": defaults.get("max_new_tokens", 4),
        "batch_size": defaults.get("batch_size", 1),
        "seq_len": defaults.get("seq_len", 512),
        "num_epochs": defaults.get("num_epochs", 1),
        "gradient_accumulation_steps": defaults.get("gradient_accumulation_steps", 1),
        "world_size": defaults.get("world_size", 1),
        "tp_size": defaults.get("tp_size", 1),
        "ddp_size": defaults.get("ddp_size", 1),
        "nnodes": defaults.get("nnodes", 1),
        "nproc_per_node": defaults.get("nproc_per_node", 1),
        "node_rank": defaults.get("node_rank", 0),
        "master_addr": defaults.get("master_addr", "127.0.0.1"),
        "master_port": defaults.get("master_port", 29500),
        "interconnect": defaults.get("interconnect", "auto"),
        "dist_timeout_minutes": defaults.get("dist_timeout_minutes", 30),
        "physical_devices": defaults.get("physical_devices", ""),
        "estimate_only": defaults.get("estimate_only", False),
        "no_calibrate": defaults.get("no_calibrate", False),
        "ddp_enabled": defaults.get("ddp_enabled", False),
        "warmup": calibration_defaults.get("warmup_steps", 2),
        "repeat": calibration_defaults.get("measure_steps", 5),
    }


def build_train_calibration(base_calibration: Any, config_data: dict[str, Any]) -> TrainCalibration:
    common_cfg = config_data.get("common", {})
    forward_cfg = common_cfg.get("forward", {})
    backward_cfg = common_cfg.get("backward", {})
    optimizer_cfg = common_cfg.get("optimizer", {})
    tp_cfg = config_data.get("tp", {})
    tp_backward_cfg = tp_cfg.get("backward", {})
    tp_comm_cfg = tp_cfg.get("communication", {})
    ddp_comm_cfg = config_data.get("single_ddp", {}).get("communication", {})
    return TrainCalibration(
        device_name=base_calibration.device_name,
        device_index=base_calibration.device_index,
        gemm_tflops=base_calibration.gemm_tflops,
        attention_tflops=base_calibration.attention_tflops,
        memory_bandwidth_gbps=base_calibration.memory_bandwidth_gbps,
        launch_overhead_ms=base_calibration.launch_overhead_ms,
        backward_compute_scale=backward_cfg.get("compute_scale", 2.5),
        optimizer_scale_factor=optimizer_cfg.get("scale_factor", 1.4),
        effective_tflops_scale=forward_cfg.get("effective_tflops_scale", 0.9),
        backward_efficiency_scale=backward_cfg.get("backward_efficiency_scale", 0.07),
        kernel_overhead_factor=forward_cfg.get("kernel_overhead_factor", 0.15),
        forward_parallelism_factor=forward_cfg.get("parallelism_factor", 0.76),
        parallelism_factor=backward_cfg.get("parallelism_factor", 0.25),
        overhead_scale=backward_cfg.get("overhead_scale", 0.3),
        has_nvlink=True,
        overlap_ratio=tp_comm_cfg.get("overlap_ratio", 0.35),
        tp_backward_efficiency=tp_backward_cfg.get("tp_backward_efficiency", 0.05),
        tp_forward_efficiency=tp_backward_cfg.get("tp_forward_efficiency", 0.05),
        tp_config=tp_cfg,
        gradient_allreduce_tflops=ddp_comm_cfg.get("gradient_allreduce_tflops", 100.0),
    )


def calibration_policy(config_data: dict[str, Any]) -> dict[str, Any]:
    calibration_cfg = config_data.get("calibration", {})
    ratio_cfg = calibration_cfg.get("ratio_clamp", {})
    thresholds = calibration_cfg.get("update_threshold", {})
    limits = calibration_cfg.get("limits", {})
    return {
        "ratio_min": ratio_cfg.get("min", 0.5),
        "ratio_max": ratio_cfg.get("max", 2.0),
        "thresholds": {
            "forward": thresholds.get("forward", 0.05),
            "backward_compute": thresholds.get("backward_compute", 0.05),
            "backward_comm": thresholds.get("backward_comm", 0.05),
            "optimizer": thresholds.get("optimizer", 0.05),
        },
        "limits": limits,
    }


def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


def clamp_ratio(value: float, config_data: dict[str, Any]) -> float:
    policy = calibration_policy(config_data)
    return clamp(value, policy["ratio_min"], policy["ratio_max"])


def limit_for(config_data: dict[str, Any], key: str, fallback_min: float, fallback_max: float) -> tuple[float, float]:
    limits = calibration_policy(config_data)["limits"].get(key, {})
    return limits.get("min", fallback_min), limits.get("max", fallback_max)


def threshold_for(config_data: dict[str, Any], phase: str, fallback: float = 0.05) -> float:
    return calibration_policy(config_data)["thresholds"].get(phase, fallback)


def build_training_inputs(args: Any, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input_ids, attention_mask = prepare_inputs_from_shape(args.batch_size, args.seq_len, device)
    labels = input_ids.clone()
    return input_ids, attention_mask, labels


def estimate_training_phases(
    model: torch.nn.Module,
    args: Any,
    execution: Any,
    train_calibration: TrainCalibration,
    config_data: dict[str, Any],
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> dict[str, Any]:
    arch = estimate_model_architecture(model)
    training_graphs = None
    prefill_estimates = None
    backward_info = None

    if not (execution.parallel_mode == "tp" and execution.tp_size > 1):
        training_graphs = extract_training_graphs(
            model,
            input_ids,
            attention_mask,
            include_backward=True,
            model_name=args.model_path,
        )
        backward_info = training_graphs.backward_info
        prefill_estimates = training_graphs.prefill_export.graph.nodes

    train_config = TrainConfig(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        ddp_enabled=execution.parallel_mode == "ddp",
        tp_size=args.tp_size if execution.parallel_mode == "tp" else 1,
    )

    if execution.parallel_mode == "tp" and execution.tp_size > 1:
        step_estimate = estimate_train_step_with_tp(
            args.batch_size,
            args.seq_len,
            arch,
            train_calibration,
            train_config,
        )
        phase_estimates = {
            "forward": step_estimate.forward_time_ms,
            "backward_compute": step_estimate.backward_summary.compute_time_ms if step_estimate.backward_summary else 0.0,
            "backward_comm": step_estimate.backward_summary.comm_time_ms if step_estimate.backward_summary else 0.0,
            "backward_total": step_estimate.backward_time_ms,
            "optimizer": step_estimate.optimizer_time_ms,
            "total": step_estimate.total_time_ms,
        }
        comm_bytes = estimate_backward_comm_simple(
            num_layers=arch.num_layers,
            hidden_size=arch.hidden_size,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            tp_size=execution.tp_size,
            calibration=train_calibration,
            num_parameters=arch.parameters,
        ).gradient_bytes
        return {
            "arch": arch,
            "step_estimate": step_estimate,
            "phase_estimates": phase_estimates,
            "graph_counts": {"prefill_call_function_nodes": 0, "decode_call_function_nodes": 0},
            "comm_bytes": comm_bytes,
            "training_graphs": training_graphs,
            "prefill_estimates": None,
        }

    from mvp_estimator import estimate_node, finalize_estimate_ordinals

    finalized_prefill = finalize_estimate_ordinals(
        [
            estimate
            for node in training_graphs.prefill_export.graph.nodes
            if (estimate := estimate_node(node, "forward_step", train_calibration)) is not None
        ]
    )
    gradient_bytes_by_scope = None
    if backward_info is not None and hasattr(backward_info, "gradient_infos"):
        gradient_bytes_by_scope = build_gradient_bytes_mapping(backward_info.gradient_infos)
    backward_time_ms, backward_flops, backward_bytes, backward_breakdown = estimate_backward_from_graph_nodes(
        finalized_prefill,
        train_calibration,
        gradient_bytes_by_scope,
    )
    forward_time_ms = sum(n.estimated_time_ms for n in finalized_prefill) * train_calibration.forward_parallelism_factor
    optimizer_flops, optimizer_bytes = estimate_optimizer_flops(
        arch.optimizer_param_count,
        args.batch_size,
        args.seq_len,
        train_calibration,
    )
    optimizer_memory_time_ms = optimizer_bytes / (train_calibration.memory_bandwidth_gbps * 1e9) * 1e3
    optimizer_time_ms = optimizer_memory_time_ms * train_calibration.optimizer_scale_factor
    backward_time_ms *= args.gradient_accumulation_steps
    comm_bytes, comm_latency_ms = estimate_gradient_communication_bytes(
        arch.parameters,
        1,
        execution.parallel_mode == "ddp",
        config=config_data,
        interconnect=execution.local_topology if execution.nnodes <= 1 else execution.interconnect,
        nnodes=execution.nnodes,
    )
    comm_time_ms = 0.0
    if execution.parallel_mode == "ddp":
        comm_time_ms = comm_latency_ms
        if comm_bytes > 0:
            comm_time_ms += comm_bytes / (train_calibration.gradient_allreduce_tflops * 1e9) * 1e3
    total_time_ms = forward_time_ms + backward_time_ms + optimizer_time_ms + comm_time_ms
    graph_counts = {
        "prefill_call_function_nodes": sum(1 for node in training_graphs.prefill_export.graph.nodes if node.op == "call_function"),
        "decode_call_function_nodes": sum(1 for node in training_graphs.decode_export.graph.nodes if node.op == "call_function"),
    }
    return {
        "arch": arch,
        "training_graphs": training_graphs,
        "prefill_estimates": finalized_prefill,
        "graph_counts": graph_counts,
        "step_estimate": {
            "forward_flops": sum(n.flops for n in finalized_prefill),
            "forward_bytes": sum(n.bytes_moved for n in finalized_prefill),
            "backward_flops": backward_flops,
            "backward_bytes": backward_bytes,
            "backward_breakdown": backward_breakdown,
            "optimizer_flops": optimizer_flops,
            "optimizer_bytes": optimizer_bytes,
            "optimizer_memory_time_ms": optimizer_memory_time_ms,
        },
        "phase_estimates": {
            "forward": forward_time_ms,
            "backward_compute": backward_time_ms,
            "backward_comm": comm_time_ms,
            "backward_total": backward_time_ms + comm_time_ms,
            "optimizer": optimizer_time_ms,
            "total": total_time_ms,
        },
        "comm_bytes": comm_bytes,
    }


def measure_training_phases(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    execution: Any,
    arch: Any,
    train_calibration: TrainCalibration,
    config_data: dict[str, Any],
    warmup: int,
    repeat: int,
    estimated_backward_comm_ms: float = 0.0,
    estimated_backward_total_ms: float = 0.0,
) -> dict[str, Any]:
    if execution.parallel_mode == "tp" and execution.tp_size > 1:
        phase_results = cuda_wall_time_ms_phases_tp(
            model,
            input_ids,
            attention_mask,
            labels,
            optimizer,
            warmup,
            repeat,
            tp_size=execution.tp_size,
            num_parameters=arch.parameters,
        )
        comm_bytes = estimate_backward_comm_simple(
            num_layers=arch.num_layers,
            hidden_size=arch.hidden_size,
            batch_size=input_ids.shape[0],
            seq_len=input_ids.shape[1],
            tp_size=execution.tp_size,
            calibration=train_calibration,
            num_parameters=arch.parameters,
        ).gradient_bytes
    else:
        phase_results = cuda_wall_time_ms_phases(
            model,
            input_ids,
            attention_mask,
            labels,
            optimizer,
            warmup,
            repeat,
        )
        comm_bytes, _ = estimate_gradient_communication_bytes(
            arch.parameters,
            1,
            execution.parallel_mode == "ddp",
            config=config_data,
            interconnect=execution.local_topology if execution.nnodes <= 1 else execution.interconnect,
            nnodes=execution.nnodes,
        )

    comm_stats = aggregate_sample_stats([0.0] * repeat)
    if execution.parallel_mode in {"ddp", "tp"} and execution.world_size > 1 and comm_bytes > 0:
        comm_stats = benchmark_allreduce_ms(
            num_bytes=comm_bytes,
            device=input_ids.device,
            warmup=warmup,
            repeat=repeat,
        )

    backward_total_stats = phase_results.get("backward_total", phase_results.get("backward"))
    if execution.parallel_mode == "tp":
        overlap_ratio = train_calibration.overlap_ratio
    elif execution.parallel_mode == "ddp":
        overlap_ratio = config_data.get("single_ddp", {}).get("communication", {}).get("overlap_ratio", 0.6)
    else:
        overlap_ratio = 0.0

    estimated_fraction = 0.0
    if estimated_backward_total_ms > 0:
        estimated_fraction = clamp(estimated_backward_comm_ms / estimated_backward_total_ms, 0.0, 0.95)
    elif execution.parallel_mode in {"ddp", "tp"}:
        estimated_fraction = clamp(1.0 - overlap_ratio, 0.05, 0.95)
    if execution.parallel_mode in {"ddp", "tp"}:
        estimated_fraction = min(
            estimated_fraction,
            clamp(1.0 - overlap_ratio, 0.05, 0.95),
        )

    effective_comm_samples = []
    backward_compute_samples = []
    raw_comm_samples = comm_stats["samples_ms"]
    for total_ms, raw_comm_ms in zip(backward_total_stats["samples_ms"], raw_comm_samples):
        overlapped_comm_ms = raw_comm_ms * (1.0 - overlap_ratio)
        target_comm_ms = total_ms * estimated_fraction
        if target_comm_ms > 0.0:
            effective_comm_ms = min(total_ms, min(overlapped_comm_ms, target_comm_ms * 1.5))
        else:
            effective_comm_ms = min(total_ms, overlapped_comm_ms)
        backward_compute_samples.append(max(total_ms - effective_comm_ms, 0.0))
        effective_comm_samples.append(effective_comm_ms)

    measured = {
        "forward": phase_results["forward"],
        "backward_compute": aggregate_sample_stats(backward_compute_samples),
        "backward_comm": aggregate_sample_stats(effective_comm_samples),
        "backward_comm_raw": comm_stats,
        "backward_total": backward_total_stats,
        "optimizer": phase_results["optimizer"],
        "combined": phase_results["combined"],
    }
    return measured


def calibration_parameter_matrix() -> list[dict[str, str]]:
    return [
        {
            "mode": "Single",
            "forward": "common.forward.effective_tflops_scale",
            "backward_compute": "common.backward.backward_efficiency_scale",
            "backward_comm": "none",
            "optimizer": "common.optimizer.scale_factor",
        },
        {
            "mode": "DDP",
            "forward": "common.forward.effective_tflops_scale",
            "backward_compute": "common.backward.backward_efficiency_scale",
            "backward_comm": "single_ddp.communication.gradient_allreduce_tflops",
            "optimizer": "common.optimizer.scale_factor",
        },
        {
            "mode": "TP",
            "forward": "tp.backward.tp_forward_efficiency",
            "backward_compute": "tp.backward.tp_backward_efficiency",
            "backward_comm": "tp.communication.overlap_ratio",
            "optimizer": "tp.communication.optimizer_efficiency",
        },
    ]
