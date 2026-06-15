#!/usr/bin/env python3
"""
Training Parameter Calibration Script

This script performs automated calibration of training estimation hyperparameters:
1. Loads config file with initial parameters
2. Runs estimation + measurement for a few steps
3. Computes error ratios for each phase
4. Adjusts hyperparameters to minimize estimation error
5. Saves calibrated parameters back to config

Usage (single card):
    python calibrate_train_params.py --config config/train_config.yaml --model-path <model>

Usage (TP mode with torchrun):
    torchrun --nproc_per_node=2 calibrate_train_params.py --config config/train_config.yaml --model-path <model> --mode tp --tp-size 2
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
import time
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import estimation functions
from mvp_calibration import build_calibration
from mvp_estimator import estimate_node, finalize_estimate_ordinals
from mvp_execution import detect_nvlink, env_int, parse_physical_devices
from mvp_runtime import extract_inference_graphs, prepare_inputs
from mvp_train_estimator import (
    estimate_backward_from_graph_nodes,
    estimate_model_architecture,
    estimate_optimizer_flops,
    build_gradient_bytes_mapping,
)
from mvp_train_graph import extract_training_graphs
from mvp_train_types import TrainCalibration

from mvp_train_unified_estimator import estimate_train_step_with_tp
from mvp_train_types import TrainConfig
from mvp_measurement import cuda_wall_time_ms_phases
from train_workflow import clamp, clamp_ratio, limit_for, threshold_for


PARAM_UPDATE_EPSILON = 1e-6


@dataclass
class CalibrationResult:
    """Result of a single calibration measurement."""
    phase: str
    estimated_ms: float
    measured_ms: float
    error_pct: float
    ratio: float


def load_config(config_path: str) -> dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def save_config(config: dict[str, Any], config_path: str) -> None:
    """Save configuration to YAML file."""
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def is_primary_rank() -> bool:
    """Check if current process is the primary rank.

    Handles both:
    1. Distributed mode (dist initialized) - check rank == 0
    2. Non-distributed mode - check LOCAL_RANK env var (set by torchrun)
    """
    # Check if running under torchrun with multiple processes
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if world_size > 1 and local_rank != 0:
        return False

    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Calibrate training estimation parameters")
    parser.add_argument("--config", default="config/train_config.yaml",
                        help="Path to config file")
    parser.add_argument("--model-path", required=True,
                        help="Path to model")
    parser.add_argument("--mode", choices=["single", "tp"], default="single",
                        help="Calibration mode")
    parser.add_argument("--tp-size", type=int, default=2,
                        help="TP size for TP mode")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size for calibration")
    parser.add_argument("--seq-len", type=int, default=128,
                        help="Sequence length for calibration")
    parser.add_argument("--warmup", type=int, default=2,
                        help="Number of warmup steps")
    parser.add_argument("--measure", type=int, default=5,
                        help="Number of measurement steps")
    parser.add_argument("--output", default=None,
                        help="Output calibrated config path (default: overwrite input)")
    parser.add_argument("--device", default="cuda:0",
                        help="Device to use")
    parser.add_argument("--dist-timeout-minutes", type=int, default=30,
                        help="Distributed timeout in minutes")
    parser.add_argument("--physical-devices", type=str, default=None,
                        help="Comma-separated list of physical device IDs (e.g., '0,1')")
    return parser.parse_args()


def init_distributed(args: argparse.Namespace) -> torch.device:
    """Initialize distributed environment if needed. Returns device."""
    dist_initialized = dist.is_available() and dist.is_initialized()
    world_size_from_env = int(os.environ.get("WORLD_SIZE", "0"))

    if args.mode == "tp" and world_size_from_env > 1:
        if not dist_initialized:
            dist.init_process_group("nccl", timeout=timedelta(minutes=args.dist_timeout_minutes))

        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        return torch.device("cuda", local_rank)

    elif args.mode == "ddp" and world_size_from_env > 1:
        if not dist_initialized:
            dist.init_process_group("nccl", timeout=timedelta(minutes=args.dist_timeout_minutes))

        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        return torch.device("cuda", local_rank)

    else:
        # Single device mode
        device = torch.device(args.device)
        return device


def build_train_calibration(config: dict[str, Any], device: torch.device) -> TrainCalibration:
    """Build TrainCalibration from config."""
    hw = config.get('hardware', {})
    calib = build_calibration(torch.bfloat16, device)
    common_cfg = config.get('common', {})
    backward_cfg = common_cfg.get('backward', {})
    optimizer_cfg = common_cfg.get('optimizer', {})

    return TrainCalibration(
        device_name=hw.get('device_name') or calib.device_name,
        device_index=calib.device_index,
        gemm_tflops=hw.get('gemm_tflops') or calib.gemm_tflops,
        attention_tflops=hw.get('attention_tflops') or calib.attention_tflops,
        memory_bandwidth_gbps=hw.get('memory_bandwidth_gbps') or calib.memory_bandwidth_gbps,
        launch_overhead_ms=hw.get('launch_overhead_ms') or calib.launch_overhead_ms,
        backward_compute_scale=backward_cfg.get('compute_scale', 2.5),
        backward_efficiency_scale=backward_cfg.get('backward_efficiency_scale', 0.07),
        optimizer_scale_factor=optimizer_cfg.get('scale_factor', 1.4),
        tp_config=config.get('tp', {}),
    )


def run_single_card_calibration(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    train_calibration: TrainCalibration,
    config: dict[str, Any],
    num_warmup: int = 2,
    num_measure: int = 5,
) -> tuple[list[CalibrationResult], dict[str, Any]]:
    """Run single card calibration."""
    arch = estimate_model_architecture(model)
    # NOTE: optimizer config is under 'common' section, not top-level
    optimizer_cfg = config.get('common', {}).get('optimizer', {})

    # Extract graphs and estimate (use extract_training_graphs for backward_info)
    training_graphs = extract_training_graphs(
        model, input_ids, attention_mask, include_backward=True,
        model_name="calibration_model"
    )

    prefill_estimates = finalize_estimate_ordinals([
        estimate_node(node, "forward_step", train_calibration)
        for node in training_graphs.prefill_export.graph.nodes
        if estimate_node(node, "forward_step", train_calibration) is not None
    ])

    forward_time_ms = sum(n.estimated_time_ms for n in prefill_estimates)

    # Build gradient bytes mapping from backward_info if available
    gradient_bytes_by_scope = None
    if training_graphs.backward_info is not None and hasattr(training_graphs.backward_info, 'gradient_infos'):
        gradient_bytes_by_scope = build_gradient_bytes_mapping(training_graphs.backward_info.gradient_infos)

    # Use per-node backward estimation (consistent with Single/DDP main prediction)
    backward_time_ms, backward_flops, backward_bytes, backward_breakdown = \
        estimate_backward_from_graph_nodes(prefill_estimates, train_calibration, gradient_bytes_by_scope)

    optimizer_flops, optimizer_bytes = estimate_optimizer_flops(
        arch.parameters, 1, input_ids.shape[1], train_calibration
    )
    optimizer_memory_time_ms = (
        optimizer_bytes / (train_calibration.memory_bandwidth_gbps * 1e9) * 1e3
    )
    optimizer_time_ms = optimizer_memory_time_ms * optimizer_cfg.get('scale_factor', 1.4)

    estimated_total_ms = forward_time_ms + backward_time_ms + optimizer_time_ms

    # Measure
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    def train_step_fn():
        model.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # Warmup
    for _ in range(num_warmup):
        train_step_fn()

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_measure):
        train_step_fn()
    torch.cuda.synchronize()
    end = time.perf_counter()

    measured_total_ms = (end - start) / num_measure * 1000
    total_ratio = measured_total_ms / estimated_total_ms if estimated_total_ms > 0 else 1.0

    results = [
        CalibrationResult(
            phase="total",
            estimated_ms=estimated_total_ms,
            measured_ms=measured_total_ms,
            error_pct=abs(estimated_total_ms - measured_total_ms) / measured_total_ms * 100,
            ratio=total_ratio,
        ),
    ]

    # Compute adjustment for single/ddp mode
    adjustment = {}
    if total_ratio > 0 and abs(1.0 - total_ratio) > 0.1:
        # ratio = measured / estimated
        # - ratio > 1: measured > estimated, 估算偏低
        # - ratio < 1: measured < estimated, 估算偏高
        #
        # backward_time = backward_flops / effective_tflops
        # backward_flops = forward_flops * compute_scale
        #
        # To adjust estimated time:
        # - ratio > 1 (too low): increase compute_scale
        # - ratio < 1 (too high): decrease compute_scale
        common_cfg = config.get('common', {})
        backward_cfg = common_cfg.get('backward', {}).copy()

        current_scale = backward_cfg.get('compute_scale', 3.5)
        new_scale = current_scale * total_ratio
        new_scale = max(1.0, min(10.0, new_scale))  # Clamp to reasonable range

        if abs(new_scale - current_scale) > 0.1:
            backward_cfg['compute_scale'] = new_scale
            adjustment['common.backward'] = backward_cfg

    return results, adjustment


def run_tp_calibration(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    train_calibration: TrainCalibration,
    config: dict[str, Any],
    tp_size: int,
    num_warmup: int = 2,
    num_measure: int = 5,
    physical_devices: list[int] = None,
) -> tuple[list[CalibrationResult], dict[str, Any]]:
    """Run TP mode calibration with four-stage independent parameter calibration.

    This function calibrates four TP parameters independently based on
    four-stage separated measurements:
    1. tp_forward_efficiency: calibrated from forward phase
    2. tp_backward_efficiency: calibrated from backward phase (compute component)
    3. overlap_ratio: calibrated from backward phase (comm component)
    4. optimizer_efficiency: calibrated from Optimizer phase
    """
    if physical_devices is None:
        physical_devices = [0, 1]  # Default for 2-GPU system

    arch = estimate_model_architecture(model)
    batch_size = input_ids.shape[0]
    seq_len = input_ids.shape[1]

    tp_cfg = config.get('tp', {})
    comm_cfg = tp_cfg.get('communication', {})
    backward_cfg = tp_cfg.get('backward', {})

    # Detect NVLink for accurate TP communication estimation
    train_calibration.has_nvlink = detect_nvlink(physical_devices)
    train_calibration.overlap_ratio = comm_cfg.get('overlap_ratio', 0.35)
    train_calibration.tp_backward_efficiency = backward_cfg.get('tp_backward_efficiency', 0.13)
    train_calibration.tp_config = tp_cfg

    train_config = TrainConfig(
        batch_size=batch_size,
        seq_len=seq_len,
        num_epochs=1,
        global_batch_size=batch_size,
        gradient_accumulation_steps=1,
        ddp_enabled=False,
        tp_size=tp_size,
    )

    # Get initial estimate to understand the estimated time breakdown
    step_estimate = estimate_train_step_with_tp(
        batch_size, seq_len, arch, train_calibration, train_config,
    )

    estimated_forward_ms = step_estimate.forward_time_ms
    estimated_backward_ms = step_estimate.backward_time_ms
    estimated_optimizer_ms = step_estimate.optimizer_time_ms
    estimated_total_ms = step_estimate.total_time_ms

    # Debug: print what was actually used
    print(f"[DEBUG TP Cal] forward_efficiency={train_calibration.tp_forward_efficiency:.4f}, "
          f"backward_efficiency={train_calibration.tp_backward_efficiency:.4f}, "
          f"overlap_ratio={train_calibration.overlap_ratio:.4f}")
    print(f"[DEBUG TP Cal] estimated: forward={estimated_forward_ms:.2f}ms, "
          f"backward={estimated_backward_ms:.2f}ms, "
          f"optimizer={estimated_optimizer_ms:.2f}ms, "
          f"total={estimated_total_ms:.2f}ms")

    # ===== Four-stage measurement using CUDA events =====
    model.train()
    # Use foreach=False for TP mode to avoid DTensor/Tensor mixed error in Adam
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, foreach=False)

    # Use cuda_wall_time_ms_phases_tp for separated phase measurements (forward, backward_compute, backward_comm, optimizer)
    phase_results = cuda_wall_time_ms_phases_tp(
        model, input_ids, attention_mask, labels, optimizer,
        num_warmup, num_measure, tp_size=tp_size
    )

    measured_forward_ms = phase_results["forward"]["median_ms"]
    measured_backward_compute_ms = phase_results["backward_compute"]["median_ms"]
    measured_backward_comm_ms = phase_results["backward_comm"]["median_ms"]
    measured_backward_total_ms = phase_results["backward_total"]["median_ms"]
    measured_optimizer_ms = phase_results["optimizer"]["median_ms"]
    measured_total_ms = phase_results["combined"]["median_ms"]

    print(f"[DEBUG TP Cal] measured: forward={measured_forward_ms:.2f}ms, "
          f"backward_compute={measured_backward_compute_ms:.2f}ms, "
          f"backward_comm={measured_backward_comm_ms:.2f}ms, "
          f"optimizer={measured_optimizer_ms:.2f}ms, "
          f"total={measured_total_ms:.2f}ms")

    # Build per-phase calibration results
    results = [
        CalibrationResult(
            phase="forward",
            estimated_ms=estimated_forward_ms,
            measured_ms=measured_forward_ms,
            error_pct=abs(estimated_forward_ms - measured_forward_ms) / measured_forward_ms * 100 if measured_forward_ms > 0 else 0,
            ratio=measured_forward_ms / estimated_forward_ms if estimated_forward_ms > 0 else 1.0,
        ),
        CalibrationResult(
            phase="backward_compute",
            estimated_ms=estimated_backward_ms,
            measured_ms=measured_backward_compute_ms,
            error_pct=abs(estimated_backward_ms - measured_backward_compute_ms) / measured_backward_compute_ms * 100 if measured_backward_compute_ms > 0 else 0,
            ratio=measured_backward_compute_ms / estimated_backward_ms if estimated_backward_ms > 0 else 1.0,
        ),
        CalibrationResult(
            phase="backward_comm",
            estimated_ms=estimated_backward_ms,
            measured_ms=measured_backward_comm_ms,
            error_pct=abs(estimated_backward_ms - measured_backward_comm_ms) / measured_backward_comm_ms * 100 if measured_backward_comm_ms > 0 else 0,
            ratio=measured_backward_comm_ms / estimated_backward_ms if estimated_backward_ms > 0 else 1.0,
        ),
        CalibrationResult(
            phase="optimizer",
            estimated_ms=estimated_optimizer_ms,
            measured_ms=measured_optimizer_ms,
            error_pct=abs(estimated_optimizer_ms - measured_optimizer_ms) / measured_optimizer_ms * 100 if measured_optimizer_ms > 0 else 0,
            ratio=measured_optimizer_ms / estimated_optimizer_ms if estimated_optimizer_ms > 0 else 1.0,
        ),
        CalibrationResult(
            phase="total",
            estimated_ms=estimated_total_ms,
            measured_ms=measured_total_ms,
            error_pct=abs(estimated_total_ms - measured_total_ms) / measured_total_ms * 100 if measured_total_ms > 0 else 0,
            ratio=measured_total_ms / estimated_total_ms if estimated_total_ms > 0 else 1.0,
        ),
    ]

    # ===== Four-parameter independent calibration =====
    adjustment = {}

    # Get current parameter values
    current_forward_eff = backward_cfg.get('tp_forward_efficiency', 0.05)
    current_backward_eff = backward_cfg.get('tp_backward_efficiency', 0.13)
    current_overlap = comm_cfg.get('overlap_ratio', 0.35)
    current_opt_eff = comm_cfg.get('optimizer_efficiency', 0.45)

    # --- 1. Forward efficiency calibration ---
    # forward_time_ms = forward_result["total_time_ms"] / tp_forward_efficiency
    # measured_forward / estimated_forward = tp_forward_efficiency (since division cancels)
    # Actually: measured = estimated_forward_time / new_efficiency
    #          => measured / estimated = old_efficiency / new_efficiency
    #          => new_efficiency = old_efficiency / (measured / estimated)
    forward_ratio = measured_forward_ms / estimated_forward_ms if estimated_forward_ms > 0 else 1.0
    forward_ratio = max(0.5, min(2.0, forward_ratio))  # Clamp to prevent extreme values

    new_forward_eff = current_forward_eff
    if abs(1.0 - forward_ratio) > 0.1:
        new_forward_eff = current_forward_eff / forward_ratio
        new_forward_eff = max(0.01, min(0.5, new_forward_eff))
        print(f"[TP Cal] Forward: ratio={forward_ratio:.4f}, "
              f"tp_forward_efficiency {current_forward_eff:.4f} -> {new_forward_eff:.4f}")
    else:
        print(f"[TP Cal] Forward: ratio={forward_ratio:.4f}, no adjustment needed")

    # --- 2. Backward efficiency calibration ---
    # With four-phase measurement, we have separate backward_compute and backward_comm measurements
    # backward_compute_time = measured directly from TP measurement
    # We calibrate tp_backward_efficiency based on backward_compute_ratio

    # Get estimated backward compute time from step_estimate
    estimated_backward_compute_ms = step_estimate.backward_summary.compute_time_ms if step_estimate.backward_summary else 0

    backward_compute_ratio = measured_backward_compute_ms / estimated_backward_compute_ms if estimated_backward_compute_ms > 0 else 1.0
    backward_compute_ratio = max(0.5, min(2.0, backward_compute_ratio))  # Clamp

    new_backward_eff = current_backward_eff
    if abs(1.0 - backward_compute_ratio) > 0.1:
        new_backward_eff = current_backward_eff / backward_compute_ratio
        new_backward_eff = max(0.005, min(0.2, new_backward_eff))
        print(f"[TP Cal] Backward Compute: ratio={backward_compute_ratio:.4f}, "
              f"tp_backward_efficiency {current_backward_eff:.4f} -> {new_backward_eff:.4f}")
    else:
        print(f"[TP Cal] Backward Compute: ratio={backward_compute_ratio:.4f}, no adjustment needed")

    # --- 3. Communication (overlap_ratio) calibration ---
    # With four-phase measurement, we have separate backward_comm measurement
    # We calibrate overlap_ratio based on backward_comm_ratio

    # Get estimated backward comm time from step_estimate
    estimated_backward_comm_ms = step_estimate.backward_summary.comm_time_ms if step_estimate.backward_summary else 0

    backward_comm_ratio = measured_backward_comm_ms / estimated_backward_comm_ms if estimated_backward_comm_ms > 0 else 1.0
    backward_comm_ratio = max(0.5, min(2.0, backward_comm_ratio))  # Clamp

    new_overlap = current_overlap
    if abs(1.0 - backward_comm_ratio) > 0.1:
        # backward_comm_ratio > 1: measured comm is slower than estimated, need more overlap to hide it
        # backward_comm_ratio < 1: measured comm is faster than estimated, need less overlap
        if backward_comm_ratio > 1.0:
            new_overlap = min(0.9, current_overlap + (backward_comm_ratio - 1.0) * 0.3)
        else:
            new_overlap = max(0.0, current_overlap - (1.0 - backward_comm_ratio) * 0.3)
        new_overlap = max(0.0, min(0.9, new_overlap))
        print(f"[TP Cal] Backward Comm: ratio={backward_comm_ratio:.4f}, "
              f"overlap_ratio {current_overlap:.4f} -> {new_overlap:.4f}")
    else:
        print(f"[TP Cal] Backward Comm: ratio={backward_comm_ratio:.4f}, no adjustment needed")

    # --- 4. Optimizer efficiency calibration ---
    # optimizer_time_ms = memory_time / optimizer_efficiency
    # measured / estimated = optimizer_efficiency (since division cancels)
    # Actually: measured = (memory_bytes / (bandwidth * optimizer_eff)) * 1000
    #          => measured / estimated = old_eff / new_eff
    #          => new_eff = old_eff * (measured / estimated)
    optimizer_ratio = measured_optimizer_ms / estimated_optimizer_ms if estimated_optimizer_ms > 0 else 1.0
    optimizer_ratio = max(0.5, min(2.0, optimizer_ratio))  # Clamp

    new_opt_eff = current_opt_eff
    if abs(1.0 - optimizer_ratio) > 0.1:
        new_opt_eff = current_opt_eff * optimizer_ratio
        new_opt_eff = max(0.1, min(0.9, new_opt_eff))
        print(f"[TP Cal] Optimizer: ratio={optimizer_ratio:.4f}, "
              f"optimizer_efficiency {current_opt_eff:.4f} -> {new_opt_eff:.4f}")
    else:
        print(f"[TP Cal] Optimizer: ratio={optimizer_ratio:.4f}, no adjustment needed")

    # Apply adjustments to config
    tp_cfg = config.get('tp', {})
    backward_cfg = tp_cfg.get('backward', {}).copy()
    comm_cfg = tp_cfg.get('communication', {}).copy()

    if abs(new_forward_eff - current_forward_eff) > 0.001:
        backward_cfg['tp_forward_efficiency'] = new_forward_eff
        adjustment['tp.backward'] = backward_cfg
        train_calibration.tp_forward_efficiency = new_forward_eff

    if abs(new_backward_eff - current_backward_eff) > 0.001:
        backward_cfg['tp_backward_efficiency'] = new_backward_eff
        adjustment['tp.backward'] = backward_cfg
        train_calibration.tp_backward_efficiency = new_backward_eff

    if abs(new_overlap - current_overlap) > 0.01:
        comm_cfg['overlap_ratio'] = new_overlap
        adjustment['tp.communication'] = comm_cfg
        train_calibration.overlap_ratio = new_overlap

    if abs(new_opt_eff - current_opt_eff) > 0.01:
        comm_cfg['optimizer_efficiency'] = new_opt_eff
        adjustment['tp.communication'] = comm_cfg

    return results, adjustment





def run_tp_calibration_inprocess(
    model,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    train_calibration: TrainCalibration,
    config: dict[str, Any],
    tp_size: int,
    rank: int,
    world_size: int,
    num_warmup: int = 2,
    num_measure: int = 5,
) -> tuple[float, float]:
    """Run TP calibration IN-PROCESS with four-phase separation.

    This function is designed to be called from mvp_train_app.py directly,
    avoiding the OOM issues that occur when spawning subprocess for calibration
    in a torchrun environment.

    Four phases measured:
    1. Forward pass
    2. Backward compute (local gradient computation)
    3. Backward communication (TP AllReduce)
    4. Optimizer update

    Args:
        model: Already loaded TP model (on GPU)
        input_ids: Input tensor [batch, seq_len]
        labels: Labels tensor [batch, seq_len]
        train_calibration: TrainCalibration object to be updated
        config: Config dict (will be modified in-place)
        tp_size: Tensor parallel size
        rank: Current process rank
        world_size: Total number of processes
        num_warmup: Number of warmup steps
        num_measure: Number of measurement steps

    Returns:
        (new_tp_backward_efficiency, error_pct_after_adjustment)
        Returns (None, None) if this rank should not run calibration.
    """
    from mvp_measurement import cuda_wall_time_ms_phases_tp

    # Only rank 0 runs the measurement
    if rank == 0:
        arch = estimate_model_architecture(model)
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]

        tp_cfg = config.get('tp', {})
        comm_cfg = tp_cfg.get('communication', {})
        backward_cfg = tp_cfg.get('backward', {})

        # Get current parameters
        current_efficiency = backward_cfg.get('tp_backward_efficiency', 0.13)
        current_overlap = comm_cfg.get('overlap_ratio', 0.35)

        # Set up calibration
        train_calibration.has_nvlink = True  # Assume NVLink for now
        train_calibration.overlap_ratio = current_overlap
        train_calibration.tp_backward_efficiency = current_efficiency
        train_calibration.tp_config = tp_cfg

        train_config = TrainConfig(
            batch_size=batch_size,
            seq_len=seq_len,
            num_epochs=1,
            global_batch_size=batch_size,
            gradient_accumulation_steps=1,
            ddp_enabled=False,
            tp_size=tp_size,
        )

        # Estimate with current parameters
        step_estimate = estimate_train_step_with_tp(
            batch_size, seq_len, arch, train_calibration, train_config,
        )
        estimated_forward_ms = step_estimate.forward_time_ms
        estimated_backward_compute_ms = step_estimate.backward_summary.compute_time_ms if step_estimate.backward_summary else 0
        estimated_backward_comm_ms = step_estimate.backward_summary.comm_time_ms if step_estimate.backward_summary else 0
        estimated_backward_total_ms = step_estimate.backward_time_ms
        estimated_optimizer_ms = step_estimate.optimizer_time_ms
        estimated_total_ms = step_estimate.total_time_ms

        # Measure actual training time with four-phase separation
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, foreach=False)

        # Use four-phase measurement for TP mode
        phase_results = cuda_wall_time_ms_phases_tp(
            model, input_ids, None, labels, optimizer,
            num_warmup, num_measure,
            tp_size=tp_size,
            num_parameters=arch.parameters,
        )

        measured_forward_ms = phase_results["forward"]["median_ms"]
        measured_backward_compute_ms = phase_results["backward_compute"]["median_ms"]
        measured_backward_comm_ms = phase_results["backward_comm"]["median_ms"]
        measured_backward_total_ms = phase_results["backward_total"]["median_ms"]
        measured_optimizer_ms = phase_results["optimizer"]["median_ms"]
        measured_total_ms = phase_results["combined"]["median_ms"]

        print(f"[TP Four-Phase Calibration] rank=0:")
        print(f"  Forward:      estimated={estimated_forward_ms:.2f}ms, measured={measured_forward_ms:.2f}ms")
        print(f"  Backward(c):  estimated={estimated_backward_compute_ms:.2f}ms, measured={measured_backward_compute_ms:.2f}ms")
        print(f"  Backward(comm): estimated={estimated_backward_comm_ms:.2f}ms, measured={measured_backward_comm_ms:.2f}ms")
        print(f"  Backward(total): estimated={estimated_backward_total_ms:.2f}ms, measured={measured_backward_total_ms:.2f}ms")
        print(f"  Optimizer:    estimated={estimated_optimizer_ms:.2f}ms, measured={measured_optimizer_ms:.2f}ms")
        print(f"  Total:        estimated={estimated_total_ms:.2f}ms, measured={measured_total_ms:.2f}ms")

        # Compute per-phase ratios
        forward_ratio = measured_forward_ms / estimated_forward_ms if estimated_forward_ms > 0 else 1.0
        backward_compute_ratio = measured_backward_compute_ms / estimated_backward_compute_ms if estimated_backward_compute_ms > 0 else 1.0
        backward_comm_ratio = measured_backward_comm_ms / estimated_backward_comm_ms if estimated_backward_comm_ms > 0 else 1.0
        optimizer_ratio = measured_optimizer_ms / estimated_optimizer_ms if estimated_optimizer_ms > 0 else 1.0
        total_ratio = measured_total_ms / estimated_total_ms if estimated_total_ms > 0 else 1.0

        error_pct_before = abs(estimated_total_ms - measured_total_ms) / measured_total_ms * 100

        print(f"[TP Four-Phase Calibration] Ratios: forward={forward_ratio:.4f}, "
              f"backward_compute={backward_compute_ratio:.4f}, "
              f"backward_comm={backward_comm_ratio:.4f}, "
              f"optimizer={optimizer_ratio:.4f}, total={total_ratio:.4f}")

        # Compute adjustments based on four-phase ratios
        current_forward_eff = backward_cfg.get('tp_forward_efficiency', 0.05)
        new_forward_eff = current_forward_eff
        new_efficiency = current_efficiency
        new_overlap = current_overlap

        # --- Forward efficiency calibration ---
        # forward_time = forward_flops / (gemm_tflops * tp_forward_efficiency)
        # ratio > 1 (measured > estimated) -> 效率太高，需要降低
        # new_efficiency = old_efficiency / forward_ratio
        if abs(1.0 - forward_ratio) > 0.1:
            new_forward_eff = current_forward_eff / forward_ratio
            new_forward_eff = max(0.01, min(0.5, new_forward_eff))

        # --- Backward efficiency calibration ---
        # Adjust backward_efficiency based on backward_compute_ratio
        # If measured > estimated (ratio > 1), current efficiency is too high, need to lower it
        # new_efficiency = old_efficiency / backward_compute_ratio
        if backward_compute_ratio > 0:
            new_efficiency = current_efficiency / backward_compute_ratio

        # Adjust overlap_ratio based on backward_comm_ratio
        # If measured > estimated (ratio > 1), comm is taking longer, need higher overlap to hide it
        # But overlap reduces effective comm time, so if comm ratio > 1, increase overlap
        # effective_comm = comm_time * (1 - overlap_ratio)
        # If measured comm is longer than expected, we could increase overlap to hide more
        if backward_comm_ratio > 1.1:
            # Comm is slower than expected, increase overlap
            new_overlap = min(0.9, current_overlap + (backward_comm_ratio - 1.0) * 0.3)
        elif backward_comm_ratio < 0.9:
            # Comm is faster than expected, decrease overlap
            new_overlap = max(0.0, current_overlap - (1.0 - backward_comm_ratio) * 0.3)
        else:
            new_overlap = current_overlap

        # Clamp to valid range
        new_forward_eff = max(0.01, min(0.5, new_forward_eff))
        new_efficiency = max(0.005, min(0.2, new_efficiency))
        new_overlap = max(0.0, min(0.9, new_overlap))

        # Update config in-place
        if abs(new_forward_eff - current_forward_eff) > 0.001:
            backward_cfg['tp_forward_efficiency'] = new_forward_eff
            train_calibration.tp_forward_efficiency = new_forward_eff
            print(f"[TP Four-Phase Calibration] Adjusted: forward_efficiency {current_forward_eff:.4f} -> {new_forward_eff:.4f}")

        backward_cfg['tp_backward_efficiency'] = new_efficiency
        comm_cfg['overlap_ratio'] = new_overlap
        train_calibration.tp_backward_efficiency = new_efficiency
        train_calibration.overlap_ratio = new_overlap

        print(f"[TP Four-Phase Calibration] Adjusted: backward_efficiency {current_efficiency:.4f} -> {new_efficiency:.4f}, "
              f"overlap {current_overlap:.4f} -> {new_overlap:.4f}")

        # Broadcast results to other ranks
        # Note: tp_forward_efficiency is also calibrated and stored in train_calibration
        calibration_data = [new_forward_eff, new_efficiency, new_overlap, error_pct_before]
        if world_size > 1:
            obj_list = [calibration_data]
            dist.broadcast_object_list(obj_list, src=0)
            calibration_data = obj_list[0]

        return calibration_data[1], calibration_data[3]  # (new_efficiency, error_pct)

    else:
        # Non-zero ranks wait for calibration result
        if world_size > 1:
            obj_list = [None]
            dist.broadcast_object_list(obj_list, src=0)
            calibration_data = obj_list[0]

            if calibration_data is not None:
                new_forward_eff, new_efficiency, new_overlap, error_pct = calibration_data
                # Update local calibration
                backward_cfg = config.get('tp', {}).get('backward', {})
                comm_cfg = config.get('tp', {}).get('communication', {})
                backward_cfg['tp_forward_efficiency'] = new_forward_eff
                backward_cfg['tp_backward_efficiency'] = new_efficiency
                comm_cfg['overlap_ratio'] = new_overlap
                train_calibration.tp_forward_efficiency = new_forward_eff
                train_calibration.tp_backward_efficiency = new_efficiency
                train_calibration.overlap_ratio = new_overlap

                return new_efficiency, error_pct
        return None, None


def run_single_card_calibration_inprocess(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    train_calibration: TrainCalibration,
    config: dict[str, Any],
    rank: int = 0,
    world_size: int = 1,
    num_warmup: int = 2,
    num_measure: int = 5,
    ddp_enabled: bool = False,
    local_device: int = 0,
) -> tuple[list[CalibrationResult], dict[str, Any]]:
    """Run single/DDP card calibration IN-PROCESS.

    This function is designed to be called from mvp_train_app.py directly,
    avoiding the OOM issues that occur when spawning subprocess for calibration
    in a torchrun environment.

    Works for both Single and DDP modes. For DDP mode, this function:
    1. Measures single-card time (without DDP wrapper)
    2. Measures DDP total time (with DDP wrapper)
    3. Computes comm_time = DDP_total_time - single_card_time
    4. Calibrates gradient_allreduce_tflops based on measured vs estimated comm time

    Args:
        model: Already loaded model (on GPU)
        input_ids: Input tensor [batch, seq_len]
        attention_mask: Attention mask tensor
        labels: Labels tensor [batch, seq_len]
        train_calibration: TrainCalibration object to be updated
        config: Config dict (will be modified in-place)
        rank: Current process rank (0 for primary)
        world_size: Total number of processes
        num_warmup: Number of warmup steps
        num_measure: Number of measurement steps
        ddp_enabled: Whether DDP is enabled (for DDP communication calibration)
        local_device: Local device ID for DDP wrapper

    Returns:
        (results, adjustment) - same as run_single_card_calibration
    """
    from mvp_train_estimator import estimate_gradient_communication_bytes
    from mvp_execution import detect_nvlink

    # Only rank 0 runs the measurement
    if rank == 0:
        arch = estimate_model_architecture(model)
        optimizer_cfg = config.get('common', {}).get('optimizer', {})

        # Extract graphs and estimate (use extract_training_graphs for backward_info)
        training_graphs = extract_training_graphs(
            model, input_ids, attention_mask, include_backward=True,
            model_name="calibration_model"
        )

        prefill_estimates = finalize_estimate_ordinals([
            estimate_node(node, "forward_step", train_calibration)
            for node in training_graphs.prefill_export.graph.nodes
            if estimate_node(node, "forward_step", train_calibration) is not None
        ])

        forward_time_ms = sum(n.estimated_time_ms for n in prefill_estimates)

        # Build gradient bytes mapping from backward_info if available
        gradient_bytes_by_scope = None
        if training_graphs.backward_info is not None and hasattr(training_graphs.backward_info, 'gradient_infos'):
            gradient_bytes_by_scope = build_gradient_bytes_mapping(training_graphs.backward_info.gradient_infos)

        # Use per-node backward estimation (consistent with Single/DDP main prediction)
        backward_time_ms, backward_flops, backward_bytes, backward_breakdown = \
            estimate_backward_from_graph_nodes(prefill_estimates, train_calibration, gradient_bytes_by_scope)

        optimizer_flops, optimizer_bytes = estimate_optimizer_flops(
            arch.parameters, 1, input_ids.shape[1], train_calibration
        )
        optimizer_memory_time_ms = (
            optimizer_bytes / (train_calibration.memory_bandwidth_gbps * 1e9) * 1e3
        )
        optimizer_time_ms = optimizer_memory_time_ms * train_calibration.optimizer_scale_factor

        # For DDP mode, get estimated comm time using current gradient_allreduce_tflops
        single_ddp_cfg = config.get('single_ddp', {})
        gradient_allreduce_tflops = single_ddp_cfg.get('communication', {}).get('gradient_allreduce_tflops', 50.0)

        comm_bytes, comm_latency_ms = estimate_gradient_communication_bytes(
            arch.parameters, 1, ddp_enabled=True,
            config=config,
            interconnect="local",  # Default to local; will be overridden in app if needed
            nnodes=1,
        )
        estimated_comm_time_ms = comm_latency_ms
        if comm_bytes > 0:
            estimated_comm_time_ms += (comm_bytes / (gradient_allreduce_tflops * 1e9) * 1e3)

        # Estimated total without comm (for backward_efficiency_scale calibration)
        estimated_total_ms_without_comm = forward_time_ms + backward_time_ms + optimizer_time_ms
        # Estimated total with comm (for DDP reporting)
        estimated_total_ms = estimated_total_ms_without_comm + (estimated_comm_time_ms if ddp_enabled and world_size > 1 else 0.0)

        # ========================================================================
        # STEP 1: Measure WITHOUT DDP wrapper (single card time)
        # ========================================================================
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        def train_step_fn():
            model.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Warmup
        for _ in range(num_warmup):
            train_step_fn()
        torch.cuda.synchronize()

        # Measure single card time
        start = time.perf_counter()
        for _ in range(num_measure):
            train_step_fn()
        torch.cuda.synchronize()
        end = time.perf_counter()

        single_card_time_ms = (end - start) / num_measure * 1000

        # ========================================================================
        # STEP 2: For DDP mode, measure WITH DDP wrapper (DDP total time)
        # ========================================================================
        ddp_total_time_ms = single_card_time_ms
        measured_comm_time_ms = 0.0

        if ddp_enabled and world_size > 1:
            # Need to re-create optimizer for DDP model
            from torch.nn.parallel import DistributedDataParallel

            # Wrap model with DDP
            model_ddp = DistributedDataParallel(model, device_ids=[local_device])
            optimizer_ddp = torch.optim.Adam(model_ddp.parameters(), lr=1e-4)

            def train_step_fn_ddp():
                model_ddp.zero_grad()
                outputs = model_ddp(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
                loss.backward()
                optimizer_ddp.step()
                optimizer_ddp.zero_grad()

            # Warmup with DDP
            for _ in range(num_warmup):
                train_step_fn_ddp()
            torch.cuda.synchronize()

            # Measure DDP total time
            start = time.perf_counter()
            for _ in range(num_measure):
                train_step_fn_ddp()
            torch.cuda.synchronize()
            end = time.perf_counter()

            ddp_total_time_ms = (end - start) / num_measure * 1000

            # Calculate measured communication time
            # DDP total time includes: forward + backward_compute + backward_comm + optimizer
            # Single card time includes: forward + backward_compute + optimizer
            # So: comm_time = ddp_total_time - single_card_time
            measured_comm_time_ms = ddp_total_time_ms - single_card_time_ms

            if is_primary_rank():
                print(f"[DDP Calibration] Single card time: {single_card_time_ms:.2f}ms, "
                      f"DDP total time: {ddp_total_time_ms:.2f}ms, "
                      f"Measured comm time: {measured_comm_time_ms:.2f}ms")

        # Use single_card_time_ms for ratio calculation (backward_efficiency_scale calibration)
        measured_for_ratio = single_card_time_ms
        total_ratio = measured_for_ratio / estimated_total_ms_without_comm if estimated_total_ms_without_comm > 0 else 1.0

        results = [
            CalibrationResult(
                phase="total",
                estimated_ms=estimated_total_ms,
                measured_ms=ddp_total_time_ms if ddp_enabled and world_size > 1 else single_card_time_ms,
                error_pct=abs(estimated_total_ms - (ddp_total_time_ms if ddp_enabled and world_size > 1 else single_card_time_ms)) / (ddp_total_time_ms if ddp_enabled and world_size > 1 else single_card_time_ms) * 100,
                ratio=total_ratio,
            ),
        ]

        print(f"[Single/DDP Calibration] rank=0: estimated={estimated_total_ms:.2f}ms "
              f"(without_comm={estimated_total_ms_without_comm:.2f}ms), "
              f"measured={ddp_total_time_ms if ddp_enabled and world_size > 1 else single_card_time_ms:.2f}ms, "
              f"ratio={total_ratio:.4f}, error={results[0].error_pct:.2f}%")

        # ========================================================================
        # STEP 3: Compute adjustments for backward_efficiency_scale and gradient_allreduce_tflops
        # ========================================================================
        adjustment = {}

        # 3a: Adjust backward_efficiency_scale based on single-card ratio
        if total_ratio > 0 and abs(1.0 - total_ratio) > 0.1:
            common_cfg = config.get('common', {})
            backward_cfg = common_cfg.get('backward', {}).copy()
            optimizer_backward_cfg = common_cfg.get('optimizer', {}).copy()

            current_efficiency = backward_cfg.get('backward_efficiency_scale', 0.07)
            current_opt_scale = optimizer_backward_cfg.get('scale_factor', 1.4)

            # new_efficiency = current_efficiency / total_ratio
            new_efficiency = current_efficiency / total_ratio
            new_efficiency = max(0.01, min(0.5, new_efficiency))

            new_opt_scale = current_opt_scale

            if abs(new_efficiency - current_efficiency) > 0.005:
                backward_cfg['backward_efficiency_scale'] = new_efficiency
                adjustment['common.backward'] = backward_cfg
                train_calibration.backward_efficiency_scale = new_efficiency
                print(f"[Single/DDP Calibration] Adjusted: backward_efficiency_scale {current_efficiency:.4f} -> {new_efficiency:.4f}")

            if abs(new_opt_scale - current_opt_scale) > 0.05:
                optimizer_backward_cfg['scale_factor'] = new_opt_scale
                adjustment['common.optimizer'] = optimizer_backward_cfg
                train_calibration.optimizer_scale_factor = new_opt_scale
                print(f"[Single/DDP Calibration] Adjusted: optimizer_scale {current_opt_scale:.4f} -> {new_opt_scale:.4f}")

        # 3b: Calibrate gradient_allreduce_tflops based on measured vs estimated comm time
        if ddp_enabled and world_size > 1 and measured_comm_time_ms > 0 and comm_bytes > 0:
            # Calibration formula:
            # measured_comm_time = ddp_total_time - single_card_time
            # estimated_comm_time = comm_bytes / (gradient_allreduce_tflops * 1e9) * 1000
            # ratio = measured_comm_time / estimated_comm_time
            # new_tflops = old_tflops / ratio

            current_comm_bandwidth_tflops = single_ddp_cfg.get('communication', {}).get('gradient_allreduce_tflops', 50.0)

            # estimated comm time excluding latency (for bandwidth calibration)
            estimated_comm_time_for_ratio = estimated_comm_time_ms - comm_latency_ms

            if estimated_comm_time_for_ratio > 0:
                comm_ratio = measured_comm_time_ms / estimated_comm_time_for_ratio
                if is_primary_rank():
                    print(f"[DDP Comm Calibration] measured_comm={measured_comm_time_ms:.2f}ms, "
                          f"estimated_comm={estimated_comm_time_for_ratio:.2f}ms (excl latency), "
                          f"ratio={comm_ratio:.4f}")

                # Clamp ratio to reasonable range [0.5, 2.0] to prevent extreme calibrations
                comm_ratio = max(0.5, min(2.0, comm_ratio))

                # new_tflops = old_tflops / comm_ratio
                # If comm_ratio > 1 (measured > estimated), bandwidth is lower, so increase tflops
                # If comm_ratio < 1 (measured < estimated), bandwidth is higher, so decrease tflops
                new_tflops = current_comm_bandwidth_tflops / comm_ratio
                new_tflops = max(10.0, min(500.0, new_tflops))  # Clamp to reasonable range

                if abs(new_tflops - current_comm_bandwidth_tflops) > 1.0:
                    single_ddp_comm_cfg = single_ddp_cfg.get('communication', {}).copy()
                    single_ddp_comm_cfg['gradient_allreduce_tflops'] = new_tflops
                    if 'single_ddp' not in adjustment:
                        adjustment['single_ddp'] = {'communication': single_ddp_comm_cfg}
                    else:
                        adjustment['single_ddp']['communication'] = single_ddp_comm_cfg
                    print(f"[DDP Comm Calibration] Adjusted: gradient_allreduce_tflops {current_comm_bandwidth_tflops:.2f} -> {new_tflops:.2f}")

        # Broadcast results to other ranks
        calibration_data = [adjustment, results[0].error_pct if results else 0]
        if world_size > 1:
            obj_list = [calibration_data]
            dist.broadcast_object_list(obj_list, src=0)
            calibration_data = obj_list[0]

        return results, calibration_data[0]

    else:
        # Non-zero ranks wait for calibration result
        if world_size > 1:
            obj_list = [None]
            dist.broadcast_object_list(obj_list, src=0)
            calibration_data = obj_list[0]

            if calibration_data is not None:
                adjustment = calibration_data[0]
                error_pct = calibration_data[1]
                # Apply adjustments
                if 'common.backward' in adjustment:
                    new_efficiency = adjustment['common.backward'].get('backward_efficiency_scale', 0.07)
                    train_calibration.backward_efficiency_scale = new_efficiency
                    config.setdefault('common', {}).setdefault('backward', {})['backward_efficiency_scale'] = new_efficiency
                if 'common.optimizer' in adjustment:
                    new_opt_scale = adjustment['common.optimizer'].get('scale_factor', 1.4)
                    train_calibration.optimizer_scale_factor = new_opt_scale
                    config.setdefault('common', {}).setdefault('optimizer', {})['scale_factor'] = new_opt_scale
                if 'single_ddp' in adjustment:
                    new_tflops = adjustment['single_ddp'].get('communication', {}).get('gradient_allreduce_tflops', 50.0)
                    config.setdefault('single_ddp', {}).setdefault('communication', {})['gradient_allreduce_tflops'] = new_tflops
                return [CalibrationResult(phase="total", estimated_ms=0, measured_ms=0, error_pct=error_pct, ratio=1.0)], adjustment
        return [], {}


# =============================================================================
# Exported calibration functions for use by mvp_train_app.py
# =============================================================================

def calibrate_single_ddp(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    train_calibration: TrainCalibration,
    config: dict[str, Any],
    forward_time_ms: float,
    backward_time_ms: float,
    optimizer_time_ms: float,
    measured_forward_ms: float,
    measured_backward_ms: float,
    measured_optimizer_ms: float,
    measured_backward_comm_ms: float = 0.0,
    rank: int = 0,
    world_size: int = 1,
    ddp_enabled: bool = False,
    comm_time_ms_original: float = 0.0,
    comm_bytes: float = 0.0,
    comm_latency_ms: float = 0.0,
    num_warmup: int = 2,
    num_measure: int = 5,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Calibrate Single/DDP mode parameters based on measured vs estimated time.

    This function replaces the inline calibration code in mvp_train_app.py.

    Args:
        model: Loaded model on GPU
        input_ids: Input tensor
        attention_mask: Attention mask tensor
        labels: Labels tensor
        train_calibration: TrainCalibration object to be updated
        config: Config dict (modified in-place)
        forward_time_ms: Estimated forward time (ms)
        backward_time_ms: Estimated backward time (ms)
        optimizer_time_ms: Estimated optimizer time (ms)
        measured_forward_ms: Measured forward time (ms)
        measured_backward_ms: Measured backward time (ms)
        measured_optimizer_ms: Measured optimizer time (ms)
        rank: Current process rank
        world_size: Total number of processes
        ddp_enabled: Whether DDP is enabled
        comm_time_ms_original: DDP communication time (ms)
        num_warmup: Warmup steps
        num_measure: Measurement steps

    Returns:
        (calibrated_times, adjustment_dict)
        - calibrated_times: dict with 'forward', 'backward', 'optimizer' keys
        - adjustment_dict: dict with parameter changes for config
    """
    adjustment = {}
    calibrated_times = {
        'forward': forward_time_ms,
        'backward': backward_time_ms,
        'optimizer': optimizer_time_ms,
    }

    if rank != 0:
        return calibrated_times, adjustment

    calibration_model = model.module if hasattr(model, "module") else model
    training_graphs = extract_training_graphs(
        calibration_model,
        input_ids,
        attention_mask,
        include_backward=True,
        model_name="ddp_calibration_model",
    )

    gradient_bytes_by_scope = None
    if training_graphs.backward_info is not None and hasattr(training_graphs.backward_info, "gradient_infos"):
        gradient_bytes_by_scope = build_gradient_bytes_mapping(training_graphs.backward_info.gradient_infos)

    def reestimate_graph_times() -> tuple[float, float]:
        updated_prefill_estimates = finalize_estimate_ordinals(
            [
                estimate
                for node in training_graphs.prefill_export.graph.nodes
                if (estimate := estimate_node(node, "forward_step", train_calibration)) is not None
            ]
        )
        updated_forward_ms = (
            sum(n.estimated_time_ms for n in updated_prefill_estimates)
            * train_calibration.forward_parallelism_factor
        )
        updated_backward_ms, _, _, _ = estimate_backward_from_graph_nodes(
            updated_prefill_estimates,
            train_calibration,
            gradient_bytes_by_scope,
        )
        return updated_forward_ms, updated_backward_ms

    # --- Forward calibration ---
    forward_ratio = clamp_ratio(
        measured_forward_ms / forward_time_ms if forward_time_ms > 0 else 1.0,
        config,
    )

    forward_cfg = config.setdefault('common', {}).setdefault('forward', {})
    old_forward_eff = train_calibration.effective_tflops_scale
    old_forward_parallelism = train_calibration.forward_parallelism_factor

    new_forward_eff = old_forward_eff / forward_ratio
    eff_min, eff_max = limit_for(config, "effective_tflops_scale", 0.5, 1.0)
    new_forward_eff = clamp(new_forward_eff, eff_min, eff_max)

    if abs(new_forward_eff - old_forward_eff) > PARAM_UPDATE_EPSILON:
        forward_cfg['effective_tflops_scale'] = new_forward_eff
        train_calibration.effective_tflops_scale = new_forward_eff
        recalculated_forward_ms, _ = reestimate_graph_times()
        calibrated_times['forward'] = recalculated_forward_ms
        print(f"[Calibration] effective_tflops_scale: {old_forward_eff:.4f} -> {new_forward_eff:.4f} (forward_ratio={forward_ratio:.3f})")

    residual_forward_ratio = (
        measured_forward_ms / calibrated_times['forward']
        if calibrated_times['forward'] > 0
        else 1.0
    )
    if abs(1.0 - residual_forward_ratio) > threshold_for(config, "forward"):
        new_forward_parallelism = old_forward_parallelism * residual_forward_ratio
        pf_min, pf_max = limit_for(config, "forward_parallelism_factor", 0.1, 1.0)
        new_forward_parallelism = clamp(new_forward_parallelism, pf_min, pf_max)
        if abs(new_forward_parallelism - old_forward_parallelism) > PARAM_UPDATE_EPSILON:
            forward_cfg['parallelism_factor'] = new_forward_parallelism
            train_calibration.forward_parallelism_factor = new_forward_parallelism
            recalculated_forward_ms, _ = reestimate_graph_times()
            calibrated_times['forward'] = recalculated_forward_ms
            print(
                f"[Calibration] forward_parallelism_factor: "
                f"{old_forward_parallelism:.4f} -> {new_forward_parallelism:.4f} "
                f"(residual_ratio={residual_forward_ratio:.3f})"
            )

    # --- Backward compute calibration ---
    measured_backward_compute_ms = max(measured_backward_ms - measured_backward_comm_ms, 0.0)
    backward_ratio = clamp_ratio(
        measured_backward_compute_ms / backward_time_ms if backward_time_ms > 0 else 1.0,
        config,
    )

    if abs(1.0 - backward_ratio) > threshold_for(config, "backward_compute"):
        old_efficiency = train_calibration.backward_efficiency_scale
        old_backward_parallelism = train_calibration.parallelism_factor
        new_efficiency = old_efficiency / backward_ratio
        eff_min, eff_max = limit_for(config, "backward_efficiency_scale", 0.01, 0.5)
        new_efficiency = clamp(new_efficiency, eff_min, eff_max)

        if abs(new_efficiency - old_efficiency) > PARAM_UPDATE_EPSILON:
            config.setdefault('common', {}).setdefault('backward', {})['backward_efficiency_scale'] = new_efficiency
            train_calibration.backward_efficiency_scale = new_efficiency
            _, recalculated_backward_ms = reestimate_graph_times()
            calibrated_times['backward'] = recalculated_backward_ms

            print(f"[Calibration] backward_efficiency_scale: {old_efficiency:.4f} -> {new_efficiency:.4f} (backward_ratio={backward_ratio:.3f})")

        residual_backward_ratio = (
            measured_backward_compute_ms / calibrated_times['backward']
            if calibrated_times['backward'] > 0
            else 1.0
        )
        if abs(1.0 - residual_backward_ratio) > threshold_for(config, "backward_compute"):
            new_backward_parallelism = old_backward_parallelism * residual_backward_ratio
            bp_min, bp_max = limit_for(config, "backward_parallelism_factor", 0.1, 1.0)
            new_backward_parallelism = clamp(new_backward_parallelism, bp_min, bp_max)
            if abs(new_backward_parallelism - old_backward_parallelism) > PARAM_UPDATE_EPSILON:
                config.setdefault('common', {}).setdefault('backward', {})['parallelism_factor'] = new_backward_parallelism
                train_calibration.parallelism_factor = new_backward_parallelism
                _, recalculated_backward_ms = reestimate_graph_times()
                calibrated_times['backward'] = recalculated_backward_ms
                print(
                    f"[Calibration] backward_parallelism_factor: "
                    f"{old_backward_parallelism:.4f} -> {new_backward_parallelism:.4f} "
                    f"(residual_ratio={residual_backward_ratio:.3f})"
                )

    # --- Overhead scale calibration (iterative, for small seq_len) ---
    for _ in range(3):
        residual_after_parallelism = (
            measured_backward_compute_ms / calibrated_times['backward']
            if calibrated_times['backward'] > 0
            else 1.0
        )
        if abs(1.0 - residual_after_parallelism) <= threshold_for(config, "backward_compute"):
            break
        old_overhead_scale = train_calibration.overhead_scale
        new_overhead_scale = old_overhead_scale * residual_after_parallelism
        oh_min, oh_max = limit_for(config, "overhead_scale", 0.05, 5.0)
        new_overhead_scale = clamp(new_overhead_scale, oh_min, oh_max)
        if abs(new_overhead_scale - old_overhead_scale) <= PARAM_UPDATE_EPSILON:
            break
        config.setdefault('common', {}).setdefault('backward', {})['overhead_scale'] = new_overhead_scale
        train_calibration.overhead_scale = new_overhead_scale
        _, recalibrated_backward_ms = reestimate_graph_times()
        calibrated_times['backward'] = recalibrated_backward_ms
        print(
            f"[Calibration] overhead_scale: "
            f"{old_overhead_scale:.4f} -> {new_overhead_scale:.4f} "
            f"(residual={residual_after_parallelism:.3f})"
        )

    # --- Optimizer calibration ---
    old_opt_factor = train_calibration.optimizer_scale_factor
    optimizer_ratio = clamp_ratio(
        measured_optimizer_ms / optimizer_time_ms if optimizer_time_ms > 0 else 1.0,
        config,
    )

    if abs(1.0 - optimizer_ratio) > threshold_for(config, "optimizer"):
        new_opt_factor = old_opt_factor * optimizer_ratio
        factor_min, factor_max = limit_for(config, "optimizer_scale_factor", 0.5, 3.0)
        new_opt_factor = clamp(new_opt_factor, factor_min, factor_max)

        if abs(new_opt_factor - old_opt_factor) > PARAM_UPDATE_EPSILON:
            config.setdefault('common', {}).setdefault('optimizer', {})['scale_factor'] = new_opt_factor
            train_calibration.optimizer_scale_factor = new_opt_factor
            calibrated_times['optimizer'] = optimizer_time_ms * (new_opt_factor / old_opt_factor)

            print(f"[Calibration] optimizer_scale_factor: {old_opt_factor:.4f} -> {new_opt_factor:.4f} (optimizer_ratio={optimizer_ratio:.3f})")

    if ddp_enabled and world_size > 1 and measured_backward_comm_ms > 0 and comm_time_ms_original > 0:
        comm_ratio = clamp_ratio(measured_backward_comm_ms / comm_time_ms_original, config)
        if abs(1.0 - comm_ratio) > threshold_for(config, "backward_comm"):
            old_bandwidth = config.setdefault("single_ddp", {}).setdefault("communication", {}).get(
                "gradient_allreduce_tflops",
                train_calibration.gradient_allreduce_tflops,
            )
            if comm_bytes > 0 and measured_backward_comm_ms > comm_latency_ms:
                payload_time_ms = max(measured_backward_comm_ms - comm_latency_ms, 1e-6)
                new_bandwidth = comm_bytes / ((payload_time_ms / 1e3) * 1e9)
            else:
                new_bandwidth = old_bandwidth / comm_ratio
            bw_min, bw_max = limit_for(config, "gradient_allreduce_tflops", 1.0, 1000.0)
            new_bandwidth = clamp(new_bandwidth, bw_min, bw_max)
            config["single_ddp"]["communication"]["gradient_allreduce_tflops"] = new_bandwidth
            train_calibration.gradient_allreduce_tflops = new_bandwidth
            adjustment["single_ddp.communication.gradient_allreduce_tflops"] = {
                "old": old_bandwidth,
                "new": new_bandwidth,
            }
            print(
                f"[Calibration] gradient_allreduce_tflops: {old_bandwidth:.4f} -> {new_bandwidth:.4f} "
                f"(backward_comm_ratio={comm_ratio:.3f})"
            )

    return calibrated_times, adjustment


def calibrate_tp(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    train_calibration: TrainCalibration,
    config: dict[str, Any],
    forward_time_ms: float,
    backward_time_ms: float,
    optimizer_time_ms: float,
    measured_forward_ms: float,
    measured_backward_ms: float,
    measured_backward_total_ms: float,
    measured_optimizer_ms: float,
    measured_backward_comm_ms: float = 0.0,
    estimated_backward_comm_ms: float = 0.0,
    rank: int = 0,
    num_warmup: int = 2,
    num_measure: int = 5,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Calibrate TP mode parameters based on measured vs estimated time.

    This function replaces the inline calibration code in mvp_train_app.py.

    Args:
        model: Loaded TP model on GPU
        input_ids: Input tensor
        labels: Labels tensor
        train_calibration: TrainCalibration object to be updated
        config: Config dict (modified in-place)
        forward_time_ms: Estimated forward time (ms)
        backward_time_ms: Estimated backward time (ms)
        optimizer_time_ms: Estimated optimizer time (ms)
        measured_forward_ms: Measured forward time (ms)
        measured_backward_ms: Measured backward compute time (ms)
        measured_backward_total_ms: Measured total backward time (ms)
        measured_optimizer_ms: Measured optimizer time (ms)
        measured_backward_comm_ms: Measured backward communication time (ms)
        estimated_backward_comm_ms: Estimated backward communication time (ms)
        rank: Current process rank
        num_warmup: Warmup steps
        num_measure: Measurement steps

    Returns:
        (calibrated_times, adjustment_dict)
        - calibrated_times: dict with 'forward', 'backward', 'optimizer' keys
        - adjustment_dict: dict with parameter changes for config
    """
    adjustment = {}
    calibrated_times = {
        'forward': forward_time_ms,
        'backward': backward_time_ms,
        'optimizer': optimizer_time_ms,
    }

    if rank != 0:
        return calibrated_times, adjustment

    tp_cfg = config.get('tp', {})
    tp_backward_cfg = tp_cfg.get('backward', {})
    tp_comm_cfg = tp_cfg.get('communication', {})

    # --- TP Forward calibration ---
    old_forward_eff = tp_backward_cfg.get('tp_forward_efficiency', train_calibration.tp_forward_efficiency)
    forward_ratio = clamp_ratio(
        measured_forward_ms / forward_time_ms if forward_time_ms > 0 else 1.0,
        config,
    )

    new_forward_eff = old_forward_eff / forward_ratio if forward_ratio > 0 else old_forward_eff
    eff_min, eff_max = limit_for(config, "tp_forward_efficiency", 0.01, 0.5)
    new_forward_eff = clamp(new_forward_eff, eff_min, eff_max)

    if abs(new_forward_eff - old_forward_eff) > PARAM_UPDATE_EPSILON:
        tp_backward_cfg['tp_forward_efficiency'] = new_forward_eff
        config.setdefault('tp', {})['backward'] = tp_backward_cfg
        train_calibration.tp_forward_efficiency = new_forward_eff
        # Recalculate forward_time_ms from base FLOPs using new efficiency
        # Forward FLOPs = forward_time_ms * gemm_tflops * old_eff * 1e12 / 1000
        base_forward_flops = forward_time_ms * train_calibration.gemm_tflops * old_forward_eff * 1e12 / 1000
        new_forward_time_ms = base_forward_flops / (train_calibration.gemm_tflops * new_forward_eff * 1e12) * 1e3
        calibrated_times['forward'] = new_forward_time_ms

        print(f"[TP Calibration] tp_forward_efficiency: {old_forward_eff:.4f} -> {new_forward_eff:.4f} (forward_ratio={forward_ratio:.3f})")

    # --- TP Backward calibration using total backward time ---
    old_efficiency = tp_backward_cfg.get('tp_backward_efficiency', train_calibration.tp_backward_efficiency)
    estimated_backward_compute_ms = max(backward_time_ms - estimated_backward_comm_ms, 0.0)
    backward_compute_ratio = clamp_ratio(
        measured_backward_ms / estimated_backward_compute_ms if estimated_backward_compute_ms > 0 else 1.0,
        config,
    )

    if abs(1.0 - backward_compute_ratio) > threshold_for(config, "backward_compute"):
        new_efficiency = old_efficiency / backward_compute_ratio
        eff_min, eff_max = limit_for(config, "tp_backward_efficiency", 0.01, 0.2)
        new_efficiency = clamp(new_efficiency, eff_min, eff_max)

        if abs(new_efficiency - old_efficiency) > PARAM_UPDATE_EPSILON:
            tp_backward_cfg['tp_backward_efficiency'] = new_efficiency
            config.setdefault('tp', {})['backward'] = tp_backward_cfg
            train_calibration.tp_backward_efficiency = new_efficiency

            print(f"[TP Calibration] tp_backward_efficiency: {old_efficiency:.4f} -> {new_efficiency:.4f} (backward_compute_ratio={backward_compute_ratio:.3f})")

    # Also adjust common.backward parameters (affects per-node estimation in unified estimator)
    common_bwd = config.setdefault('common', {}).setdefault('backward', {})
    old_bwd_eff = common_bwd.get('backward_efficiency_scale', 0.07)
    new_bwd_eff = old_bwd_eff / backward_compute_ratio
    bwd_eff_min, bwd_eff_max = limit_for(config, "backward_efficiency_scale", 0.0005, 0.5)
    new_bwd_eff = clamp(new_bwd_eff, bwd_eff_min, bwd_eff_max)
    if abs(new_bwd_eff - old_bwd_eff) > PARAM_UPDATE_EPSILON:
        common_bwd['backward_efficiency_scale'] = new_bwd_eff
        train_calibration.backward_efficiency_scale = new_bwd_eff
        print(f"[TP Calibration] backward_efficiency_scale: {old_bwd_eff:.4f} -> {new_bwd_eff:.4f}")

    old_overhead = common_bwd.get('overhead_scale', 0.3)
    new_overhead = old_overhead * backward_compute_ratio
    oh_min, oh_max = limit_for(config, "overhead_scale", 0.05, 5.0)
    new_overhead = clamp(new_overhead, oh_min, oh_max)
    if abs(new_overhead - old_overhead) > PARAM_UPDATE_EPSILON:
        common_bwd['overhead_scale'] = new_overhead
        train_calibration.overhead_scale = new_overhead
        print(f"[TP Calibration] overhead_scale: {old_overhead:.4f} -> {new_overhead:.4f}")

    # --- TP Backward Comm calibration (overlap_ratio) ---
    if measured_backward_comm_ms > 0 and estimated_backward_comm_ms > 0:
        old_overlap = tp_comm_cfg.get('overlap_ratio', train_calibration.overlap_ratio)
        backward_comm_ratio = clamp_ratio(measured_backward_comm_ms / estimated_backward_comm_ms, config)

        if abs(1.0 - backward_comm_ratio) > threshold_for(config, "backward_comm"):
            if old_overlap < 1.0:
                estimated_raw_comm_ms = estimated_backward_comm_ms / max(1.0 - old_overlap, 1e-6)
                new_overlap = 1.0 - (measured_backward_comm_ms / max(estimated_raw_comm_ms, 1e-6))
            else:
                new_overlap = old_overlap / backward_comm_ratio
            overlap_min, overlap_max = limit_for(config, "overlap_ratio", 0.0, 0.9)
            new_overlap = clamp(new_overlap, overlap_min, overlap_max)

            if abs(new_overlap - old_overlap) > PARAM_UPDATE_EPSILON:
                tp_comm_cfg['overlap_ratio'] = new_overlap
                config.setdefault('tp', {})['communication'] = tp_comm_cfg
                train_calibration.overlap_ratio = new_overlap

                print(f"[TP Calibration] overlap_ratio: {old_overlap:.4f} -> {new_overlap:.4f} (backward_comm_ratio={backward_comm_ratio:.3f})")

    calibrated_backward_compute_ms = measured_backward_ms if measured_backward_ms > 0 else estimated_backward_compute_ms
    calibrated_backward_comm_ms = measured_backward_comm_ms if measured_backward_comm_ms > 0 else estimated_backward_comm_ms
    calibrated_times["backward"] = calibrated_backward_compute_ms + calibrated_backward_comm_ms

    # --- Optimizer calibration ---
    old_opt_eff = tp_comm_cfg.get("optimizer_efficiency", 0.45)
    optimizer_ratio = clamp_ratio(
        measured_optimizer_ms / optimizer_time_ms if optimizer_time_ms > 0 else 1.0,
        config,
    )
    new_opt_eff = old_opt_eff / optimizer_ratio if optimizer_ratio > 0 else old_opt_eff
    eff_min, eff_max = limit_for(config, "optimizer_efficiency", 0.1, 0.9)
    new_opt_eff = clamp(new_opt_eff, eff_min, eff_max)

    if abs(new_opt_eff - old_opt_eff) > PARAM_UPDATE_EPSILON:
        config.setdefault("tp", {}).setdefault("communication", {})["optimizer_efficiency"] = new_opt_eff
        train_calibration.tp_config.setdefault("communication", {})["optimizer_efficiency"] = new_opt_eff
        calibrated_times["optimizer"] = optimizer_time_ms * (old_opt_eff / new_opt_eff) if new_opt_eff > 0 else optimizer_time_ms

        print(f"[TP Calibration] optimizer_efficiency: {old_opt_eff:.4f} -> {new_opt_eff:.4f} (optimizer_ratio={optimizer_ratio:.3f})")

    return calibrated_times, adjustment


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for calibration")

    # Initialize distributed if needed
    device = init_distributed(args)
    is_primary = is_primary_rank()

    # Load config
    config = load_config(args.config)

    if is_primary:
        print(f"\n{'='*60}")
        print("Training Parameter Calibration")
        print(f"{'='*60}")
        print(f"Mode: {args.mode}")
        print(f"Model: {args.model_path}")
        print(f"TP size: {args.tp_size if args.mode == 'tp' else 1}")
        print(f"Batch size: {args.batch_size}, Seq len: {args.seq_len}")
        print(f"Warmup: {args.warmup}, Measure: {args.measure}")
        print()

    # Load model
    if is_primary:
        print("Loading model...")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    # Create a prompt that can be tokenized to the specified seq_len
    # Repeat a short prompt to reach the desired length
    prompt = "Hello world "
    # Calculate how many times to repeat
    single_token_len = len(tokenizer(prompt, return_tensors="pt")["input_ids"][0])
    repeat_times = (args.seq_len // single_token_len) + 2  # Add extra to ensure we have enough
    prompt = (prompt * repeat_times)[:args.seq_len * 2]  # Approximate, will be truncated after tokenize
    input_ids, attention_mask = prepare_inputs(tokenizer, prompt, device)
    # Truncate or pad to exact seq_len
    if input_ids.shape[1] > args.seq_len:
        input_ids = input_ids[:, :args.seq_len]
        attention_mask = attention_mask[:, :args.seq_len]
    elif input_ids.shape[1] < args.seq_len:
        pad_len = args.seq_len - input_ids.shape[1]
        input_ids = torch.cat([input_ids, torch.zeros((1, pad_len), dtype=input_ids.dtype, device=device)], dim=1)
        attention_mask = torch.cat([attention_mask, torch.ones((1, pad_len), dtype=attention_mask.dtype, device=device)], dim=1)
    labels = input_ids.clone()

    # For TP mode, we need to load the TP model for accurate measurement
    if args.mode == "tp" and args.tp_size > 1:
        if is_primary:
            print("Loading TP model...")
        # Load with TP support
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            tp_plan="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)

    model.eval()
    model.to(device)

    # Build calibration
    train_calibration = build_train_calibration(config, device)

    if is_primary:
        print(f"Hardware: {train_calibration.device_name}")
        print(f"GEMM TFLOPs: {train_calibration.gemm_tflops:.2f}")
        print(f"Memory BW: {train_calibration.memory_bandwidth_gbps:.2f} GB/s")
        print()

    # Parse physical devices
    if args.physical_devices:
        physical_devices = [int(x.strip()) for x in args.physical_devices.split(',')]
    else:
        # Default: use [0, 1] for 2-GPU system
        physical_devices = [0, 1] if args.tp_size > 1 else [0]

    # Run calibration
    if args.mode == "tp":
        results, adjustment = run_tp_calibration(
            model, input_ids, attention_mask, labels,
            train_calibration, config, args.tp_size,
            args.warmup, args.measure,
            physical_devices=physical_devices
        )
    else:
        results, adjustment = run_single_card_calibration(
            model, input_ids, attention_mask, labels,
            train_calibration, config,
            args.warmup, args.measure
        )

    if is_primary:
        print("=== Calibration Results ===")
        for r in results:
            print(f"  {r.phase:12s}: estimated={r.estimated_ms:8.2f}ms, "
                  f"measured={r.measured_ms:8.2f}ms, "
                  f"error={r.error_pct:6.2f}%, ratio={r.ratio:.4f}")

        print("\n=== Adjusted Parameters ===")
        if adjustment:
            for section_key, params in adjustment.items():
                # Handle nested paths like 'common.backward' or 'tp.backward'
                parts = section_key.split('.')
                target = config
                for part in parts:
                    if part not in target:
                        target[part] = {}
                    target = target[part]

                for param_key, value in params.items():
                    old_value = target.get(param_key)
                    target[param_key] = value
                    print(f"  {section_key}.{param_key}: {old_value} -> {value:.4f}")
        else:
            print("  No adjustments needed (error < 10%)")

        # Save calibrated config
        output_path = args.output or args.config
        save_config(config, output_path)
        print(f"\nCalibrated config saved to: {output_path}")
        print(f"{'='*60}\n")

    # Cleanup distributed
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
