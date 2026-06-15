from __future__ import annotations

import statistics
import time
from typing import Any

import torch

from mvp_estimator import collective_bandwidth_gbps, collective_latency_ms
from mvp_llama_train_runtime import benchmark_llama_backbone_probe
from mvp_types import ExecutionConfig, HardwareCalibration


INIT_STD = 0.02
LOSS_SCALE = 1.0e-3
WARMUP_STEPS = 6
MEASURE_ITERATIONS = 2


def _stable_avg(vals: list[float]) -> tuple[float, float, list[float]]:
    median_val = statistics.median(vals)
    stable_cutoff = median_val * 0.5
    stable_vals = [value for value in vals if value >= stable_cutoff]
    if not stable_vals:
        stable_vals = list(vals)
    return sum(stable_vals) / len(stable_vals), stable_cutoff, stable_vals


def _dtype_num_bytes(dtype_name: str) -> int:
    text = str(dtype_name or "float16").lower()
    if text in {"bf16", "bfloat16", "fp16", "float16", "half"}:
        return 2
    if text in {"fp32", "float32"}:
        return 4
    return 2


def _gemm_time_ms(
    m: int, k: int, n: int, calibration: HardwareCalibration, dtype_bytes: int
) -> float:
    flops = float(2 * m * k * n)
    bytes_moved = float((m * k + k * n + m * n) * dtype_bytes)
    compute_ms = flops / (calibration.gemm_tflops * 1.0e12) * 1.0e3
    memory_ms = bytes_moved / (calibration.memory_bandwidth_gbps * 1.0e9) * 1.0e3
    return max(compute_ms, memory_ms) + calibration.launch_overhead_ms


def _elementwise_time_ms(
    numel: int, calibration: HardwareCalibration, dtype_bytes: int, scale: float = 2.0
) -> float:
    bytes_moved = float(numel * dtype_bytes * scale)
    return (
        bytes_moved / (calibration.memory_bandwidth_gbps * 1.0e9) * 1.0e3
        + calibration.launch_overhead_ms
    )


def _reduction_time_ms(
    numel: int, calibration: HardwareCalibration, dtype_bytes: int
) -> float:
    bytes_moved = float(numel * dtype_bytes * 2.0)
    return (
        bytes_moved / (calibration.memory_bandwidth_gbps * 1.0e9) * 1.0e3
        + calibration.launch_overhead_ms
    )


def _transfer_time_ms(activation_bytes: int, execution: ExecutionConfig) -> float:
    if execution.world_size <= 1:
        return 0.0
    bandwidth_gbps = collective_bandwidth_gbps(execution)
    base_latency_ms = collective_latency_ms(execution)
    return base_latency_ms + activation_bytes / (bandwidth_gbps * 1.0e9) * 1.0e3


def _synchronize(device_backend: str, device_ids: list[int]) -> None:
    if device_backend == "musa" and hasattr(torch, "musa"):
        for device_id in device_ids:
            torch.musa.synchronize(device_id)
    elif device_backend == "cuda":
        for device_id in device_ids:
            torch.cuda.synchronize(device_id)


def _create_pp1_state(model_desc: dict[str, Any], device: str, dtype: torch.dtype) -> dict[str, Any]:
    m = int(model_desc.get("sequence_hidden_tokens", 768))
    k = int(model_desc.get("hidden_size", 768))
    n = int(model_desc.get("stage0_out_features", 768))
    stage1_out = int(model_desc.get("stage1_out_features", k))
    x = torch.randn((m, k), device=device, dtype=dtype) * INIT_STD
    w0 = torch.nn.Parameter(torch.randn((k, n), device=device, dtype=dtype) * INIT_STD)
    w1 = torch.nn.Parameter(
        torch.randn((n, stage1_out), device=device, dtype=dtype) * INIT_STD
    )
    probe = torch.nn.Parameter(torch.tensor(1.0, device=device, dtype=torch.float32))
    opt = torch.optim.SGD([w0, w1, probe], lr=1e-3)
    return {
        "x": x,
        "w0": w0,
        "w1": w1,
        "probe": probe,
        "opt": opt,
    }


def _create_pp2_state(
    model_desc: dict[str, Any], device0: str, device1: str, dtype: torch.dtype
) -> dict[str, Any]:
    m = int(model_desc.get("sequence_hidden_tokens", 768))
    k = int(model_desc.get("hidden_size", 768))
    n = int(model_desc.get("stage0_out_features", 768))
    stage1_out = int(model_desc.get("stage1_out_features", k))
    x0 = torch.randn((m, k), device=device0, dtype=dtype) * INIT_STD
    w0 = torch.nn.Parameter(torch.randn((k, n), device=device0, dtype=dtype) * INIT_STD)
    w1 = torch.nn.Parameter(
        torch.randn((n, stage1_out), device=device1, dtype=dtype) * INIT_STD
    )
    probe0 = torch.nn.Parameter(torch.tensor(1.0, device=device0, dtype=torch.float32))
    probe1 = torch.nn.Parameter(torch.tensor(1.0, device=device1, dtype=torch.float32))
    opt0 = torch.optim.SGD([w0, probe0], lr=1e-3)
    opt1 = torch.optim.SGD([w1, probe1], lr=1e-3)
    return {
        "x0": x0,
        "w0": w0,
        "w1": w1,
        "probe0": probe0,
        "probe1": probe1,
        "opt0": opt0,
        "opt1": opt1,
    }


def _pp1_microstep(state: dict[str, Any]) -> None:
    state["opt"].zero_grad(set_to_none=True)
    y = torch.matmul(state["x"], state["w0"])
    y = torch.relu(y)
    y = torch.matmul(y, state["w1"])
    loss = ((state["probe"] * y.float() * LOSS_SCALE) ** 2).mean()
    loss.backward()
    state["opt"].step()


def _pp2_microstep(state: dict[str, Any], device1: str) -> None:
    state["opt0"].zero_grad(set_to_none=True)
    state["opt1"].zero_grad(set_to_none=True)
    y0 = torch.matmul(state["x0"], state["w0"])
    y0 = torch.relu(y0)
    y1 = y0.detach().to("cpu").to(device1, dtype=state["w1"].dtype)
    y1 = torch.matmul(y1, state["w1"])
    y1 = torch.relu(y1)
    loss0 = ((state["probe0"] * y0.float() * LOSS_SCALE) ** 2).mean() * 0.5
    loss1 = ((state["probe1"] * y1.float() * LOSS_SCALE) ** 2).mean()
    loss0.backward()
    loss1.backward()
    state["opt0"].step()
    state["opt1"].step()


def benchmark_train_microbatch_ms(
    model_desc: dict[str, Any],
    parallel_cfg: dict[str, Any],
    device_backend: str,
    runs: int = 3,
) -> dict[str, Any]:
    if model_desc.get("train_workload") == "llama_backbone_probe":
        return benchmark_llama_backbone_probe(
            model_path=str(model_desc["model_path"]),
            samples_path=str(model_desc["train_samples_path"]),
            pipeline_parallel_size=int(parallel_cfg.get("pipeline_parallel_size", 1)),
            tensor_parallel_size=int(parallel_cfg.get("tensor_parallel_size", 1)),
            microbatch_num=int(parallel_cfg.get("microbatch_num", 1)),
            global_batch_size=int(parallel_cfg.get("global_batch_size", 8)),
            device_backend=device_backend,
            runs=runs,
            warmups=2,
            max_seq_len=int(model_desc.get("max_seq_len", 8)),
            split_index=int(model_desc.get("pipeline_split_index", 16)),
            lora_rank=int(model_desc.get("lora_rank", 8)),
            adapter_only=bool(model_desc.get("adapter_only", False)),
        )
    if model_desc.get("train_workload") == "lora_feature_probe":
        raise RuntimeError(
            "lora_feature_probe was removed; use lora_vocab_adapter or "
            "llama_backbone_probe so training matches train-infer-estimation."
        )

    pp_size = int(parallel_cfg.get("pipeline_parallel_size", 1))
    dtype_name = str(parallel_cfg.get("dtype") or model_desc.get("dtype") or "float16").lower()
    if dtype_name in {"bf16", "bfloat16"}:
        dtype = torch.bfloat16
    elif dtype_name in {"fp16", "float16", "half"}:
        dtype = torch.float16
    else:
        dtype = torch.float32
    if pp_size <= 1:
        device0 = f"{device_backend}:0" if device_backend != "cpu" else "cpu"
        state = _create_pp1_state(model_desc, device0, dtype)
        for _ in range(WARMUP_STEPS):
            _pp1_microstep(state)
        _synchronize(device_backend, [0])
        vals = []
        for _ in range(runs):
            _synchronize(device_backend, [0])
            start = time.perf_counter()
            for _ in range(MEASURE_ITERATIONS):
                _pp1_microstep(state)
            _synchronize(device_backend, [0])
            vals.append((time.perf_counter() - start) * 1000.0 / MEASURE_ITERATIONS)
        avg_val, stable_cutoff, stable_vals = _stable_avg(vals)
        return {
            "profile_kind": "online_microbatch_probe",
            "timings_ms": vals,
            "avg_ms": avg_val,
            "median_ms": statistics.median(vals),
            "stable_cutoff_ms": stable_cutoff,
            "stable_timings_ms": stable_vals,
        }

    device0 = f"{device_backend}:0"
    device1 = f"{device_backend}:1"
    state = _create_pp2_state(model_desc, device0, device1, dtype)
    for _ in range(WARMUP_STEPS):
        _pp2_microstep(state, device1)
    _synchronize(device_backend, [0, 1])
    vals = []
    for _ in range(runs):
        _synchronize(device_backend, [0, 1])
        start = time.perf_counter()
        for _ in range(MEASURE_ITERATIONS):
            _pp2_microstep(state, device1)
        _synchronize(device_backend, [0, 1])
        vals.append((time.perf_counter() - start) * 1000.0 / MEASURE_ITERATIONS)
    avg_val, stable_cutoff, stable_vals = _stable_avg(vals)
    return {
        "profile_kind": "online_microbatch_probe",
        "timings_ms": vals,
        "avg_ms": avg_val,
        "median_ms": statistics.median(vals),
        "stable_cutoff_ms": stable_cutoff,
        "stable_timings_ms": stable_vals,
    }


def estimate_train_iteration(
    model_desc: dict[str, Any],
    parallel_cfg: dict[str, Any],
    hardware_topology: dict[str, Any],
    calibration: HardwareCalibration,
    execution: ExecutionConfig,
    runtime_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    hidden_size = int(model_desc.get("hidden_size", 768))
    stage0_out = int(model_desc.get("stage0_out_features", hidden_size))
    stage1_out = int(model_desc.get("stage1_out_features", hidden_size))
    seq_m = int(model_desc.get("sequence_hidden_tokens", hidden_size))
    dtype_name = str(
        parallel_cfg.get("dtype") or model_desc.get("dtype") or "float16"
    )
    dtype_bytes = _dtype_num_bytes(dtype_name)
    pp_size = int(parallel_cfg.get("pipeline_parallel_size", 1))
    tp_size = int(parallel_cfg.get("tensor_parallel_size", 1))
    microbatch_num = int(parallel_cfg.get("microbatch_num", 1))

    gemm0_ms = _gemm_time_ms(
        m=seq_m,
        k=hidden_size,
        n=stage0_out,
        calibration=calibration,
        dtype_bytes=dtype_bytes,
    )
    relu0_ms = _elementwise_time_ms(
        seq_m * stage0_out, calibration=calibration, dtype_bytes=dtype_bytes
    )
    loss0_ms = _elementwise_time_ms(
        seq_m * stage0_out, calibration=calibration, dtype_bytes=4, scale=3.0
    ) + _reduction_time_ms(
        seq_m * stage0_out, calibration=calibration, dtype_bytes=4
    )

    activation_bytes = seq_m * stage0_out * dtype_bytes

    if runtime_profile is not None:
        profile_kind = str(runtime_profile.get("profile_kind", "online_microbatch_probe"))
        if profile_kind == "online_llama_backbone_probe":
            train_iteration_time_ms = float(runtime_profile["avg_ms"])
            microbatch_time_ms = train_iteration_time_ms / max(microbatch_num, 1)
            estimation_formula = (
                "train_iteration_time_ms = directly profiled optimizer-step time "
                "for the same Llama backbone training workload and topology"
            )
        else:
            microbatch_time_ms = float(runtime_profile["avg_ms"])
            train_iteration_time_ms = microbatch_time_ms * microbatch_num
            estimation_formula = (
                "train_iteration_time_ms = microbatch_num * independently profiled "
                "microbatch_time_ms on the same topology"
            )
        return {
            "model_name": model_desc.get("name", "train_probe"),
            "hardware_topology": hardware_topology,
            "parallel_config": parallel_cfg,
            "microbatch_time_ms": microbatch_time_ms,
            "train_iteration_time_ms": train_iteration_time_ms,
            "stage_breakdown_ms": {
                "stage0_ms": None,
                "stage1_ms": None,
                "loss_and_optimizer_ms": None,
                "activation_transfer_ms": None,
            },
            "estimation_formula": estimation_formula,
            "probe_shape": {
                "sequence_hidden_tokens": seq_m,
                "hidden_size": hidden_size,
                "stage0_out_features": stage0_out,
                "stage1_out_features": stage1_out,
                "dtype": dtype_name,
                "tensor_parallel_size": tp_size,
            },
            "runtime_profile": runtime_profile,
        }

    if pp_size <= 1 and tp_size <= 1:
        gemm1_ms = _gemm_time_ms(
            m=seq_m,
            k=stage0_out,
            n=stage1_out,
            calibration=calibration,
            dtype_bytes=dtype_bytes,
        )
        microbatch_time_ms = (
            gemm0_ms
            + relu0_ms
            + gemm1_ms
            + _elementwise_time_ms(
                seq_m * stage1_out, calibration=calibration, dtype_bytes=dtype_bytes
            )
            + _elementwise_time_ms(
                seq_m * stage1_out, calibration=calibration, dtype_bytes=4, scale=3.0
            )
            + _reduction_time_ms(
                seq_m * stage1_out, calibration=calibration, dtype_bytes=4
            )
            + calibration.launch_overhead_ms * 2.0
        )
        stage_breakdown = {
            "stage0_ms": gemm0_ms + relu0_ms,
            "stage1_ms": gemm1_ms,
            "loss_and_optimizer_ms": microbatch_time_ms - gemm0_ms - relu0_ms - gemm1_ms,
            "activation_transfer_ms": 0.0,
        }
    elif pp_size <= 1:
        shard_out = max(1, stage1_out // max(tp_size, 1))
        gemm1_ms = _gemm_time_ms(
            m=seq_m,
            k=stage0_out,
            n=shard_out,
            calibration=calibration,
            dtype_bytes=dtype_bytes,
        )
        tp_transfer_ms = _transfer_time_ms(seq_m * shard_out * dtype_bytes, execution) * 2.0
        microbatch_time_ms = (
            gemm0_ms
            + relu0_ms
            + gemm1_ms
            + tp_transfer_ms
            + _elementwise_time_ms(
                seq_m * stage1_out, calibration=calibration, dtype_bytes=4, scale=3.0
            )
            + _reduction_time_ms(
                seq_m * stage1_out, calibration=calibration, dtype_bytes=4
            )
            + calibration.launch_overhead_ms * 3.0
        )
        stage_breakdown = {
            "stage0_ms": gemm0_ms + relu0_ms,
            "stage1_ms": gemm1_ms,
            "loss_and_optimizer_ms": microbatch_time_ms - gemm0_ms - relu0_ms - gemm1_ms - tp_transfer_ms,
            "activation_transfer_ms": tp_transfer_ms,
        }
    else:
        gemm1_ms = _gemm_time_ms(
            m=seq_m,
            k=stage0_out,
            n=stage1_out,
            calibration=calibration,
            dtype_bytes=dtype_bytes,
        )
        transfer_ms = _transfer_time_ms(activation_bytes, execution) * 2.0
        microbatch_time_ms = (
            gemm0_ms
            + relu0_ms
            + transfer_ms
            + gemm1_ms
            + _elementwise_time_ms(
                seq_m * stage1_out, calibration=calibration, dtype_bytes=dtype_bytes
            )
            + loss0_ms
            + _elementwise_time_ms(
                seq_m * stage1_out, calibration=calibration, dtype_bytes=4, scale=3.0
            )
            + _reduction_time_ms(
                seq_m * stage1_out, calibration=calibration, dtype_bytes=4
            )
            + calibration.launch_overhead_ms * 4.0
        )
        stage_breakdown = {
            "stage0_ms": gemm0_ms + relu0_ms + loss0_ms,
            "stage1_ms": gemm1_ms
            + _elementwise_time_ms(
                seq_m * stage1_out, calibration=calibration, dtype_bytes=dtype_bytes
            ),
            "loss_and_optimizer_ms": microbatch_time_ms
            - gemm0_ms
            - relu0_ms
            - gemm1_ms
            - transfer_ms,
            "activation_transfer_ms": transfer_ms,
        }

    train_iteration_time_ms = microbatch_time_ms * microbatch_num
    return {
        "model_name": model_desc.get("name", "train_probe"),
        "hardware_topology": hardware_topology,
        "parallel_config": parallel_cfg,
        "microbatch_time_ms": microbatch_time_ms,
        "train_iteration_time_ms": train_iteration_time_ms,
        "stage_breakdown_ms": stage_breakdown,
        "estimation_formula": (
            "train_iteration_time_ms = microbatch_num * "
            "(stage compute + activation/loss memory + topology-aware transfer)"
        ),
        "probe_shape": {
            "sequence_hidden_tokens": seq_m,
            "hidden_size": hidden_size,
            "stage0_out_features": stage0_out,
            "stage1_out_features": stage1_out,
            "dtype": dtype_name,
            "tensor_parallel_size": tp_size,
        },
    }
