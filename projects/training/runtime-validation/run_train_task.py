#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(ROOT, "../../.."))
MODEL_PATH = os.path.join(REPO_ROOT, "clj-proj", "model", "Meta-Llama-3.1-8B")
SAMPLES_PATH = os.path.join(ROOT, "train_samples.jsonl")
TOOL_ROOT = os.path.join(REPO_ROOT, "projects", "shared", "train-infer-estimation")

if TOOL_ROOT not in sys.path:
    sys.path.insert(0, TOOL_ROOT)

from mvp_llama_train_runtime import LlamaTrainRuntime, _synchronize  # type: ignore


def detect_backend() -> dict[str, Any]:
    try:
        import torch_musa  # noqa: F401
        import torch

        if hasattr(torch, "musa") and torch.musa.is_available():
            count = int(torch.musa.device_count())
            return {
                "backend": "musa",
                "device_count": count,
                "device_names": [torch.musa.get_device_name(i) for i in range(count)],
            }
    except Exception:
        pass

    try:
        import torch

        if torch.cuda.is_available():
            count = int(torch.cuda.device_count())
            return {
                "backend": "cuda",
                "device_count": count,
                "device_names": [torch.cuda.get_device_name(i) for i in range(count)],
            }
    except Exception:
        pass

    return {"backend": "cpu", "device_count": 0, "device_names": []}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real 5.1.6 training task with shared MVP runtime")
    parser.add_argument("--mode", choices=["single", "dual", "tp"], required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--samples-path", default=SAMPLES_PATH)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--microbatch-num", type=int, default=1)
    parser.add_argument("--global-batch-size", type=int, default=1)
    parser.add_argument("--max-seq-len", type=int, default=8)
    parser.add_argument("--split-index", type=int, default=16)
    parser.add_argument("--lora-rank", type=int, default=8)
    return parser.parse_args()


def _module_state_to_cpu(module) -> dict[str, Any]:
    state = {}
    for key, value in module.state_dict().items():
        state[key] = value.detach().cpu()
    return state


def _trainable_modules(runtime: LlamaTrainRuntime) -> list[Any]:
    modules = []
    if hasattr(runtime, "head"):
        modules.append(runtime.head)
    if hasattr(runtime, "tp_heads"):
        modules.append(runtime.tp_heads)
    return modules


def _parameter_norm(modules: list[Any]) -> float:
    total = 0.0
    for module in modules:
        for param in module.parameters():
            total += float(param.detach().float().norm().item())
    return total


def _checkpoint_payload(runtime: LlamaTrainRuntime, mode: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "mode": mode,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if hasattr(runtime, "head"):
        payload["head"] = _module_state_to_cpu(runtime.head)
    if hasattr(runtime, "tp_heads"):
        payload["tp_heads"] = [_module_state_to_cpu(module) for module in runtime.tp_heads]
    return payload


def _memory_snapshot(backend: str, sync_ids: list[int]) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "backend": backend,
        "per_device": [],
    }
    try:
        import torch

        if backend == "musa" and hasattr(torch, "musa"):
            for device_id in sync_ids:
                snapshot["per_device"].append(
                    {
                        "device_id": device_id,
                        "allocated_bytes": int(torch.musa.memory_allocated(device_id)),
                        "reserved_bytes": int(torch.musa.memory_reserved(device_id)),
                        "max_allocated_bytes": int(torch.musa.max_memory_allocated(device_id)),
                        "max_reserved_bytes": int(torch.musa.max_memory_reserved(device_id)),
                    }
                )
        elif backend == "cuda":
            for device_id in sync_ids:
                snapshot["per_device"].append(
                    {
                        "device_id": device_id,
                        "allocated_bytes": int(torch.cuda.memory_allocated(device_id)),
                        "reserved_bytes": int(torch.cuda.memory_reserved(device_id)),
                        "max_allocated_bytes": int(torch.cuda.max_memory_allocated(device_id)),
                        "max_reserved_bytes": int(torch.cuda.max_memory_reserved(device_id)),
                    }
                )
    except Exception as exc:
        snapshot["collection_error"] = repr(exc)
    return snapshot


def _reset_peak_memory(backend: str, sync_ids: list[int]) -> None:
    try:
        import torch

        if backend == "musa" and hasattr(torch, "musa"):
            for device_id in sync_ids:
                if hasattr(torch.musa, "reset_peak_memory_stats"):
                    torch.musa.reset_peak_memory_stats(device_id)
        elif backend == "cuda":
            for device_id in sync_ids:
                torch.cuda.reset_peak_memory_stats(device_id)
    except Exception:
        pass


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = detect_backend()
    required_devices = 2 if args.mode in {"dual", "tp"} else 1
    if env["backend"] == "cpu":
        raise RuntimeError("No MUSA/CUDA accelerator available")
    if int(env["device_count"]) < required_devices:
        raise RuntimeError(
            f"Mode={args.mode} requires {required_devices} devices, found {env['device_count']}"
        )

    pp_size = 2 if args.mode == "dual" else 1
    tp_size = 2 if args.mode == "tp" else 1
    sync_ids = [0, 1] if max(pp_size, tp_size) > 1 else [0]
    runtime = LlamaTrainRuntime(
        model_path=args.model_path,
        samples_path=args.samples_path,
        device_backend=env["backend"],
        pipeline_parallel_size=pp_size,
        tensor_parallel_size=tp_size,
        max_seq_len=args.max_seq_len,
        split_index=args.split_index,
        lora_rank=args.lora_rank,
        adapter_only=False,
    )
    modules = _trainable_modules(runtime)
    _reset_peak_memory(env["backend"], sync_ids)

    timings_ms = []
    norm_trace = []
    step_memory_trace = []
    for step in range(args.steps):
        _synchronize(env["backend"], sync_ids)
        started = time.perf_counter()
        runtime.train_iteration(
            microbatch_num=args.microbatch_num,
            global_batch_size=args.global_batch_size,
        )
        _synchronize(env["backend"], sync_ids)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        timings_ms.append(elapsed_ms)
        norm_trace.append(
            {
                "step": step + 1,
                "trainable_parameter_norm": _parameter_norm(modules),
                "elapsed_ms": elapsed_ms,
            }
        )
        step_memory_trace.append(
            {
                "step": step + 1,
                **_memory_snapshot(env["backend"], sync_ids),
            }
        )

    checkpoint_path = output_dir / f"{args.mode}_adapter_checkpoint.pt"
    import torch

    torch.save(_checkpoint_payload(runtime, args.mode), checkpoint_path)

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "success": True,
        "model_path": os.path.abspath(args.model_path),
        "samples_path": os.path.abspath(args.samples_path),
        "backend": env["backend"],
        "device_count": env["device_count"],
        "device_names": env["device_names"],
        "pipeline_parallel_size": pp_size,
        "tensor_parallel_size": tp_size,
        "parallel_mode": "tp" if tp_size > 1 else "pp" if pp_size > 1 else "single",
        "microbatch_num": args.microbatch_num,
        "global_batch_size": args.global_batch_size,
        "steps": args.steps,
        "timings_ms": timings_ms,
        "avg_step_ms": sum(timings_ms) / len(timings_ms),
        "checkpoint_path": str(checkpoint_path),
        "parameter_norm_trace": norm_trace,
        "step_memory_trace": step_memory_trace,
        "final_memory_snapshot": _memory_snapshot(env["backend"], sync_ids),
        "trainable_parameter_count": int(
            sum(param.numel() for module in modules for param in module.parameters())
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(output_dir / "summary.json")


if __name__ == "__main__":
    main()
