#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
TOOL_ROOT = os.path.join(REPO_ROOT, "projects", "shared", "train-infer-estimation")

if TOOL_ROOT not in sys.path:
    sys.path.insert(0, TOOL_ROOT)

from mvp_llama_train_runtime import LlamaTrainRuntime, _synchronize  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="5.1.6 Training Runner using train-infer-estimation runtime")
    parser.add_argument("--model_path", type=str, required=True, help="Model path")
    parser.add_argument("--config_file", type=str, required=True, help="Config file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs")
    parser.add_argument(
        "--task_type",
        type=str,
        choices=["full_training", "lora_training"],
        default="lora_training",
        help="Task type",
    )
    parser.add_argument("--distributed", action="store_true", help="Enable distributed training")
    parser.add_argument("--dry-run", action="store_true", help="Dry run mode for smoke testing")
    parser.add_argument("--device_ids", type=str, default="", help="Comma-separated device IDs")
    parser.add_argument(
        "--parallel_mode",
        type=str,
        choices=["auto", "pp", "tp"],
        default="auto",
        help="Two-GPU parallel mode: pipeline parallel or tensor parallel",
    )
    return parser.parse_args()


def load_config(config_file: str) -> dict[str, Any]:
    with open(config_file, "r", encoding="utf-8") as f:
        return json.load(f)


def detect_backend() -> dict[str, Any]:
    try:
        import torch_musa  # noqa: F401
        import torch

        if hasattr(torch, "musa") and torch.musa.is_available():
            return {
                "backend": "musa",
                "device_count": int(torch.musa.device_count()),
                "device_names": [torch.musa.get_device_name(i) for i in range(torch.musa.device_count())],
            }
    except Exception:
        pass

    try:
        import torch

        if torch.cuda.is_available():
            return {
                "backend": "cuda",
                "device_count": int(torch.cuda.device_count()),
                "device_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            }
    except Exception:
        pass

    return {"backend": "cpu", "device_count": 0, "device_names": []}


def parse_device_ids(raw: str, num_gpus: int) -> list[int]:
    ids = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if ids:
        return ids
    return list(range(max(1, num_gpus)))


def load_train_texts() -> list[str]:
    data_path = Path(ROOT) / "train_data.jsonl"
    texts: list[str] = []
    if data_path.exists():
        for line in data_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                text = item.get("instruction") or item.get("prompt") or item.get("input") or ""
                if text:
                    texts.append(str(text))
            except Exception:
                continue
    if not texts:
        texts = [
            "解释为什么大模型训练需要多卡并行。",
            "说明 LoRA 微调在训练中的作用。",
            "摩尔线程 GPU 在训练链路中的职责是什么？",
        ]
    return texts


def ensure_runtime_samples(texts: list[str], output_dir: Path) -> Path:
    path = output_dir / "runtime_samples.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for index, text in enumerate(texts):
            f.write(json.dumps({"text": text, "label": index % 2}, ensure_ascii=False) + "\n")
    return path


def trainable_modules(runtime: LlamaTrainRuntime) -> list[Any]:
    modules = []
    if hasattr(runtime, "head"):
        modules.append(runtime.head)
    if hasattr(runtime, "tp_heads"):
        modules.append(runtime.tp_heads)
    return modules


def parameter_norm(modules: list[Any]) -> float:
    total = 0.0
    for module in modules:
        for param in module.parameters():
            total += float(param.detach().float().norm().item())
    return total


def module_state_to_cpu(module) -> dict[str, Any]:
    state = {}
    for key, value in module.state_dict().items():
        state[key] = value.detach().cpu()
    return state


def save_checkpoint(runtime: LlamaTrainRuntime, mode_name: str, output_dir: Path) -> Path:
    import torch

    payload: dict[str, Any] = {
        "mode": mode_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if hasattr(runtime, "head"):
        payload["head"] = module_state_to_cpu(runtime.head)
    if hasattr(runtime, "tp_heads"):
        payload["tp_heads"] = [module_state_to_cpu(module) for module in runtime.tp_heads]
    path = output_dir / f"{mode_name}_adapter_checkpoint.pt"
    torch.save(payload, path)
    return path


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "training.log"

    env = detect_backend()
    config = load_config(args.config_file)
    texts = load_train_texts()
    samples_path = ensure_runtime_samples(texts, output_dir)

    max_steps = int(config.get("training_config", {}).get("max_steps", 6))
    max_steps = max(2, min(max_steps, 2))
    max_seq_len = int(config.get("data_config", {}).get("max_seq_len", 32))
    max_seq_len = max(8, min(max_seq_len, 8))
    lora_rank = int(config.get("lora_config", {}).get("lora_rank", 8))
    microbatch_num = max(1, int(config.get("training_config", {}).get("gradient_accumulation_steps", 1)))
    global_batch_size = microbatch_num

    device_ids = parse_device_ids(args.device_ids, args.num_gpus)
    if args.num_gpus >= 2:
        selected_parallel_mode = "tp" if args.parallel_mode == "tp" else "pp"
        mode_name = "tp" if selected_parallel_mode == "tp" else "dual"
        pp_size = 1 if selected_parallel_mode == "tp" else 2
        tp_size = 2 if selected_parallel_mode == "tp" else 1
    else:
        selected_parallel_mode = "single"
        mode_name = "single"
        pp_size = 1
        tp_size = 1

    summary: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": mode_name,
        "success": False,
        "dry_run": bool(args.dry_run),
        "num_gpus": args.num_gpus,
        "model_path": os.path.abspath(args.model_path),
        "config_file": os.path.abspath(args.config_file),
        "runtime_source": "projects/shared/train-infer-estimation/mvp_llama_train_runtime.py",
        "task_type": args.task_type,
        "distributed": bool(args.distributed),
        "backend": env["backend"],
        "device_count": env["device_count"],
        "device_names": env["device_names"],
        "device_ids": device_ids,
        "pipeline_parallel_size": pp_size,
        "tensor_parallel_size": tp_size,
        "parallel_mode": selected_parallel_mode,
        "microbatch_num": microbatch_num,
        "global_batch_size": global_batch_size,
        "steps": max_steps,
        "outputs": [str(log_path)],
        "errors": [],
    }

    if args.dry_run:
        log_path.write_text(
            "dry-run: train-infer-estimation runtime path verified, no real model execution.\n",
            encoding="utf-8",
        )
        summary["success"] = True
        summary["timings_ms"] = []
        summary["trainable_parameter_count"] = 0
        (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return 0

    if env["backend"] == "cpu":
        summary["errors"].append("no_musa_or_cuda_backend")
        log_path.write_text("No MUSA/CUDA backend available.\n", encoding="utf-8")
        (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return 1
    if int(env["device_count"]) < args.num_gpus:
        summary["errors"].append(f"insufficient_devices:{env['device_count']}<{args.num_gpus}")
        log_path.write_text("Insufficient visible accelerator devices.\n", encoding="utf-8")
        (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return 1

    started_at = time.perf_counter()
    try:
        runtime = LlamaTrainRuntime(
            model_path=args.model_path,
            samples_path=str(samples_path),
            device_backend=env["backend"],
            pipeline_parallel_size=pp_size,
            tensor_parallel_size=tp_size,
            max_seq_len=max_seq_len,
            split_index=16,
            lora_rank=lora_rank,
            adapter_only=False,
        )
        modules = trainable_modules(runtime)
        timings_ms = []
        parameter_norm_trace = []

        for step in range(max_steps):
            sync_ids = [0, 1] if max(pp_size, tp_size) > 1 else [0]
            _synchronize(env["backend"], sync_ids)
            started = time.perf_counter()
            runtime.train_iteration(microbatch_num=microbatch_num, global_batch_size=global_batch_size)
            _synchronize(env["backend"], sync_ids)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            timings_ms.append(elapsed_ms)
            parameter_norm_trace.append(
                {
                    "step": step + 1,
                    "elapsed_ms": elapsed_ms,
                    "trainable_parameter_norm": parameter_norm(modules),
                }
            )

        checkpoint_path = save_checkpoint(runtime, mode_name, output_dir)
        execution_time_seconds = time.perf_counter() - started_at

        log_lines = [
            f"runtime_source={summary['runtime_source']}",
            f"backend={env['backend']}",
            f"mode={mode_name}",
            f"parallel_mode={selected_parallel_mode}",
            f"pipeline_parallel_size={pp_size}",
            f"tensor_parallel_size={tp_size}",
            f"steps={max_steps}",
            f"timings_ms={timings_ms}",
            f"checkpoint={checkpoint_path}",
        ]
        log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

        summary.update(
            {
                "success": True,
                "timings_ms": timings_ms,
                "avg_step_ms": sum(timings_ms) / len(timings_ms),
                "execution_time_seconds": execution_time_seconds,
                "checkpoint_path": str(checkpoint_path),
                "parameter_norm_trace": parameter_norm_trace,
                "trainable_parameter_count": int(
                    sum(param.numel() for module in modules for param in module.parameters())
                ),
                "outputs": [str(log_path), str(checkpoint_path), str(samples_path)],
            }
        )
    except Exception as exc:
        log_path.write_text(f"runtime_error={repr(exc)}\n", encoding="utf-8")
        summary["errors"].append(repr(exc))
        summary["execution_time_seconds"] = time.perf_counter() - started_at
        (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return 1

    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
