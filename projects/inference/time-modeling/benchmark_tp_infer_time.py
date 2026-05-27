#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone


ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(ROOT, "../../.."))
ARTIFACT_ROOT = os.path.join(ROOT, "artifacts")
DEFAULT_MODEL = os.path.join(REPO_ROOT, "clj-proj", "model", "Meta-Llama-3.1-8B")
TRAIN_SAMPLES = os.path.join(REPO_ROOT, "projects", "training", "time-modeling", "train_samples.jsonl")
TOOL_ROOT = os.path.join(REPO_ROOT, "projects", "shared", "train-infer-estimation")

if TOOL_ROOT not in sys.path:
    sys.path.insert(0, TOOL_ROOT)

from mvp_llama_train_runtime import LlamaTrainRuntime  # noqa: E402


def utc_stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_configs():
    return load_json(os.path.join(ROOT, "tp_parallel_configs.json"))


def detect_backend():
    try:
        import torch_musa  # noqa: F401
        import torch

        if hasattr(torch, "musa") and torch.musa.is_available():
            count = int(torch.musa.device_count())
            return {
                "backend": "musa",
                "device_count": count,
                "device_names": [torch.musa.get_device_name(i) for i in range(count)],
                "mode": "real_llama_tp_inference_probe",
                "topology": "pcie" if count >= 2 else "local",
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
                "mode": "real_llama_tp_inference_probe",
                "topology": "pcie" if count >= 2 else "local",
            }
    except Exception:
        pass

    return {
        "backend": "cpu",
        "device_count": 0,
        "device_names": [],
        "mode": "unsupported",
        "topology": "local",
    }


def synchronize(backend, device_ids):
    import torch

    if backend == "musa" and hasattr(torch, "musa"):
        for device_id in device_ids:
            torch.musa.synchronize(device_id)
    elif backend == "cuda":
        for device_id in device_ids:
            torch.cuda.synchronize(device_id)


def stable_summary(values, runs, warmups):
    vals = list(values)
    vals_sorted = sorted(vals)
    median = vals_sorted[len(vals_sorted) // 2]
    stable_cutoff = median * 0.8
    stable_vals = [value for value in vals if value >= stable_cutoff] or vals
    return {
        "profile_kind": "online_llama_tp_inference_probe",
        "timings_ms": vals,
        "avg_ms": sum(stable_vals) / len(stable_vals),
        "median_ms": median,
        "min_ms": min(vals),
        "max_ms": max(vals),
        "runs": runs,
        "warmups": warmups,
        "stable_cutoff_ms": stable_cutoff,
        "stable_timings_ms": stable_vals,
        "stable_count": len(stable_vals),
    }


def infer_iteration(runtime, microbatch_num, global_batch_size):
    import torch

    batch_size = max(1, int(global_batch_size) // max(1, int(microbatch_num)))
    with torch.no_grad():
        for microbatch_index in range(int(microbatch_num)):
            if int(runtime.tensor_parallel_size) > 1:
                runtime._run_tp2_microbatch(microbatch_index, batch_size)
            else:
                runtime._run_pp1_microbatch(microbatch_index, batch_size)


def benchmark_runtime(runtime, microbatch_num, global_batch_size, runs, warmups):
    for _ in range(warmups):
        infer_iteration(runtime, microbatch_num, global_batch_size)
    timings = []
    for _ in range(runs):
        synchronize(runtime.device_backend, [0, 1])
        start = time.perf_counter()
        infer_iteration(runtime, microbatch_num, global_batch_size)
        synchronize(runtime.device_backend, [0, 1])
        timings.append((time.perf_counter() - start) * 1000.0)
    return stable_summary(timings, runs=runs, warmups=warmups)


def parse_args():
    parser = argparse.ArgumentParser(description="5.2.15 tensor-parallel inference supplement benchmark")
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--runs-per-config", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--max-seq-len", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(ARTIFACT_ROOT, exist_ok=True)
    artifact_dir = os.path.join(ARTIFACT_ROOT, utc_stamp())
    os.makedirs(artifact_dir, exist_ok=True)

    env = detect_backend()
    if env["backend"] == "cpu" or env["device_count"] < 2:
        raise RuntimeError("TP inference supplement requires at least two MUSA/CUDA devices")

    model_cfg = load_json(os.path.join(args.model_path, "config.json"))
    runtime = LlamaTrainRuntime(
        model_path=args.model_path,
        samples_path=TRAIN_SAMPLES,
        device_backend=env["backend"],
        pipeline_parallel_size=1,
        tensor_parallel_size=2,
        max_seq_len=args.max_seq_len,
        adapter_only=False,
    )

    results = []
    for cfg in load_configs():
        real = benchmark_runtime(
            runtime,
            microbatch_num=int(cfg["microbatch_num"]),
            global_batch_size=int(cfg["global_batch_size"]),
            runs=args.runs_per_config,
            warmups=args.warmups,
        )
        results.append({**cfg, "real": real})

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "task_id": "MTT-PARALLEL-INFER-TIME-TEST-TP-SUPPLEMENT",
        "model_reference": {
            "name": "Meta-Llama-3.1-8B",
            "model_path": args.model_path,
            "hidden_size": int(model_cfg["hidden_size"]),
            "intermediate_size": int(model_cfg["intermediate_size"]),
            "num_hidden_layers": int(model_cfg["num_hidden_layers"]),
            "num_attention_heads": int(model_cfg["num_attention_heads"]),
            "num_key_value_heads": int(model_cfg["num_key_value_heads"]),
            "requested_dtype": str(model_cfg.get("torch_dtype") or "float16"),
        },
        "inference_task": {
            "task_kind": "llama_backbone_probe_inference_tp_supplement",
            "samples_path": TRAIN_SAMPLES,
            "max_seq_len": args.max_seq_len,
            "tensor_parallel_size": 2,
            "runtime_scope": "llama_backbone_forward_with_tp_sharded_head",
            "note": "Backbone executes on device0; final low-rank/classification head is sharded across two devices to provide a real TP supplement probe.",
        },
        "environment": env,
        "configs": results,
    }

    with open(os.path.join(artifact_dir, "tp_benchmark_results.json"), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    with open(os.path.join(ROOT, "latest_tp_artifact.txt"), "w", encoding="utf-8") as handle:
        handle.write(artifact_dir)
    print(f"artifact_dir={artifact_dir}")


if __name__ == "__main__":
    main()
