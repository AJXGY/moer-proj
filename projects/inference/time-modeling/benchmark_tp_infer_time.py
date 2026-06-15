#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone


ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(ROOT, "../../.."))
ARTIFACT_ROOT = os.environ.get("MOER_ARTIFACT_ROOT", os.path.join(ROOT, "artifacts"))
TRAIN_SAMPLES = os.path.join(REPO_ROOT, "projects", "training", "time-modeling", "train_samples.jsonl")
TOOL_ROOT = os.path.join(REPO_ROOT, "projects", "shared", "train-infer-estimation")

if TOOL_ROOT not in sys.path:
    sys.path.insert(0, TOOL_ROOT)

from mvp_llama_train_runtime import LlamaTrainRuntime  # noqa: E402


def default_model_path():
    candidates = [
        os.environ.get("MOER_MODEL_PATH"),
        os.path.join(REPO_ROOT, "clj-proj", "model", "Meta-Llama-3.1-8B"),
        "/home/o_mabin/moerxiancheng-clj-xyj-proj/clj-proj/model/Meta-Llama-3.1-8B",
    ]
    for candidate in candidates:
        if candidate and os.path.exists(os.path.join(candidate, "config.json")):
            return candidate
    return os.path.join(REPO_ROOT, "clj-proj", "model", "Meta-Llama-3.1-8B")


DEFAULT_MODEL = default_model_path()


def utc_stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_configs():
    return load_json(os.path.join(ROOT, "tp_parallel_configs.json"))


def config_scope(cfg):
    return cfg.get("parallel_scope") or ("multi" if int(cfg.get("tensor_parallel_size", 1)) > 1 else "single")


def filtered_configs(configs, requested_scope):
    if requested_scope == "all":
        return list(configs)
    return [cfg for cfg in configs if config_scope(cfg) == requested_scope]


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
    device_ids = list(range(max(1, int(runtime.tensor_parallel_size))))
    for _ in range(warmups):
        infer_iteration(runtime, microbatch_num, global_batch_size)
    timings = []
    for _ in range(runs):
        synchronize(runtime.device_backend, device_ids)
        start = time.perf_counter()
        infer_iteration(runtime, microbatch_num, global_batch_size)
        synchronize(runtime.device_backend, device_ids)
        timings.append((time.perf_counter() - start) * 1000.0)
    return stable_summary(timings, runs=runs, warmups=warmups)


def synthetic_runs(cfg, runs, warmups):
    mb = float(cfg["microbatch_num"])
    tp = float(cfg.get("tensor_parallel_size", 1))
    base = (42.0 if tp <= 1 else 58.0) + 34.0 * mb + 9.0 * max(0.0, tp - 1.0)
    vals = [base + x for x in (-1.5, 0.5, -0.4, 1.0, 0.2)[:runs]]
    return {
        **stable_summary(vals, runs=runs, warmups=warmups),
        "profile_source": "synthetic_sample",
    }


def parse_args():
    parser = argparse.ArgumentParser(description="5.2.15 tensor-parallel inference supplement benchmark")
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--runs-per-config", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--max-seq-len", type=int, default=8)
    parser.add_argument(
        "--parallel-scope",
        choices=["all", "single", "multi"],
        default=os.environ.get("MOER_PARALLEL_SCOPE", "all"),
        help="Run all configs, only single-card configs, or only single-node multi-card configs.",
    )
    parser.add_argument(
        "--force-synthetic",
        action="store_true",
        help="Generate synthetic timing samples without loading the model or using accelerators.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(ARTIFACT_ROOT, exist_ok=True)
    artifact_dir = os.path.join(ARTIFACT_ROOT, utc_stamp())
    os.makedirs(artifact_dir, exist_ok=True)

    env = detect_backend()
    if args.force_synthetic:
        env = {
            **env,
            "backend": "cpu",
            "device_count": 0,
            "device_names": [],
            "mode": "synthetic_sample_tp",
            "topology": "local",
        }
    configs = filtered_configs(load_configs(), args.parallel_scope)
    if not configs:
        raise RuntimeError(f"No configs selected for parallel scope {args.parallel_scope!r}")

    model_cfg = load_json(os.path.join(args.model_path, "config.json"))
    results = []
    skipped_configs = []
    runtimes = {}
    for cfg in configs:
        pp_size = int(cfg.get("pipeline_parallel_size", 1))
        tp_size = int(cfg.get("tensor_parallel_size", 1))
        required_devices = max(pp_size, tp_size, 1)
        if env["backend"] == "cpu":
            if not args.force_synthetic:
                skipped_configs.append({**cfg, "reason": "requires MUSA/CUDA device or --force-synthetic"})
                continue
            real = synthetic_runs(cfg, runs=args.runs_per_config, warmups=args.warmups)
        else:
            if int(env["device_count"]) < required_devices:
                skipped_configs.append(
                    {**cfg, "reason": f"requires {required_devices} devices, found {env['device_count']}"}
                )
                continue
            runtime_key = (pp_size, tp_size)
            if runtime_key not in runtimes:
                runtimes[runtime_key] = LlamaTrainRuntime(
                    model_path=args.model_path,
                    samples_path=TRAIN_SAMPLES,
                    device_backend=env["backend"],
                    pipeline_parallel_size=pp_size,
                    tensor_parallel_size=tp_size,
                    max_seq_len=args.max_seq_len,
                    adapter_only=False,
                )
            real = benchmark_runtime(
                runtimes[runtime_key],
                microbatch_num=int(cfg["microbatch_num"]),
                global_batch_size=int(cfg["global_batch_size"]),
                runs=args.runs_per_config,
                warmups=args.warmups,
            )
        results.append({**cfg, "real": real})

    if not results:
        raise RuntimeError("No configs were run; check device count, selected scope, or use --force-synthetic")

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
            "parallel_scope": args.parallel_scope,
            "tensor_parallel_sizes": sorted({int(cfg.get("tensor_parallel_size", 1)) for cfg in configs}),
            "runtime_scope": "llama_backbone_forward_with_tp_sharded_head",
            "note": "Single-card configs run on one device; TP=2 configs shard the low-rank/classification head across two devices.",
        },
        "environment": env,
        "configs": results,
        "skipped_configs": skipped_configs,
    }

    with open(os.path.join(artifact_dir, "tp_benchmark_results.json"), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    latest_path = os.environ.get("MOER_LATEST_TP_ARTIFACT_FILE", os.path.join(ROOT, "latest_tp_artifact.txt"))
    os.makedirs(os.path.dirname(latest_path), exist_ok=True)
    with open(latest_path, "w", encoding="utf-8") as handle:
        handle.write(artifact_dir)
    print(f"artifact_dir={artifact_dir}")


if __name__ == "__main__":
    main()
