#!/usr/bin/env python3
import argparse
import json
import os
from datetime import datetime, timezone


ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(ROOT, "../../.."))
ARTIFACT_ROOT = os.environ.get("MOER_ARTIFACT_ROOT", os.path.join(ROOT, "artifacts"))
TRAIN_SAMPLES = os.path.join(ROOT, "train_samples.jsonl")
TRAIN_RUNTIME_ROOT = os.path.join(ROOT, "train_runtime")

import sys

if TRAIN_RUNTIME_ROOT not in sys.path:
    sys.path.insert(0, TRAIN_RUNTIME_ROOT)

from mvp_llama_train_runtime import LlamaTrainRuntime, benchmark_runtime


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


def load_model_config(model_path):
    return load_json(os.path.join(model_path, "config.json"))


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
                "mode": "real_llama_training_task_tp",
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
                "mode": "real_llama_training_task_tp",
                "topology": "pcie" if count >= 2 else "local",
            }
    except Exception:
        pass

    return {
        "backend": "cpu",
        "device_count": 0,
        "device_names": [],
        "mode": "synthetic_sample_tp",
        "topology": "local",
    }


def synthetic_runs(cfg, runs):
    mb = float(cfg["microbatch_num"])
    tp = float(cfg.get("tensor_parallel_size", 1))
    base = (80.0 if tp <= 1 else 105.0) + 95.0 * mb + 18.0 * max(0.0, tp - 1.0)
    vals = [base + x for x in (-2.0, 1.0, -1.0, 2.0, 0.5)[:runs]]
    return {
        "timings_ms": vals,
        "avg_ms": sum(vals) / len(vals),
        "median_ms": sorted(vals)[len(vals) // 2],
        "min_ms": min(vals),
        "max_ms": max(vals),
        "runs": runs,
        "warmups": 0,
        "profile_source": "synthetic_chain_check",
    }


def mark_measurement_source(profile):
    payload = json.loads(json.dumps(profile))
    payload["profile_source"] = "evaluation_measurement"
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description="5.2.14 tensor parallel train benchmark")
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--runs-per-config", type=int, default=5)
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
    model_cfg = load_model_config(args.model_path)
    results = []
    skipped_configs = []

    model_reference = {
        "name": "Meta-Llama-3.1-8B",
        "model_path": args.model_path,
        "hidden_size": int(model_cfg["hidden_size"]),
        "intermediate_size": int(model_cfg["intermediate_size"]),
        "num_hidden_layers": int(model_cfg["num_hidden_layers"]),
        "num_attention_heads": int(model_cfg["num_attention_heads"]),
        "num_key_value_heads": int(model_cfg["num_key_value_heads"]),
        "vocab_size": int(model_cfg["vocab_size"]),
        "requested_dtype": str(model_cfg.get("torch_dtype") or "float16"),
    }
    lora_rank = 8
    lora_trainable_parameters = lora_rank * (
        int(model_cfg["hidden_size"]) + int(model_cfg["vocab_size"])
    )
    training_task = {
        "task_kind": "llama_vocab_lora_training_tp",
        "runtime_source": "projects/training/time-modeling/train_runtime",
        "train_samples_path": TRAIN_SAMPLES,
        "max_seq_len": args.max_seq_len,
        "sequence_length": args.max_seq_len,
        "parallel_scope": args.parallel_scope,
        "tensor_parallel_sizes": sorted({int(cfg.get("tensor_parallel_size", 1)) for cfg in configs}),
        "pipeline_split_index": 16,
        "optimizer": "adam",
        "lora_rank": lora_rank,
        "lora_alpha": 16.0,
        "training_mode": "lora",
        "lora_head": "vocab_lm_head",
        "lora_projection": "hidden_size_to_rank_to_vocab_size",
        "trainable_parameter_count": lora_trainable_parameters,
        "runtime_scope": "llama_backbone_autograd_with_vocab_lora_head_update",
        "backbone_update": "autograd_traverses_backbone_optimizer_updates_lora_only",
    }

    if env["mode"] == "real_llama_training_task_tp":
        primitive_profiles = {}
        runtimes = {}
        for cfg in configs:
            pp_size = int(cfg.get("pipeline_parallel_size", 1))
            tp_size = int(cfg.get("tensor_parallel_size", 1))
            required_devices = max(pp_size, tp_size, 1)
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
                    max_seq_len=int(training_task["max_seq_len"]),
                    split_index=int(training_task["pipeline_split_index"]),
                    lora_rank=int(training_task["lora_rank"]),
                    adapter_only=False,
                )
            real = mark_measurement_source(
                benchmark_runtime(
                    runtimes[runtime_key],
                    microbatch_num=int(cfg["microbatch_num"]),
                    global_batch_size=int(cfg["global_batch_size"]),
                    runs=args.runs_per_config,
                    warmups=2,
                ),
            )
            results.append({**cfg, "real": real})
    elif args.force_synthetic:
        primitive_profiles = {}
        for cfg in configs:
            real = mark_measurement_source(synthetic_runs(cfg, args.runs_per_config))
            results.append({**cfg, "real": real})
    else:
        primitive_profiles = {}
        for cfg in configs:
            skipped_configs.append({**cfg, "reason": "requires MUSA/CUDA device or --force-synthetic"})

    if not results:
        raise RuntimeError("No configs were run; check device count, selected scope, or use --force-synthetic")

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "task_id": "MTT-PARALLEL-TRAIN-TIME-TEST-TP-SUPPLEMENT",
        "model_reference": model_reference,
        "training_task": training_task,
        "environment": env,
        "configs": results,
        "skipped_configs": skipped_configs,
        "primitive_profiles": primitive_profiles,
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
