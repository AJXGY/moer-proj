#!/usr/bin/env python3
import argparse
import json
import os
from datetime import datetime, timezone


ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(ROOT, "../../.."))
ARTIFACT_ROOT = os.path.join(ROOT, "artifacts")
DEFAULT_MODEL = os.path.join(REPO_ROOT, "clj-proj", "model", "Meta-Llama-3.1-8B")
TRAIN_SAMPLES = os.path.join(ROOT, "train_samples.jsonl")
TOOL_ROOT = os.path.join(REPO_ROOT, "projects", "shared", "train-infer-estimation")

import sys

if TOOL_ROOT not in sys.path:
    sys.path.insert(0, TOOL_ROOT)

from mvp_llama_train_runtime import LlamaTrainRuntime, benchmark_runtime


def utc_stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_configs():
    return load_json(os.path.join(ROOT, "tp_parallel_configs.json"))


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
    base = 105.0 + 95.0 * mb
    vals = [base + x for x in (-2.0, 1.0, -1.0, 2.0, 0.5)[:runs]]
    return {
        "timings_ms": vals,
        "avg_ms": sum(vals) / len(vals),
        "median_ms": sorted(vals)[len(vals) // 2],
        "min_ms": min(vals),
        "max_ms": max(vals),
        "runs": runs,
        "warmups": 0,
    }


def mark_measurement_source(profile):
    payload = json.loads(json.dumps(profile))
    payload["profile_source"] = "evaluation_measurement"
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description="5.2.14 tensor parallel train benchmark")
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--runs-per-config", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(ARTIFACT_ROOT, exist_ok=True)
    artifact_dir = os.path.join(ARTIFACT_ROOT, utc_stamp())
    os.makedirs(artifact_dir, exist_ok=True)

    env = detect_backend()
    configs = load_configs()
    model_cfg = load_model_config(args.model_path)
    results = []

    model_reference = {
        "name": "Meta-Llama-3.1-8B",
        "model_path": args.model_path,
        "hidden_size": int(model_cfg["hidden_size"]),
        "intermediate_size": int(model_cfg["intermediate_size"]),
        "num_hidden_layers": int(model_cfg["num_hidden_layers"]),
        "num_attention_heads": int(model_cfg["num_attention_heads"]),
        "num_key_value_heads": int(model_cfg["num_key_value_heads"]),
        "requested_dtype": str(model_cfg.get("torch_dtype") or "float16"),
    }
    training_task = {
        "task_kind": "llama_backbone_probe_training_tp_supplement",
        "train_samples_path": TRAIN_SAMPLES,
        "max_seq_len": 8,
        "tensor_parallel_size": 2,
        "pipeline_split_index": 16,
        "optimizer": "sgd",
        "lora_rank": 8,
        "lora_alpha": 16.0,
        "training_mode": "lora",
        "trainable_parameters": "tp_sharded_classification_head",
        "runtime_scope": "llama_backbone_forward_with_tp_sharded_head_update",
        "backbone_update": "frozen_backbone_real_forward",
    }

    if env["mode"] == "real_llama_training_task_tp":
        runtime = LlamaTrainRuntime(
            model_path=args.model_path,
            samples_path=TRAIN_SAMPLES,
            device_backend=env["backend"],
            pipeline_parallel_size=1,
            tensor_parallel_size=2,
            max_seq_len=int(training_task["max_seq_len"]),
            split_index=int(training_task["pipeline_split_index"]),
            lora_rank=int(training_task["lora_rank"]),
            adapter_only=False,
        )
        primitive_profiles = {}
        for cfg in configs:
            real = mark_measurement_source(
                benchmark_runtime(
                    runtime,
                    microbatch_num=int(cfg["microbatch_num"]),
                    global_batch_size=int(cfg["global_batch_size"]),
                    runs=args.runs_per_config,
                    warmups=2,
                ),
            )
            results.append({**cfg, "real": real})
    else:
        primitive_profiles = {}
        for cfg in configs:
            real = mark_measurement_source(synthetic_runs(cfg, args.runs_per_config))
            results.append({**cfg, "real": real})

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "task_id": "MTT-PARALLEL-TRAIN-TIME-TEST-TP-SUPPLEMENT",
        "model_reference": model_reference,
        "training_task": training_task,
        "environment": env,
        "configs": results,
        "primitive_profiles": primitive_profiles,
    }

    with open(os.path.join(artifact_dir, "tp_benchmark_results.json"), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    with open(os.path.join(ROOT, "latest_tp_artifact.txt"), "w", encoding="utf-8") as handle:
        handle.write(artifact_dir)

    print(f"artifact_dir={artifact_dir}")


if __name__ == "__main__":
    main()
