#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.abspath(__file__))
ARTIFACT = os.path.join(ROOT, "artifacts", "20260415T094800Z")
REPO_ROOT = os.path.abspath(os.path.join(ROOT, "../../.."))
TOOL_ROOT = os.path.join(REPO_ROOT, "projects", "shared", "train-infer-estimation")
MODEL_PATH = "/home/o_mabin/moerxiancheng-clj-xyj-proj/clj-proj/model/Meta-Llama-3.1-8B"
SAMPLES_PATH = os.path.join(REPO_ROOT, "projects", "training", "runtime-validation", "train_samples.jsonl")

if TOOL_ROOT not in sys.path:
    sys.path.insert(0, TOOL_ROOT)

from mvp_llama_train_runtime import LlamaTrainRuntime, _synchronize  # type: ignore


def dump_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def detect_backend():
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
    return {"backend": "cpu", "device_count": 0, "device_names": []}


def main():
    env = detect_backend()
    observation = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "task_id": "MTT-TRAIN-MODEL-STRUCT-TEST",
        "mode": "runtime_consistency_probe",
        "backend": env["backend"],
        "device_count": env["device_count"],
        "device_names": env["device_names"],
        "pipeline_parallel_size": 2,
        "tensor_parallel_size": 1,
        "microbatch_num": 1,
        "global_batch_size": 1,
        "split_index": 16,
        "model_path": MODEL_PATH,
        "samples_path": SAMPLES_PATH,
        "success": False,
    }
    try:
        if env["backend"] == "cpu" or int(env["device_count"]) < 2:
            raise RuntimeError("runtime consistency probe requires two visible MUSA devices")
        runtime = LlamaTrainRuntime(
            model_path=MODEL_PATH,
            samples_path=SAMPLES_PATH,
            device_backend=env["backend"],
            pipeline_parallel_size=2,
            tensor_parallel_size=1,
            max_seq_len=8,
            split_index=16,
            lora_rank=8,
            adapter_only=False,
        )
        _synchronize(env["backend"], [0, 1])
        started = time.perf_counter()
        runtime.train_iteration(microbatch_num=1, global_batch_size=1)
        _synchronize(env["backend"], [0, 1])
        observation["elapsed_ms"] = (time.perf_counter() - started) * 1000.0
        observation["success"] = True
    except Exception as exc:
        observation["error"] = repr(exc)

    dump_json(os.path.join(ARTIFACT, "runtime_observation.json"), observation)
    if not observation["success"]:
        raise SystemExit(observation.get("error", "runtime observation failed"))


if __name__ == "__main__":
    main()
