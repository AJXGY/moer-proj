#!/usr/bin/env python3
import json
import os
import sys


ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(ROOT, "../../.."))
TRAIN_MVP_ROOT = os.path.join(REPO_ROOT, "projects", "shared", "train-infer-estimation")
MODEL_PATH = "/home/o_mabin/moerxiancheng-clj-xyj-proj/clj-proj/model/Meta-Llama-3.1-8B"

if TRAIN_MVP_ROOT not in sys.path:
    sys.path.insert(0, TRAIN_MVP_ROOT)

from mvp_llama_train_runtime import LoraFeatureTrainRuntime, benchmark_runtime


def main():
    with open(os.path.join(MODEL_PATH, "config.json"), "r", encoding="utf-8") as handle:
        model_cfg = json.load(handle)

    runtime = LoraFeatureTrainRuntime(
        hidden_size=int(model_cfg["hidden_size"]),
        num_labels=2,
        device_backend="musa",
        pipeline_parallel_size=1,
        tensor_parallel_size=2,
        lora_rank=8,
    )

    for mb in (1, 2, 3, 4, 5, 6, 8):
        result = benchmark_runtime(
            runtime,
            microbatch_num=mb,
            global_batch_size=mb,
            runs=3,
            warmups=1,
        )
        timings = [float(value) for value in result.get("timings_ms", [])]
        if timings:
            print(
                json.dumps(
                    {
                        "mb": mb,
                        "avg_ms": result["avg_ms"],
                        "min_ms": min(timings),
                        "max_ms": max(timings),
                        "timings_ms": timings,
                    },
                    ensure_ascii=False,
                )
            )


if __name__ == "__main__":
    main()
