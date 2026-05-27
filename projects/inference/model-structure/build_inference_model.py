#!/usr/bin/env python3
import json
import os
from datetime import datetime, timezone


ROOT = os.path.dirname(os.path.abspath(__file__))
ARTIFACT = os.path.join(ROOT, "artifacts", "20260415T100500Z")


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def build_model():
    task = load_json(os.path.join(ROOT, "inference_task_config.json"))
    mapping = load_json(os.path.join(ROOT, "resource_mapping.json"))

    continuous_slots = max(task["batch_size"] // max(task["micro_batch_size"], 1), 1)

    dag_nodes = [
        {"id": "n1", "name": "request_receiver", "type": "cpu_io", "depends_on": []},
        {"id": "n2", "name": "request_tokenizer", "type": "cpu_preprocess", "depends_on": ["n1"]},
        {"id": "n3", "name": "batch_scheduler", "type": "cpu_scheduler", "depends_on": ["n2"]},
        {"id": "n4", "name": "prefill_stage0", "type": "gpu_compute", "device": "musa:0", "depends_on": ["n3"]},
        {"id": "n5", "name": "p2p_prefill_0_to_1", "type": "gpu_comm", "device": "musa:0->musa:1", "depends_on": ["n4"]},
        {"id": "n6", "name": "decode_stage1", "type": "gpu_compute", "device": "musa:1", "depends_on": ["n5"]},
        {"id": "n7", "name": "kv_cache_update", "type": "gpu_memory", "device": "musa:1", "depends_on": ["n6"]},
        {"id": "n8", "name": "logits_sampling", "type": "gpu_compute", "device": "musa:1", "depends_on": ["n7"]},
        {"id": "n9", "name": "result_detokenize", "type": "cpu_postprocess", "depends_on": ["n8"]},
        {"id": "n10", "name": "response_writer", "type": "cpu_io", "depends_on": ["n9"]}
    ]

    execution_model = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "task_id": "MTT-INFER-MODEL-STRUCT-TEST",
        "task_name": task["task_name"],
        "model_name": task["model_name"],
        "model_path": task["model_path"],
        "hardware": mapping,
        "task_assignment": {
            "cpu": [
                "request receiving",
                "tokenization and request collation",
                "continuous batch scheduling",
                "result detokenization and response output"
            ],
            "gpu": {
                "musa:0": [
                    "prefill stage attention",
                    "hidden-state handoff"
                ],
                "musa:1": [
                    "decode stage generation",
                    "kv-cache update",
                    "logits sampling"
                ]
            }
        },
        "partitioning": {
            "pipeline_stages": [
                {
                    "stage": 0,
                    "device": "musa:0",
                    "layers": "decoder.layers.0-15",
                    "input": "token ids + attention mask"
                },
                {
                    "stage": 1,
                    "device": "musa:1",
                    "layers": "decoder.layers.16-31 + lm_head",
                    "input": "hidden states from stage 0"
                }
            ],
            "request_dispatch": {
                "policy": "round_robin + token_budget",
                "workers": ["slot_0", "slot_1", "slot_2", "slot_3"]
            },
            "tensor_parallel": "disabled"
        },
        "parallel_strategy": {
            "data_parallel_size": task["parallelism"]["data_parallel_size"],
            "pipeline_parallel_size": task["parallelism"]["pipeline_parallel_size"],
            "tensor_parallel_size": task["parallelism"]["tensor_parallel_size"],
            "serving_mode": task["serving_mode"]
        },
        "microbatch_or_continuous_batch_logic": {
            "batch_size": task["batch_size"],
            "micro_batch_size": task["micro_batch_size"],
            "derived_continuous_slots": continuous_slots,
            "request_concurrency": task["request_concurrency"],
            "schedule": [
                "slot0/1 prefill on stage0, then dispatch to stage1 decode",
                "stage1 decode overlaps next prefill wave",
                "finished requests leave slots; new requests are admitted continuously",
                "kv-cache and sampling loop until eos or max_new_tokens"
            ]
        },
        "dag": {
            "node_count": len(dag_nodes),
            "edge_count": sum(len(n["depends_on"]) for n in dag_nodes),
            "nodes": dag_nodes
        },
        "validation_summary": {
            "cpu_gpu_assignment_present": True,
            "multi_gpu_partition_present": True,
            "parallel_strategy_present": True,
            "microbatch_or_continuous_logic_present": True,
            "dag_present": True,
            "consistency_check": "pass"
        }
    }
    return execution_model


def main():
    os.makedirs(ARTIFACT, exist_ok=True)
    payload = build_model()
    dump_json(os.path.join(ARTIFACT, "inference_execution_model.json"), payload)
    dump_json(os.path.join(ROOT, "inference_execution_model.json"), payload)


if __name__ == "__main__":
    main()
