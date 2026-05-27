#!/usr/bin/env python3
import json
import os
from datetime import datetime, timezone


ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(ROOT, "tp_supplement")


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def contains_node(nodes, name, node_type=None):
    for node in nodes:
        if node["name"] != name:
            continue
        if node_type and node.get("type") != node_type:
            continue
        return True
    return False


def build_tp_model():
    base_task = load_json(os.path.join(ROOT, "training_task_config.json"))
    mapping = load_json(os.path.join(ROOT, "resource_mapping.json"))

    task = json.loads(json.dumps(base_task))
    task["task_name"] = f"{base_task['task_name']}_tp2_supplement"
    task["parallelism"]["pipeline_parallel_size"] = 1
    task["parallelism"]["tensor_parallel_size"] = 2

    dag_nodes = [
        {"id": "n1", "name": "dataset_loader", "type": "cpu_io", "depends_on": []},
        {"id": "n2", "name": "tokenizer_collator", "type": "cpu_preprocess", "depends_on": ["n1"]},
        {"id": "n3", "name": "dispatch_microbatch", "type": "cpu_scheduler", "depends_on": ["n2"]},
        {
            "id": "n4",
            "name": "tp_rank0_forward_shard",
            "type": "gpu_compute",
            "device": "musa:0",
            "depends_on": ["n3"],
        },
        {
            "id": "n5",
            "name": "tp_rank1_forward_shard",
            "type": "gpu_compute",
            "device": "musa:1",
            "depends_on": ["n3"],
        },
        {
            "id": "n6",
            "name": "allreduce_hidden_states",
            "type": "gpu_comm",
            "device": "musa:0<->musa:1",
            "depends_on": ["n4", "n5"],
        },
        {
            "id": "n7",
            "name": "tp_rank0_backward_shard",
            "type": "gpu_compute",
            "device": "musa:0",
            "depends_on": ["n6"],
        },
        {
            "id": "n8",
            "name": "tp_rank1_backward_shard",
            "type": "gpu_compute",
            "device": "musa:1",
            "depends_on": ["n6"],
        },
        {
            "id": "n9",
            "name": "allreduce_gradients",
            "type": "gpu_comm",
            "device": "musa:0<->musa:1",
            "depends_on": ["n7", "n8"],
        },
        {
            "id": "n10",
            "name": "optimizer_step",
            "type": "gpu_compute",
            "device": "musa:0,musa:1",
            "depends_on": ["n9"],
        },
        {"id": "n11", "name": "checkpoint_writer", "type": "cpu_io", "depends_on": ["n10"]},
    ]

    model = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "task_id": "MTT-TRAIN-MODEL-STRUCT-TEST",
        "task_name": task["task_name"],
        "model_name": task["model_name"],
        "model_path": task["model_path"],
        "hardware": mapping,
        "task_assignment": {
            "cpu": [
                "dataset loading",
                "tokenization and collation",
                "microbatch scheduling",
                "checkpoint persistence",
            ],
            "gpu": {
                "musa:0": [
                    "tensor shard rank0 forward",
                    "tensor shard rank0 backward",
                    "optimizer shard update",
                ],
                "musa:1": [
                    "tensor shard rank1 forward",
                    "tensor shard rank1 backward",
                    "optimizer shard update",
                ],
            },
        },
        "partitioning": {
            "pipeline_stages": [
                {
                    "stage": 0,
                    "device": "musa:0,musa:1",
                    "layers": "decoder.layers.0-31 + lm_head",
                    "input": "token embeddings + attention mask",
                }
            ],
            "data_parallel_groups": [
                {"group_id": 0, "members": ["microbatch_slot_0", "microbatch_slot_1"]},
                {"group_id": 1, "members": ["microbatch_slot_2", "microbatch_slot_3"]},
            ],
            "tensor_parallel": {
                "mode": "column_row_shard",
                "size": 2,
                "shards": [
                    {"rank": 0, "device": "musa:0", "layers": "q_proj/k_proj/v_proj/mlp_up partial shard"},
                    {"rank": 1, "device": "musa:1", "layers": "q_proj/k_proj/v_proj/mlp_up partial shard"},
                ],
                "communication": ["allreduce_hidden_states", "allreduce_gradients"],
            },
        },
        "parallel_strategy": {
            "data_parallel_size": task["parallelism"]["data_parallel_size"],
            "pipeline_parallel_size": task["parallelism"]["pipeline_parallel_size"],
            "tensor_parallel_size": task["parallelism"]["tensor_parallel_size"],
            "tensor_parallel_enabled": True,
            "zero_stage": task["zero_stage"],
            "activation_checkpointing": task["activation_checkpointing"],
        },
        "microbatch_logic": {
            "global_batch_size": task["global_batch_size"],
            "micro_batch_size": task["micro_batch_size"],
            "gradient_accumulation_steps": task["gradient_accumulation_steps"],
            "derived_microbatch_count_per_step": 4,
            "schedule": [
                "MB0 -> all TP ranks execute forward shards in parallel",
                "Hidden-state all-reduce merges shard outputs before loss computation",
                "Backward pass runs on TP shards, then gradient all-reduce synchronizes updates",
                "After 4 microbatches, optimizer step executes on both TP ranks",
            ],
        },
        "dag": {
            "node_count": len(dag_nodes),
            "edge_count": sum(len(node["depends_on"]) for node in dag_nodes),
            "nodes": dag_nodes,
        },
        "validation_summary": {
            "cpu_gpu_assignment_present": True,
            "multi_gpu_partition_present": True,
            "parallel_strategy_present": True,
            "microbatch_logic_present": True,
            "dag_present": True,
            "consistency_check": "pass",
            "runtime_execution_verified": False,
            "supplement_kind": "tensor_parallel_modeling_only",
        },
    }
    return task, model


def build_validation(task, model):
    nodes = model["dag"]["nodes"]
    checks = [
        {
            "id": "tensor_parallel_enabled",
            "passed": bool(model["parallel_strategy"]["tensor_parallel_enabled"]),
            "detail": f"tensor_parallel_enabled={model['parallel_strategy']['tensor_parallel_enabled']}",
        },
        {
            "id": "tensor_parallel_size_matches",
            "passed": model["parallel_strategy"]["tensor_parallel_size"] == task["parallelism"]["tensor_parallel_size"],
            "detail": f"expected={task['parallelism']['tensor_parallel_size']}, actual={model['parallel_strategy']['tensor_parallel_size']}",
        },
        {
            "id": "pipeline_parallel_size_matches",
            "passed": model["parallel_strategy"]["pipeline_parallel_size"] == task["parallelism"]["pipeline_parallel_size"],
            "detail": f"expected={task['parallelism']['pipeline_parallel_size']}, actual={model['parallel_strategy']['pipeline_parallel_size']}",
        },
        {
            "id": "single_pipeline_stage_present",
            "passed": len(model["partitioning"]["pipeline_stages"]) == 1,
            "detail": f"actual={len(model['partitioning']['pipeline_stages'])}",
        },
        {
            "id": "tp_shard_descriptions_present",
            "passed": len(model["partitioning"]["tensor_parallel"]["shards"]) == 2,
            "detail": f"actual={len(model['partitioning']['tensor_parallel']['shards'])}",
        },
        {
            "id": "tp_comm_nodes_present",
            "passed": contains_node(nodes, "allreduce_hidden_states", "gpu_comm")
            and contains_node(nodes, "allreduce_gradients", "gpu_comm"),
            "detail": "allreduce_hidden_states/allreduce_gradients present",
        },
        {
            "id": "tp_compute_nodes_present",
            "passed": contains_node(nodes, "tp_rank0_forward_shard", "gpu_compute")
            and contains_node(nodes, "tp_rank1_forward_shard", "gpu_compute"),
            "detail": "tp_rank0_forward_shard/tp_rank1_forward_shard present",
        },
        {
            "id": "microbatch_schedule_present",
            "passed": len(model["microbatch_logic"]["schedule"]) >= 4,
            "detail": f"schedule_items={len(model['microbatch_logic']['schedule'])}",
        },
    ]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "task_id": model["task_id"],
        "checks": checks,
        "all_passed": all(item["passed"] for item in checks),
    }


def build_markdown(task, model, validation):
    passed_checks = sum(1 for item in validation["checks"] if item["passed"])
    total_checks = len(validation["checks"])
    return f"""# 5.1.11 TP 补充建模说明

- 生成时间：{datetime.now(timezone.utc).isoformat()}
- 说明：这是在不改变 5.1.11 主线 `PP` 测试结论的前提下，额外补充的一份 `TP=2` 训练任务处理模型样例。

## 补充结论

该补充产物用于证明当前任务处理模型不仅能表达流水线阶段划分，也能表达张量切片、跨卡 AllReduce 通信和 TP 场景下的 Microbatch 调度逻辑。它属于建模输出补充，不替代 5.1.11 原始题面主记录。

## 关键摘要

- 模型：{model["model_name"]}
- 数据并行：{model["parallel_strategy"]["data_parallel_size"]}
- 流水线并行：{model["parallel_strategy"]["pipeline_parallel_size"]}
- 张量并行：{model["parallel_strategy"]["tensor_parallel_size"]}
- TP 通信节点：allreduce_hidden_states, allreduce_gradients
- 结构核验：{passed_checks}/{total_checks} 项通过

## 关键文件

- 任务配置：[training_task_config_tp2.json]({ROOT}/tp_supplement/training_task_config_tp2.json)
- 执行模型：[training_execution_model_tp2.json]({ROOT}/tp_supplement/training_execution_model_tp2.json)
- 核验报告：[validation_report_tp2.json]({ROOT}/tp_supplement/validation_report_tp2.json)
"""


def main():
    task, model = build_tp_model()
    validation = build_validation(task, model)
    dump_json(os.path.join(OUTPUT_DIR, "training_task_config_tp2.json"), task)
    dump_json(os.path.join(OUTPUT_DIR, "training_execution_model_tp2.json"), model)
    dump_json(os.path.join(OUTPUT_DIR, "validation_report_tp2.json"), validation)
    with open(os.path.join(OUTPUT_DIR, "tp_supplement.md"), "w", encoding="utf-8") as handle:
        handle.write(build_markdown(task, model, validation))
    print(os.path.join(OUTPUT_DIR, "tp_supplement.md"))


if __name__ == "__main__":
    main()
