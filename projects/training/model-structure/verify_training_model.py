#!/usr/bin/env python3
import json
import os
from datetime import datetime, timezone


ROOT = os.path.dirname(os.path.abspath(__file__))
ARTIFACT = os.path.join(ROOT, "artifacts", "20260415T094800Z")


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


def main():
    task = load_json(os.path.join(ROOT, "training_task_config.json"))
    mapping = load_json(os.path.join(ROOT, "resource_mapping.json"))
    model = load_json(os.path.join(ARTIFACT, "training_execution_model.json"))

    expected_microbatches = task["global_batch_size"] // (
        task["micro_batch_size"] * task["parallelism"]["data_parallel_size"]
    )
    expected_microbatches = max(expected_microbatches, task["gradient_accumulation_steps"])

    expected_gpu_devices = [f"musa:{gpu['id']}" for gpu in mapping["gpus"]]
    actual_gpu_devices = sorted(model["task_assignment"]["gpu"].keys())
    pipeline_stages = model["partitioning"]["pipeline_stages"]
    dag_nodes = model["dag"]["nodes"]
    runtime_observation = model.get("runtime_observation")

    checks = [
        {
            "id": "model_name_matches",
            "passed": model["model_name"] == task["model_name"],
            "detail": f"model={model['model_name']}, task={task['model_name']}",
        },
        {
            "id": "resource_gpu_count_matches",
            "passed": len(expected_gpu_devices) == len(actual_gpu_devices),
            "detail": f"expected={len(expected_gpu_devices)}, actual={len(actual_gpu_devices)}",
        },
        {
            "id": "resource_gpu_devices_match",
            "passed": sorted(expected_gpu_devices) == actual_gpu_devices,
            "detail": f"expected={expected_gpu_devices}, actual={actual_gpu_devices}",
        },
        {
            "id": "pipeline_stage_count_matches",
            "passed": len(pipeline_stages) == task["parallelism"]["pipeline_parallel_size"],
            "detail": f"expected={task['parallelism']['pipeline_parallel_size']}, actual={len(pipeline_stages)}",
        },
        {
            "id": "data_parallel_size_matches",
            "passed": model["parallel_strategy"]["data_parallel_size"] == task["parallelism"]["data_parallel_size"],
            "detail": f"expected={task['parallelism']['data_parallel_size']}, actual={model['parallel_strategy']['data_parallel_size']}",
        },
        {
            "id": "pipeline_parallel_size_matches",
            "passed": model["parallel_strategy"]["pipeline_parallel_size"] == task["parallelism"]["pipeline_parallel_size"],
            "detail": f"expected={task['parallelism']['pipeline_parallel_size']}, actual={model['parallel_strategy']['pipeline_parallel_size']}",
        },
        {
            "id": "tensor_parallel_size_matches",
            "passed": model["parallel_strategy"]["tensor_parallel_size"] == task["parallelism"]["tensor_parallel_size"],
            "detail": f"expected={task['parallelism']['tensor_parallel_size']}, actual={model['parallel_strategy']['tensor_parallel_size']}",
        },
        {
            "id": "microbatch_count_matches",
            "passed": model["microbatch_logic"]["derived_microbatch_count_per_step"] == expected_microbatches,
            "detail": f"expected={expected_microbatches}, actual={model['microbatch_logic']['derived_microbatch_count_per_step']}",
        },
        {
            "id": "cpu_assignment_present",
            "passed": len(model["task_assignment"]["cpu"]) >= 3,
            "detail": f"cpu_roles={len(model['task_assignment']['cpu'])}",
        },
        {
            "id": "gpu_partition_present",
            "passed": all(stage.get("device") in expected_gpu_devices for stage in pipeline_stages),
            "detail": f"stage_devices={[stage.get('device') for stage in pipeline_stages]}",
        },
        {
            "id": "parallel_strategy_present",
            "passed": bool(model["parallel_strategy"]),
            "detail": "parallel_strategy populated",
        },
        {
            "id": "microbatch_schedule_present",
            "passed": len(model["microbatch_logic"]["schedule"]) >= 3,
            "detail": f"schedule_items={len(model['microbatch_logic']['schedule'])}",
        },
        {
            "id": "dag_has_compute_and_comm_nodes",
            "passed": contains_node(dag_nodes, "stage0_forward", "gpu_compute")
            and contains_node(dag_nodes, "p2p_transfer_0_to_1", "gpu_comm")
            and contains_node(dag_nodes, "stage1_forward", "gpu_compute"),
            "detail": "stage0_forward/p2p_transfer_0_to_1/stage1_forward present",
        },
        {
            "id": "dag_edge_count_consistent",
            "passed": model["dag"]["edge_count"] == sum(len(node["depends_on"]) for node in dag_nodes),
            "detail": f"edge_count={model['dag']['edge_count']}",
        },
        {
            "id": "runtime_observation_present",
            "passed": runtime_observation is not None,
            "detail": "runtime_observation embedded in execution model" if runtime_observation is not None else "missing",
        },
        {
            "id": "runtime_backend_matches_mapping",
            "passed": runtime_observation is not None and runtime_observation.get("backend") == "musa",
            "detail": f"backend={None if runtime_observation is None else runtime_observation.get('backend')}",
        },
        {
            "id": "runtime_pipeline_matches_task",
            "passed": runtime_observation is not None and int(runtime_observation.get("pipeline_parallel_size", -1)) == task["parallelism"]["pipeline_parallel_size"],
            "detail": f"expected={task['parallelism']['pipeline_parallel_size']}, actual={None if runtime_observation is None else runtime_observation.get('pipeline_parallel_size')}",
        },
        {
            "id": "runtime_probe_succeeded",
            "passed": runtime_observation is not None and bool(runtime_observation.get("success")),
            "detail": "success=True" if runtime_observation is not None and runtime_observation.get("success") else f"observation={runtime_observation}",
        },
    ]

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "task_id": model["task_id"],
        "checks": checks,
        "all_passed": all(check["passed"] for check in checks),
    }
    dump_json(os.path.join(ARTIFACT, "validation_report.json"), payload)
    print(os.path.join(ARTIFACT, "validation_report.json"))


if __name__ == "__main__":
    main()
