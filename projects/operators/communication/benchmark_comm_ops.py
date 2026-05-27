#!/usr/bin/env python3
import json
import os
import statistics
import time
from datetime import datetime, timezone

import torch
import torch.distributed as dist
import torch_musa  # noqa: F401


ROOT = os.path.dirname(os.path.abspath(__file__))
ARTIFACT = os.path.join(ROOT, "artifacts", "20260415T113500Z")
WORLD_SIZE = 2
RUNS = 5
WARMUPS = 4
INNER_LOOPS = 20


def load_specs():
    with open(os.path.join(ROOT, "operator_specs.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def dtype_from_name(name):
    return {"float16": torch.float16, "float32": torch.float32}[name]


def numel_from_bytes(bytes_count, dtype):
    element_size = torch.tensor([], dtype=dtype).element_size()
    return bytes_count // element_size


def ensure_rank_world():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    if world_size != WORLD_SIZE:
        raise RuntimeError(f"Expected world size {WORLD_SIZE}, got {world_size}")
    return rank, local_rank, world_size


def init_dist():
    rank, local_rank, world_size = ensure_rank_world()
    dist.init_process_group(backend="gloo")
    if torch.musa.device_count() < WORLD_SIZE:
        raise RuntimeError(f"Expected at least {WORLD_SIZE} MUSA devices, got {torch.musa.device_count()}")
    torch.musa.set_device(local_rank)
    return rank, local_rank, world_size


def prepare_state(spec, local_rank):
    dtype = dtype_from_name(spec["dtype"])
    numel = numel_from_bytes(spec["bytes"], dtype)
    device = f"musa:{local_rank}"
    src_gpu = torch.full((numel,), fill_value=float(local_rank + 1), device=device, dtype=dtype)
    dst_gpu = torch.zeros((numel,), device=device, dtype=dtype)
    send_cpu = torch.empty((numel,), device="cpu", dtype=dtype)
    recv_cpu = torch.empty((numel,), device="cpu", dtype=dtype)
    work_cpu = torch.empty((numel,), device="cpu", dtype=dtype)
    return {
        "device": device,
        "src_gpu": src_gpu,
        "dst_gpu": dst_gpu,
        "send_cpu": send_cpu,
        "recv_cpu": recv_cpu,
        "work_cpu": work_cpu,
    }


def run_send_recv(state, rank, local_rank):
    state["send_cpu"].copy_(state["src_gpu"])
    torch.musa.synchronize(local_rank)
    if rank == 0:
        dist.send(state["send_cpu"], dst=1)
        dist.recv(state["recv_cpu"], src=1)
    else:
        dist.recv(state["recv_cpu"], src=0)
        dist.send(state["send_cpu"], dst=0)
    state["dst_gpu"].copy_(state["recv_cpu"])
    torch.musa.synchronize(local_rank)


def run_all_reduce(state, local_rank):
    state["work_cpu"].copy_(state["src_gpu"])
    torch.musa.synchronize(local_rank)
    dist.all_reduce(state["work_cpu"], op=dist.ReduceOp.SUM)
    state["dst_gpu"].copy_(state["work_cpu"])
    torch.musa.synchronize(local_rank)


def run_once(spec, state, rank, local_rank, inner_loops=INNER_LOOPS):
    dist.barrier()
    start = time.perf_counter()
    for _ in range(inner_loops):
        if spec["kind"] == "send_recv":
            run_send_recv(state, rank, local_rank)
        elif spec["kind"] == "all_reduce":
            run_all_reduce(state, local_rank)
        else:
            raise ValueError(f"Unsupported communication operator kind: {spec['kind']}")
    dist.barrier()
    elapsed_ms = (time.perf_counter() - start) * 1000.0 / inner_loops
    collected = [None for _ in range(WORLD_SIZE)]
    dist.all_gather_object(collected, elapsed_ms)
    return max(collected)


def bench(spec, rank, local_rank, runs=RUNS, warmups=WARMUPS):
    state = prepare_state(spec, local_rank)
    for _ in range(warmups):
        run_once(spec, state, rank, local_rank)
    timings = []
    for _ in range(runs):
        timings.append(run_once(spec, state, rank, local_rank))
    return {
        "timings_ms": timings,
        "avg_ms": sum(timings) / len(timings),
        "median_ms": statistics.median(timings),
        "min_ms": min(timings),
        "max_ms": max(timings),
        "max_min_ratio": max(timings) / min(timings),
        "cv_percent": statistics.pstdev(timings) / (sum(timings) / len(timings)) * 100.0,
        "runs": runs,
        "warmups": warmups,
        "inner_loops": INNER_LOOPS,
    }


def write_results(payload, rank):
    if rank != 0:
        return
    os.makedirs(ARTIFACT, exist_ok=True)
    with open(os.path.join(ARTIFACT, "benchmark_results.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main():
    rank, local_rank, _ = init_dist()
    specs = load_specs()
    results = []
    for spec in specs:
        metrics = bench(spec, rank, local_rank)
        results.append(
            {
                "id": spec["id"],
                "name": spec["name"],
                "kind": spec["kind"],
                "bytes": spec["bytes"],
                "dtype": spec["dtype"],
                "real": metrics,
            }
        )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "task_id": "MTT-COMM-OP-SPACE-TEST",
        "device_backend": "musa",
        "device_count": torch.musa.device_count(),
        "device_names": [torch.musa.get_device_name(i) for i in range(torch.musa.device_count())],
        "distributed_backend": "gloo",
        "communication_path": "torch.distributed.gloo + cpu_staging + musa_device_buffers",
        "operators": results,
    }
    write_results(payload, rank)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
