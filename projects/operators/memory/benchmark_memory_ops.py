#!/usr/bin/env python3
import json
import multiprocessing as mp
import os
import statistics
import time
from datetime import datetime, timezone

import torch
import torch_musa  # noqa: F401


ROOT = os.path.dirname(os.path.abspath(__file__))
ARTIFACT = os.path.join(ROOT, "artifacts", "20260415T101500Z")
RUNS = 5
WARMUPS = 4
INNER_LOOPS = 20


def load_specs():
    with open(os.path.join(ROOT, "operator_specs.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def dtype_from_name(name):
    return {"float16": torch.float16, "float32": torch.float32}[name]


def bytes_for_spec(spec):
    dtype_size = 2 if spec["dtype"] == "float16" else 4
    if spec["kind"] in ("copy", "slice"):
        numel = 1
        for d in spec["shape"]:
            numel *= d
        return numel * dtype_size
    if spec["kind"] == "cat":
        numel = 1
        for d in spec["shape_a"]:
            numel *= d
        return numel * dtype_size * 2
    return 0


def scaled_shape(shape, dim, scale):
    result = list(shape)
    result[dim] = max(1, int(result[dim] * scale))
    return result


def prepare_state(spec, device, scale=1.0):
    torch.musa.set_device(device)
    dtype = dtype_from_name(spec["dtype"])
    musa_device = f"musa:{device}"
    if spec["kind"] == "copy":
        shape = scaled_shape(spec["shape"], 0, scale)
        x = torch.randn(shape, device=musa_device, dtype=dtype)
        y = torch.empty_like(x)
        return {"x": x, "y": y}
    if spec["kind"] == "slice":
        shape = scaled_shape(spec["shape"], 1, scale)
        x = torch.randn(shape, device=musa_device, dtype=dtype)
        return {"x": x, "slice_stop": max(1, shape[1] // 2)}
    if spec["kind"] == "cat":
        shape_a = scaled_shape(spec["shape_a"], 0, scale)
        shape_b = scaled_shape(spec["shape_b"], 0, scale)
        a = torch.randn(shape_a, device=musa_device, dtype=dtype)
        b = torch.randn(shape_b, device=musa_device, dtype=dtype)
        return {"a": a, "b": b}
    raise ValueError(f"unsupported kind={spec['kind']}")


def run_prepared_op(spec, state):
    if spec["kind"] == "copy":
        state["y"].copy_(state["x"])
        return state["y"]
    if spec["kind"] == "slice":
        # Materialize the slice so the benchmark measures real memory traffic, not only view creation.
        return state["x"][:, : state["slice_stop"], :, :].contiguous()
    if spec["kind"] == "cat":
        return torch.cat([state["a"], state["b"]], dim=spec["dim"])
    raise ValueError(f"unsupported kind={spec['kind']}")


def bench_one(spec, device, runs=RUNS, warmups=WARMUPS, scale=1.0, inner_loops=INNER_LOOPS):
    state = prepare_state(spec, device, scale=scale)
    for _ in range(warmups):
        run_prepared_op(spec, state)
    torch.musa.synchronize(device)
    timings = []
    for _ in range(runs):
        start = time.perf_counter()
        for _ in range(inner_loops):
            run_prepared_op(spec, state)
        torch.musa.synchronize(device)
        end = time.perf_counter()
        timings.append((end - start) * 1000.0 / inner_loops)
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
        "inner_loops": inner_loops,
    }


def dual_worker(spec, device, queue):
    res = bench_one(spec, device=device, scale=0.5)
    queue.put({"device": f"musa:{device}", **res})


def dual_bench(spec):
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    procs = [
        ctx.Process(target=dual_worker, args=(spec, 0, queue)),
        ctx.Process(target=dual_worker, args=(spec, 1, queue)),
    ]
    wall_start = time.perf_counter()
    for p in procs:
        p.start()
    payloads = [queue.get(), queue.get()]
    for p in procs:
        p.join()
    wall_end = time.perf_counter()
    return {
        "wall_ms": (wall_end - wall_start) * 1000.0,
        "workers": sorted(payloads, key=lambda x: x["device"]),
        "effective_avg_ms": max(p["avg_ms"] for p in payloads),
    }


def main():
    os.makedirs(ARTIFACT, exist_ok=True)
    specs = load_specs()
    ops = []
    for spec in specs:
        single = bench_one(spec, device=0)
        dual = dual_bench(spec)
        ops.append(
            {
                "id": spec["id"],
                "name": spec["name"],
                "kind": spec["kind"],
                "llama_component": spec["llama_component"],
                "dtype": spec["dtype"],
                "bytes": bytes_for_spec(spec),
                "single_card": single,
                "dual_card": dual,
            }
        )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "task_id": "MTT-MEM-OP-SPACE-TEST",
        "device_backend": "musa",
        "device_count": torch.musa.device_count(),
        "device_names": [torch.musa.get_device_name(i) for i in range(torch.musa.device_count())],
        "operators": ops,
    }
    with open(os.path.join(ARTIFACT, "benchmark_results.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
