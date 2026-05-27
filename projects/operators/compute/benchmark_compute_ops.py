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
ARTIFACT = os.path.join(ROOT, "artifacts", "20260415T100500Z")


def load_specs():
    with open(os.path.join(ROOT, "operator_specs.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def dtype_from_name(name):
    return {"float16": torch.float16, "float32": torch.float32}[name]


def bench_matmul(spec, device, runs=5, warmups=2, split_m=None):
    torch.musa.set_device(device)
    dtype = dtype_from_name(spec["dtype"])
    m = split_m or spec["m"]
    a = torch.randn((m, spec["k"]), device=f"musa:{device}", dtype=dtype)
    b = torch.randn((spec["k"], spec["n"]), device=f"musa:{device}", dtype=dtype)

    for _ in range(warmups):
        _ = a @ b
    torch.musa.synchronize(device)

    timings = []
    for _ in range(runs):
        start = time.perf_counter()
        _ = a @ b
        torch.musa.synchronize(device)
        end = time.perf_counter()
        timings.append((end - start) * 1000.0)

    avg_ms = sum(timings) / len(timings)
    return {
        "timings_ms": timings,
        "avg_ms": avg_ms,
        "median_ms": statistics.median(timings),
        "min_ms": min(timings),
        "max_ms": max(timings),
    }


def bench_flash_attention(spec, device, runs=5, warmups=2, split_seq_len=None):
    torch.musa.set_device(device)
    dtype = dtype_from_name(spec["dtype"])
    seq_len = split_seq_len or spec["seq_len"]
    shape = (spec["batch"], spec["heads"], seq_len, spec["head_dim"])
    q = torch.randn(shape, device=f"musa:{device}", dtype=dtype)
    k = torch.randn(shape, device=f"musa:{device}", dtype=dtype)
    v = torch.randn(shape, device=f"musa:{device}", dtype=dtype)

    for _ in range(warmups):
        _ = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    torch.musa.synchronize(device)

    timings = []
    for _ in range(runs):
        start = time.perf_counter()
        _ = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        torch.musa.synchronize(device)
        end = time.perf_counter()
        timings.append((end - start) * 1000.0)

    avg_ms = sum(timings) / len(timings)
    return {
        "timings_ms": timings,
        "avg_ms": avg_ms,
        "median_ms": statistics.median(timings),
        "min_ms": min(timings),
        "max_ms": max(timings),
    }


def bench_one(spec, device, runs=5, warmups=2, split_m=None, split_seq_len=None):
    if spec["kind"] == "matmul":
        return bench_matmul(spec, device, runs=runs, warmups=warmups, split_m=split_m)
    if spec["kind"] == "flash_attention":
        return bench_flash_attention(
            spec, device, runs=runs, warmups=warmups, split_seq_len=split_seq_len
        )
    raise ValueError(f"Unsupported compute operator kind: {spec['kind']}")


def dual_worker(spec, device, queue):
    shard_m = spec["m"] // 2 if spec["kind"] == "matmul" else None
    shard_seq_len = spec["seq_len"] // 2 if spec["kind"] == "flash_attention" else None
    res = bench_one(spec, device=device, split_m=shard_m, split_seq_len=shard_seq_len)
    queue.put(
        {
            "device": f"musa:{device}",
            "avg_ms": res["avg_ms"],
            "median_ms": res["median_ms"],
            "timings_ms": res["timings_ms"],
        }
    )


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


def flops_matmul(m, k, n):
    return 2.0 * m * k * n


def flops_flash_attention(batch, heads, seq_len, head_dim):
    return 4.0 * batch * heads * seq_len * seq_len * head_dim


def operator_shape(spec):
    if spec["kind"] == "matmul":
        return {"m": spec["m"], "k": spec["k"], "n": spec["n"]}
    if spec["kind"] == "flash_attention":
        return {
            "batch": spec["batch"],
            "heads": spec["heads"],
            "seq_len": spec["seq_len"],
            "head_dim": spec["head_dim"],
        }
    raise ValueError(f"Unsupported compute operator kind: {spec['kind']}")


def operator_flops(spec):
    if spec["kind"] == "matmul":
        return flops_matmul(spec["m"], spec["k"], spec["n"])
    if spec["kind"] == "flash_attention":
        return flops_flash_attention(
            spec["batch"], spec["heads"], spec["seq_len"], spec["head_dim"]
        )
    raise ValueError(f"Unsupported compute operator kind: {spec['kind']}")


def build_attention_calibration_spec(specs):
    for spec in specs:
        if spec["kind"] == "flash_attention":
            return {
                **spec,
                "id": "calib_flash_attention_seq2048",
                "name": "FlashAttention Calibration Seq2048",
                "seq_len": 1024,
                "llama_component": "attention.flash_attention.calibration",
            }
    return None


def bench_calibration(spec):
    if spec is None:
        return None
    single = bench_one(spec, device=0)
    dual = dual_bench(spec)
    return {
        "id": spec["id"],
        "name": spec["name"],
        "kind": spec["kind"],
        "shape": operator_shape(spec),
        "dtype": spec["dtype"],
        "flops": operator_flops(spec),
        "single_card": single,
        "dual_card": dual,
    }


def main():
    os.makedirs(ARTIFACT, exist_ok=True)
    specs = load_specs()
    results = []
    for spec in specs:
        single = bench_one(spec, device=0)
        dual = dual_bench(spec)
        results.append(
            {
                "id": spec["id"],
                "name": spec["name"],
                "kind": spec["kind"],
                "llama_component": spec["llama_component"],
                "shape": operator_shape(spec),
                "dtype": spec["dtype"],
                "flops": operator_flops(spec),
                "single_card": single,
                "dual_card": dual,
            }
        )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "task_id": "MTT-COMPUTE-OP-SPACE-TEST",
        "device_backend": "musa",
        "device_count": torch.musa.device_count(),
        "device_names": [torch.musa.get_device_name(i) for i in range(torch.musa.device_count())],
        "calibration": {
            "attention": bench_calibration(build_attention_calibration_spec(specs))
        },
        "operators": results,
    }
    with open(os.path.join(ARTIFACT, "benchmark_results.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
