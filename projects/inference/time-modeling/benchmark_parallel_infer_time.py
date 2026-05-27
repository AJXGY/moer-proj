#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_ROOT = os.path.join(ROOT, "artifacts")
REPO_ROOT = os.path.abspath(os.path.join(ROOT, "../../.."))
INFER_RUNNER = os.path.join(REPO_ROOT, "projects", "inference", "runtime-validation", "infer_runner.py")
BASE_PROMPTS = os.path.join(REPO_ROOT, "projects", "inference", "runtime-validation", "prompts.jsonl")
DEFAULT_MODEL = os.path.join(REPO_ROOT, "clj-proj", "model", "Meta-Llama-3.1-8B")


def utc_stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def load_configs():
    path = os.path.join(ROOT, "parallel_configs.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def select_configs(configs, single_only):
    if not single_only:
        return configs
    return [cfg for cfg in configs if int(cfg.get("pipeline_parallel_size", 0)) == 1]


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
                "mode": "real_inference_benchmark",
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
                "mode": "real_inference_benchmark",
            }
    except Exception:
        pass

    return {
        "backend": "cpu",
        "device_count": 0,
        "device_names": [],
        "mode": "unsupported",
    }


def load_base_prompts(path):
    prompts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            prompts.append(json.loads(line))
    return prompts


def build_prompt_file(base_prompts, target_count, output_path):
    selected = []
    for idx in range(target_count):
        base = dict(base_prompts[idx % len(base_prompts)])
        base["id"] = f"bench-{idx:03d}"
        selected.append(base)

    with open(output_path, "w", encoding="utf-8") as f:
        for item in selected:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def run_one_benchmark(cfg, run_idx, env, model_path, artifact_dir, runs_max_new_tokens, single_device_ids, dual_device_ids, single_only):
    pp = int(cfg["pipeline_parallel_size"])
    mb = int(cfg["microbatch_num"])
    mode_name = "single" if single_only or pp == 1 else "dual"
    device_ids = single_device_ids if mode_name == "single" else dual_device_ids
    parsed_ids = [item.strip() for item in device_ids.split(",") if item.strip()]
    num_devices = 1 if mode_name == "single" else (len(parsed_ids) if parsed_ids else 2)

    prompt_count = max(1, mb * 2)
    cfg_dir = os.path.join(artifact_dir, "raw_runs", cfg["id"])
    os.makedirs(cfg_dir, exist_ok=True)
    prompts_path = os.path.join(cfg_dir, f"prompts_mb{mb}.jsonl")

    base_prompts = load_base_prompts(BASE_PROMPTS)
    build_prompt_file(base_prompts, prompt_count, prompts_path)

    out_dir = os.path.join(cfg_dir, f"run_{run_idx}")
    cmd = [
        sys.executable,
        INFER_RUNNER,
        "--model-path",
        model_path,
        "--prompts-file",
        prompts_path,
        "--output-dir",
        out_dir,
        "--mode-name",
        mode_name,
        "--num-devices",
        str(num_devices),
        "--device-type",
        env["backend"] if env["backend"] in {"musa", "cuda", "cpu"} else "auto",
        "--device-ids",
        device_ids,
        "--max-new-tokens",
        str(runs_max_new_tokens),
    ]

    started = time.perf_counter()
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    summary_path = os.path.join(out_dir, "summary.json")
    summary = None
    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
    # Prefer internal per-request generation timings (gen_ms) reported by infer_runner
    measured_elapsed_ms = None
    if summary and isinstance(summary.get("worker_payloads"), list):
        total_gen = 0.0
        found = False
        for worker in summary.get("worker_payloads", []):
            for res in worker.get("results", []):
                if res.get("gen_ms") is not None:
                    total_gen += float(res.get("gen_ms"))
                    found = True
        if found:
            measured_elapsed_ms = total_gen

    # If no internal gen measurements, fall back to subprocess wall-clock
    used_elapsed = measured_elapsed_ms if measured_elapsed_ms is not None else elapsed_ms

    success = proc.returncode == 0 and summary is not None and bool(summary.get("success"))
    return {
        "run_index": run_idx,
        "elapsed_ms": used_elapsed,
        "raw_elapsed_ms": elapsed_ms,
        "measured_elapsed_ms": measured_elapsed_ms,
        "returncode": proc.returncode,
        "stdout_tail": "\n".join(proc.stdout.splitlines()[-10:]),
        "stderr_tail": "\n".join(proc.stderr.splitlines()[-20:]),
        "summary_path": summary_path,
        "summary": summary,
        "success": success,
    }


def compute_metrics(run_payloads):
    vals = [item["elapsed_ms"] for item in run_payloads if item.get("success")]
    vals.sort()
    if not vals:
        return None
    # Use median as the reported central value to reduce sensitivity to outliers
    median_ms = vals[len(vals) // 2] if len(vals) % 2 == 1 else (vals[len(vals) // 2 - 1] + vals[len(vals) // 2]) / 2
    avg_ms = median_ms
    return {
        "timings_ms": vals,
        "avg_ms": avg_ms,
        "median_ms": median_ms,
        "min_ms": min(vals),
        "max_ms": max(vals),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="5.2.15 real inference benchmark")
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--runs-per-config", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--single-device-ids", default="0")
    parser.add_argument("--dual-device-ids", default="0,1")
    parser.add_argument("--single-only", action="store_true", help="benchmark only single-card pp=1 configs")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(ARTIFACT_ROOT, exist_ok=True)
    artifact_dir = os.path.join(ARTIFACT_ROOT, utc_stamp())
    os.makedirs(artifact_dir, exist_ok=True)

    env = detect_backend()
    if env["backend"] == "cpu":
        raise RuntimeError("No MUSA/CUDA accelerator available for real benchmark")

    configs = select_configs(load_configs(), args.single_only)
    if not configs:
        raise RuntimeError("No benchmark configs selected after applying single-only filter")

    results = []
    has_failure = False
    for cfg in configs:
        run_payloads = []
        for run_idx in range(args.runs_per_config):
            payload = run_one_benchmark(
                cfg,
                run_idx,
                env,
                args.model_path,
                artifact_dir,
                args.max_new_tokens,
                args.single_device_ids,
                args.dual_device_ids,
                args.single_only,
            )
            run_payloads.append(payload)

        metrics = compute_metrics(run_payloads)
        cfg_success = metrics is not None and all(item.get("success") for item in run_payloads)
        if not cfg_success:
            has_failure = True

        results.append(
            {
                "id": cfg["id"],
                "name": cfg["name"],
                "pipeline_parallel_size": cfg["pipeline_parallel_size"],
                "microbatch_num": cfg["microbatch_num"],
                "input_tokens": cfg["input_tokens"],
                "dtype": cfg["dtype"],
                "real": metrics,
                "runs": run_payloads,
                "success": cfg_success,
            }
        )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "task_id": "MTT-PARALLEL-INFER-TIME-TEST",
        "model_name": "Llama-3.1-8B",
        "model_path": os.path.abspath(args.model_path),
        "single_only": args.single_only,
        "hardware_scope": "single_node_single_gpu" if args.single_only else "single_node_dual_gpu",
        "measurement_type": "real_inference_benchmark",
        "measurement_note": "T_real comes from the median of per-config run latencies; when available, each run uses internal gen_ms excluding model load.",
        "runs_per_config": args.runs_per_config,
        "single_device_ids": args.single_device_ids,
        "dual_device_ids": args.dual_device_ids,
        "device_backend": env["backend"],
        "device_count": env["device_count"],
        "benchmark_device_count": 1 if args.single_only else env["device_count"],
        "device_names": env["device_names"][:1] if args.single_only else env["device_names"],
        "benchmark_mode": "real_inference_benchmark_single_card" if args.single_only else env["mode"],
        "all_configs_success": not has_failure,
        "configs": results,
    }

    with open(os.path.join(artifact_dir, "benchmark_results.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    with open(os.path.join(ROOT, "latest_artifact.txt"), "w", encoding="utf-8") as f:
        f.write(artifact_dir)

    print(os.path.join(artifact_dir, "benchmark_results.json"))

    if has_failure:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
