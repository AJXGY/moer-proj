#!/usr/bin/env python3
import json
import os
from datetime import datetime, timezone


ROOT = os.path.dirname(os.path.abspath(__file__))


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def read_latest_artifact():
    path = os.environ.get("MOER_LATEST_TP_ARTIFACT_FILE", os.path.join(ROOT, "latest_tp_artifact.txt"))
    if not os.path.exists(path):
        raise FileNotFoundError("latest_tp_artifact.txt not found, run benchmark_tp_infer_time.py first")
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read().strip()


def main():
    artifact = read_latest_artifact()
    bench = load_json(os.path.join(artifact, "tp_benchmark_results.json"))
    model = load_json(os.path.join(artifact, "tp_time_model_results.json"))
    configs = model.get("configs", [])
    skipped = bench.get("skipped_configs", [])
    tp_sizes = sorted({int(item.get("tensor_parallel_size", 1)) for item in configs})
    scopes = sorted({item.get("parallel_scope", "single" if int(item.get("tensor_parallel_size", 1)) == 1 else "multi") for item in configs})
    passed = bool(model.get("all_within_20_percent"))
    acceptance_status = model.get("acceptance_status", "unknown")

    lines = [
        "# Inference Time Modeling Summary",
        "",
        f"- generated_at: {datetime.now(timezone.utc).isoformat()}",
        f"- artifact: {artifact}",
        f"- backend: {bench.get('environment', {}).get('backend', 'unknown')}",
        f"- device_count: {bench.get('environment', {}).get('device_count', 'unknown')}",
        f"- scopes: {', '.join(scopes) if scopes else 'none'}",
        f"- tensor_parallel_sizes: {', '.join(str(value) for value in tp_sizes) if tp_sizes else 'none'}",
        f"- acceptance_status: {acceptance_status}",
        f"- result: {'pass' if passed else 'not_acceptance'}",
        "",
        "## Config Results",
        "",
        "| config | scope | TP | MB | T_real(ms) | T_sim(ms) | error | mode |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in configs:
        lines.append(
            "| {id} | {scope} | {tp} | {mb} | {real:.3f} | {sim:.3f} | {err:.2f}% | {mode} |".format(
                id=item["id"],
                scope=item.get("parallel_scope", "single" if int(item.get("tensor_parallel_size", 1)) == 1 else "multi"),
                tp=int(item.get("tensor_parallel_size", 1)),
                mb=int(item.get("microbatch_num", 1)),
                real=float(item["t_real_ms"]),
                sim=float(item["t_sim_ms"]),
                err=float(item["error_percent"]),
                mode=item.get("prediction_mode", "unknown"),
            )
        )

    if skipped:
        lines.extend(["", "## Skipped Configs", "", "| config | reason |", "| --- | --- |"])
        for item in skipped:
            lines.append(f"| {item.get('id', 'unknown')} | {item.get('reason', 'unknown')} |")

    lines.extend(
        [
            "",
            "## Files",
            "",
            f"- benchmark: {os.path.join(artifact, 'tp_benchmark_results.json')}",
            f"- model: {os.path.join(artifact, 'tp_time_model_results.json')}",
        ]
    )
    text = "\n".join(lines) + "\n"
    for output_path in [
        os.path.join(artifact, "summary.md"),
        os.path.join(ROOT, "inference_time_model_summary.md"),
    ]:
        with open(output_path, "w", encoding="utf-8") as handle:
            handle.write(text)


if __name__ == "__main__":
    main()
