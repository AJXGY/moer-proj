#!/usr/bin/env python3
import json
import os
from datetime import datetime, timezone


ROOT = os.path.dirname(os.path.abspath(__file__))


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_latest_artifact():
    path = os.path.join(ROOT, "latest_artifact.txt")
    if not os.path.exists(path):
        raise FileNotFoundError("latest_artifact.txt not found, run benchmark_parallel_infer_time.py first")
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def error_percent(real_ms, pred_ms):
    return abs(real_ms - pred_ms) / real_ms * 100.0


def solve_linear_system(matrix, rhs):
    size = len(matrix)
    m = [row[:] + [rhs_value] for row, rhs_value in zip(matrix, rhs)]

    for col in range(size):
        pivot = col
        for r in range(col + 1, size):
            if abs(m[r][col]) > abs(m[pivot][col]):
                pivot = r
        if abs(m[pivot][col]) < 1e-12:
            raise ValueError("Singular matrix while fitting model")
        if pivot != col:
            m[col], m[pivot] = m[pivot], m[col]

        div = m[col][col]
        for c in range(col, size + 1):
            m[col][c] /= div

        for r in range(size):
            if r == col:
                continue
            factor = m[r][col]
            for c in range(col, size + 1):
                m[r][c] -= factor * m[col][c]

    return [m[i][size] for i in range(size)]


def fit_coefficients(configs, single_only=False):
    # Single-card mode uses a reduced model to avoid singular matrices.
    if single_only:
        feature_fn = lambda cfg: [1.0, 1.0 / float(cfg["microbatch_num"])]
    else:
        feature_fn = lambda cfg: [1.0, float(cfg["pipeline_parallel_size"]), 1.0 / float(cfg["microbatch_num"])]

    feature_count = len(feature_fn(configs[0]))
    xtx = [[0.0 for _ in range(feature_count)] for _ in range(feature_count)]
    xty = [0.0 for _ in range(feature_count)]

    for cfg in configs:
        x = feature_fn(cfg)
        # Prefer median-based reported value when available (more robust to outliers)
        y = float(cfg["real"].get("median_ms", cfg["real"].get("avg_ms", 0.0)))
        for i in range(feature_count):
            xty[i] += x[i] * y
            for j in range(feature_count):
                xtx[i][j] += x[i] * x[j]

    coeffs = solve_linear_system(xtx, xty)
    return coeffs


def main():
    artifact = read_latest_artifact()
    bench = load_json(os.path.join(artifact, "benchmark_results.json"))
    configs = bench["configs"]
    single_only = bool(bench.get("single_only", False)) or all(int(cfg["pipeline_parallel_size"]) == 1 for cfg in configs)

    coeffs = fit_coefficients(configs, single_only=single_only)
    if single_only:
        alpha, gamma = coeffs
        beta = None
    else:
        alpha, beta, gamma = coeffs

    evaluated = []
    for cfg in configs:
        pp = float(cfg["pipeline_parallel_size"])
        inv_mb = 1.0 / float(cfg["microbatch_num"])
        t_real = float(cfg["real"].get("median_ms", cfg["real"].get("avg_ms", 0.0)))
        t_sim = alpha + gamma * inv_mb if single_only else alpha + beta * pp + gamma * inv_mb
        evaluated.append({
            "id": cfg["id"],
            "name": cfg["name"],
            "pipeline_parallel_size": cfg["pipeline_parallel_size"],
            "microbatch_num": cfg["microbatch_num"],
            "t_real_ms": t_real,
            "t_sim_ms": t_sim,
            "error_percent": error_percent(t_real, t_sim)
        })

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "task_id": "MTT-PARALLEL-INFER-TIME-TEST",
        "model": {
            "alpha_ms": alpha,
            "gamma_ms": gamma,
            "single_only": single_only,
            "formula": "T_sim = alpha + gamma * (1/microbatch_num)" if single_only else "T_sim = alpha + beta * pipeline_parallel_size + gamma * (1/microbatch_num)",
            **({} if single_only else {"beta_ms_per_pp": beta})
        },
        "configs": evaluated,
        "single_only": single_only,
        "all_within_20_percent": all(item["error_percent"] <= 20.0 for item in evaluated)
    }

    with open(os.path.join(artifact, "time_model_results.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
