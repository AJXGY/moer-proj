#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_run_common.sh"
REPO_ROOT="$(moer_repo_root)"
moer_setup_ld_library_path
moer_prepare_run_dir "${REPO_ROOT}" "operator-communication"

bash "${REPO_ROOT}/projects/operators/communication/run_529_suite.sh" "$@"

python3 - <<'PY'
import json
import os
from pathlib import Path

artifact_root = Path(os.environ.get("MOER_ARTIFACT_ROOT", ""))
if not artifact_root:
    artifact_root = max(Path("/home/o_mabin/moer-proj/results/operator-communication").glob("*/artifacts"), key=lambda p: p.stat().st_mtime)
result_path = artifact_root / "space_model_results.json"
if not result_path.exists():
    raise SystemExit(f"missing result JSON: {result_path}")

data = json.loads(result_path.read_text())
labels = {
    "all_gather": "AllGather",
    "all_reduce": "AllReduce",
    "reduce_scatter": "ReduceScatter",
}
print("=== 摩尔线程通信密集型算子本次运行结果 ===")
print(f"结果文件: {result_path}")
for kind, label in labels.items():
    rows = [op for op in data.get("operators", []) if op.get("kind") == kind]
    if not rows:
        continue
    worst = max(rows, key=lambda op: float(op.get("error_percent", 0.0)))
    size_mb = float(worst.get("bytes", 0)) / 1024 / 1024
    real = float(worst.get("t_real_ms", 0.0))
    sim = float(worst.get("t_sim_ms", 0.0))
    err = float(worst.get("error_percent", 0.0))
    status = "通过" if err <= 20.0 else "不通过"
    print(f"{label} {kind}: worst_size={size_mb:.0f} MB  measured={real:.3f} ms  predicted={sim:.3f} ms  max_error={err:.2f}%  结果={status}")
print(f"整体: <=20% {'通过' if data.get('all_within_20_percent') else '不通过'}")
PY

