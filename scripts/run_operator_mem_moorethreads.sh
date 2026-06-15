#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_run_common.sh"
REPO_ROOT="$(moer_repo_root)"
moer_setup_ld_library_path
moer_prepare_run_dir "${REPO_ROOT}" "operator-memory"

bash "${REPO_ROOT}/projects/operators/memory/run_526_suite.sh" "$@"

python3 - <<'PY'
import json
import os
from pathlib import Path

artifact_root = Path(os.environ.get("MOER_ARTIFACT_ROOT", ""))
if not artifact_root:
    artifact_root = max(Path("/home/o_mabin/moer-proj/results/operator-memory").glob("*/artifacts"), key=lambda p: p.stat().st_mtime)
result_path = artifact_root / "space_model_results.json"
if not result_path.exists():
    raise SystemExit(f"missing result JSON: {result_path}")

data = json.loads(result_path.read_text())
labels = {
    "data_copy": "数据拷贝",
    "slice_copy": "张量切片",
    "concat": "张量拼接",
}
print("=== 摩尔线程访存密集型算子本次运行结果 ===")
print(f"结果文件: {result_path}")
for kind, label in labels.items():
    rows = [op for op in data.get("operators", []) if op.get("kind") == kind]
    if not rows:
        continue
    print(f"{label} {kind}:")
    for mode, mode_label in [("single_card", "单卡"), ("dual_card", "双卡")]:
        candidates = [op for op in rows if mode in op]
        if not candidates:
            continue
        worst = max(candidates, key=lambda op: float((op.get(mode) or {}).get("error_percent", 0.0)))
        item = worst.get(mode) or {}
        size_mb = float(worst.get("bytes", 0)) / 1024 / 1024
        real = float(item.get("t_real_ms", 0.0))
        sim = float(item.get("t_sim_ms", 0.0))
        err = float(item.get("error_percent", 0.0))
        status = "通过" if err <= 20.0 else "不通过"
        print(f"  {mode_label}: size={size_mb:.0f} MB  measured={real:.3f} ms  predicted={sim:.3f} ms  max_error={err:.2f}%  结果={status}")
print(f"整体: <=20% {'通过' if data.get('all_within_20_percent') else '不通过'}")
PY

