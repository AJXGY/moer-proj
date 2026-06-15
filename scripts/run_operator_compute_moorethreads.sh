#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_run_common.sh"
REPO_ROOT="$(moer_repo_root)"
moer_setup_ld_library_path
moer_prepare_run_dir "${REPO_ROOT}" "operator-compute"

bash "${REPO_ROOT}/projects/operators/compute/run_523_suite.sh" "$@"

python3 - <<'PY'
import json
import os
from pathlib import Path

artifact_root = Path(os.environ.get("MOER_ARTIFACT_ROOT", ""))
if not artifact_root:
    artifact_root = max(Path("/home/o_mabin/moer-proj/results/operator-compute").glob("*/artifacts"), key=lambda p: p.stat().st_mtime)
result_path = artifact_root / "space_model_results.json"
if not result_path.exists():
    raise SystemExit(f"missing result JSON: {result_path}")

data = json.loads(result_path.read_text())
labels = {
    "mlp_up_gemm": "矩阵乘法(MLP up)",
    "flash_attention": "注意力算子",
    "attention_output_proj_gemm": "注意力输出投影",
}
print("=== 摩尔线程计算密集型算子本次运行结果 ===")
print(f"结果文件: {result_path}")
for op in data.get("operators", []):
    op_id = op.get("id")
    if op_id not in labels:
        continue
    print(f"{labels[op_id]} {op_id}:")
    for mode, mode_label in [("single_card", "单卡"), ("dual_card", "双卡")]:
        item = op.get(mode) or {}
        real = float(item.get("t_real_ms", 0.0))
        sim = float(item.get("t_sim_ms", 0.0))
        err = float(item.get("error_percent", 0.0))
        status = "通过" if err <= 20.0 else "不通过"
        print(f"  {mode_label}: measured={real:.3f} ms  predicted={sim:.3f} ms  error={err:.2f}%  结果={status}")
print(f"整体: <=20% {'通过' if data.get('all_within_20_percent') else '不通过'}; <=10% {'通过' if data.get('all_within_10_percent') else '不通过'}")
PY

