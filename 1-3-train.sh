#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/scripts/_run_common.sh"

REPO_ROOT="${SCRIPT_DIR}"
moer_setup_ld_library_path
moer_prepare_run_dir "${REPO_ROOT}" "1-3-train"

DEFAULT_MODEL_PATH="/home/o_mabin/moerxiancheng-clj-xyj-proj/clj-proj/model/Meta-Llama-3.1-8B"
MODEL_PATH="${MOER_MODEL_PATH:-${MODEL_PATH:-${DEFAULT_MODEL_PATH}}}"
RUNS_PER_CONFIG="${RUNS_PER_CONFIG:-1}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-8}"
DAG_DTYPE="${DAG_DTYPE:-fp16}"
DAG_DEVICE="${DAG_DEVICE:-musa:0}"

export MOER_MODEL_PATH="${MODEL_PATH}"
export MOER_PARALLEL_SCOPE=single
export MVP_DEVICE_BACKEND="${MVP_DEVICE_BACKEND:-musa}"
export MUSA_VISIBLE_DEVICES="${MUSA_VISIBLE_DEVICES:-0}"

if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
	echo "Missing Llama3.1-8B config.json under MODEL_PATH=${MODEL_PATH}" >&2
	echo "Set MOER_MODEL_PATH=/path/to/Meta-Llama-3.1-8B and rerun." >&2
	exit 2
fi

echo "task=1-3-train"
echo "platform=MooreThreads/MUSA"
echo "parallel_scope=${MOER_PARALLEL_SCOPE}"
echo "visible_devices=${MUSA_VISIBLE_DEVICES}"
echo "model_path=${MODEL_PATH}"

python3 "${REPO_ROOT}/projects/training/time-modeling/benchmark_tp_train_time.py" \
	--model-path "${MODEL_PATH}" \
	--parallel-scope single \
	--runs-per-config "${RUNS_PER_CONFIG}" \
	--max-seq-len "${MAX_SEQ_LEN}" \
	"$@"

ARTIFACT_DIR="$(< "${MOER_LATEST_TP_ARTIFACT_FILE}")"
REPORT_JSON="${ARTIFACT_DIR}/tp_benchmark_results.json"
DAG_DIR="${ARTIFACT_DIR}/dag"
mkdir -p "${DAG_DIR}"

python3 "${REPO_ROOT}/projects/shared/train-infer-estimation/export_train_graph.py" \
	--model-path "${MODEL_PATH}" \
	--device "${DAG_DEVICE}" \
	--dtype "${DAG_DTYPE}" \
	--output-dir "${DAG_DIR}"

python3 - "${REPORT_JSON}" "${DAG_DIR}" "${MODEL_PATH}" <<'PY'
import json
import re
import sys
from pathlib import Path

report_path = Path(sys.argv[1])
dag_dir = Path(sys.argv[2])
model_path = Path(sys.argv[3])

report = json.loads(report_path.read_text(encoding="utf-8"))
model_cfg = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
dag_summary_path = dag_dir / "summary.json"
dag_nodes_path = dag_dir / "backward_graph_nodes.json"
dag_summary = json.loads(dag_summary_path.read_text(encoding="utf-8"))
dag_nodes = json.loads(dag_nodes_path.read_text(encoding="utf-8"))

expected_layers = int(model_cfg.get("num_hidden_layers", 32))


def ok(name, passed, detail):
    return {"name": name, "passed": bool(passed), "detail": detail}


def positive(value):
    try:
        return float(value) > 0.0
    except Exception:
        return False


def layer_indices_from_dag(payload):
    found = set()
    for node in payload.get("nodes", []):
        text = " ".join(str(node.get(key, "")) for key in ("id", "label", "group"))
        match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", text)
        if match:
            found.add(int(match.group(1)))
    for grad in payload.get("gradients", []):
        match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", str(grad.get("name", "")))
        if match:
            found.add(int(match.group(1)))
    return found


def is_acyclic(nodes, edges):
    ids = {str(node.get("id")) for node in nodes}
    graph = {node_id: [] for node_id in ids}
    indegree = {node_id: 0 for node_id in ids}
    for src, dst in edges:
        src = str(src)
        dst = str(dst)
        if src not in graph or dst not in graph:
            continue
        graph[src].append(dst)
        indegree[dst] += 1
    queue = [node_id for node_id, degree in indegree.items() if degree == 0]
    seen = 0
    while queue:
        node_id = queue.pop()
        seen += 1
        for dst in graph[node_id]:
            indegree[dst] -= 1
            if indegree[dst] == 0:
                queue.append(dst)
    return seen == len(ids)


def write_logic_svg(path):
    phases = ["input", "embed"] + [f"layer {idx:02d}" for idx in range(expected_layers)] + [
        "norm",
        "lm_head",
        "loss",
        "backward",
        "optimizer",
    ]
    width = 980
    node_h = 34
    gap = 18
    height = 80 + len(phases) * (node_h + gap) + 40
    x = 330
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        '<text x="32" y="42" font-size="24" font-family="monospace" fill="#111827">1-3 train logic DAG</text>',
        '<text x="32" y="66" font-size="13" font-family="monospace" fill="#475569">single-card Llama3.1-8B training: forward chain, backward gradients, optimizer update</text>',
    ]
    for idx, label in enumerate(phases):
        y = 96 + idx * (node_h + gap)
        fill = "#2563eb"
        if label.startswith("layer"):
            fill = "#475569"
        elif label in {"backward", "optimizer"}:
            fill = "#dc2626"
        elif label in {"lm_head", "loss"}:
            fill = "#15803d"
        lines.append(f'<rect x="{x}" y="{y}" width="300" height="{node_h}" rx="8" fill="{fill}" opacity="0.94"/>')
        lines.append(f'<text x="{x + 14}" y="{y + 22}" font-size="13" font-family="monospace" fill="#ffffff">{label}</text>')
        if idx:
            prev_y = 96 + (idx - 1) * (node_h + gap)
            lines.append(f'<path d="M {x + 150} {prev_y + node_h} L {x + 150} {y}" stroke="#94a3b8" stroke-width="2" marker-end="url(#arrow)"/>')
    lines.insert(
        2,
        '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="5" refY="3" orient="auto"><path d="M0,0 L0,6 L6,3 z" fill="#94a3b8"/></marker></defs>',
    )
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


configs = report.get("configs", [])
layer_indices = layer_indices_from_dag(dag_nodes)
missing_layers = [idx for idx in range(expected_layers) if idx not in layer_indices]
timing_bad = [
    cfg.get("id", cfg.get("name", "unknown"))
    for cfg in configs
    if not positive((cfg.get("real") or {}).get("avg_ms"))
]

checks = [
    ok("platform_backend_is_musa", (report.get("environment") or {}).get("backend") == "musa", json.dumps(report.get("environment", {}), ensure_ascii=False)),
    ok("single_card_config", bool(configs) and all(int(cfg.get("tensor_parallel_size", -1)) == 1 and int(cfg.get("pipeline_parallel_size", -1)) == 1 and cfg.get("parallel_scope") == "single" for cfg in configs), f"configs={len(configs)}"),
    ok("llama31_8b_structure_complete", int(report.get("model_reference", {}).get("num_hidden_layers", -1)) == expected_layers and int(report.get("model_reference", {}).get("hidden_size", -1)) == int(model_cfg.get("hidden_size", -2)), json.dumps(report.get("model_reference", {}), ensure_ascii=False)),
    ok("training_timings_positive", not timing_bad, f"bad_configs={timing_bad}"),
    ok("training_runtime_scope_matches", report.get("training_task", {}).get("runtime_scope") == "llama_backbone_autograd_with_vocab_lora_head_update", report.get("training_task", {}).get("runtime_scope")),
    ok("dag_artifacts_exist", (dag_dir / "backward_layer_graph.svg").stat().st_size > 0 and dag_summary_path.stat().st_size > 0 and dag_nodes_path.stat().st_size > 0, str(dag_dir)),
    ok("dag_has_backward_nodes", int(dag_summary.get("backward", {}).get("node_count", 0)) > 0, json.dumps(dag_summary.get("backward", {}), ensure_ascii=False)),
    ok("dag_covers_all_transformer_layers", not missing_layers, f"missing_layers={missing_layers[:10]} total_missing={len(missing_layers)}"),
    ok("dag_is_acyclic", is_acyclic(dag_nodes.get("nodes", []), dag_nodes.get("edges", [])), f"nodes={len(dag_nodes.get('nodes', []))} edges={len(dag_nodes.get('edges', []))}"),
]

write_logic_svg(dag_dir / "logic_dag.svg")
overall = all(item["passed"] for item in checks)
summary = {
    "task": "1-3-train",
    "overall": "PASS" if overall else "FAIL",
    "report_json": str(report_path),
    "dag_dir": str(dag_dir),
    "logic_dag": str(dag_dir / "logic_dag.svg"),
    "checks": checks,
}
(report_path.parent / "validation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

lines = [
    "# 1-3 Train Validation",
    "",
    f"Overall: **{summary['overall']}**",
    "",
    f"- Report: `{report_path}`",
    f"- DAG: `{dag_dir / 'backward_layer_graph.svg'}`",
    f"- Logic DAG: `{dag_dir / 'logic_dag.svg'}`",
    "",
    "## Checks",
]
for item in checks:
    mark = "PASS" if item["passed"] else "FAIL"
    lines.append(f"- {mark} `{item['name']}`: {item['detail']}")
(report_path.parent / "validation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

print(json.dumps(summary, ensure_ascii=False, indent=2))
if not overall:
    raise SystemExit(1)
PY

echo "validation_report=${ARTIFACT_DIR}/validation_report.md"
echo "validation_summary=${ARTIFACT_DIR}/validation_summary.json"
echo "dag_dir=${DAG_DIR}"
