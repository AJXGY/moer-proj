#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

required_paths=(
  "projects/operators/communication/benchmark_comm_ops.py"
  "projects/operators/communication/run_529_suite.sh"
  "projects/operators/memory/benchmark_memory_ops.py"
  "projects/operators/memory/run_526_suite.sh"
  "projects/operators/compute/benchmark_compute_ops.py"
  "projects/operators/compute/run_523_suite.sh"
  "projects/training/runtime-validation/run_516_suite.sh"
  "projects/training/runtime-validation/train_samples.jsonl"
  "projects/training/lora-runtime-validation/run_516_suite.sh"
  "projects/training/lora-runtime-validation/train_runner.py"
  "projects/training/model-structure/run_5111_suite.sh"
  "projects/training/model-structure/build_training_model.py"
  "projects/training/time-modeling/benchmark_tp_train_time.py"
  "projects/training/time-modeling/run_5214_tp_suite.sh"
  "projects/training/time-modeling/train_samples.jsonl"
  "projects/inference/runtime-validation/run_515_suite.sh"
  "projects/inference/runtime-validation/infer_runner.py"
  "projects/inference/model-structure/run_512_suite.sh"
  "projects/inference/model-structure/build_inference_model.py"
  "projects/inference/time-modeling/benchmark_tp_infer_time.py"
  "projects/inference/time-modeling/run_5215_tp_suite.sh"
  "projects/shared/train-infer-estimation/tools/python_with_env.sh"
  "projects/shared/train-infer-estimation/torch_train_mvp.py"
  "projects/reports/README.md"
  "scripts/run_inference_runtime_moorethreads.sh"
  "scripts/run_inference_model_structure_moorethreads.sh"
  "scripts/run_operator_comm_moorethreads.sh"
  "scripts/run_operator_mem_moorethreads.sh"
  "scripts/run_operator_compute_moorethreads.sh"
  "scripts/run_training_runtime_moorethreads.sh"
  "scripts/run_training_lora_moorethreads.sh"
  "scripts/run_training_model_structure_moorethreads.sh"
  "scripts/run_training_moorethreads_smoke.sh"
  "scripts/run_inference_moorethreads_smoke.sh"
)

for path in "${required_paths[@]}"; do
  if [[ ! -e "${REPO_ROOT}/${path}" ]]; then
    printf 'missing required path: %s\n' "${path}" >&2
    exit 1
  fi
done

mapfile -t shell_files < <(find "${REPO_ROOT}/scripts" "${REPO_ROOT}/projects" "${REPO_ROOT}/clj-proj/5.1.5" "${REPO_ROOT}/clj-proj/5.1.6" "${REPO_ROOT}/clj-proj/5.1.11" "${REPO_ROOT}/clj-proj/5.2.3" "${REPO_ROOT}/clj-proj/5.2.6" "${REPO_ROOT}/clj-proj/5.2.9" "${REPO_ROOT}/clj-proj/5.2.14" "${REPO_ROOT}/clj-proj/train-infer-estimation-release-2026-04-11" "${REPO_ROOT}/xyj/5.1.6" "${REPO_ROOT}/xyj/5.1.12" "${REPO_ROOT}/xyj/5.2.15" -maxdepth 5 -type f -name '*.sh' | sort)
if [[ ${#shell_files[@]} -gt 0 ]]; then
  bash -n "${shell_files[@]}"
fi

mapfile -t python_files < <(find "${REPO_ROOT}/projects/operators" "${REPO_ROOT}/projects/training" "${REPO_ROOT}/projects/inference" "${REPO_ROOT}/projects/shared" -type f -name '*.py' | sort)
if [[ ${#python_files[@]} -gt 0 ]]; then
  python3 -m py_compile "${python_files[@]}"
fi

printf 'Project layout check passed.\n'
