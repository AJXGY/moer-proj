#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
REPO_ROOT="$(cd "../../.." && pwd)"
PYTHON_WITH_ENV="${REPO_ROOT}/projects/shared/train-infer-estimation/tools/python_with_env.sh"

EXTRA_LD_PATHS=()
for candidate in \
  "/home/o_mabin/.local/gfortran/usr/lib/x86_64-linux-gnu" \
  "/home/o_mabin/.local/openblas/usr/lib/x86_64-linux-gnu/openblas-pthread" \
  "/home/o_mabin/.local/musa_toolkits/musa_toolkits_4.2.0/lib" \
  "/home/o_mabin/.local/mudnn/mudnn/lib" \
  "/usr/local/musa/lib"
do
  if [[ -d "${candidate}" ]]; then
    EXTRA_LD_PATHS+=("${candidate}")
  fi
done

if [[ ${#EXTRA_LD_PATHS[@]} -gt 0 ]]; then
  EXTRA_JOINED="$(IFS=:; echo "${EXTRA_LD_PATHS[*]}")"
  export LD_LIBRARY_PATH="${EXTRA_JOINED}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

"${PYTHON_WITH_ENV}" capture_runtime_observation.py
python3 build_training_model.py
python3 verify_training_model.py
python3 generate_charts.py
python3 summarize_results.py
