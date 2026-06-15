#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

export MUSA_HOME="${MUSA_HOME:-/home/o_mabin/.local/musa_toolkits/musa_toolkits_4.2.0}"
export MUSA_PATH="${MUSA_PATH:-${MUSA_HOME}}"
export PATH="${MUSA_HOME}/bin:${PATH}"
EXTRA_LD_PATHS=()
for candidate in \
	"${MUSA_HOME}/lib" \
	"/home/o_mabin/.local/gfortran/usr/lib/x86_64-linux-gnu" \
	"/home/o_mabin/.local/openblas/usr/lib/x86_64-linux-gnu/openblas-pthread" \
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

python3 benchmark_tp_infer_time.py --parallel-scope "${MOER_PARALLEL_SCOPE:-all}" "$@"
python3 fit_tp_time_model.py
python3 summarize_tp_results.py

echo "5.2.15 TP supplement suite finished."
