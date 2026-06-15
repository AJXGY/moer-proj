#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
export MUSA_HOME="${MUSA_HOME:-/home/o_mabin/.local/musa_toolkits/musa_toolkits_4.2.0}"
export MUSA_PATH="${MUSA_PATH:-${MUSA_HOME}}"
export PATH="${MUSA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH=${MUSA_HOME}/lib:/home/o_mabin/.local/gfortran/usr/lib/x86_64-linux-gnu:/home/o_mabin/.local/openblas/usr/lib/x86_64-linux-gnu/openblas-pthread:/home/o_mabin/.local/mudnn/mudnn/lib:/usr/local/musa/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export MOER_COMM_RUNS="${MOER_COMM_RUNS:-5}"
export MOER_COMM_WARMUPS="${MOER_COMM_WARMUPS:-2}"
export MOER_COMM_INNER_LOOPS="${MOER_COMM_INNER_LOOPS:-4}"

python3 -m torch.distributed.run --nproc_per_node=2 benchmark_comm_ops.py
python3 fit_space_model.py
python3 generate_charts.py
python3 summarize_results.py
