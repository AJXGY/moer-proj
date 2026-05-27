#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
export LD_LIBRARY_PATH=/home/o_mabin/.local/gfortran/usr/lib/x86_64-linux-gnu:/home/o_mabin/.local/openblas/usr/lib/x86_64-linux-gnu/openblas-pthread:/home/o_mabin/.local/musa_toolkits/musa_toolkits_4.2.0/lib:/home/o_mabin/.local/mudnn/mudnn/lib:/usr/local/musa/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

python3 benchmark_compute_ops.py
python3 fit_space_model.py
python3 generate_charts.py
python3 summarize_results.py
