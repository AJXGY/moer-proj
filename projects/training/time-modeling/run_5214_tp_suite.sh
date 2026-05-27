#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
export LD_LIBRARY_PATH=/home/o_mabin/.local/gfortran/usr/lib/x86_64-linux-gnu:/home/o_mabin/.local/openblas/usr/lib/x86_64-linux-gnu/openblas-pthread:/home/o_mabin/.local/musa_toolkits/musa_toolkits_4.2.0/lib:/home/o_mabin/.local/mudnn/mudnn/lib:/usr/local/musa/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

python3 benchmark_tp_train_time.py
python3 fit_tp_time_model.py
python3 summarize_tp_results.py
