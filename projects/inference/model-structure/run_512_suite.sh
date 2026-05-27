#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
python3 build_inference_model.py
python3 generate_charts.py
python3 summarize_results.py
