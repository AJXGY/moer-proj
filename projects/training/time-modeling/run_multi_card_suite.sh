#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
MOER_PARALLEL_SCOPE=multi bash run_5214_tp_suite.sh "$@"
