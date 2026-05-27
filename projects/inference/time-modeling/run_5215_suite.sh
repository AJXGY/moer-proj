#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

exec bash run_5215_tp_suite.sh "$@"
