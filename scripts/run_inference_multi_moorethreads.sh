#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_run_common.sh"
REPO_ROOT="$(moer_repo_root)"
moer_setup_ld_library_path
moer_prepare_run_dir "${REPO_ROOT}" "inference-multi"

export MOER_PARALLEL_SCOPE=multi
bash "${REPO_ROOT}/projects/inference/time-modeling/run_5215_tp_suite.sh" "$@"
