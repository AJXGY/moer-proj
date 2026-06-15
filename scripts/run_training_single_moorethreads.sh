#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_run_common.sh"
REPO_ROOT="$(moer_repo_root)"
moer_setup_ld_library_path
moer_prepare_run_dir "${REPO_ROOT}" "training-single"

export MOER_PARALLEL_SCOPE=single
bash "${REPO_ROOT}/projects/training/time-modeling/run_5214_tp_suite.sh" "$@"
