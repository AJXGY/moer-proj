#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MUSA_LD=(
  "/home/o_mabin/.local/gfortran/usr/lib/x86_64-linux-gnu"
  "/home/o_mabin/.local/openblas/usr/lib/x86_64-linux-gnu/openblas-pthread"
  "/home/o_mabin/.local/musa_toolkits/musa_toolkits_4.2.0/lib"
  "/home/o_mabin/.local/mudnn/mudnn/lib"
  "/usr/local/musa/lib"
)

LD_PREFIX=""
for path in "${MUSA_LD[@]}"; do
  if [[ -d "$path" ]]; then
    if [[ -n "$LD_PREFIX" ]]; then
      LD_PREFIX="${LD_PREFIX}:"
    fi
    LD_PREFIX="${LD_PREFIX}${path}"
  fi
done

if [[ -n "$LD_PREFIX" ]]; then
  export LD_LIBRARY_PATH="${LD_PREFIX}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

exec python3 "$@"
