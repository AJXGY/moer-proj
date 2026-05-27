#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TORCH_MUSA_BUNDLE="${TORCH_MUSA_BUNDLE:-/tmp/S80-v2.1.1.tar.gz}"
MUSA_TOOLKITS_TAR="${MUSA_TOOLKITS_TAR:-/tmp/musa_toolkits_4.2.0.tar.gz}"
MUDNN_TAR="${MUDNN_TAR:-/tmp/mudnn_3.0.0.CC2.1.tar.gz}"

python3 -m pip install --user --force-reinstall -r "${ROOT_DIR}/requirements.txt"

if [[ ! -f "${TORCH_MUSA_BUNDLE}" ]]; then
  echo "Missing torch_musa bundle: ${TORCH_MUSA_BUNDLE}" >&2
  exit 1
fi

mkdir -p /tmp/torch_musa_211
tar -xf "${TORCH_MUSA_BUNDLE}" -C /tmp/torch_musa_211
python3 -m pip install --user --force-reinstall \
  /tmp/torch_musa_211/qy1/torch-2.5.0-cp310-cp310-linux_x86_64.whl \
  /tmp/torch_musa_211/qy1/torch_musa-2.1.1-cp310-cp310-linux_x86_64.whl

apt download libopenblas0-pthread libgfortran5
mkdir -p /home/o_mabin/.local/openblas /home/o_mabin/.local/gfortran
dpkg-deb -x /home/o_mabin/libopenblas0-pthread_*_amd64.deb /home/o_mabin/.local/openblas
dpkg-deb -x /home/o_mabin/libgfortran5_*_amd64.deb /home/o_mabin/.local/gfortran

if [[ -f "${MUSA_TOOLKITS_TAR}" ]]; then
  mkdir -p /home/o_mabin/.local/musa_toolkits
  tar -xf "${MUSA_TOOLKITS_TAR}" -C /home/o_mabin/.local/musa_toolkits
fi

if [[ -f "${MUDNN_TAR}" ]]; then
  mkdir -p /home/o_mabin/.local/mudnn
  tar -xf "${MUDNN_TAR}" -C /home/o_mabin/.local/mudnn
fi

cat <<'EOF'
export LD_LIBRARY_PATH=/home/o_mabin/.local/gfortran/usr/lib/x86_64-linux-gnu:/home/o_mabin/.local/openblas/usr/lib/x86_64-linux-gnu/openblas-pthread:/home/o_mabin/.local/musa_toolkits/musa_toolkits_4.2.0/lib:/home/o_mabin/.local/mudnn/mudnn/lib:/usr/local/musa/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
EOF
