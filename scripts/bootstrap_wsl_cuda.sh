#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/bootstrap_wsl_cuda.sh [--install]

Without --install this script performs read-only checks and prints the plan.
With --install it creates/reuses .venv and installs fixed PyTorch cu128 wheels,
then core and development dependencies. It never installs CUDA Toolkit, NVIDIA
drivers, binds USB, or changes system Python.
EOF
}

install=false
case "${1:-}" in
  "") ;;
  --install) install=true ;;
  -h|--help) usage; exit 0 ;;
  *) usage >&2; exit 2 ;;
esac

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

command -v python3 >/dev/null || { echo "ERROR: python3 not found" >&2; exit 1; }
python3 -c 'import sys; assert sys.version_info >= (3, 10); print("Python:", sys.version.split()[0])'
available_kib="$(df -Pk . | awk 'NR==2 {print $4}')"
echo "Available disk: $((available_kib / 1024)) MiB"
if command -v nvidia-smi >/dev/null; then nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader; else echo "WARNING: nvidia-smi not available"; fi
if grep -qi microsoft /proc/version 2>/dev/null; then
  echo "Platform: WSL2; use the Windows NVIDIA driver, not a Linux display driver. USB remains unbound."
else
  echo "Platform: native Linux; system driver management remains outside this script."
fi

cat <<'EOF'
Plan:
  1. Create/reuse .venv.
  2. Upgrade pip/setuptools/wheel inside .venv.
  3. Install torch 2.10.0, torchvision 0.25.0, torchaudio 2.10.0 from cu128 index.
  4. Install requirements/core.txt and requirements/dev.txt.
EOF

if ! $install; then
  echo "CHECK_ONLY: pass --install to execute the plan."
  exit 0
fi

python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements/core.txt
python -m pip install -r requirements/dev.txt
python -m pip check
echo "Installation completed in $project_root/.venv"
