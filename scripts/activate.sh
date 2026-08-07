#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$PROJECT_DIR"
if [[ ! -f "$PROJECT_DIR/.venv/bin/activate" ]]; then
  echo "Missing .venv. Run scripts/bootstrap_wsl_cuda.sh for checks, then --install if needed." >&2
  return 1 2>/dev/null || exit 1
fi
# shellcheck disable=SC1091
source "$PROJECT_DIR/.venv/bin/activate"

echo "Speck Reflex environment activated"
echo "Project: ${PROJECT_DIR/#$HOME/~}"
echo "Python: $(python --version)"
echo "CUDA_HOME: ${CUDA_HOME:-unset}"

python - <<'PY'
import torch

print("Torch:", torch.__version__)
print("Torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
