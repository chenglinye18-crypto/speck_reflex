#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$HOME/projects/speck_reflex"

cd "$PROJECT_DIR"
source "$PROJECT_DIR/.venv/bin/activate"

echo "Speck Reflex environment activated"
echo "Project: $PROJECT_DIR"
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
