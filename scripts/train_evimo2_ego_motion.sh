#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# shellcheck disable=SC1091
source "$PROJECT_DIR/scripts/activate.sh"
cd "$PROJECT_DIR"

exec python -m software.training.train_evimo2_ego_motion \
  --config configs/evimo2_ego_motion_v0.1.yaml \
  "$@"
