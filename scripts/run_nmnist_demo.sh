#!/usr/bin/env bash
set -Eeuo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/speck-reflex-mpl}" PYTHONPATH=src python experiments/official_baselines/nmnist/evaluate_nmnist_baseline.py "$@"
