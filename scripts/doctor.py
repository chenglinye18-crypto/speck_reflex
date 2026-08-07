#!/usr/bin/env python3
"""Read-only project diagnostics; device discovery requires two explicit gates."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from speck_reflex.environment import collect_environment  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardware", action="store_true", help="allow device-map query only when SPECK_ALLOW_HARDWARE_TESTS=1")
    args = parser.parse_args()
    allow = os.environ.get("SPECK_ALLOW_HARDWARE_TESTS") == "1"
    info = collect_environment(ROOT)
    for key, value in info.items():
        print(f"{key}: {value}")
    status = subprocess.run(["git", "submodule", "status", "--recursive"], cwd=ROOT, text=True, capture_output=True, check=False)
    print("submodules:")
    print(status.stdout.rstrip() or "  none")
    print(f"nmnist_data_present: {(ROOT / 'data/official/NMNIST/test.zip').exists()}")
    print(f"official_checkpoint_present: {(ROOT / 'third_party/synsense/sinabs/docs/tutorials/scnn_mnist.nir').exists()}")
    print(f"local_training_checkpoint_present: {(ROOT / 'experiments/official_baselines/nmnist/checkpoints/best.pt').exists()}")
    print("hardware_access_mode: disabled_by_default")
    print(f"hardware_tests_allowed: {allow}")
    if args.hardware:
        if not allow:
            print("ERROR: --hardware also requires SPECK_ALLOW_HARDWARE_TESTS=1", file=sys.stderr)
            return 2
        from sinabs.backend.dynapcnn.io import get_device_map

        print(f"device_map: {get_device_map()}")
    else:
        print("device_discovery_called: False")
    return 0 if status.returncode == 0 else status.returncode


if __name__ == "__main__":
    raise SystemExit(main())
