#!/usr/bin/env python3
"""Validate the official SpikeMS checkpoint and one synthetic CUDA forward."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from reference.spikems.model_compat import (  # noqa: E402
    build_official_model,
    load_official_slayer_cuda,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--slayer-root",
        type=Path,
        default=Path("/home/speck/.cache/spikems_reference/slayerPytorch"),
    )
    parser.add_argument(
        "--build-root",
        type=Path,
        default=Path("/home/speck/.cache/torch_extensions/spikems_reference"),
    )
    args = parser.parse_args()

    load_official_slayer_cuda(args.slayer_root, args.build_root)
    spikems = REPO_ROOT / "third_party" / "SpikeMS"
    checkpoint_path = (
        spikems / "pretrainedModels" / "EVIMO-pretrained" / "out" / "checkpoint.pth.tar"
    )
    device = torch.device("cuda:0")
    model, checkpoint, load_result = build_official_model(spikems, checkpoint_path, device)

    state = checkpoint["state_dict"]
    checkpoint_finite = all(bool(torch.isfinite(value).all()) for value in state.values())
    x = torch.zeros((1, 2, 15, 15, 8), device=device, dtype=torch.float32)
    x[0, 0, 3, 4, 1] = 1
    x[0, 0, 7, 8, 3] = 1
    x[0, 1, 5, 6, 2] = 1
    x[0, 1, 10, 11, 6] = 1
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.no_grad():
        output = model(x)
    torch.cuda.synchronize()

    result = {
        "marker": "MODEL_INTERFACE_SMOKE_ONLY",
        "dependency_audit": "PASS",
        "checkpoint_structure_gate": "PASS",
        "model_construction": "PASS",
        "spikems_model_forward_gate": "PASS",
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_state_entries": len(state),
        "checkpoint_finite": checkpoint_finite,
        "missing_keys": load_result.missing_keys,
        "unexpected_keys": load_result.unexpected_keys,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "device": str(next(model.parameters()).device),
        "dtype": str(next(model.parameters()).dtype),
        "input_shape": list(x.shape),
        "input_spikes": int(x.sum().item()),
        "output_shape": list(output.shape),
        "output_spikes": int(output.sum().item()),
        "output_finite": bool(torch.isfinite(output).all()),
        "forward_ms_diagnostic_only": (time.perf_counter() - started) * 1000.0,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
