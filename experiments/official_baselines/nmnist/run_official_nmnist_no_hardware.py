#!/usr/bin/env python3
"""No-hardware port of the Sinabs 3.1.3 N-MNIST quick-start baseline.

Official sources:
- third_party/synsense/sinabs/docs/speck/notebooks/nmnist_quick_start.ipynb
- third_party/synsense/sinabs/docs/speck/specksim.md
- third_party/synsense/sinabs/docs/speck/faqs/save_hardware_config_as_binary.md

This file deliberately never discovers, opens, or writes to a physical device.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import traceback
from pathlib import Path

import numpy as np
import samna
import sinabs
import sinabs.layers as sl
import tonic
import torch
from sinabs.activation.surrogate_gradient_fn import PeriodicExponential
from sinabs.backend.dynapcnn import DynapcnnNetwork
from sinabs.backend.dynapcnn.specksim import SpecksimNetwork, from_sequential
from tonic.datasets import NMNIST
from tonic.transforms import ToFrame
from torch import nn


SEED = 17
INPUT_SHAPE = (2, 34, 34)
N_TIME_STEPS = 100
DEVICE_TYPE = "speck2fdevkit"


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def set_reproducible_seed() -> None:
    # COMPATIBILITY_PATCH: the official notebook does not fix a seed; the audit requires seed 17.
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def build_official_snn(batch_size: int = 1) -> nn.Sequential:
    """Build the BPTT SNN architecture from the Sinabs 3.1.3 quick start."""
    # COMPATIBILITY_PATCH: batch_size is 1 instead of the notebook's training batch of 4
    # because this is a one-sample, inference-only compatibility smoke test.
    snn_bptt = nn.Sequential(
        nn.Conv2d(
            in_channels=2,
            out_channels=8,
            kernel_size=(3, 3),
            padding=(1, 1),
            bias=False,
        ),
        sl.IAFSqueeze(
            batch_size=batch_size,
            min_v_mem=-1.0,
            surrogate_grad_fn=PeriodicExponential(),
        ),
        nn.AvgPool2d(2, 2),
        nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=(3, 3),
            padding=(1, 1),
            bias=False,
        ),
        sl.IAFSqueeze(
            batch_size=batch_size,
            min_v_mem=-1.0,
            surrogate_grad_fn=PeriodicExponential(),
        ),
        nn.AvgPool2d(2, 2),
        nn.Conv2d(
            in_channels=16,
            out_channels=16,
            kernel_size=(3, 3),
            padding=(1, 1),
            stride=(2, 2),
            bias=False,
        ),
        sl.IAFSqueeze(
            batch_size=batch_size,
            min_v_mem=-1.0,
            surrogate_grad_fn=PeriodicExponential(),
        ),
        nn.Flatten(),
        nn.Linear(16 * 4 * 4, 10, bias=False),
        sl.IAFSqueeze(
            batch_size=batch_size,
            min_v_mem=-1.0,
            surrogate_grad_fn=PeriodicExponential(),
        ),
    )

    for layer in snn_bptt.modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_normal_(layer.weight.data)
    return snn_bptt


def reset_sinabs_states(model: nn.Module) -> None:
    for layer in model.modules():
        if isinstance(layer, sl.StatefulLayer):
            # COMPATIBILITY_PATCH: Sinabs 3.1.3 calls detach_() on state views, which
            # Torch 2.10 rejects. Replace each local state buffer with a detached zero
            # tensor instead; this preserves reset semantics without patching Sinabs.
            for name, buffer in list(layer.named_buffers(recurse=False)):
                layer.register_buffer(name, torch.zeros_like(buffer.detach()))


def load_one_official_sample(data_root: Path) -> tuple[np.ndarray, np.ndarray, int]:
    """Load one genuine N-MNIST test sample, raw and with official rasterization."""
    # COMPATIBILITY_PATCH: only the test split is downloaded; the full training split is
    # unnecessary because this baseline checks the inference/deployment chain, not accuracy.
    raw_dataset = NMNIST(save_to=str(data_root), train=False)
    raw_events, label = raw_dataset[0]

    to_raster = ToFrame(sensor_size=NMNIST.sensor_size, n_time_bins=N_TIME_STEPS)
    raster_dataset = NMNIST(save_to=str(data_root), train=False, transform=to_raster)
    raster, raster_label = raster_dataset[0]
    if int(label) != int(raster_label):
        raise RuntimeError("Raw and rasterized views returned different labels")
    return raw_events, raster, int(label)


def normalize_events_for_specksim(raw_events: np.ndarray) -> np.ndarray:
    # COMPATIBILITY_PATCH: preserve the real N-MNIST events while converting Tonic's
    # platform-sized integer fields to the uint32 event dtype required by Specksim.
    events = np.empty(len(raw_events), dtype=SpecksimNetwork.output_dtype)
    for field in ("x", "y", "t", "p"):
        events[field] = raw_events[field]
    events["t"] -= events["t"][0]
    return events


def spike_counts_from_raster(output: torch.Tensor) -> list[int]:
    return output.reshape(N_TIME_STEPS, -1).sum(dim=0).to(torch.int64).tolist()


def spike_counts_from_events(output: np.ndarray, classes: int = 10) -> list[int]:
    if len(output) == 0:
        return [0] * classes
    return np.bincount(output["p"].astype(np.int64), minlength=classes).tolist()


def write_flash_binary(config: object, destination: Path) -> int:
    # This is the official local conversion API. No FlashWrite/device method is called.
    config.factory_config.io_sel = 24
    binary = samna.speck2f.configuration_to_flash_binary(config)
    destination.write_bytes(bytes(binary))
    return len(binary)


def run(data_root: Path, output_dir: Path) -> dict[str, object]:
    set_reproducible_seed()
    data_root.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[1/7] Loading the genuine Tonic N-MNIST test sample", flush=True)
    raw_events, raster_np, label = load_one_official_sample(data_root)
    raster = torch.as_tensor(raster_np, dtype=torch.float32)
    print(
        f"sample=0 label={label} raw_events={len(raw_events)} raster_shape={tuple(raster.shape)}",
        flush=True,
    )

    print("[2/7] Building the official Sinabs 3.1.3 BPTT SNN", flush=True)
    snn = build_official_snn(batch_size=1).cpu().eval()

    print("[3/7] Running Sinabs software inference", flush=True)
    reset_sinabs_states(snn)
    with torch.no_grad():
        sinabs_output = snn(raster)
    sinabs_counts = spike_counts_from_raster(sinabs_output)
    print(f"sinabs_output_shape={tuple(sinabs_output.shape)} counts={sinabs_counts}", flush=True)

    print("[4/7] Constructing DynapcnnNetwork and generating Speck 2f config", flush=True)
    dynapcnn = DynapcnnNetwork(
        snn=snn,
        input_shape=INPUT_SHAPE,
        discretize=True,
        dvs_input=False,
    )
    compatible = dynapcnn.is_compatible_with(DEVICE_TYPE)
    # COMPATIBILITY_PATCH: use current 3.1.3 layer2core_map API instead of the
    # notebook's obsolete chip_layers_ordering argument, and never call .to(device).
    config = dynapcnn.make_config(device=DEVICE_TYPE, layer2core_map="auto")
    config_valid, validation_message = samna.speck2f.validate_configuration(config)
    print(
        f"compatible={compatible} layer2core_map={dynapcnn.layer2core_map} "
        f"config_valid={config_valid} validation_message={validation_message!r}",
        flush=True,
    )

    print("[5/7] Running quantized DynapcnnNetwork in PyTorch", flush=True)
    reset_sinabs_states(dynapcnn)
    with torch.no_grad():
        dynapcnn_output = dynapcnn(raster)
    dynapcnn_counts = spike_counts_from_raster(dynapcnn_output)
    print(
        f"dynapcnn_output_shape={tuple(dynapcnn_output.shape)} counts={dynapcnn_counts}",
        flush=True,
    )

    print("[6/7] Running Specksim on the same genuine N-MNIST events", flush=True)
    specksim_input = normalize_events_for_specksim(raw_events)
    specksim = from_sequential(dynapcnn, input_shape=INPUT_SHAPE)
    specksim_output = specksim(specksim_input)
    specksim_counts = spike_counts_from_events(specksim_output)
    print(
        f"specksim_input_events={len(specksim_input)} "
        f"specksim_output_events={len(specksim_output)} counts={specksim_counts}",
        flush=True,
    )

    print("[7/7] Generating a local flash binary without writing flash", flush=True)
    flash_path = output_dir / "nmnist_speck2f_config.bin"
    flash_bytes = write_flash_binary(config, flash_path)
    print(f"flash_binary={flash_path} bytes={flash_bytes}", flush=True)

    results: dict[str, object] = {
        "seed": SEED,
        "versions": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "sinabs": sinabs.__version__,
            "samna": samna.__version__,
            "tonic": tonic.__version__,
        },
        "source": "Sinabs v3.1.3 N-MNIST quick-start BPTT architecture",
        "sample": {
            "split": "test",
            "index": 0,
            "label": label,
            "raw_event_count": len(raw_events),
            "raster_shape": list(raster.shape),
        },
        "sinabs_software": {
            "passed": True,
            "output_shape": list(sinabs_output.shape),
            "spike_counts": sinabs_counts,
        },
        "dynapcnn": {
            "construction_passed": True,
            "software_forward_passed": True,
            "compatible_with_speck2fdevkit": bool(compatible),
            "layer2core_map": {str(k): int(v) for k, v in dynapcnn.layer2core_map.items()},
            "output_shape": list(dynapcnn_output.shape),
            "spike_counts": dynapcnn_counts,
        },
        "samna_config": {
            "generated": True,
            "type": f"{type(config).__module__}.{type(config).__name__}",
            "valid": bool(config_valid),
            "validation_message": validation_message,
        },
        "specksim": {
            "passed": True,
            "input_event_count": len(specksim_input),
            "output_event_count": len(specksim_output),
            "spike_counts": specksim_counts,
        },
        "flash_binary": {
            "generated": True,
            "path": str(flash_path.relative_to(repository_root())),
            "bytes": flash_bytes,
            "flash_write_called": False,
        },
        "hardware": {
            "connected": False,
            "device_discovery_called": False,
            "device_open_called": False,
            "dynapcnn_to_hardware_called": False,
        },
    }
    results_path = output_dir / "results.json"
    results_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"results={results_path}", flush=True)
    return results


def parse_args() -> argparse.Namespace:
    root = repository_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=root / "data" / "official" / "NMNIST",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "artifacts",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        run(args.data_root, args.output_dir)
    except Exception:
        # Keep the full traceback in stdout/stderr so tee captures an audit log.
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
