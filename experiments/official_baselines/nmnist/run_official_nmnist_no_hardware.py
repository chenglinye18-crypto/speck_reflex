#!/usr/bin/env python3
"""N-MNIST deployment pipeline smoke test (random weights; no accuracy claim)."""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

import samna  # noqa: E402
import torch  # noqa: E402

from speck_reflex.environment import collect_environment  # noqa: E402
from speck_reflex.official_baseline.data import load_sample  # noqa: E402
from speck_reflex.official_baseline.dynapcnn import (  # noqa: E402
    build_dynapcnn,
    generate_flash_binary,
    run_specksim,
)
from speck_reflex.official_baseline.inference import event_counts, tensor_counts  # noqa: E402
from speck_reflex.official_baseline.model import (  # noqa: E402
    SEED,
    build_random_quick_start_model,
    reset_states,
    set_seed,
)
from speck_reflex.official_baseline.results import write_results  # noqa: E402

DEVICE_TYPE = "speck2fdevkit"


def run(data_root: Path, output_dir: Path, *, generate_flash: bool = False, io_sel: int | None = None) -> dict[str, object]:
    if io_sel is not None and not generate_flash:
        raise ValueError("--io-sel is only valid with --generate-flash-binary")
    set_seed()
    raw, frames, label = load_sample(data_root, 0, n_time_bins=100)
    raster = torch.as_tensor(frames, dtype=torch.float32)
    model = build_random_quick_start_model().eval()
    reset_states(model)
    with torch.no_grad():
        software = model(raster)
    dynapcnn, config, valid, validation_message = build_dynapcnn(model, device_type=DEVICE_TYPE)
    reset_states(dynapcnn)
    with torch.no_grad():
        quantized = dynapcnn(raster)
    simulated = run_specksim(dynapcnn, raw)

    flash_generated = False
    flash_bytes = 0
    if generate_flash:
        flash_bytes = generate_flash_binary(config, output_dir / "nmnist_speck2f_config.bin", io_sel)
        flash_generated = True

    result = {
        "mode": "pipeline_smoke",
        "weights": "deterministic_random_xavier",
        "trained": False,
        "functional_accuracy_claimed": False,
        "seed": SEED,
        "versions": collect_environment(ROOT),
        "source": "Sinabs v3.1.3 N-MNIST quick-start topology",
        "input_mode": "host_injected_events",
        "sample": {"split": "test", "index": 0, "label": label, "raw_event_count": len(raw), "raster_shape": list(raster.shape)},
        "sinabs_software": {"passed": True, "spike_counts": tensor_counts(software)},
        "dynapcnn": {"construction_passed": True, "software_forward_passed": True, "layer2core_map": {str(k): int(v) for k, v in dynapcnn.layer2core_map.items()}, "spike_counts": tensor_counts(quantized)},
        "samna_config": {"generated": True, "valid": valid, "validation_message": validation_message},
        "specksim": {"passed": True, "input_event_count": len(raw), "output_event_count": len(simulated), "spike_counts": event_counts(simulated)},
        "flash_binary_requested": generate_flash,
        "flash_binary_generated": flash_generated,
        "flash_binary_bytes": flash_bytes,
        "flash_write_called": False,
        "hardware": {"connected": False, "device_discovery_called": False, "device_open_called": False, "dynapcnn_to_hardware_called": False},
    }
    destination = output_dir / "results.json"
    write_results(destination, result)
    print(f"sample=0 label={label} raw_events={len(raw)}")
    print(f"sinabs_counts={result['sinabs_software']['spike_counts']}")
    print(f"dynapcnn_counts={result['dynapcnn']['spike_counts']} map={result['dynapcnn']['layer2core_map']}")
    print(f"specksim_counts={result['specksim']['spike_counts']} config_valid={valid}")
    print(f"flash_binary_requested={generate_flash} flash_binary_generated={flash_generated}")
    print(f"results={destination}")
    print("PIPELINE_SMOKE_ONLY")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/official/NMNIST")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "artifacts")
    parser.add_argument("--generate-flash-binary", action="store_true", help="generate a local binary only; never writes Flash")
    parser.add_argument("--io-sel", type=int, help="explicit board-specific factory io_sel required for binary generation")
    return parser.parse_args()


def main() -> int:
    try:
        args = parse_args()
        run(args.data_root, args.output_dir, generate_flash=args.generate_flash_binary, io_sel=args.io_sel)
        return 0
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
