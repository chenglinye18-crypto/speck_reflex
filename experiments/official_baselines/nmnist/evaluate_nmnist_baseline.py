#!/usr/bin/env python3
"""Evaluate the official Sinabs v3.1.3 N-MNIST NIR checkpoint without hardware."""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from speck_reflex.environment import collect_environment  # noqa: E402
from speck_reflex.official_baseline.data import load_sample  # noqa: E402
from speck_reflex.official_baseline.dynapcnn import build_dynapcnn, run_specksim  # noqa: E402
from speck_reflex.official_baseline.inference import event_counts, prediction, tensor_counts  # noqa: E402
from speck_reflex.official_baseline.model import (  # noqa: E402
    SEED,
    load_official_nir_model,
    nir_output_tensor,
    reset_states,
    sequentialize_nir_model,
    set_seed,
)
from speck_reflex.official_baseline.results import sha256_file, write_results  # noqa: E402

CHECKPOINT = ROOT / "third_party/synsense/sinabs/docs/tutorials/scnn_mnist.nir"
DEVICE_TYPE = "speck2fmodule"


def evaluate_accuracy(graph, data_root: Path, count: int) -> tuple[int, float]:
    correct = 0
    for ordinal in range(count):
        index = ordinal * 200  # Matches the fixed sparse subset in the official notebook.
        _, frames, label = load_sample(data_root, index, n_time_bins=None, time_window=1000)
        reset_states(graph)
        with torch.no_grad():
            output = nir_output_tensor(graph(torch.as_tensor(frames, dtype=torch.float32)))
        correct += prediction(tensor_counts(output)) == label
    return correct, correct / count


def run(data_root: Path, output_dir: Path, sample_index: int = 0, accuracy_samples: int = 50) -> dict[str, object]:
    set_seed()
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"pinned official checkpoint missing: {CHECKPOINT}")
    checkpoint_hash = sha256_file(CHECKPOINT)
    raw, frames, label = load_sample(data_root, sample_index, n_time_bins=None, time_window=1000)
    raster = torch.as_tensor(frames, dtype=torch.float32)

    graph = load_official_nir_model(CHECKPOINT).eval()
    reset_states(graph)
    with torch.no_grad():
        software = nir_output_tensor(graph(raster))
    software_counts = tensor_counts(software)

    sequential = sequentialize_nir_model(graph).eval()
    dynapcnn, config, valid, validation_message = build_dynapcnn(sequential, device_type=DEVICE_TYPE)
    reset_states(dynapcnn)
    with torch.no_grad():
        quantized = dynapcnn(raster)
    quantized_counts = tensor_counts(quantized)

    # COMPATIBILITY_PATCH: Specksim 3.1.3 cannot consume the graph executor and
    # traversing DynapcnnNetwork duplicates internal graph nodes. Convert the same
    # official bias-free sequential network; report any disagreement explicitly.
    simulated = run_specksim(sequential, raw)
    simulated_counts = event_counts(simulated)

    started = time.monotonic()
    accuracy_graph = load_official_nir_model(CHECKPOINT).eval()
    correct, accuracy = evaluate_accuracy(accuracy_graph, data_root, accuracy_samples)
    elapsed = time.monotonic() - started
    predictions = {
        "sinabs": prediction(software_counts),
        "dynapcnn_quantized": prediction(quantized_counts),
        "specksim": prediction(simulated_counts),
    }
    if any(sum(counts) == 0 for counts in (software_counts, quantized_counts, simulated_counts)):
        raise RuntimeError("functional demo requires non-zero output from all three paths")

    result = {
        "mode": "functional_nmnist",
        "weights": "official_sinabs_v3.1.3_scNN_NIR",
        "trained": True,
        "functional_accuracy_claimed": True,
        "seed": SEED,
        "versions": collect_environment(ROOT),
        "input_mode": "host_injected_events",
        "checkpoint": {"path": str(CHECKPOINT.relative_to(ROOT)), "sha256": checkpoint_hash, "source": "Sinabs v3.1.3 pinned submodule", "license": "Apache-2.0"},
        "sample": {"split": "test", "index": sample_index, "label": label, "raw_event_count": len(raw), "raster_shape": list(raster.shape)},
        "sinabs_software": {"passed": True, "prediction": predictions["sinabs"], "spike_counts": software_counts},
        "dynapcnn": {"construction_passed": True, "software_forward_passed": True, "prediction": predictions["dynapcnn_quantized"], "spike_counts": quantized_counts, "layer2core_map": {str(k): int(v) for k, v in dynapcnn.layer2core_map.items()}},
        "samna_config": {"generated": True, "valid": valid, "validation_message": validation_message},
        "specksim": {"passed": True, "prediction": predictions["specksim"], "input_event_count": len(raw), "output_event_count": len(simulated), "spike_counts": simulated_counts, "conversion_source": "official_pre_quantization_sequential"},
        "agreement": len(set(predictions.values())) == 1,
        "predictions": predictions,
        "accuracy": {"subset_scheme": "official_notebook_indices_i_times_200", "samples": accuracy_samples, "correct": correct, "value": accuracy, "elapsed_seconds": elapsed},
        "flash_binary_requested": False,
        "flash_binary_generated": False,
        "flash_write_called": False,
        "hardware": {"connected": False, "device_discovery_called": False, "device_open_called": False, "dynapcnn_to_hardware_called": False},
    }
    destination = output_dir / "functional_results.json"
    write_results(destination, result)
    print(f"sample={sample_index} label={label} checkpoint_sha256={checkpoint_hash}")
    for name, counts in (("sinabs", software_counts), ("dynapcnn_quantized", quantized_counts), ("specksim", simulated_counts)):
        print(f"{name}: prediction={predictions[name]} total_spikes={sum(counts)} counts={counts}")
    print(f"agreement={result['agreement']} config_valid={valid} layer2core_map={result['dynapcnn']['layer2core_map']}")
    print(f"test_subset_accuracy={correct}/{accuracy_samples}={accuracy:.4f} elapsed_seconds={elapsed:.2f}")
    print(f"results={destination}")
    print("FUNCTIONAL_NMNIST_DEMO_PASSED")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/official/NMNIST")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "artifacts")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--accuracy-samples", type=int, default=50)
    return parser.parse_args()


def main() -> int:
    try:
        args = parse_args()
        if args.accuracy_samples <= 0:
            raise ValueError("--accuracy-samples must be positive")
        run(args.data_root, args.output_dir, args.sample_index, args.accuracy_samples)
        return 0
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
