#!/usr/bin/env python3
"""Forward-only SpikeMS layer-delay and spikeTime window-end diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from reference.spikems.evimo2_adapter import load_frame_aligned_sample  # noqa: E402
from reference.spikems.model_compat import load_official_slayer_cuda  # noqa: E402
from reference.spikems.train_engineering import build_training_components  # noqa: E402
from scripts.diagnose_spikems_membrane_combined_loss import (  # noqa: E402
    FIXED_CROP,
    SEQUENCE,
    crop_sample,
    raw_crop_statistics,
)


LAYER_META_NAMES = [
    ("input", "input"),
    ("conv1", "conv1"),
    ("conv2", "conv2"),
    ("conv3", "conv3"),
    ("deconv4", "conv4"),
    ("deconv5", "conv5"),
    ("deconv6", "conv6"),
]


def temporal_counts(tensor: torch.Tensor) -> list[int]:
    return [int(value) for value in (tensor > 0).sum(dim=(0, 1, 2, 3)).tolist()]


def first_spike_timestep(counts: list[int]) -> int | None:
    return next((index for index, count in enumerate(counts) if count > 0), None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT
        / "outputs"
        / "spikems_training"
        / "temporal_window_diagnostic",
    )
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
    device = torch.device("cuda:0")
    full_sample = load_frame_aligned_sample(
        SEQUENCE, frame_index=57, physical_window_ms=10.0, num_time_bins=10
    )
    crop_spec = {**FIXED_CROP, "rule": "frozen_from_da68d4e"}
    raw_stats = raw_crop_statistics(SEQUENCE, full_sample, crop_spec)
    sample = crop_sample(full_sample, crop_spec, raw_stats)
    if (
        raw_stats["raw_events"],
        raw_stats["foreground_events"],
        raw_stats["background_events"],
    ) != (1057, 913, 144):
        raise RuntimeError(f"Frozen crop statistics changed: {raw_stats}")

    spikems_root = REPO_ROOT / "third_party" / "SpikeMS"
    checkpoint_path = (
        spikems_root
        / "pretrainedModels"
        / "EVIMO-pretrained"
        / "out"
        / "checkpoint.pth.tar"
    )
    model, criterion, _optimizer, _checkpoint, load_result = build_training_components(
        spikems_root, checkpoint_path, device
    )
    model.eval()
    model_input = sample.spike_tensor.unsqueeze(0).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        prediction = model(model_input)

    meta = model.getMetaTensorDict()
    missing = [meta_name for _, meta_name in LAYER_META_NAMES if meta_name not in meta]
    if missing:
        raise RuntimeError(f"Missing official model MetaTensors: {missing}")
    layer_temporal = {}
    for display_name, meta_name in LAYER_META_NAMES:
        tensor = meta[meta_name].getTensor()
        counts = temporal_counts(tensor)
        layer_temporal[display_name] = {
            "shape": list(tensor.shape),
            "spikes_by_timestep": counts,
            "total_spikes": sum(counts),
            "first_spike_timestep": first_spike_timestep(counts),
        }
    if not torch.equal(meta["conv6"].getTensor(), prediction.detach()):
        raise RuntimeError("Official conv6 MetaTensor does not equal model prediction")

    synthetic_shape = list(prediction.shape)
    gt = torch.zeros(synthetic_shape, device=device, dtype=torch.float32)
    synthetic_losses = []
    fixed_index = {"batch": 0, "polarity": 0, "y": 63, "x": 63}
    with torch.no_grad():
        for timestep in range(synthetic_shape[-1]):
            pred = torch.zeros_like(gt)
            pred[
                fixed_index["batch"],
                fixed_index["polarity"],
                fixed_index["y"],
                fixed_index["x"],
                timestep,
            ] = 1.0
            loss = criterion.spikeTime(pred, gt)
            synthetic_losses.append(
                {"timestep": timestep, "loss": float(loss.item())}
            )

    kernel = criterion.slayer.srmKernel.detach().cpu()
    loss_values = [item["loss"] for item in synthetic_losses]
    nonincreasing = all(
        later <= earlier for earlier, later in zip(loss_values, loss_values[1:])
    )
    end_bias = nonincreasing and loss_values[-1] < loss_values[0]

    first_times = [
        layer_temporal[display_name]["first_spike_timestep"]
        for display_name, _ in LAYER_META_NAMES
    ]
    finite_first_times = [value for value in first_times if value is not None]
    nondecreasing_first_times = all(
        later >= earlier
        for earlier, later in zip(finite_first_times, finite_first_times[1:])
    )
    has_strict_delay = any(
        later > earlier
        for earlier, later in zip(finite_first_times, finite_first_times[1:])
    )
    # An all-silent final layer is evidence of failure to respond in this short
    # window, but by itself is not assigned an artificial numeric first time.
    layer_delay = nondecreasing_first_times and has_strict_delay

    result = {
        "marker": "SPIKEMS_TEMPORAL_WINDOW_FORWARD_ONLY_DIAGNOSTIC",
        "sample": {
            "sequence": SEQUENCE.name,
            "frame_index": 57,
            "timestamp_s": full_sample.mask_timestamp_s,
            "window_start_s": full_sample.start_time_s,
            "window_end_s": full_sample.end_time_s,
            "crop": FIXED_CROP,
            **raw_stats,
            "input_shape": list(model_input.shape),
            "prediction_shape": list(prediction.shape),
        },
        "experiment_a": {
            "layers": layer_temporal,
            "first_spike_timesteps_in_order": first_times,
            "finite_first_times_nondecreasing": nondecreasing_first_times,
            "has_strict_interlayer_delay": has_strict_delay,
            "layer_temporal_delay": layer_delay,
        },
        "experiment_b": {
            "synthetic_shape": synthetic_shape,
            "gt": "all_zero",
            "fixed_spatial_polarity_index": fixed_index,
            "single_error_spike_losses": synthetic_losses,
            "loss_nonincreasing_toward_window_end": nonincreasing,
            "spiketime_end_bias": end_bias,
            "psp_kernel_length": int(kernel.numel()),
            "psp_kernel": [float(value) for value in kernel.tolist()],
            "simulation_Ts": criterion.simulation["Ts"],
            "simulation_tSample": criterion.simulation["tSample"],
        },
        "state_dict_missing_keys": list(load_result.missing_keys),
        "state_dict_unexpected_keys": list(load_result.unexpected_keys),
        "training_performed": False,
        "backward_performed": False,
        "optimizer_step_performed": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
