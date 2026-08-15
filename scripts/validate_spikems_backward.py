#!/usr/bin/env python3
"""Run one full SpikeMS L_spike forward/backward/optimizer engineering gate."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from reference.spikems.evimo2_adapter import load_frame_aligned_sample  # noqa: E402
from reference.spikems.model_compat import load_official_slayer_cuda  # noqa: E402
from reference.spikems.train_engineering import (  # noqa: E402
    BASELINE_MARKER,
    build_training_components,
    clone_trainable_parameters,
    forward_loss_metrics,
    gradient_statistics,
    parameter_change_statistics,
)


DEFAULT_SEQUENCE = Path(
    "/home/speck/datasets/evimo2/motion_segmentation/right_camera/imo/eval/"
    "scene15_dyn_test_05_000000"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=Path, default=DEFAULT_SEQUENCE)
    parser.add_argument("--frame-index", type=int, default=272)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=REPO_ROOT / "outputs" / "spikems_training" / "backward_gate.json",
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
    torch.cuda.reset_peak_memory_stats(device)
    sample = load_frame_aligned_sample(
        args.sequence,
        frame_index=args.frame_index,
        physical_window_ms=10.0,
        num_time_bins=10,
    )
    spikems_root = REPO_ROOT / "third_party" / "SpikeMS"
    checkpoint_path = (
        spikems_root
        / "pretrainedModels"
        / "EVIMO-pretrained"
        / "out"
        / "checkpoint.pth.tar"
    )
    model, criterion, optimizer, _, load_result = build_training_components(
        spikems_root, checkpoint_path, device
    )

    optimizer.zero_grad(set_to_none=True)
    prediction, _, loss, metrics = forward_loss_metrics(model, criterion, sample, device)
    loss.backward()
    gradients = gradient_statistics(model.parameters())
    before = clone_trainable_parameters(model)
    gradients_valid = (
        gradients["parameter_tensors_with_nonzero_grad"] > 0
        and gradients["gradients_finite"]
    )
    if gradients_valid:
        optimizer.step()
    parameter_change = parameter_change_statistics(model, before)
    peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024**2)

    gate_pass = bool(
        metrics.prediction_finite
        and math.isfinite(metrics.loss_spike)
        and gradients_valid
        and parameter_change["parameter_change_finite"]
        and parameter_change["parameter_change_norm"] > 0
    )
    result = {
        "baseline_marker": BASELINE_MARKER,
        "selection_marker": "DEBUG_SAMPLE_SELECTION_USING_GT_STATS",
        "spikems_backward_gate": "PASS" if gate_pass else "FAIL",
        "sequence": str(args.sequence.resolve()),
        "frame_index": sample.frame_index,
        "timestamp_s": sample.mask_timestamp_s,
        "window_start_s": sample.start_time_s,
        "window_end_s": sample.end_time_s,
        "raw_events": sample.raw_event_count,
        "foreground_events": sample.foreground_event_count,
        "background_events": sample.background_event_count,
        "off_events": sample.polarity_0_count,
        "on_events": sample.polarity_1_count,
        "input_shape": [1, *sample.spike_tensor.shape],
        "gt_shape": [1, *sample.masked_spike_tensor.shape],
        "prediction_shape": list(prediction.shape),
        "prediction_spikes": metrics.prediction_spikes,
        "prediction_finite": metrics.prediction_finite,
        "loss_spike": metrics.loss_spike,
        "loss_finite": math.isfinite(metrics.loss_spike),
        "gt_alignment": "top_left_crop_to_prediction_support_no_resize",
        "state_dict_missing_keys": list(load_result.missing_keys),
        "state_dict_unexpected_keys": list(load_result.unexpected_keys),
        **gradients,
        **parameter_change,
        "peak_vram_mb": peak_vram_mb,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not gate_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
