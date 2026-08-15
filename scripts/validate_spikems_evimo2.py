#!/usr/bin/env python3
"""Validate one EVIMO2 adapter sample and optionally run zero-shot SpikeMS."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from reference.spikems.evimo2_adapter import (  # noqa: E402
    load_frame_aligned_sample,
    save_adapter_visualizations,
    save_prediction_visualization,
)
from reference.spikems.model_compat import (  # noqa: E402
    build_official_model,
    load_official_slayer_cuda,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sequence", type=Path)
    parser.add_argument("--frame-index", type=int, default=5)
    parser.add_argument("--physical-window-ms", type=float, default=20.0)
    parser.add_argument("--time-bins", type=int, default=100)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "spikems_evimo2_adapter",
    )
    parser.add_argument("--zero-shot", action="store_true")
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

    sample = load_frame_aligned_sample(
        args.sequence,
        frame_index=args.frame_index,
        physical_window_ms=args.physical_window_ms,
        num_time_bins=args.time_bins,
    )
    save_adapter_visualizations(sample, args.output_dir)
    result = {
        "alignment_gate": "PASS",
        "adapter_gate": "PASS",
        "sequence": str(args.sequence.resolve()),
        "camera": sample.camera_name,
        "polarity_channel_0": sample.polarity_channel_semantics[0],
        "polarity_channel_1": sample.polarity_channel_semantics[1],
        "frame_index": sample.frame_index,
        "mask_id": sample.mask_id,
        "mask_timestamp_s": sample.mask_timestamp_s,
        "start_time_s": sample.start_time_s,
        "end_time_s": sample.end_time_s,
        "physical_window_ms": sample.physical_window_ms,
        "num_time_bins": sample.num_time_bins,
        "dt_per_bin_ms": sample.dt_per_bin_ms,
        "spike_tensor_shape": list(sample.spike_tensor.shape),
        "masked_spike_tensor_shape": list(sample.masked_spike_tensor.shape),
        "raw_events": sample.raw_event_count,
        "polarity_0_events": sample.polarity_0_count,
        "polarity_1_events": sample.polarity_1_count,
        "foreground_events": sample.foreground_event_count,
        "background_events": sample.background_event_count,
        "input_voxels": sample.input_voxel_count,
        "foreground_voxels": sample.foreground_voxel_count,
        "event_to_voxel_collisions": sample.raw_event_count - sample.input_voxel_count,
        "foreground_ratio": sample.foreground_event_count / sample.raw_event_count,
        "visualizations": {
            "raw_events": str((args.output_dir / "raw_events.png").resolve()),
            "object_mask": str((args.output_dir / "object_mask.png").resolve()),
            "gt_foreground_events": str(
                (args.output_dir / "gt_foreground_events.png").resolve()
            ),
        },
        "zero_shot": "NOT_RUN",
    }

    if args.zero_shot:
        load_official_slayer_cuda(args.slayer_root, args.build_root)
        spikems = REPO_ROOT / "third_party" / "SpikeMS"
        checkpoint = (
            spikems
            / "pretrainedModels"
            / "EVIMO-pretrained"
            / "out"
            / "checkpoint.pth.tar"
        )
        device = torch.device("cuda:0")
        model, _, _ = build_official_model(spikems, checkpoint, device)
        model_input = sample.spike_tensor.unsqueeze(0).to(device)
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.no_grad():
            prediction = model(model_input)
        torch.cuda.synchronize()
        runtime_ms = (time.perf_counter() - started) * 1000.0
        prediction_cpu = prediction.detach().cpu()
        gt = sample.masked_spike_tensor[
            :, : prediction.shape[2], : prediction.shape[3], : prediction.shape[4]
        ]
        pred_binary = prediction_cpu[0] > 0
        gt_binary = gt > 0
        intersection = int(torch.logical_and(pred_binary, gt_binary).sum())
        union = int(torch.logical_or(pred_binary, gt_binary).sum())
        gt_count = int(gt_binary.sum())
        pred_count = int(pred_binary.sum())
        spatial_foreground = torch.from_numpy(sample.object_mask)[
            : prediction.shape[2], : prediction.shape[3]
        ]
        outside = ~spatial_foreground.unsqueeze(0).unsqueeze(-1)
        leaked = int(torch.logical_and(pred_binary, outside).sum())
        prediction_path = args.output_dir / "pred_foreground_events.png"
        save_prediction_visualization(prediction_cpu, prediction_path)
        result["zero_shot"] = "RUN"
        result["zero_shot_marker"] = "EVIMO_TO_EVIMO2_ZERO_SHOT"
        result["prediction_shape"] = list(prediction.shape)
        result["predicted_spikes"] = pred_count
        result["prediction_finite"] = bool(torch.isfinite(prediction_cpu).all())
        result["gt_foreground_voxels_on_output_support"] = gt_count
        result["background_leaked_spikes"] = leaked
        result["iou"] = intersection / union if union else math.nan
        result["foreground_recall"] = intersection / gt_count if gt_count else math.nan
        result["background_leakage"] = leaked / pred_count if pred_count else math.nan
        result["runtime_ms_diagnostic_only"] = runtime_ms
        result["visualizations"]["pred_foreground_events"] = str(
            prediction_path.resolve()
        )

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
