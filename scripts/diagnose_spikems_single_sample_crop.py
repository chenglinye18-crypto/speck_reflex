#!/usr/bin/env python3
"""Compare one-sample SpikeMS overfit on full-frame and a fixed GT crop."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageDraw
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from reference.spikems.evimo2_adapter import (  # noqa: E402
    EVIMO2SpikeMSSample,
    load_frame_aligned_sample,
    save_adapter_visualizations,
    save_prediction_visualization,
)
from reference.spikems.model_compat import load_official_slayer_cuda  # noqa: E402
from reference.spikems.train_engineering import (  # noqa: E402
    BASELINE_MARKER,
    build_training_components,
    forward_loss_metrics,
    gradient_statistics,
    named_gradient_norms,
)


DEFAULT_SEQUENCE = Path(
    "/home/speck/datasets/evimo2/motion_segmentation/right_camera/imo/train/"
    "scene14_dyn_test_02_000000"
)
SCHEDULE = {0, 10, 20, 50, 100, 200, 300, 400, 500}


def deterministic_peak_center_crop(sample: EVIMO2SpikeMSSample, size: int):
    """Center a fixed crop on the first maximum of the GT event projection."""

    projection = sample.masked_spike_tensor.sum(dim=(0, 3)).numpy()
    peak_y, peak_x = np.unravel_index(np.argmax(projection), projection.shape)
    height, width = projection.shape
    x0 = min(max(int(peak_x) - size // 2, 0), width - size)
    y0 = min(max(int(peak_y) - size // 2, 0), height - size)
    return {
        "x0": x0,
        "y0": y0,
        "width": size,
        "height": size,
        "peak_x": int(peak_x),
        "peak_y": int(peak_y),
        "peak_foreground_voxels": int(projection[peak_y, peak_x]),
        "rule": "row_major_first_max_gt_projection_centered_and_boundary_clamped",
    }


def raw_crop_statistics(
    sequence: Path, sample: EVIMO2SpikeMSSample, crop: dict[str, int | str]
):
    timestamps = np.load(sequence / "dataset_events_t.npy", mmap_mode="r")
    xy = np.load(sequence / "dataset_events_xy.npy", mmap_mode="r")
    polarity = np.load(sequence / "dataset_events_p.npy", mmap_mode="r")
    begin = int(np.searchsorted(timestamps, sample.start_time_s, side="left"))
    end = int(np.searchsorted(timestamps, sample.end_time_s, side="right"))
    points = np.asarray(xy[begin:end], dtype=np.int64)
    polarities = np.asarray(polarity[begin:end], dtype=np.int64)
    x0, y0 = int(crop["x0"]), int(crop["y0"])
    x1, y1 = x0 + int(crop["width"]), y0 + int(crop["height"])
    inside = (
        (points[:, 0] >= x0)
        & (points[:, 0] < x1)
        & (points[:, 1] >= y0)
        & (points[:, 1] < y1)
    )
    points = points[inside]
    polarities = polarities[inside]
    foreground = sample.object_mask[points[:, 1], points[:, 0]]
    return {
        "raw_events": int(points.shape[0]),
        "foreground_events": int(foreground.sum()),
        "background_events": int((~foreground).sum()),
        "off_events": int(np.count_nonzero(polarities == 0)),
        "on_events": int(np.count_nonzero(polarities == 1)),
    }


def crop_sample(
    sample: EVIMO2SpikeMSSample,
    crop: dict[str, int | str],
    raw_stats: dict[str, int],
) -> EVIMO2SpikeMSSample:
    x0, y0 = int(crop["x0"]), int(crop["y0"])
    x1, y1 = x0 + int(crop["width"]), y0 + int(crop["height"])
    spike = sample.spike_tensor[:, y0:y1, x0:x1, :].clone()
    foreground = sample.masked_spike_tensor[:, y0:y1, x0:x1, :].clone()
    if torch.any(foreground > spike):
        raise AssertionError("Cropped foreground GT is not a subset of cropped input")
    return replace(
        sample,
        spike_tensor=spike,
        masked_spike_tensor=foreground,
        object_mask=sample.object_mask[y0:y1, x0:x1].copy(),
        raw_event_count=raw_stats["raw_events"],
        polarity_0_count=raw_stats["off_events"],
        polarity_1_count=raw_stats["on_events"],
        foreground_event_count=raw_stats["foreground_events"],
        background_event_count=raw_stats["background_events"],
        input_voxel_count=int(spike.sum().item()),
        foreground_voxel_count=int(foreground.sum().item()),
    )


def evaluate(model, criterion, sample, device):
    model.eval()
    with torch.no_grad():
        prediction, _, _, metrics = forward_loss_metrics(
            model, criterion, sample, device
        )
    model.train()
    return prediction.detach().cpu(), metrics


def run_experiment(
    label: str,
    sample: EVIMO2SpikeMSSample,
    output_dir: Path,
    spikems_root: Path,
    checkpoint_path: Path,
    device: torch.device,
    seed: int,
):
    torch.manual_seed(seed)
    model, criterion, optimizer, _, load_result = build_training_components(
        spikems_root, checkpoint_path, device
    )
    torch.cuda.reset_peak_memory_stats(device)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_adapter_visualizations(sample, output_dir)
    before_prediction, initial = evaluate(model, criterion, sample, device)
    save_prediction_visualization(before_prediction, output_dir / "pred_before.png")

    # Measure the checkpoint's initial learning signal without changing weights.
    optimizer.zero_grad(set_to_none=True)
    _, _, initial_loss, _ = forward_loss_metrics(model, criterion, sample, device)
    initial_loss.backward()
    initial_gradient = gradient_statistics(model.parameters())
    initial_layer_gradients = named_gradient_norms(model)

    history = [
        {
            "step": 0,
            **initial.__dict__,
            "global_gradient_norm": initial_gradient["global_gradient_norm"],
        }
    ]
    final_gradient = initial_gradient
    final_layer_gradients = initial_layer_gradients
    numerical_issue = None
    started = time.perf_counter()
    for step in range(1, 501):
        optimizer.zero_grad(set_to_none=True)
        _, _, loss, training_metrics = forward_loss_metrics(
            model, criterion, sample, device
        )
        if not math.isfinite(training_metrics.loss_spike):
            numerical_issue = f"non-finite loss at step {step}"
            break
        loss.backward()
        step_gradient = gradient_statistics(model.parameters())
        if not step_gradient["gradients_finite"]:
            numerical_issue = f"non-finite gradient at step {step}"
            break
        step_layer_gradients = named_gradient_norms(model)
        optimizer.step()

        if step in SCHEDULE:
            _, post_update_metrics = evaluate(model, criterion, sample, device)
            history.append(
                {
                    "step": step,
                    **post_update_metrics.__dict__,
                    "global_gradient_norm": step_gradient["global_gradient_norm"],
                }
            )
        final_gradient = step_gradient
        final_layer_gradients = step_layer_gradients

    after_prediction, final = evaluate(model, criterion, sample, device)
    save_prediction_visualization(after_prediction, output_dir / "pred_after.png")
    runtime_s = time.perf_counter() - started
    total_voxels = int(np.prod(sample.spike_tensor.shape))
    return {
        "label": label,
        "shape": [1, *sample.spike_tensor.shape],
        "prediction_shape": list(after_prediction.shape),
        "input_spikes": sample.input_voxel_count,
        "foreground_spikes": sample.foreground_voxel_count,
        "background_spikes": sample.input_voxel_count - sample.foreground_voxel_count,
        "foreground_density": sample.foreground_voxel_count / total_voxels,
        "foreground_density_denominator": total_voxels,
        "initial": initial.__dict__,
        "final": final.__dict__,
        "loss_ratio": final.loss_spike / initial.loss_spike,
        "history": history,
        "initial_gradient": initial_gradient,
        "final_gradient": final_gradient,
        "initial_layer_gradient_norms": initial_layer_gradients,
        "final_layer_gradient_norms": final_layer_gradients,
        "runtime_s": runtime_s,
        "peak_vram_mb": torch.cuda.max_memory_allocated(device) / (1024**2),
        "numerical_issue": numerical_issue,
        "state_dict_missing_keys": list(load_result.missing_keys),
        "state_dict_unexpected_keys": list(load_result.unexpected_keys),
    }


def save_crop_location(full_output: Path, crop: dict[str, int | str], destination: Path):
    image = Image.open(full_output / "raw_events.png").convert("RGB")
    draw = ImageDraw.Draw(image)
    x0, y0 = int(crop["x0"]), int(crop["y0"])
    x1 = x0 + int(crop["width"]) - 1
    y1 = y0 + int(crop["height"]) - 1
    draw.rectangle((x0, y0, x1, y1), outline=(0, 255, 0), width=3)
    image.save(destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=Path, default=DEFAULT_SEQUENCE)
    parser.add_argument("--frame-index", type=int, default=57)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument(
        "--full-output",
        type=Path,
        default=REPO_ROOT / "outputs" / "spikems_training" / "single_sample_fullframe",
    )
    parser.add_argument(
        "--crop-output",
        type=Path,
        default=REPO_ROOT / "outputs" / "spikems_training" / "single_sample_crop128",
    )
    parser.add_argument(
        "--result-json",
        type=Path,
        default=REPO_ROOT
        / "outputs"
        / "spikems_training"
        / "single_sample_crop_ab_result.json",
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
    if args.crop_size != 128:
        raise ValueError("This diagnostic is frozen to a 128x128 crop")

    load_official_slayer_cuda(args.slayer_root, args.build_root)
    device = torch.device("cuda:0")
    full_sample = load_frame_aligned_sample(
        args.sequence,
        frame_index=args.frame_index,
        physical_window_ms=10.0,
        num_time_bins=10,
    )
    crop = deterministic_peak_center_crop(full_sample, args.crop_size)
    crop_raw = raw_crop_statistics(args.sequence, full_sample, crop)
    cropped_sample = crop_sample(full_sample, crop, crop_raw)
    if cropped_sample.foreground_event_count <= 0 or cropped_sample.background_event_count <= 0:
        raise ValueError("Diagnostic crop must contain both foreground and background events")

    spikems_root = REPO_ROOT / "third_party" / "SpikeMS"
    checkpoint_path = (
        spikems_root
        / "pretrainedModels"
        / "EVIMO-pretrained"
        / "out"
        / "checkpoint.pth.tar"
    )
    full_result = run_experiment(
        "full_frame",
        full_sample,
        args.full_output,
        spikems_root,
        checkpoint_path,
        device,
        args.seed,
    )
    crop_result = run_experiment(
        "crop_128",
        cropped_sample,
        args.crop_output,
        spikems_root,
        checkpoint_path,
        device,
        args.seed,
    )
    save_crop_location(
        args.full_output, crop, args.crop_output / "crop_location_on_full_frame.png"
    )

    result = {
        "baseline_marker": BASELINE_MARKER,
        "diagnostic_marker": "DIAGNOSTIC_ONLY_GT_ASSISTED_CROP",
        "same_seed": args.seed,
        "sample": {
            "sequence": args.sequence.name,
            "sequence_path": str(args.sequence.resolve()),
            "frame_index": full_sample.frame_index,
            "timestamp_s": full_sample.mask_timestamp_s,
            "window_start_s": full_sample.start_time_s,
            "window_end_s": full_sample.end_time_s,
            "raw_events": full_sample.raw_event_count,
            "foreground_events": full_sample.foreground_event_count,
            "background_events": full_sample.background_event_count,
            "off_events": full_sample.polarity_0_count,
            "on_events": full_sample.polarity_1_count,
            "object_mask_pixels": int(full_sample.object_mask.sum()),
            "object_mask_coverage": float(full_sample.object_mask.mean()),
        },
        "crop": {
            **crop,
            **crop_raw,
            "foreground_ratio": cropped_sample.foreground_event_count
            / cropped_sample.raw_event_count,
            "object_mask_coverage": float(cropped_sample.object_mask.mean()),
        },
        "experiment_a": full_result,
        "experiment_b": crop_result,
    }
    args.result_json.parent.mkdir(parents=True, exist_ok=True)
    args.result_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if full_result["numerical_issue"] or crop_result["numerical_issue"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
