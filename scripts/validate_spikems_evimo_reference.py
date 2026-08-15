#!/usr/bin/env python3
"""Compare SpikeMS 10 ms and official 20 ms windows on EVIMO eval_wall seq_00."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from PIL import Image
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from reference.spikems.evimo2_adapter import save_prediction_visualization  # noqa: E402
from reference.spikems.evimo_reference import (  # noqa: E402
    load_evimo1_npz,
    make_reference_sample,
    official_valid_frame_indices,
)
from reference.spikems.model_compat import (  # noqa: E402
    build_official_model,
    load_official_slayer_cuda,
)
from reference.spikems.train_engineering import align_gt_to_prediction  # noqa: E402


DEFAULT_DATASET = Path(
    "/home/speck/datasets/evimo1/reference/eval/wall/npz/seq_00.npz"
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


def first_spike(counts: list[int]) -> int | None:
    return next((index for index, count in enumerate(counts) if count > 0), None)


def layer_counts(model) -> dict:
    meta = model.getMetaTensorDict()
    result = {}
    for display_name, meta_name in LAYER_META_NAMES:
        tensor = meta[meta_name].getTensor() > 0
        counts = [int(value) for value in tensor.sum(dim=(0, 1, 2, 3)).tolist()]
        result[display_name] = {
            "shape": list(tensor.shape),
            "spikes_by_timestep": counts,
            "total_spikes": sum(counts),
            "first_spike_timestep": first_spike(counts),
        }
    return result


def metrics(prediction: torch.Tensor, gt: torch.Tensor) -> dict:
    pred = prediction > 0
    truth = gt > 0
    intersection = int(torch.logical_and(pred, truth).sum().item())
    union = int(torch.logical_or(pred, truth).sum().item())
    pred_count = int(pred.sum().item())
    gt_count = int(truth.sum().item())
    pred_spatial = torch.any(pred, dim=(1, 4))
    gt_spatial = torch.any(truth, dim=(1, 4))
    spatial_intersection = int(torch.logical_and(pred_spatial, gt_spatial).sum().item())
    spatial_union = int(torch.logical_or(pred_spatial, gt_spatial).sum().item())
    pred_pixels = int(pred_spatial.sum().item())
    gt_pixels = int(gt_spatial.sum().item())
    precision = spatial_intersection / pred_pixels if pred_pixels else 0.0
    recall = spatial_intersection / gt_pixels if gt_pixels else 0.0
    return {
        "gt_foreground_spikes": gt_count,
        "prediction_spikes": pred_count,
        "event_iou": intersection / union if union else 0.0,
        "event_recall": intersection / gt_count if gt_count else 0.0,
        "spatial_iou": spatial_intersection / spatial_union if spatial_union else 0.0,
        "spatial_precision": precision,
        "spatial_recall": recall,
        "spatial_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "gt_active_pixels": gt_pixels,
        "predicted_active_pixels": pred_pixels,
        "gt_spikes_by_timestep": [
            int(value) for value in truth.sum(dim=(0, 1, 2, 3)).tolist()
        ],
        "prediction_spikes_by_timestep": [
            int(value) for value in pred.sum(dim=(0, 1, 2, 3)).tolist()
        ],
    }


def save_visualizations(sample, prediction, gt, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    save_prediction_visualization(sample.spike_tensor, output_dir / "input_events.png")
    save_prediction_visualization(gt, output_dir / "gt_foreground_events.png")
    save_prediction_visualization(prediction, output_dir / "prediction.png")
    Image.fromarray(sample.dilated_object_mask.astype(np.uint8) * 255).save(
        output_dir / "dilated_object_mask.png"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs/spikems_training/evimo_reference_inference",
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
    source = load_evimo1_npz(args.dataset)
    chosen = official_valid_frame_indices(source, args.num_samples)
    device = torch.device("cuda:0")
    spikems_root = REPO_ROOT / "third_party/SpikeMS"
    checkpoint_path = spikems_root / "pretrainedModels/EVIMO-pretrained/out/checkpoint.pth.tar"
    model, _, load_result = build_official_model(spikems_root, checkpoint_path, device)
    model.eval()

    experiments = []
    for frame_array_index, official_crop in chosen:
        frame_result = {
            "frame_array_index": frame_array_index,
            "frame_id": int(source.frames[frame_array_index]["id"]),
            "mask_timestamp_s": float(source.frames[frame_array_index]["cam"]["ts"]),
            "crop_from_official_20ms_gt": official_crop,
            "windows": {},
        }
        for label, window_ms, bins in (
            ("10ms_T10", 10.0, 10),
            ("official_20ms_T20", 20.0, 20),
        ):
            sample = make_reference_sample(
                source,
                frame_array_index,
                window_ms=window_ms,
                num_time_bins=bins,
                crop=official_crop,
            )
            model_input = sample.spike_tensor.unsqueeze(0).to(device)
            gt_full = sample.gt_foreground_events.unsqueeze(0).to(device)
            with torch.no_grad():
                prediction = model(model_input)
            gt = align_gt_to_prediction(gt_full, prediction)
            window_result = {
                "window_ms": window_ms,
                "num_time_bins": bins,
                "input_shape": list(model_input.shape),
                "prediction_shape": list(prediction.shape),
                "raw_event_rows_full_frame": sample.raw_event_rows,
                "input_spike_voxels_full_frame": sample.input_spike_voxels_full,
                "foreground_spike_voxels_full_frame": sample.foreground_spike_voxels_full,
                "background_spike_voxels_full_frame": sample.background_spike_voxels_full,
                "background_foreground_ratio_full_frame": sample.background_foreground_ratio,
                "metrics": metrics(prediction, gt),
                "layers": layer_counts(model),
            }
            frame_result["windows"][label] = window_result
            save_visualizations(
                sample,
                prediction.detach().cpu(),
                gt.detach().cpu(),
                args.output_dir / f"frame_{frame_result['frame_id']}" / label,
            )
        experiments.append(frame_result)

    result = {
        "marker": "EVIMO1_SPIKEMS_REFERENCE_FORWARD_ONLY",
        "dataset": str(args.dataset.resolve()),
        "dataset_source": "https://obj.umiacs.umd.edu/evimo1npz/eval_wall_npz.tar.gz",
        "sequence": "eval_wall/seq_00",
        "preprocessing": {
            "sensor_resolution": [260, 346],
            "polarity_channels": "released EVIMO 0/1 used directly",
            "official_valid_time_s": 0.01,
            "official_physical_window_ms": 20.0,
            "mask_time": "frames[i]['cam']['ts']",
            "mask": "object_id > 0, then 5x5 dilation",
            "gt": "input spike tensor intersect dilated object mask",
            "preprocessing_min_mask_sum": 1000,
            "dataloader_min_boolean_mask_pixels": 30,
            "official_event_slice_off_by_one": "final stored event omitted",
            "max_background_foreground_ratio": 1.5,
            "crop": "128x128 centered on densest GT foreground-event pixel",
            "temporal_bin_mapping": "trunc((T-1)*(event_t-start)/(stop-start))",
            "comparison_crop_policy": "official 20ms GT crop frozen for both windows",
        },
        "samples": experiments,
        "state_dict_missing_keys": list(load_result.missing_keys),
        "state_dict_unexpected_keys": list(load_result.unexpected_keys),
        "training_performed": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
