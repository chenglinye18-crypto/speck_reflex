#!/usr/bin/env python3
"""Run the frozen RECONSTRUCTED_SPATIAL_BCE_V1 single-sample diagnostic."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from reference.spikems.evimo2_adapter import (  # noqa: E402
    load_frame_aligned_sample,
    save_adapter_visualizations,
    save_prediction_visualization,
)
from reference.spikems.losses import (  # noqa: E402
    RECONSTRUCTED_SPATIAL_BCE_V1,
    reconstructed_spatial_bce_v1,
)
from reference.spikems.model_compat import load_official_slayer_cuda  # noqa: E402
from reference.spikems.train_engineering import (  # noqa: E402
    align_gt_to_prediction,
    build_model_and_optimizer,
    gradient_statistics,
    named_gradient_norms,
    spatial_mask_on_prediction_support,
)
from scripts.diagnose_spikems_single_sample_crop import (  # noqa: E402
    crop_sample,
    raw_crop_statistics,
)


SEQUENCE = Path(
    "/home/speck/datasets/evimo2/motion_segmentation/right_camera/imo/train/"
    "scene14_dyn_test_02_000000"
)
FIXED_CROP = {"x0": 248, "y0": 0, "width": 128, "height": 128}
SCHEDULE = {0, 10, 20, 50, 100, 200, 300, 400, 500}
EPS = 1e-6
SAVED_LSPIKE_CROP_RESULT = {
    "initial": {
        "background_leakage": 0.0,
        "foreground_recall": 0.0,
        "iou": 0.0,
        "loss_spike": 1368.896728515625,
        "prediction_spikes": 0,
    },
    "final": {
        "background_leakage": 0.026905829596412557,
        "foreground_recall": 0.03699551569506727,
        "iou": 0.030499075785582256,
        "loss_spike": 1293.0499267578125,
        "prediction_spikes": 223,
    },
    "loss_ratio": 0.944592751098136,
    "provenance": "SINGLE_SAMPLE_CROP_DIAGNOSTIC_20260815.md",
}


def spatial_metrics(probability: torch.Tensor, target: torch.Tensor) -> dict[str, float | int]:
    prediction = probability >= 0.5
    truth = target > 0
    intersection = int(torch.logical_and(prediction, truth).sum().item())
    union = int(torch.logical_or(prediction, truth).sum().item())
    predicted = int(prediction.sum().item())
    positive = int(truth.sum().item())
    precision = intersection / predicted if predicted else 0.0
    recall = intersection / positive if positive else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "spatial_iou": intersection / union if union else 0.0,
        "spatial_precision": precision,
        "spatial_recall": recall,
        "spatial_f1": f1,
        "gt_active_pixels": positive,
        "predicted_active_pixels": predicted,
    }


def event_metrics(
    prediction: torch.Tensor,
    gt: torch.Tensor,
    spatial_foreground_mask: torch.Tensor,
) -> dict[str, float | int]:
    pred_binary = prediction > 0
    gt_binary = gt > 0
    intersection = int(torch.logical_and(pred_binary, gt_binary).sum().item())
    union = int(torch.logical_or(pred_binary, gt_binary).sum().item())
    gt_count = int(gt_binary.sum().item())
    pred_count = int(pred_binary.sum().item())
    outside = ~spatial_foreground_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
    leaked = int(torch.logical_and(pred_binary, outside).sum().item())
    return {
        "event_iou": intersection / union if union else 0.0,
        "event_recall": intersection / gt_count if gt_count else 0.0,
        "background_leakage": leaked / pred_count if pred_count else 0.0,
        "prediction_spikes": pred_count,
    }


def evaluate(model, sample, device):
    model.eval()
    with torch.no_grad():
        model_input = sample.spike_tensor.unsqueeze(0).to(device)
        gt_full = sample.masked_spike_tensor.unsqueeze(0).to(device)
        prediction = model(model_input)
        gt = align_gt_to_prediction(gt_full, prediction)
        bce = reconstructed_spatial_bce_v1(prediction, gt, eps=EPS)
        metrics = {
            "loss_bce": float(bce.loss.item()),
            **spatial_metrics(bce.probability, bce.gt_spatial),
            **event_metrics(
                prediction,
                gt,
                spatial_mask_on_prediction_support(sample, prediction, device),
            ),
            "prediction_finite": bool(torch.isfinite(prediction).all().item()),
            "probability_finite": bool(torch.isfinite(bce.probability).all().item()),
        }
    model.train()
    return prediction.detach().cpu(), bce.probability.detach().cpu(), bce.gt_spatial.cpu(), metrics


def save_spatial_map(tensor: torch.Tensor, path: Path) -> None:
    if tensor.ndim == 3:
        if tensor.shape[0] != 1:
            raise ValueError("Visualization supports one sample")
        tensor = tensor[0]
    image = np.asarray(torch.clamp(tensor, 0, 1).numpy() * 255.0, dtype=np.uint8)
    Image.fromarray(image).save(path)


def forward_bce(model, sample, device):
    model_input = sample.spike_tensor.unsqueeze(0).to(device)
    gt_full = sample.masked_spike_tensor.unsqueeze(0).to(device)
    prediction = model(model_input)
    gt = align_gt_to_prediction(gt_full, prediction)
    bce = reconstructed_spatial_bce_v1(prediction, gt, eps=EPS)
    return prediction, gt, bce


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "spikems_training" / "bce_single_sample",
    )
    parser.add_argument(
        "--previous-lspike-result",
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

    load_official_slayer_cuda(args.slayer_root, args.build_root)
    device = torch.device("cuda:0")
    full_sample = load_frame_aligned_sample(
        SEQUENCE, frame_index=57, physical_window_ms=10.0, num_time_bins=10
    )
    crop_spec = {
        **FIXED_CROP,
        "rule": "frozen_from_single_sample_crop_diagnostic_da68d4e",
    }
    raw_stats = raw_crop_statistics(SEQUENCE, full_sample, crop_spec)
    sample = crop_sample(full_sample, crop_spec, raw_stats)
    if (raw_stats["raw_events"], raw_stats["foreground_events"], raw_stats["background_events"]) != (
        1057,
        913,
        144,
    ):
        raise ValueError(f"Frozen crop statistics changed: {raw_stats}")

    previous_lspike = dict(SAVED_LSPIKE_CROP_RESULT)
    if args.previous_lspike_result.is_file():
        previous = json.loads(args.previous_lspike_result.read_text())
        if previous["crop"]["x0"] != 248 or previous["crop"]["y0"] != 0:
            raise ValueError("Previous L_spike result does not use the frozen crop")
        previous_lspike = {
            **previous["experiment_b"],
            "provenance": str(args.previous_lspike_result.resolve()),
        }

    torch.manual_seed(args.seed)
    spikems_root = REPO_ROOT / "third_party" / "SpikeMS"
    checkpoint_path = (
        spikems_root
        / "pretrainedModels"
        / "EVIMO-pretrained"
        / "out"
        / "checkpoint.pth.tar"
    )
    model, optimizer, _, load_result = build_model_and_optimizer(
        spikems_root, checkpoint_path, device
    )
    torch.cuda.reset_peak_memory_stats(device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_adapter_visualizations(sample, args.output_dir)

    before_events, before_probability, gt_spatial, initial = evaluate(model, sample, device)
    save_spatial_map(gt_spatial, args.output_dir / "gt_spatial_map.png")
    save_spatial_map(before_probability, args.output_dir / "pred_spatial_before.png")
    save_prediction_visualization(
        before_events, args.output_dir / "pred_foreground_events_before.png"
    )

    # Initial backward also verifies the positive/zero-count probability path.
    optimizer.zero_grad(set_to_none=True)
    _, _, initial_bce = forward_bce(model, sample, device)
    initial_bce.pred_count.retain_grad()
    positive_with_zero_prediction = torch.logical_and(
        initial_bce.gt_spatial > 0, initial_bce.pred_count == 0
    )
    locations = torch.nonzero(positive_with_zero_prediction, as_tuple=False)
    if locations.numel() == 0:
        raise ValueError("No GT-positive zero-prediction pixel available for sanity check")
    sanity_location = locations[0]
    initial_bce.loss.backward()
    count_gradient = initial_bce.pred_count.grad[tuple(sanity_location)].item()
    if not math.isfinite(count_gradient) or count_gradient == 0:
        raise RuntimeError("GT-positive zero-count BCE path has no finite learning signal")
    initial_gradient = gradient_statistics(model.parameters())
    initial_layers = named_gradient_norms(model)
    history = [
        {
            "step": 0,
            **initial,
            "global_gradient_norm": initial_gradient["global_gradient_norm"],
            "layer_gradient_norms": initial_layers,
        }
    ]

    numerical_issue = None
    final_gradient = initial_gradient
    final_layers = initial_layers
    started = time.perf_counter()
    completed_steps = 0
    for step in range(1, 501):
        optimizer.zero_grad(set_to_none=True)
        _, _, bce = forward_bce(model, sample, device)
        if not math.isfinite(float(bce.loss.item())):
            numerical_issue = f"non-finite BCE at step {step}"
            break
        bce.loss.backward()
        step_gradient = gradient_statistics(model.parameters())
        step_layers = named_gradient_norms(model)
        if not step_gradient["gradients_finite"]:
            numerical_issue = f"non-finite gradient at step {step}"
            break
        optimizer.step()
        completed_steps = step
        if step in SCHEDULE:
            _, _, _, post_update = evaluate(model, sample, device)
            history.append(
                {
                    "step": step,
                    **post_update,
                    "global_gradient_norm": step_gradient["global_gradient_norm"],
                    "layer_gradient_norms": step_layers,
                }
            )
        final_gradient = step_gradient
        final_layers = step_layers

    after_events, after_probability, _, final = evaluate(model, sample, device)
    save_spatial_map(after_probability, args.output_dir / "pred_spatial_after.png")
    save_prediction_visualization(
        after_events, args.output_dir / "pred_foreground_events_after.png"
    )
    runtime_s = time.perf_counter() - started

    result = {
        "loss_marker": RECONSTRUCTED_SPATIAL_BCE_V1,
        "provenance": "ENGINEERING_RECONSTRUCTION",
        "diagnostic_marker": "DIAGNOSTIC_ONLY_GT_ASSISTED_CROP",
        "sample": {
            "sequence": SEQUENCE.name,
            "frame_index": 57,
            "timestamp_s": full_sample.mask_timestamp_s,
            "window_start_s": full_sample.start_time_s,
            "window_end_s": full_sample.end_time_s,
            "crop": FIXED_CROP,
            **raw_stats,
            "input_shape": [1, *sample.spike_tensor.shape],
        },
        "bce_definition": {
            "eps": EPS,
            "num_positive": initial_bce.num_positive,
            "num_negative": initial_bce.num_negative,
            "positive_weight": initial_bce.positive_weight,
            "probability_mapping": "eps + (1 - 2*eps) * (1 - exp(-pred_count))",
            "threshold": 0.5,
            "reduction": "mean_over_BHW",
        },
        "zero_prediction_positive_path_sanity": {
            "batch": int(sanity_location[0]),
            "y": int(sanity_location[1]),
            "x": int(sanity_location[2]),
            "pred_count": float(initial_bce.pred_count[tuple(sanity_location)].item()),
            "d_loss_d_pred_count": count_gradient,
            "finite_nonzero": True,
        },
        "initial": initial,
        "final": final,
        "loss_ratio": final["loss_bce"] / initial["loss_bce"],
        "initial_gradient": initial_gradient,
        "final_gradient": final_gradient,
        "initial_layer_gradient_norms": initial_layers,
        "final_layer_gradient_norms": final_layers,
        "history": history,
        "training": {
            "steps_requested": 500,
            "steps_completed": completed_steps,
            "optimizer": "Adam",
            "learning_rate": 1e-4,
            "seed": args.seed,
            "runtime_s": runtime_s,
            "peak_vram_mb": torch.cuda.max_memory_allocated(device) / (1024**2),
            "numerical_issue": numerical_issue,
        },
        "previous_lspike_crop": {
            "initial": previous_lspike["initial"],
            "final": previous_lspike["final"],
            "loss_ratio": previous_lspike["loss_ratio"],
        },
        "state_dict_missing_keys": list(load_result.missing_keys),
        "state_dict_unexpected_keys": list(load_result.unexpected_keys),
    }
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if numerical_issue or completed_steps != 500:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
