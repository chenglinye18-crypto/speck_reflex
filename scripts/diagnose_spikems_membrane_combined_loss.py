#!/usr/bin/env python3
"""Diagnose final-membrane spatial BCE and a gradient-balanced combined loss."""

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
    MEMBRANE_SPATIAL_BCE_V2,
    membrane_spatial_bce_v2,
)
from reference.spikems.model_compat import load_official_slayer_cuda  # noqa: E402
from reference.spikems.train_engineering import (  # noqa: E402
    align_gt_to_prediction,
    build_training_components,
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
THETA6 = 0.22
SCREEN_SCHEDULE = {0, 20, 50, 100}
TRAIN_SCHEDULE = {0, 10, 20, 50, 100, 200, 300, 400, 500}
OLD_BASELINES = {
    "l_spike_only": {
        "spatial_iou": None,
        "spatial_precision": None,
        "spatial_recall": None,
        "spatial_f1": None,
        "event_iou": 0.030499075785582256,
        "event_recall": 0.03699551569506727,
        "background_leakage": 0.026905829596412557,
        "prediction_spikes": 223,
        "provenance": "SINGLE_SAMPLE_CROP_DIAGNOSTIC_20260815.md",
    },
    "spike_count_bce": {
        "spatial_iou": 0.069929,
        "spatial_precision": 0.069929,
        "spatial_recall": 1.0,
        "spatial_f1": 0.130717,
        "predicted_active_pixels": 10382,
        "event_iou": 0.004438,
        "event_recall": 0.246637,
        "background_leakage": 0.162093,
        "prediction_spikes": 48898,
        "provenance": "BCE_SINGLE_SAMPLE_DIAGNOSTIC_20260815.md",
    },
}


class FinalMembraneHook:
    def __init__(self, model):
        self.output = None
        self.handle = model.deconv6.register_forward_hook(self._capture)

    def _capture(self, _module, _inputs, output):
        self.output = output

    def close(self):
        self.handle.remove()


def spatial_metrics(probability: torch.Tensor, target: torch.Tensor) -> dict:
    prediction = probability >= 0.5
    truth = target > 0
    intersection = int(torch.logical_and(prediction, truth).sum().item())
    union = int(torch.logical_or(prediction, truth).sum().item())
    predicted = int(prediction.sum().item())
    positive = int(truth.sum().item())
    precision = intersection / predicted if predicted else 0.0
    recall = intersection / positive if positive else 0.0
    return {
        "spatial_iou": intersection / union if union else 0.0,
        "spatial_precision": precision,
        "spatial_recall": recall,
        "spatial_f1": (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        ),
        "gt_active_pixels": positive,
        "predicted_active_pixels": predicted,
    }


def event_metrics(prediction, gt, spatial_foreground_mask) -> dict:
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
        "event_precision": intersection / pred_count if pred_count else 0.0,
        "event_recall": intersection / gt_count if gt_count else 0.0,
        "background_leakage": leaked / pred_count if pred_count else 0.0,
        "prediction_spikes": pred_count,
    }


def save_map(tensor: torch.Tensor, path: Path) -> None:
    tensor = tensor.detach().cpu()
    if tensor.ndim == 3:
        tensor = tensor[0]
    image = np.asarray(torch.clamp(tensor, 0, 1).numpy() * 255.0, dtype=np.uint8)
    Image.fromarray(image).save(path)


def model_forward(model, hook, sample, device):
    hook.output = None
    model_input = sample.spike_tensor.unsqueeze(0).to(device=device, dtype=torch.float32)
    gt_full = sample.masked_spike_tensor.unsqueeze(0).to(
        device=device, dtype=torch.float32
    )
    prediction = model(model_input)
    membrane = hook.output
    if membrane is None:
        raise RuntimeError("deconv6 forward hook did not capture a membrane tensor")
    if membrane.shape != prediction.shape:
        raise RuntimeError(
            f"Membrane shape {membrane.shape} != prediction shape {prediction.shape}"
        )
    gt = align_gt_to_prediction(gt_full, prediction)
    return prediction, membrane, gt


def evaluate(model, hook, criterion, sample, device, positive_weight) -> tuple:
    model.eval()
    with torch.no_grad():
        prediction, membrane, gt = model_forward(model, hook, sample, device)
        spatial = membrane_spatial_bce_v2(
            membrane, gt, positive_weight=positive_weight, theta=THETA6
        )
        loss_time = criterion.spikeTime(prediction, gt)
        metrics = {
            "loss_space": float(spatial.loss.item()),
            "loss_time": float(loss_time.item()),
            **spatial_metrics(spatial.probability, spatial.gt_spatial),
            **event_metrics(
                prediction,
                gt,
                spatial_mask_on_prediction_support(sample, prediction, device),
            ),
            "prediction_finite": bool(torch.isfinite(prediction).all().item()),
            "membrane_finite": bool(torch.isfinite(membrane).all().item()),
        }
    model.train()
    return (
        prediction.detach().cpu(),
        spatial.probability.detach().cpu(),
        spatial.gt_spatial.detach().cpu(),
        metrics,
    )


def build_components(spikems_root, checkpoint_path, device, seed):
    torch.manual_seed(seed)
    model, criterion, optimizer, _, load_result = build_training_components(
        spikems_root, checkpoint_path, device
    )
    hook = FinalMembraneHook(model)
    return model, criterion, optimizer, hook, load_result


def record_gradient(model, step, metrics, loss_total, weighted_time=None):
    record = {
        "step": step,
        **metrics,
        "loss_total": float(loss_total),
        "global_gradient_norm": gradient_statistics(model.parameters())[
            "global_gradient_norm"
        ],
        "layer_gradient_norms": named_gradient_norms(model),
    }
    if weighted_time is not None:
        record["weighted_loss_time"] = float(weighted_time)
    return record


def train_run(
    mode,
    steps,
    schedule,
    sample,
    device,
    spikems_root,
    checkpoint_path,
    seed,
    positive_weight,
    output_dir=None,
    lambda_time=None,
):
    model, criterion, optimizer, hook, load_result = build_components(
        spikems_root, checkpoint_path, device, seed
    )
    before_events, before_probability, gt_spatial, initial = evaluate(
        model, hook, criterion, sample, device, positive_weight
    )
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        save_adapter_visualizations(sample, output_dir)
        save_map(gt_spatial, output_dir / "gt_spatial_map.png")
        save_map(
            (before_probability >= 0.5).to(before_probability.dtype),
            output_dir / "pred_spatial_before.png",
        )
        save_map(before_probability, output_dir / "membrane_spatial_score_before.png")
        save_prediction_visualization(
            before_events, output_dir / "pred_foreground_events_before.png"
        )

    history = []
    numerical_issue = None
    started = time.perf_counter()
    completed_steps = 0
    for step in range(0, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        prediction, membrane, gt = model_forward(model, hook, sample, device)
        spatial = membrane_spatial_bce_v2(
            membrane, gt, positive_weight=positive_weight, theta=THETA6
        )
        loss_time = criterion.spikeTime(prediction, gt)
        if mode == "space":
            total = spatial.loss
            weighted_time = None
        elif mode == "combined":
            total = spatial.loss + lambda_time * loss_time
            weighted_time = lambda_time * loss_time
        else:
            raise ValueError(mode)
        if not bool(torch.isfinite(total).item()):
            numerical_issue = f"non-finite loss at step {step}"
            break
        total.backward()
        gradients = gradient_statistics(model.parameters())
        if not gradients["gradients_finite"]:
            numerical_issue = f"non-finite gradient at step {step}"
            break
        if step in schedule:
            _, _, _, current = evaluate(
                model, hook, criterion, sample, device, positive_weight
            )
            history.append(
                record_gradient(
                    model,
                    step,
                    current,
                    float(total.detach().item()),
                    None if weighted_time is None else float(weighted_time.detach().item()),
                )
            )
        if step == steps:
            break
        optimizer.step()
        completed_steps += 1

    after_events, after_probability, _, final = evaluate(
        model, hook, criterion, sample, device, positive_weight
    )
    if output_dir is not None:
        save_map(
            (after_probability >= 0.5).to(after_probability.dtype),
            output_dir / "pred_spatial_after.png",
        )
        save_map(after_probability, output_dir / "membrane_spatial_score_after.png")
        save_prediction_visualization(
            after_events, output_dir / "pred_foreground_events_after.png"
        )
    hook.close()
    return {
        "initial": initial,
        "final": final,
        "history": history,
        "steps_completed": completed_steps,
        "runtime_s": time.perf_counter() - started,
        "numerical_issue": numerical_issue,
        "state_dict_missing_keys": list(load_result.missing_keys),
        "state_dict_unexpected_keys": list(load_result.unexpected_keys),
    }


def separate_initial_gradient_scales(
    sample, device, spikems_root, checkpoint_path, seed, positive_weight
):
    model, criterion, optimizer, hook, _ = build_components(
        spikems_root, checkpoint_path, device, seed
    )
    optimizer.zero_grad(set_to_none=True)
    prediction, membrane, gt = model_forward(model, hook, sample, device)
    if not membrane.requires_grad:
        raise RuntimeError("Captured final membrane is detached from autograd")
    hook_validation = {
        "membrane_shape": list(membrane.shape),
        "prediction_shape": list(prediction.shape),
        "shapes_equal": membrane.shape == prediction.shape,
        "membrane_requires_grad": bool(membrane.requires_grad),
        "membrane_detached": False,
    }
    space = membrane_spatial_bce_v2(
        membrane, gt, positive_weight=positive_weight, theta=THETA6
    )
    initial_space = float(space.loss.detach().item())
    space.loss.backward()
    space_grad = gradient_statistics(model.parameters())

    optimizer.zero_grad(set_to_none=True)
    prediction, _, gt = model_forward(model, hook, sample, device)
    time_loss = criterion.spikeTime(prediction, gt)
    initial_time = float(time_loss.detach().item())
    time_loss.backward()
    time_grad = gradient_statistics(model.parameters())
    hook.close()
    g_space = float(space_grad["global_gradient_norm"])
    g_time = float(time_grad["global_gradient_norm"])
    if not (math.isfinite(g_space) and math.isfinite(g_time) and g_space > 0 and g_time > 0):
        raise RuntimeError(f"Invalid initial gradient scales: {g_space=}, {g_time=}")
    return {
        "initial_l_space": initial_space,
        "initial_l_time": initial_time,
        "initial_g_space": g_space,
        "initial_g_time": g_time,
        "lambda_time": g_space / g_time,
        "space_gradient": space_grad,
        "time_gradient": time_grad,
        "membrane_hook_validation": hook_validation,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "spikems_training" / "membrane_combined",
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
    if (raw_stats["raw_events"], raw_stats["foreground_events"], raw_stats["background_events"]) != (1057, 913, 144):
        raise RuntimeError(f"Frozen crop statistics changed: {raw_stats}")

    spikems_root = REPO_ROOT / "third_party" / "SpikeMS"
    checkpoint_path = spikems_root / "pretrainedModels/EVIMO-pretrained/out/checkpoint.pth.tar"
    gt = sample.masked_spike_tensor[:, :127, :127, :]
    gt_spatial = torch.any(gt > 0, dim=(0, 3))
    n_pos = int(gt_spatial.sum().item())
    n_neg = int(gt_spatial.numel() - n_pos)
    ratio = n_neg / n_pos
    weights = [1.0, math.sqrt(ratio), ratio]

    torch.cuda.reset_peak_memory_stats(device)
    screen = []
    for weight in weights:
        run = train_run(
            "space", 100, SCREEN_SCHEDULE, sample, device, spikems_root,
            checkpoint_path, args.seed, weight
        )
        screen.append({"positive_weight": weight, **run})
    valid = [
        item for item in screen
        if item["numerical_issue"] is None
        and 0 < item["final"]["predicted_active_pixels"] < 127 * 127
        and item["final"]["spatial_precision"] > 0
        and item["final"]["spatial_recall"] > 0
    ]
    if not valid:
        raise RuntimeError("All positive-weight screen cases were degenerate")
    selected = max(
        valid,
        key=lambda item: (item["final"]["spatial_f1"], item["final"]["spatial_iou"]),
    )
    selected_weight = float(selected["positive_weight"])

    stage_a = train_run(
        "space", 500, TRAIN_SCHEDULE, sample, device, spikems_root,
        checkpoint_path, args.seed, selected_weight, args.output_dir / "membrane_bce"
    )
    scales = separate_initial_gradient_scales(
        sample, device, spikems_root, checkpoint_path, args.seed, selected_weight
    )
    scales["weighted_initial_temporal_contribution"] = (
        scales["lambda_time"] * scales["initial_l_time"]
    )
    stage_b = train_run(
        "combined", 500, TRAIN_SCHEDULE, sample, device, spikems_root,
        checkpoint_path, args.seed, selected_weight, args.output_dir / "combined",
        lambda_time=scales["lambda_time"]
    )

    result = {
        "loss_marker": MEMBRANE_SPATIAL_BCE_V2,
        "combined_marker": "GRADIENT_BALANCED_COMBINED_V1",
        "provenance": "ENGINEERING_CHOICE",
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
            "prediction_and_membrane_shape": [1, 2, 127, 127, 10],
        },
        "definition": {
            "theta6": THETA6,
            "gt": "ANY(gt_foreground_events over polarity,time)",
            "membrane_aggregation": "MAX(final_deconv6_membrane over polarity,time)",
            "spatial_logit": "membrane_max - theta6",
            "loss": "binary_cross_entropy_with_logits(mean, pos_weight)",
        },
        "positive_weight_screen": screen,
        "selected_positive_weight": selected_weight,
        "selection_rule": "highest final spatial F1; IoU tie-break; reject degenerate maps",
        "stage_a_membrane_bce": stage_a,
        "initial_gradient_balance": scales,
        "stage_b_combined": stage_b,
        "old_baselines": OLD_BASELINES,
        "training": {
            "optimizer": "Adam",
            "learning_rate": 1e-4,
            "betas": [0.9, 0.999],
            "weight_decay": 0.0,
            "amsgrad": True,
            "seed": args.seed,
            "peak_vram_mb": torch.cuda.max_memory_allocated(device) / (1024**2),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if stage_a["numerical_issue"] or stage_b["numerical_issue"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
