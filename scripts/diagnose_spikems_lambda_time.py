#!/usr/bin/env python3
"""Bounded lambda_time screen for the frozen SpikeMS combined-loss sample."""

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
from scripts.diagnose_spikems_membrane_combined_loss import (  # noqa: E402
    FIXED_CROP,
    SEQUENCE,
    crop_sample,
    raw_crop_statistics,
    train_run,
)


LAMBDAS = [0.00513924, 0.01027848, 0.02055696, 0.04111392]
SPATIAL_F1_FLOOR = 0.35
SCREEN_SCHEDULE = {0, 20, 50, 100}
FINAL_SCHEDULE = {0, 10, 20, 50, 100, 200, 300, 400, 500}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument(
        "--baseline-result",
        type=Path,
        default=REPO_ROOT
        / "outputs"
        / "spikems_training"
        / "membrane_combined"
        / "result.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT
        / "outputs"
        / "spikems_training"
        / "lambda_time_screen",
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

    baseline = json.loads(args.baseline_result.read_text())
    baseline_lambda = float(baseline["initial_gradient_balance"]["lambda_time"])
    if abs(baseline_lambda - LAMBDAS[0]) > 1e-8:
        raise RuntimeError(f"Unexpected baseline lambda: {baseline_lambda}")
    selected_positive_weight = float(baseline["selected_positive_weight"])
    baseline_100 = next(
        item
        for item in baseline["stage_b_combined"]["history"]
        if item["step"] == 100
    )
    baseline_500 = baseline["stage_b_combined"]["final"]

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
    screen = [
        {
            "lambda_time": LAMBDAS[0],
            "source": "REUSED_MEMBRANE_COMBINED_BASELINE_43f1660",
            "final": baseline_100,
        }
    ]
    for lambda_time in LAMBDAS[1:]:
        run = train_run(
            "combined",
            100,
            SCREEN_SCHEDULE,
            sample,
            device,
            spikems_root,
            checkpoint_path,
            args.seed,
            selected_positive_weight,
            lambda_time=lambda_time,
        )
        screen.append(
            {
                "lambda_time": lambda_time,
                "source": "NEW_100_STEP_SCREEN",
                **run,
            }
        )

    eligible = [
        item
        for item in screen
        if item["final"]["spatial_f1"] >= SPATIAL_F1_FLOOR
        and item.get("numerical_issue") is None
    ]
    selected_lambda = None
    final_run = None
    stop_reason = None
    if eligible:
        selected = max(eligible, key=lambda item: item["final"]["event_iou"])
        selected_lambda = float(selected["lambda_time"])
        final_run = train_run(
            "combined",
            500,
            FINAL_SCHEDULE,
            sample,
            device,
            spikems_root,
            checkpoint_path,
            args.seed,
            selected_positive_weight,
            args.output_dir / "selected_500",
            lambda_time=selected_lambda,
        )
    else:
        stop_reason = "NO_100_STEP_CANDIDATE_MEETS_SPATIAL_F1_FLOOR"
    result = {
        "marker": "SPIKEMS_LAMBDA_TIME_BOUNDED_SCREEN",
        "fixed_definition": {
            "loss": "L_space + lambda_time * official spikeTime",
            "space_loss": "MEMBRANE_SPATIAL_BCE_V2_UNCHANGED",
            "positive_weight": selected_positive_weight,
            "optimizer": "Adam(lr=1e-4, betas=(0.9,0.999), weight_decay=0, amsgrad=True)",
            "checkpoint": str(checkpoint_path),
        },
        "sample": {
            "sequence": SEQUENCE.name,
            "frame_index": 57,
            "timestamp_s": full_sample.mask_timestamp_s,
            "crop": FIXED_CROP,
            **raw_stats,
        },
        "lambda_candidates": LAMBDAS,
        "screen_steps": 100,
        "spatial_f1_floor": SPATIAL_F1_FLOOR,
        "screen": screen,
        "selection_rule": "Spatial F1 >= 0.35, then highest 5D event IoU",
        "selected_lambda_time": selected_lambda,
        "selected_500_step_run": final_run,
        "stop_reason": stop_reason,
        "references": {
            "previous_combined_500": baseline_500,
            "l_spike_only_event_iou": 0.030499075785582256,
            "previous_combined_result": str(args.baseline_result.resolve()),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if final_run is not None and (
        final_run["numerical_issue"] or final_run["steps_completed"] != 500
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
