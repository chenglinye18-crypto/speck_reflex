#!/usr/bin/env python3
"""Overfit the released SpikeMS architecture to exactly eight EVIMO2 samples."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import random
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
from reference.spikems.model_compat import load_official_slayer_cuda  # noqa: E402
from reference.spikems.train_engineering import (  # noqa: E402
    BASELINE_MARKER,
    Metrics,
    build_training_components,
    forward_loss_metrics,
    gradient_statistics,
    mean_metrics,
)


def load_manifest(path: Path):
    manifest = json.loads(path.read_text())
    entries = manifest["samples"]
    if len(entries) != 8:
        raise ValueError(f"Expected exactly 8 manifest samples, found {len(entries)}")
    samples = []
    for entry in entries:
        sample = load_frame_aligned_sample(
            Path(entry["sequence_path"]),
            frame_index=int(entry["frame_index"]),
            physical_window_ms=10.0,
            num_time_bins=10,
        )
        for key, actual in (
            ("raw_events", sample.raw_event_count),
            ("foreground_events", sample.foreground_event_count),
            ("background_events", sample.background_event_count),
        ):
            if int(entry[key]) != actual:
                raise ValueError(
                    f"Manifest drift for {entry['sequence']} frame {entry['frame_index']} "
                    f"{key}: expected {entry[key]}, loaded {actual}"
                )
        samples.append(sample)
    return manifest, entries, samples


def evaluate_all(model, criterion, samples, device):
    model.eval()
    metrics: list[Metrics] = []
    predictions = []
    with torch.no_grad():
        for sample in samples:
            prediction, _, _, item = forward_loss_metrics(
                model, criterion, sample, device
            )
            metrics.append(item)
            predictions.append(prediction.detach().cpu())
    model.train()
    return mean_metrics(metrics), metrics, predictions


def metric_record(step: int, aggregate: dict[str, float]) -> dict[str, float | int]:
    return {"step": step, **aggregate}


def finite_aggregate(metrics: dict[str, float]) -> bool:
    return all(math.isfinite(value) for value in metrics.values())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT / "reference" / "spikems" / "overfit8_manifest.json",
    )
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--eval-interval", type=int, default=20)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "spikems_training" / "overfit8",
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
    if args.steps < 20 or args.eval_interval != 20:
        raise ValueError("Engineering protocol requires at least 20 steps and 20-step evaluation")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    load_official_slayer_cuda(args.slayer_root, args.build_root)
    device = torch.device("cuda:0")
    torch.cuda.reset_peak_memory_stats(device)
    manifest, entries, samples = load_manifest(args.manifest)

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
    args.output_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    initial, initial_items, before_predictions = evaluate_all(
        model, criterion, samples, device
    )
    history = [metric_record(0, initial)]
    for sample_index in (0, 7):
        visual_dir = args.output_dir / f"sample_{sample_index:02d}"
        save_adapter_visualizations(samples[sample_index], visual_dir)
        save_prediction_visualization(
            before_predictions[sample_index], visual_dir / "pred_before_training.png"
        )

    order = list(range(8))
    train_log = []
    numerical_issue = None
    completed_steps = 0
    for step in range(1, args.steps + 1):
        if (step - 1) % 8 == 0:
            random.shuffle(order)
        sample_index = order[(step - 1) % 8]
        optimizer.zero_grad(set_to_none=True)
        _, _, loss, item = forward_loss_metrics(
            model, criterion, samples[sample_index], device
        )
        if not math.isfinite(item.loss_spike) or not item.prediction_finite:
            numerical_issue = f"non-finite forward/loss at step {step}"
            break
        loss.backward()
        gradients = gradient_statistics(model.parameters())
        if not gradients["gradients_finite"]:
            numerical_issue = f"non-finite gradient at step {step}"
            break
        if gradients["parameter_tensors_with_nonzero_grad"] == 0:
            numerical_issue = f"all-zero gradients at step {step}"
            break
        optimizer.step()
        completed_steps = step

        if step == 1 or step % args.log_interval == 0:
            train_log.append(
                {
                    "step": step,
                    "sample_index": sample_index,
                    "loss_spike": item.loss_spike,
                    "iou": item.iou,
                    "foreground_recall": item.foreground_recall,
                    "background_leakage": item.background_leakage,
                    "prediction_spikes": item.prediction_spikes,
                    "global_gradient_norm": gradients["global_gradient_norm"],
                }
            )

        if step % args.eval_interval == 0:
            aggregate, _, _ = evaluate_all(model, criterion, samples, device)
            history.append(metric_record(step, aggregate))
            if not finite_aggregate(aggregate):
                numerical_issue = f"non-finite aggregate metric at step {step}"
                break
            if step == 20 and aggregate["mean_loss_spike"] >= initial["mean_loss_spike"]:
                numerical_issue = "20-step mean loss did not decrease"
                break

    final, final_items, after_predictions = evaluate_all(
        model, criterion, samples, device
    )
    if not history or history[-1]["step"] != completed_steps:
        history.append(metric_record(completed_steps, final))
    for sample_index in (0, 7):
        visual_dir = args.output_dir / f"sample_{sample_index:02d}"
        save_prediction_visualization(
            after_predictions[sample_index], visual_dir / "pred_after_training.png"
        )

    loss_ratio = final["mean_loss_spike"] / initial["mean_loss_spike"]
    changes = {
        "loss_ratio": loss_ratio,
        "iou_change": final["mean_iou"] - initial["mean_iou"],
        "foreground_recall_change": final["mean_foreground_recall"]
        - initial["mean_foreground_recall"],
        "background_leakage_change": final["mean_background_leakage"]
        - initial["mean_background_leakage"],
    }
    correspondence_improved = bool(
        changes["iou_change"] > 0
        or changes["foreground_recall_change"] > 0
        or changes["background_leakage_change"] < 0
    )
    gate_pass = bool(
        numerical_issue is None
        and completed_steps == args.steps
        and finite_aggregate(final)
        and loss_ratio < 0.8
        and correspondence_improved
    )
    runtime_s = time.perf_counter() - started
    result = {
        "baseline_marker": BASELINE_MARKER,
        "selection_marker": manifest["selection_marker"],
        "spikems_overfit8_gate": "PASS" if gate_pass else "FAIL",
        "initial": initial,
        "final": final,
        "learning_trend": changes,
        "training": {
            "optimizer": "Adam",
            "learning_rate": 1e-4,
            "steps_requested": args.steps,
            "steps_completed": completed_steps,
            "batch_size": 1,
            "spatial_strategy": "full_frame",
            "runtime_s": runtime_s,
            "peak_vram_mb": torch.cuda.max_memory_allocated(device) / (1024**2),
            "numerical_issue": numerical_issue,
            "seed": args.seed,
        },
        "history": history,
        "per_sample_initial": [item.__dict__ for item in initial_items],
        "per_sample_final": [item.__dict__ for item in final_items],
        "manifest_samples": entries,
        "state_dict_missing_keys": list(load_result.missing_keys),
        "state_dict_unexpected_keys": list(load_result.unexpected_keys),
        "visualizations": [
            str((args.output_dir / "sample_00").resolve()),
            str((args.output_dir / "sample_07").resolve()),
        ],
    }
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    with (args.output_dir / "training_log.csv").open("w", newline="") as stream:
        if train_log:
            writer = csv.DictWriter(stream, fieldnames=list(train_log[0].keys()))
            writer.writeheader()
            writer.writerows(train_log)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not gate_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
