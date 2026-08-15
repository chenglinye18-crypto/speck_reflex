"""Small SpikeMS L_spike-only engineering training helpers.

This is deliberately not a reconstruction of the unpublished combined loss.
The official model, checkpoint, SLAYER kernels, and spikeTime loss stay unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from reference.spikems.evimo2_adapter import EVIMO2SpikeMSSample
from reference.spikems.model_compat import build_official_model


BASELINE_MARKER = "SPIKEMS_LSPIKE_ENGINEERING_BASELINE"
SIMULATION = {"Ts": 1.0, "tSample": 10, "tStartLoss": 0}
CHECKPOINT_KERNEL_SIMULATION = {"Ts": 1, "tSample": 100, "tStartLoss": 50}
GLOBAL_NEURON = {
    "type": "SRMALPHA",
    "theta": 0.22,
    "tauSr": 2.0,
    "tauRef": 1.0,
    "scaleRef": 2,
    "tauRho": 1,
    "scaleRho": 0.20,
}


@dataclass(frozen=True)
class Metrics:
    loss_spike: float
    iou: float
    foreground_recall: float
    background_leakage: float
    prediction_spikes: int
    prediction_finite: bool


def build_training_components(
    spikems_root: Path,
    checkpoint_path: Path,
    device: torch.device,
):
    """Build the released model/loss and a fresh optimizer with checkpoint hyperparameters."""

    model, optimizer, checkpoint, load_result = build_model_and_optimizer(
        spikems_root, checkpoint_path, device
    )
    snn = importlib.import_module("slayerpytorch")
    criterion = snn.loss(
        {
            "simulation": dict(CHECKPOINT_KERNEL_SIMULATION),
            "neuron": dict(GLOBAL_NEURON),
        }
    ).to(device)
    return model, criterion, optimizer, checkpoint, load_result


def build_model_and_optimizer(
    spikems_root: Path,
    checkpoint_path: Path,
    device: torch.device,
):
    """Build official checkpoint weights and a fresh checkpoint-configured Adam."""

    # The checkpoint state dict includes registered SRM/refractory kernels built
    # with tSample=100. Strict-load those buffers unchanged. Runtime duration is
    # still the input tensor's final dimension (T=10).
    model, checkpoint, load_result = build_official_model(
        spikems_root, checkpoint_path, device
    )
    model.train()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        amsgrad=True,
    )
    return model, optimizer, checkpoint, load_result


def align_gt_to_prediction(gt: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
    """Match the released runner's deterministic top-left support crop; never resize."""

    if gt.ndim != 5 or prediction.ndim != 5:
        raise ValueError("GT and prediction must have [B,C,H,W,T] layout")
    if gt.shape[:2] != prediction.shape[:2]:
        raise ValueError(f"Batch/channel mismatch: GT {gt.shape}, prediction {prediction.shape}")
    if any(g < p for g, p in zip(gt.shape[2:], prediction.shape[2:])):
        raise ValueError(f"GT support {gt.shape} is smaller than prediction {prediction.shape}")
    return gt[
        :,
        :,
        : prediction.shape[2],
        : prediction.shape[3],
        : prediction.shape[4],
    ]


def spatial_mask_on_prediction_support(
    sample: EVIMO2SpikeMSSample, prediction: torch.Tensor, device: torch.device
) -> torch.Tensor:
    mask = torch.from_numpy(np.asarray(sample.object_mask, dtype=np.bool_)).to(device)
    return mask[: prediction.shape[2], : prediction.shape[3]]


def calculate_metrics(
    prediction: torch.Tensor,
    gt: torch.Tensor,
    spatial_foreground_mask: torch.Tensor,
    loss: torch.Tensor,
) -> Metrics:
    pred_binary = prediction > 0
    gt_binary = gt > 0
    intersection = int(torch.logical_and(pred_binary, gt_binary).sum().item())
    union = int(torch.logical_or(pred_binary, gt_binary).sum().item())
    gt_count = int(gt_binary.sum().item())
    pred_count = int(pred_binary.sum().item())
    outside = ~spatial_foreground_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
    leaked = int(torch.logical_and(pred_binary, outside).sum().item())
    return Metrics(
        loss_spike=float(loss.detach().item()),
        iou=intersection / union if union else math.nan,
        foreground_recall=intersection / gt_count if gt_count else math.nan,
        # With no predicted spikes there are also no leaked spikes. Keep this
        # diagnostic finite at zero; recall and prediction_spikes expose the
        # undesirable all-zero case separately.
        background_leakage=leaked / pred_count if pred_count else 0.0,
        prediction_spikes=pred_count,
        prediction_finite=bool(torch.isfinite(prediction).all().item()),
    )


def forward_loss_metrics(model, criterion, sample: EVIMO2SpikeMSSample, device):
    model_input = sample.spike_tensor.unsqueeze(0).to(device=device, dtype=torch.float32)
    gt_full = sample.masked_spike_tensor.unsqueeze(0).to(device=device, dtype=torch.float32)
    prediction = model(model_input)
    gt = align_gt_to_prediction(gt_full, prediction)
    loss = criterion.spikeTime(prediction, gt)
    spatial_mask = spatial_mask_on_prediction_support(sample, prediction, device)
    metrics = calculate_metrics(prediction, gt, spatial_mask, loss)
    return prediction, gt, loss, metrics


def gradient_statistics(parameters: Iterable[torch.nn.Parameter]) -> dict[str, int | float | bool]:
    with_grad = 0
    nonzero = 0
    sum_squares = 0.0
    max_norm = 0.0
    finite = True
    for parameter in parameters:
        if parameter.grad is None:
            continue
        with_grad += 1
        gradient = parameter.grad.detach()
        finite = finite and bool(torch.isfinite(gradient).all().item())
        norm = float(torch.linalg.vector_norm(gradient.float()).item())
        if norm > 0:
            nonzero += 1
        sum_squares += norm * norm
        max_norm = max(max_norm, norm)
    return {
        "parameter_tensors_with_grad": with_grad,
        "parameter_tensors_with_nonzero_grad": nonzero,
        "global_gradient_norm": math.sqrt(sum_squares),
        "max_parameter_gradient_norm": max_norm,
        "gradients_finite": finite,
    }


def named_gradient_norms(model) -> dict[str, float | None]:
    """Return one diagnostic norm for each of SpikeMS's six trainable layers."""

    result = {}
    for name, parameter in model.named_parameters():
        result[name] = (
            float(torch.linalg.vector_norm(parameter.grad.detach().float()).item())
            if parameter.grad is not None
            else None
        )
    return result


def clone_trainable_parameters(model) -> list[torch.Tensor]:
    return [parameter.detach().clone() for parameter in model.parameters()]


def parameter_change_statistics(model, before: list[torch.Tensor]) -> dict[str, float | bool]:
    sum_squares = 0.0
    finite = True
    for parameter, old in zip(model.parameters(), before):
        difference = parameter.detach() - old
        finite = finite and bool(torch.isfinite(difference).all().item())
        norm = float(torch.linalg.vector_norm(difference.float()).item())
        sum_squares += norm * norm
    return {
        "parameter_change_norm": math.sqrt(sum_squares),
        "parameter_change_finite": finite,
    }


def mean_metrics(metrics: Iterable[Metrics]) -> dict[str, float]:
    values = list(metrics)
    if not values:
        raise ValueError("Cannot average an empty metric list")
    return {
        "mean_loss_spike": float(np.mean([item.loss_spike for item in values])),
        "mean_iou": float(np.mean([item.iou for item in values])),
        "mean_foreground_recall": float(
            np.mean([item.foreground_recall for item in values])
        ),
        "mean_background_leakage": float(
            np.mean([item.background_leakage for item in values])
        ),
        "mean_prediction_spikes": float(
            np.mean([item.prediction_spikes for item in values])
        ),
    }
