"""Explicit engineering loss reconstructions for SpikeMS experiments."""

from __future__ import annotations

from dataclasses import dataclass

import torch


RECONSTRUCTED_SPATIAL_BCE_V1 = "RECONSTRUCTED_SPATIAL_BCE_V1"


@dataclass(frozen=True)
class SpatialBCEResult:
    loss: torch.Tensor
    gt_spatial: torch.Tensor
    pred_count: torch.Tensor
    probability: torch.Tensor
    num_positive: int
    num_negative: int
    positive_weight: float


def reconstructed_spatial_bce_v1(
    pred_foreground_events: torch.Tensor,
    gt_foreground_events: torch.Tensor,
    eps: float = 1e-6,
) -> SpatialBCEResult:
    """Balanced BCE over 2D event activity, not the unpublished SpikeMS BCE.

    SpikeMS predicts binary foreground ON/OFF spikes with [B,2,H,W,T]
    layout. The GT target marks a pixel positive when at least one foreground
    event of either polarity occurs at any timestep. Predicted spike count is
    mapped monotonically to probability as

        eps + (1 - 2 eps) * (1 - exp(-count)).

    No clamp, detach, loss normalization beyond the specified mean, or temporal
    supervision is introduced here.
    """

    if pred_foreground_events.shape != gt_foreground_events.shape:
        raise ValueError(
            f"Prediction {pred_foreground_events.shape} and GT "
            f"{gt_foreground_events.shape} must match"
        )
    if pred_foreground_events.ndim != 5 or pred_foreground_events.shape[1] != 2:
        raise ValueError("Expected [B,2,H,W,T] SpikeMS tensors")
    if not 0 < eps < 0.5:
        raise ValueError("eps must be between zero and 0.5")

    gt_spatial = torch.any(gt_foreground_events > 0, dim=(1, 4)).to(
        pred_foreground_events.dtype
    )
    pred_count = torch.sum(pred_foreground_events, dim=(1, 4))
    probability = eps + (1.0 - 2.0 * eps) * (1.0 - torch.exp(-pred_count))
    num_positive = int(torch.count_nonzero(gt_spatial).item())
    num_negative = int(gt_spatial.numel() - num_positive)
    if num_positive == 0:
        raise ValueError("Spatial BCE requires at least one positive GT pixel")
    positive_weight = num_negative / num_positive
    per_pixel = -(
        positive_weight * gt_spatial * torch.log(probability)
        + (1.0 - gt_spatial) * torch.log1p(-probability)
    )
    return SpatialBCEResult(
        loss=torch.mean(per_pixel),
        gt_spatial=gt_spatial,
        pred_count=pred_count,
        probability=probability,
        num_positive=num_positive,
        num_negative=num_negative,
        positive_weight=positive_weight,
    )
