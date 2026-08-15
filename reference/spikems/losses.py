"""Explicit engineering loss reconstructions for SpikeMS experiments."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


RECONSTRUCTED_SPATIAL_BCE_V1 = "RECONSTRUCTED_SPATIAL_BCE_V1"
MEMBRANE_SPATIAL_BCE_V2 = "MEMBRANE_SPATIAL_BCE_V2"


@dataclass(frozen=True)
class SpatialBCEResult:
    loss: torch.Tensor
    gt_spatial: torch.Tensor
    pred_count: torch.Tensor
    probability: torch.Tensor
    num_positive: int
    num_negative: int
    positive_weight: float


@dataclass(frozen=True)
class MembraneSpatialBCEResult:
    loss: torch.Tensor
    gt_spatial: torch.Tensor
    membrane_max: torch.Tensor
    spatial_logit: torch.Tensor
    probability: torch.Tensor
    num_positive: int
    num_negative: int
    positive_weight: float


def membrane_spatial_bce_v2(
    final_membrane: torch.Tensor,
    gt_foreground_events: torch.Tensor,
    positive_weight: float,
    theta: float = 0.22,
) -> MembraneSpatialBCEResult:
    """Engineering spatial BCE on the continuous final decoder membrane.

    The target is event-derived: a pixel is positive if either polarity has a
    foreground event at any point in the sample window. The matching prediction
    logit is max(final_membrane over polarity and time) minus the final-layer
    firing threshold. This is an engineering interface, not the unpublished
    SpikeMS spatial BCE.
    """

    if final_membrane.shape != gt_foreground_events.shape:
        raise ValueError(
            f"Membrane {final_membrane.shape} and GT "
            f"{gt_foreground_events.shape} must match"
        )
    if final_membrane.ndim != 5 or final_membrane.shape[1] != 2:
        raise ValueError("Expected [B,2,H,W,T] SpikeMS tensors")
    if not positive_weight > 0:
        raise ValueError("positive_weight must be positive")

    gt_spatial = torch.any(gt_foreground_events > 0, dim=(1, 4)).to(
        final_membrane.dtype
    )
    membrane_max = torch.amax(final_membrane, dim=(1, 4))
    spatial_logit = membrane_max - theta
    pos_weight = torch.as_tensor(
        positive_weight, dtype=spatial_logit.dtype, device=spatial_logit.device
    )
    loss = F.binary_cross_entropy_with_logits(
        spatial_logit, gt_spatial, pos_weight=pos_weight, reduction="mean"
    )
    num_positive = int(torch.count_nonzero(gt_spatial).item())
    num_negative = int(gt_spatial.numel() - num_positive)
    if num_positive == 0:
        raise ValueError("Membrane spatial BCE requires positive GT pixels")
    return MembraneSpatialBCEResult(
        loss=loss,
        gt_spatial=gt_spatial,
        membrane_max=membrane_max,
        spatial_logit=spatial_logit,
        probability=torch.sigmoid(spatial_logit),
        num_positive=num_positive,
        num_negative=num_negative,
        positive_weight=float(positive_weight),
    )


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
