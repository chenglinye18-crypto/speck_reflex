"""Minimal EVIMO1 NPZ adapter reproducing SpikeMS EVIMO preprocessing semantics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class EVIMOReferenceSource:
    events: np.ndarray
    masks: np.ndarray
    frames: list[dict]
    height: int = 260
    width: int = 346


@dataclass(frozen=True)
class EVIMOReferenceSample:
    frame_array_index: int
    frame_id: int
    center_time_s: float
    window_ms: float
    num_time_bins: int
    crop: dict[str, int]
    spike_tensor: torch.Tensor
    gt_foreground_events: torch.Tensor
    dilated_object_mask: np.ndarray
    raw_event_rows: int
    input_spike_voxels_full: int
    foreground_spike_voxels_full: int
    background_spike_voxels_full: int
    background_foreground_ratio: float


def load_evimo1_npz(path: Path) -> EVIMOReferenceSource:
    """Load one official legacy EVIMO sequence from its released NPZ archive."""

    with np.load(path, allow_pickle=True) as archive:
        events = archive["events"]
        masks = archive["mask"]
        frames = archive["meta"].item()["frames"]
    if events.ndim != 2 or events.shape[1] != 4:
        raise ValueError(f"Expected [N,4] events, got {events.shape}")
    if masks.shape[0] != len(frames) or masks.shape[1:] != (260, 346):
        raise ValueError(f"Unexpected mask/frame alignment: {masks.shape}, {len(frames)}")
    return EVIMOReferenceSource(events=events, masks=masks, frames=frames)


def official_mask(source: EVIMOReferenceSource, frame_array_index: int) -> np.ndarray:
    """Convert EVIMO object IDs to the binary mask used by SpikeMS, then dilate."""

    mask = torch.from_numpy(
        (source.masks[frame_array_index] > 0).astype(np.float32)
    ).unsqueeze(0).unsqueeze(0)
    dilated = F.max_pool2d(mask, kernel_size=5, stride=1, padding=2)
    return dilated[0, 0].numpy().astype(bool)


def official_gt_assisted_crop(
    gt_full: torch.Tensor, height: int = 128, width: int = 128
) -> dict[str, int]:
    """Reproduce EVIMODatasetBase's argmax-centered foreground-event crop."""

    image_height, image_width = gt_full.shape[1:3]
    summed = torch.sum(gt_full, dim=(0, 3))
    if not torch.any(summed > 0):
        raise ValueError("Cannot form official crop without foreground event voxels")
    maximum = int(torch.argmax(summed).item())
    center_x = maximum % image_width
    center_y = maximum // image_width
    x0 = max(center_x - width // 2, 0)
    y0 = max(center_y - height // 2, 0)
    if center_x + width // 2 > image_width - 1:
        x0 = image_width - 1 - width
    if center_y + height // 2 > image_height - 1:
        y0 = image_height - 1 - height
    return {"x0": int(x0), "y0": int(y0), "width": width, "height": height}


def make_reference_sample(
    source: EVIMOReferenceSource,
    frame_array_index: int,
    window_ms: float,
    num_time_bins: int,
    crop: dict[str, int] | None = None,
) -> EVIMOReferenceSample:
    """Apply the released SpikeMS EVIMO window, voxel, mask, and crop rules."""

    frame = source.frames[frame_array_index]
    # SpikeMS preprocessing uses frames[i]['cam']['ts'], not frames[i]['ts'].
    center = float(frame["cam"]["ts"])
    half_window_s = window_ms / 2000.0
    start = center - half_window_s
    stop = center + half_window_s
    times = source.events[:, 0]
    begin = int(np.searchsorted(times, start, side="left"))
    end = int(np.searchsorted(times, stop, side="right"))
    events = source.events[begin:end]
    # EVIMODatasetBase sets idx1 = events_idx[next] - 1 and then uses the
    # exclusive Python slice [idx0:idx1], dropping the final stored event.
    if events.shape[0]:
        events = events[:-1]

    polarity = events[:, 3].astype(np.int64)
    x = events[:, 1].astype(np.int64)
    y = events[:, 2].astype(np.int64)
    if events.shape[0]:
        time_bin = ((num_time_bins - 1) * (events[:, 0] - start) / (stop - start))
        # The released code uses normalized times directly as indices. Integer
        # truncation captures its intended bin assignment on current PyTorch.
        time_bin = np.clip(time_bin.astype(np.int64), 0, num_time_bins - 1)
    else:
        time_bin = np.empty(0, dtype=np.int64)
    if not set(np.unique(polarity)).issubset({0, 1}):
        raise ValueError(f"EVIMO polarity is not 0/1: {np.unique(polarity)}")

    spikes = torch.zeros((2, source.height, source.width, num_time_bins))
    if events.shape[0]:
        spikes[
            torch.from_numpy(polarity),
            torch.from_numpy(y),
            torch.from_numpy(x),
            torch.from_numpy(time_bin),
        ] = 1.0
    mask = official_mask(source, frame_array_index)
    mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(-1)
    foreground = spikes * mask_tensor
    background = spikes * torch.logical_not(mask_tensor)
    foreground_count = int(torch.count_nonzero(foreground).item())
    background_count = int(torch.count_nonzero(background).item())
    ratio = background_count / foreground_count if foreground_count else float("inf")

    if crop is None:
        crop = official_gt_assisted_crop(foreground)
    x0, y0 = crop["x0"], crop["y0"]
    x1, y1 = x0 + crop["width"], y0 + crop["height"]
    return EVIMOReferenceSample(
        frame_array_index=frame_array_index,
        frame_id=int(frame["id"]),
        center_time_s=center,
        window_ms=window_ms,
        num_time_bins=num_time_bins,
        crop=dict(crop),
        spike_tensor=spikes[:, y0:y1, x0:x1, :],
        gt_foreground_events=foreground[:, y0:y1, x0:x1, :],
        dilated_object_mask=mask[y0:y1, x0:x1],
        raw_event_rows=int(events.shape[0]),
        input_spike_voxels_full=int(torch.count_nonzero(spikes).item()),
        foreground_spike_voxels_full=foreground_count,
        background_spike_voxels_full=background_count,
        background_foreground_ratio=ratio,
    )


def official_valid_frame_indices(
    source: EVIMOReferenceSource,
    count: int,
    max_background_ratio: float = 1.5,
    preprocessing_min_mask_sum: int = 1000,
) -> list[tuple[int, dict[str, int]]]:
    """Choose first frames that survive released preprocessing and test filters."""

    selected = []
    for index in range(len(source.frames)):
        # EVIMO NPZ stores object IDs multiplied by 1000. The upstream HDF5
        # converter names this min_events but applies it to the raw mask sum.
        mask_id_sum = float(np.sum(source.masks[index] / 1000.0))
        if mask_id_sum < preprocessing_min_mask_sum:
            continue
        sample = make_reference_sample(source, index, window_ms=20.0, num_time_bins=20)
        if sample.foreground_spike_voxels_full == 0:
            continue
        if sample.background_foreground_ratio > max_background_ratio:
            continue
        selected.append((index, sample.crop))
        if len(selected) == count:
            return selected
    raise RuntimeError(f"Only found {len(selected)} valid EVIMO reference frames")
