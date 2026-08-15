"""Minimal EVIMO2v2 -> SpikeMS event tensor adapter.

This module intentionally handles one frame-aligned sample at a time.  It does
not crop from ground truth and does not filter samples by foreground ratio.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image


@dataclass(frozen=True)
class EVIMO2SpikeMSSample:
    spike_tensor: torch.Tensor
    masked_spike_tensor: torch.Tensor
    object_mask: np.ndarray
    camera_name: str
    polarity_channel_semantics: tuple[str, str]
    frame_index: int
    mask_id: int
    mask_timestamp_s: float
    start_time_s: float
    end_time_s: float
    physical_window_ms: float
    num_time_bins: int
    dt_per_bin_ms: float
    raw_event_count: int
    polarity_0_count: int
    polarity_1_count: int
    foreground_event_count: int
    background_event_count: int
    input_voxel_count: int
    foreground_voxel_count: int


def _load_metadata(sequence_path: Path):
    with np.load(sequence_path / "dataset_info.npz", allow_pickle=True) as info:
        metadata = info["meta"].item()
        frames = metadata["frames"]
        camera = metadata["meta"]
    return frames, camera


def load_frame_aligned_sample(
    sequence_path: Path,
    frame_index: int = 5,
    physical_window_ms: float = 20.0,
    num_time_bins: int = 100,
) -> EVIMO2SpikeMSSample:
    """Create full-frame SpikeMS tensors around one timestamped object mask.

    EVIMO2 event timestamps and ``meta.frames[*].ts`` are both in seconds.  A
    symmetric window around the selected mask is discretized with floor-based
    bins.  Polarity values are used directly as SpikeMS channel indices.
    """

    sequence_path = sequence_path.resolve()
    if physical_window_ms <= 0 or num_time_bins <= 0:
        raise ValueError("physical_window_ms and num_time_bins must be positive")
    if sequence_path.parents[1].name != "imo":
        raise ValueError("Adapter only accepts EVIMO2 Motion Segmentation 'imo' sequences")
    camera_name = sequence_path.parents[2].name
    if camera_name in {"left_camera", "right_camera"}:
        polarity_channel_semantics = ("negative/OFF", "positive/ON")
    elif camera_name == "samsung_mono":
        polarity_channel_semantics = ("positive/ON", "negative/OFF")
    else:
        raise ValueError(f"Unknown EVIMO2 camera polarity convention: {camera_name}")
    frames, camera = _load_metadata(sequence_path)
    frame = frames[frame_index]
    mask_id = int(frame["id"])
    mask_timestamp_s = float(frame["ts"])
    mask_key = f"mask_{mask_id:010d}"
    with np.load(sequence_path / "dataset_mask.npz") as masks:
        if mask_key not in masks:
            raise KeyError(f"Mask {mask_key} referenced by metadata is missing")
        object_mask = np.asarray(masks[mask_key] > 0, dtype=np.bool_)

    height = int(camera["res_y"])
    width = int(camera["res_x"])
    if object_mask.shape != (height, width):
        raise ValueError(
            f"Mask shape {object_mask.shape} != metadata resolution {(height, width)}"
        )

    timestamps = np.load(sequence_path / "dataset_events_t.npy", mmap_mode="r")
    xy = np.load(sequence_path / "dataset_events_xy.npy", mmap_mode="r")
    polarity = np.load(sequence_path / "dataset_events_p.npy", mmap_mode="r")
    if not (len(timestamps) == len(xy) == len(polarity)):
        raise ValueError("EVIMO2 event arrays have inconsistent lengths")
    if np.any(timestamps[1:] < timestamps[:-1]):
        raise ValueError("EVIMO2 event timestamps are not monotonic")

    half_window_s = physical_window_ms / 2000.0
    start_time_s = mask_timestamp_s - half_window_s
    end_time_s = mask_timestamp_s + half_window_s
    begin = int(np.searchsorted(timestamps, start_time_s, side="left"))
    end = int(np.searchsorted(timestamps, end_time_s, side="right"))
    event_t = np.asarray(timestamps[begin:end], dtype=np.float64)
    event_xy = np.asarray(xy[begin:end], dtype=np.int64)
    event_p = np.asarray(polarity[begin:end], dtype=np.int64)
    if event_t.size == 0:
        raise ValueError("Selected mask window contains no events")
    if set(np.unique(event_p).tolist()) - {0, 1}:
        raise ValueError(f"Unsupported polarity values: {np.unique(event_p).tolist()}")
    if (
        event_xy[:, 0].min() < 0
        or event_xy[:, 0].max() >= width
        or event_xy[:, 1].min() < 0
        or event_xy[:, 1].max() >= height
    ):
        raise ValueError("Event coordinates are outside the camera resolution")

    duration_s = end_time_s - start_time_s
    time_bins = np.floor((event_t - start_time_s) / duration_s * num_time_bins)
    time_bins = np.clip(time_bins.astype(np.int64), 0, num_time_bins - 1)
    x = event_xy[:, 0]
    y = event_xy[:, 1]
    foreground = object_mask[y, x]

    spike = np.zeros((2, height, width, num_time_bins), dtype=np.uint8)
    spike[event_p, y, x, time_bins] = 1
    masked = np.zeros_like(spike)
    masked[event_p[foreground], y[foreground], x[foreground], time_bins[foreground]] = 1
    if np.any(masked > spike):
        raise AssertionError("Foreground supervision is not a subset of the input spikes")
    if not np.isfinite(spike).all() or not np.isfinite(masked).all():
        raise ValueError("Adapter output contains NaN or Inf")

    return EVIMO2SpikeMSSample(
        spike_tensor=torch.from_numpy(spike).float(),
        masked_spike_tensor=torch.from_numpy(masked).float(),
        object_mask=object_mask,
        camera_name=camera_name,
        polarity_channel_semantics=polarity_channel_semantics,
        frame_index=frame_index,
        mask_id=mask_id,
        mask_timestamp_s=mask_timestamp_s,
        start_time_s=start_time_s,
        end_time_s=end_time_s,
        physical_window_ms=physical_window_ms,
        num_time_bins=num_time_bins,
        dt_per_bin_ms=physical_window_ms / num_time_bins,
        raw_event_count=int(event_t.size),
        polarity_0_count=int(np.count_nonzero(event_p == 0)),
        polarity_1_count=int(np.count_nonzero(event_p == 1)),
        foreground_event_count=int(np.count_nonzero(foreground)),
        background_event_count=int(np.count_nonzero(~foreground)),
        input_voxel_count=int(spike.sum()),
        foreground_voxel_count=int(masked.sum()),
    )


def _event_image(tensor: torch.Tensor) -> np.ndarray:
    counts = tensor.detach().cpu().numpy().sum(axis=-1)
    image = np.zeros((counts.shape[1], counts.shape[2], 3), dtype=np.float32)
    image[..., 0] = counts[0]
    image[..., 2] = counts[1]
    maximum = float(image.max())
    if maximum > 0:
        image = np.log1p(image) / np.log1p(maximum)
    return np.asarray(np.clip(image * 255.0, 0, 255), dtype=np.uint8)


def save_adapter_visualizations(sample: EVIMO2SpikeMSSample, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_event_image(sample.spike_tensor)).save(output_dir / "raw_events.png")
    Image.fromarray(np.asarray(sample.object_mask, dtype=np.uint8) * 255).save(
        output_dir / "object_mask.png"
    )
    Image.fromarray(_event_image(sample.masked_spike_tensor)).save(
        output_dir / "gt_foreground_events.png"
    )


def save_prediction_visualization(prediction: torch.Tensor, output_path: Path) -> None:
    if prediction.ndim == 5:
        if prediction.shape[0] != 1:
            raise ValueError("Only a single prediction is supported")
        prediction = prediction[0]
    Image.fromarray(_event_image(prediction)).save(output_path)
