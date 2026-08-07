"""Tonic N-MNIST loading for host-injected event baselines."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from tonic.datasets import NMNIST
from tonic.transforms import ToFrame


def dataset_available(data_root: Path) -> bool:
    return (data_root / "test.zip").exists() or (data_root / "Test").exists()


def load_sample(data_root: Path, index: int = 0, *, n_time_bins: int | None = 100, time_window: int | None = None):
    raw_dataset = NMNIST(save_to=str(data_root), train=False)
    raw_events, label = raw_dataset[index]
    kwargs = {"sensor_size": NMNIST.sensor_size}
    if time_window is not None:
        kwargs["time_window"] = time_window
    else:
        kwargs["n_time_bins"] = n_time_bins
    framed = NMNIST(save_to=str(data_root), train=False, transform=ToFrame(**kwargs))
    frames, framed_label = framed[index]
    if int(label) != int(framed_label):
        raise RuntimeError("raw and framed sample labels differ")
    return raw_events, np.asarray(frames), int(label)
