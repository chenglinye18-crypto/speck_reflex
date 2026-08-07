"""Prediction and spike-count helpers."""

from __future__ import annotations

import numpy as np
import torch


def tensor_counts(output: torch.Tensor) -> list[int]:
    if output.ndim < 2:
        raise ValueError("expected time and class dimensions")
    return output.reshape(output.shape[0], output.shape[1], -1).sum((0, 2)).detach().to(torch.int64).cpu().tolist()


def event_counts(output: np.ndarray, classes: int = 10) -> list[int]:
    if len(output) == 0:
        return [0] * classes
    return np.bincount(output["p"].astype(np.int64), minlength=classes).tolist()


def prediction(counts: list[int]) -> int:
    if not counts or sum(counts) == 0:
        return -1
    return int(np.argmax(counts))
