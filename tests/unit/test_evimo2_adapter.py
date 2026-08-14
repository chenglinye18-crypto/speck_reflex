from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from software.datasets import (
    EVIMO2EgoMotionDataset,
    EVIMO2EventConfig,
    EVIMO2Sensor,
    EVIMO2Sequence,
    TargetNormalization,
    build_ego_motion_index,
)


def _pose(timestamp: float, x: float, yaw: float = 0.0) -> dict:
    return {
        "ts": timestamp,
        "pos": {
            "t": {"x": x, "y": 0.0, "z": 0.0},
            "q": {
                "w": math.cos(yaw / 2.0),
                "x": 0.0,
                "y": 0.0,
                "z": math.sin(yaw / 2.0),
            },
            "rpy": {"r": 0.0, "p": 0.0, "y": yaw},
        },
    }


def _write_sequence(
    path: Path,
    *,
    translation_m: float = 0.01,
    yaw_rad: float = 0.0,
    width: int = 4,
    height: int = 3,
) -> None:
    path.mkdir(parents=True)
    np.save(path / "dataset_events_t.npy", np.array([0.0005, 0.0012, 0.0014, 0.0039, 0.0040], dtype=np.float32))
    np.save(
        path / "dataset_events_xy.npy",
        np.array([[1, 1], [2, 1], [2, 1], [3, 2], [0, 0]], dtype=np.uint16),
    )
    np.save(path / "dataset_events_p.npy", np.array([0, 1, 1, 0, 1], dtype=np.uint8))
    meta = {
        "frames": [{"id": 7, "ts": 0.004, "cam": _pose(0.004, 0.004)["pos"]}],
        "full_trajectory": [
            {"ts": 0.0, "cam": _pose(0.0, 0.0, 0.0)},
            {"ts": 0.01, "cam": _pose(0.01, translation_m, yaw_rad)},
        ],
        "imu": {},
        "meta": {"res_x": width, "res_y": height},
    }
    np.savez(
        path / "dataset_info.npz",
        K=np.eye(3),
        D=np.zeros(4),
        index=np.array([0], dtype=np.uint32),
        discretization=np.array(0.01),
        meta=np.array(meta, dtype=object),
    )


@pytest.mark.unit
def test_event_count_binning_and_end_exclusion(tmp_path: Path) -> None:
    sequence_path = tmp_path / "sequence"
    _write_sequence(sequence_path)
    sequence = EVIMO2Sequence(
        sequence_path,
        sensor=EVIMO2Sensor.LEFT_CAMERA,
        config=EVIMO2EventConfig(timesteps=4, dt_ms=1.0),
    )
    sample = sequence.sample_at_frame(7)
    assert sample.events.shape == (4, 2, 3, 4)
    assert sample.events.dtype is torch.float32
    assert sample.event_count == 4
    assert sample.events.sum().item() == 4
    assert sample.events[0, 0, 1, 1].item() == 1
    assert sample.events[1, 1, 1, 2].item() == 2
    assert sample.events[3, 0, 2, 3].item() == 1
    assert sample.events[:, :, 0, 0].sum().item() == 0


@pytest.mark.unit
def test_samsung_polarity_is_canonicalized_without_changing_counts(tmp_path: Path) -> None:
    sequence_path = tmp_path / "sequence"
    _write_sequence(sequence_path)
    config = EVIMO2EventConfig(timesteps=4, dt_ms=1.0)
    prophesee = EVIMO2Sequence(
        sequence_path, sensor=EVIMO2Sensor.RIGHT_CAMERA, config=config
    ).sample_at_frame(7)
    samsung = EVIMO2Sequence(
        sequence_path, sensor=EVIMO2Sensor.SAMSUNG_MONO, config=config
    ).sample_at_frame(7)
    assert torch.equal(samsung.events[:, 0], prophesee.events[:, 1])
    assert torch.equal(samsung.events[:, 1], prophesee.events[:, 0])
    assert samsung.events.sum().item() == prophesee.events.sum().item()


@pytest.mark.unit
def test_interpolated_translation_produces_camera_local_velocity(tmp_path: Path) -> None:
    sequence_path = tmp_path / "sequence"
    _write_sequence(sequence_path, translation_m=0.01)
    sample = EVIMO2Sequence(
        sequence_path,
        sensor=EVIMO2Sensor.LEFT_CAMERA,
        config=EVIMO2EventConfig(timesteps=4, dt_ms=1.0),
    ).sample_at_frame(7)
    assert torch.allclose(sample.ego_motion[:3], torch.tensor([1.0, 0.0, 0.0]), atol=1e-6)
    assert torch.count_nonzero(sample.ego_motion[3:]) == 0


@pytest.mark.unit
def test_interpolated_rotation_produces_angular_velocity(tmp_path: Path) -> None:
    sequence_path = tmp_path / "sequence"
    _write_sequence(sequence_path, translation_m=0.0, yaw_rad=0.01)
    sample = EVIMO2Sequence(
        sequence_path,
        sensor=EVIMO2Sensor.LEFT_CAMERA,
        config=EVIMO2EventConfig(timesteps=4, dt_ms=1.0),
    ).sample_at_frame(7)
    assert torch.count_nonzero(sample.ego_motion[:3]) == 0
    assert torch.allclose(sample.ego_motion[3:], torch.tensor([0.0, 0.0, 1.0]), atol=1e-5)


@pytest.mark.unit
def test_adapter_is_deterministic_and_read_only(tmp_path: Path) -> None:
    sequence_path = tmp_path / "sequence"
    _write_sequence(sequence_path)
    source_files = {path.name: path.stat().st_mtime_ns for path in sequence_path.iterdir()}
    sequence = EVIMO2Sequence(
        sequence_path,
        sensor=EVIMO2Sensor.LEFT_CAMERA,
        config=EVIMO2EventConfig(timesteps=4, dt_ms=1.0),
    )
    first = sequence.sample_at_frame(7)
    second = sequence.sample_at_frame(7)
    assert torch.equal(first.events, second.events)
    assert torch.equal(first.ego_motion, second.ego_motion)
    assert source_files == {path.name: path.stat().st_mtime_ns for path in sequence_path.iterdir()}


@pytest.mark.unit
def test_window_outside_trajectory_is_rejected(tmp_path: Path) -> None:
    sequence_path = tmp_path / "sequence"
    _write_sequence(sequence_path)
    sequence = EVIMO2Sequence(sequence_path, sensor=EVIMO2Sensor.LEFT_CAMERA)
    with pytest.raises(ValueError, match="outside the camera trajectory"):
        sequence.sample_at_time(0.02)


@pytest.mark.unit
def test_event_coordinate_reduction_preserves_counts_and_polarity(tmp_path: Path) -> None:
    sequence_path = tmp_path / "sequence"
    _write_sequence(sequence_path, width=10, height=10)
    native = EVIMO2Sequence(
        sequence_path,
        sensor=EVIMO2Sensor.LEFT_CAMERA,
        config=EVIMO2EventConfig(timesteps=4, dt_ms=1.0),
    ).sample_at_frame(7)
    reduced = EVIMO2Sequence(
        sequence_path,
        sensor=EVIMO2Sensor.LEFT_CAMERA,
        config=EVIMO2EventConfig(timesteps=4, dt_ms=1.0, spatial_reduction=2),
    ).sample_at_frame(7)
    assert reduced.events.shape == (4, 2, 5, 5)
    assert reduced.events.sum() == native.events.sum() == 4
    assert reduced.events[1, 1, 0, 1].item() == 2


@pytest.mark.unit
def test_sequence_split_index_and_target_normalization_are_deterministic(
    tmp_path: Path,
) -> None:
    root = tmp_path / "samsung_mono"
    for index in range(3):
        _write_sequence(
            root / "sfm" / "train" / f"train_{index}",
            translation_m=0.01 * (index + 1),
            width=10,
            height=10,
        )
    _write_sequence(
        root / "sfm" / "eval" / "test_0",
        translation_m=0.04,
        width=10,
        height=10,
    )
    config = EVIMO2EventConfig(timesteps=4, dt_ms=1.0, spatial_reduction=2)
    first = build_ego_motion_index(
        root,
        event_config=config,
        frame_stride=1,
        validation_fraction=0.34,
        seed=17,
    )
    second = build_ego_motion_index(
        root,
        event_config=config,
        frame_stride=1,
        validation_fraction=0.34,
        seed=17,
    )
    assert first == second
    assert len(first.train) == 2
    assert len(first.validation) == 1
    assert len(first.test) == 1
    split_sequences = [
        {record.sequence for record in records}
        for records in (first.train, first.validation, first.test)
    ]
    assert split_sequences[0].isdisjoint(split_sequences[1])
    assert split_sequences[0].isdisjoint(split_sequences[2])
    assert split_sequences[1].isdisjoint(split_sequences[2])

    normalization = TargetNormalization.fit(first.train, epsilon=1e-6)
    targets = torch.tensor([record.ego_motion for record in first.train])
    assert torch.allclose(
        normalization.denormalize(normalization.normalize(targets)), targets
    )
    sample = EVIMO2EgoMotionDataset(
        root, first.train, event_config=config
    )[0]
    assert sample["events"].shape == (4, 2, 5, 5)
    assert sample["ego_motion"].shape == (6,)
