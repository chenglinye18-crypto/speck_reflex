"""Read-only EVIMO2v2 event-window adapter for camera ego-motion supervision.

This module deliberately stops at one deterministic sample contract:
canonical OFF/ON event counts plus a window-level camera twist. It does not
construct object-motion labels, losses, training loops, or task claims.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor


class EVIMO2Sensor(str, Enum):
    """DVS sensors present in the downloaded EVIMO2 subset."""

    SAMSUNG_MONO = "samsung_mono"
    LEFT_CAMERA = "left_camera"
    RIGHT_CAMERA = "right_camera"


@dataclass(frozen=True, slots=True)
class EVIMO2EventConfig:
    """Temporal discretization for one event-native model input window."""

    timesteps: int = 32
    dt_ms: float = 1.0

    def __post_init__(self) -> None:
        if self.timesteps <= 0:
            raise ValueError("timesteps must be positive")
        if not math.isfinite(self.dt_ms) or self.dt_ms <= 0.0:
            raise ValueError("dt_ms must be finite and positive")

    @property
    def duration_s(self) -> float:
        return self.timesteps * self.dt_ms / 1000.0


@dataclass(frozen=True, slots=True)
class EVIMO2EgoMotionSample:
    """One event window and its camera-local average SE(3) twist label."""

    events: Tensor
    ego_motion: Tensor
    start_time_s: float
    end_time_s: float
    sensor: EVIMO2Sensor
    sequence: str
    event_count: int
    frame_id: int | None


@dataclass(frozen=True, slots=True)
class _Pose:
    translation: np.ndarray
    quaternion_wxyz: np.ndarray


def _canonical_polarity(raw_polarity: np.ndarray, sensor: EVIMO2Sensor) -> np.ndarray:
    if raw_polarity.ndim != 1:
        raise ValueError("event polarity must be a one-dimensional array")
    if raw_polarity.size and not np.isin(raw_polarity, (0, 1)).all():
        raise ValueError("EVIMO2 polarity values must be 0 or 1")
    canonical = raw_polarity.astype(np.int64, copy=False)
    if sensor is EVIMO2Sensor.SAMSUNG_MONO:
        # COMPATIBILITY_PATCH: EVIMO2 documents Samsung polarity as inverted
        # relative to the Prophesee reference convention. Keep raw files intact
        # and canonicalize only in the read path (OFF=0, ON=1).
        canonical = 1 - canonical
    return canonical


def _normalize_quaternion(quaternion: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(quaternion))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("camera quaternion must be finite and non-zero")
    return quaternion / norm


def _slerp(q0: np.ndarray, q1: np.ndarray, fraction: float) -> np.ndarray:
    q0 = _normalize_quaternion(q0)
    q1 = _normalize_quaternion(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        return _normalize_quaternion(q0 + fraction * (q1 - q0))
    angle = math.acos(dot)
    sin_angle = math.sin(angle)
    return (
        math.sin((1.0 - fraction) * angle) / sin_angle * q0
        + math.sin(fraction * angle) / sin_angle * q1
    )


def _quaternion_to_rotation(q: np.ndarray) -> np.ndarray:
    w, x, y, z = _normalize_quaternion(q)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _so3_log(rotation: np.ndarray) -> np.ndarray:
    cosine = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    angle = math.acos(cosine)
    skew_vector = np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ],
        dtype=np.float64,
    )
    if angle < 1e-8:
        return 0.5 * skew_vector
    return angle / (2.0 * math.sin(angle)) * skew_vector


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def _se3_twist_per_second(start: _Pose, end: _Pose, duration_s: float) -> np.ndarray:
    """Return log(inv(T_wc_start) @ T_wc_end) / duration in start-camera axes."""

    if duration_s <= 0.0:
        raise ValueError("camera-motion duration must be positive")
    rotation_start = _quaternion_to_rotation(start.quaternion_wxyz)
    rotation_end = _quaternion_to_rotation(end.quaternion_wxyz)
    relative_rotation = rotation_start.T @ rotation_end
    relative_translation = rotation_start.T @ (end.translation - start.translation)

    phi = _so3_log(relative_rotation)
    angle = float(np.linalg.norm(phi))
    phi_hat = _skew(phi)
    if angle < 1e-6:
        inverse_left_jacobian = np.eye(3) - 0.5 * phi_hat + (phi_hat @ phi_hat) / 12.0
    else:
        coefficient = 1.0 / (angle * angle) - (
            1.0 + math.cos(angle)
        ) / (2.0 * angle * math.sin(angle))
        inverse_left_jacobian = (
            np.eye(3) - 0.5 * phi_hat + coefficient * (phi_hat @ phi_hat)
        )
    rho = inverse_left_jacobian @ relative_translation
    return np.concatenate((rho, phi)) / duration_s


class EVIMO2Sequence:
    """Memory-mapped, read-only view of one official EVIMO2v2 sequence."""

    def __init__(
        self,
        sequence_directory: str | Path,
        *,
        sensor: EVIMO2Sensor | str,
        config: EVIMO2EventConfig | None = None,
    ) -> None:
        self.path = Path(sequence_directory).expanduser().resolve()
        self.sensor = EVIMO2Sensor(sensor)
        self.config = config or EVIMO2EventConfig()
        self._event_t = np.load(self.path / "dataset_events_t.npy", mmap_mode="r").reshape(-1)
        self._event_xy = np.load(self.path / "dataset_events_xy.npy", mmap_mode="r")
        self._event_p = np.load(self.path / "dataset_events_p.npy", mmap_mode="r").reshape(-1)
        if self._event_xy.ndim != 2 or self._event_xy.shape[1] != 2:
            raise ValueError("event xy array must have shape [N, 2]")
        if not (len(self._event_t) == len(self._event_xy) == len(self._event_p)):
            raise ValueError("EVIMO2 event arrays must contain the same number of events")

        # Official EVIMO2 meta is an object dictionary and therefore requires
        # pickle-enabled NumPy loading. Only trusted official archives belong here.
        with np.load(self.path / "dataset_info.npz", allow_pickle=True) as info:
            meta = info["meta"].item()
        camera_meta = meta["meta"]
        self.height = int(camera_meta["res_y"])
        self.width = int(camera_meta["res_x"])
        self._frames = {
            int(frame["id"]): frame
            for frame in meta["frames"]
            if isinstance(frame.get("id"), (int, np.integer))
            and int(frame["id"]) < np.iinfo(np.uint64).max
        }
        self._trajectory_times, self._trajectory_poses = self._camera_trajectory(
            meta["full_trajectory"]
        )

    @staticmethod
    def _camera_trajectory(records: list[dict[str, Any]]) -> tuple[np.ndarray, tuple[_Pose, ...]]:
        timed_poses: list[tuple[float, _Pose]] = []
        for record in records:
            camera = record.get("cam")
            if not isinstance(camera, dict) or "pos" not in camera:
                continue
            position = camera["pos"]
            translation = position["t"]
            quaternion = position["q"]
            timestamp = float(camera.get("ts", record.get("ts")))
            pose = _Pose(
                translation=np.array(
                    [translation["x"], translation["y"], translation["z"]],
                    dtype=np.float64,
                ),
                quaternion_wxyz=_normalize_quaternion(
                    np.array(
                        [quaternion["w"], quaternion["x"], quaternion["y"], quaternion["z"]],
                        dtype=np.float64,
                    )
                ),
            )
            if np.isfinite(pose.translation).all() and math.isfinite(timestamp):
                timed_poses.append((timestamp, pose))
        if len(timed_poses) < 2:
            raise ValueError("EVIMO2 sequence needs at least two valid camera poses")
        timed_poses.sort(key=lambda item: item[0])
        times = np.array([item[0] for item in timed_poses], dtype=np.float64)
        if np.any(np.diff(times) <= 0.0):
            raise ValueError("camera trajectory timestamps must be strictly increasing")
        return times, tuple(item[1] for item in timed_poses)

    def _pose_at(self, timestamp_s: float) -> _Pose:
        if not self._trajectory_times[0] <= timestamp_s <= self._trajectory_times[-1]:
            raise ValueError("requested time is outside the camera trajectory")
        upper = int(np.searchsorted(self._trajectory_times, timestamp_s, side="left"))
        if upper < len(self._trajectory_times) and self._trajectory_times[upper] == timestamp_s:
            return self._trajectory_poses[upper]
        lower = upper - 1
        span = self._trajectory_times[upper] - self._trajectory_times[lower]
        fraction = float((timestamp_s - self._trajectory_times[lower]) / span)
        first = self._trajectory_poses[lower]
        second = self._trajectory_poses[upper]
        return _Pose(
            translation=first.translation + fraction * (second.translation - first.translation),
            quaternion_wxyz=_slerp(
                first.quaternion_wxyz, second.quaternion_wxyz, fraction
            ),
        )

    def sample_at_frame(self, frame_id: int) -> EVIMO2EgoMotionSample:
        """Create the event window ending at one official ground-truth frame."""

        try:
            frame = self._frames[frame_id]
        except KeyError as error:
            raise KeyError(f"unknown EVIMO2 ground-truth frame id: {frame_id}") from error
        return self.sample_at_time(float(frame["ts"]), frame_id=frame_id)

    def sample_at_time(
        self, end_time_s: float, *, frame_id: int | None = None
    ) -> EVIMO2EgoMotionSample:
        """Create a canonical count tensor and camera twist for one time window."""

        if not math.isfinite(end_time_s):
            raise ValueError("end_time_s must be finite")
        start_time_s = end_time_s - self.config.duration_s
        start_pose = self._pose_at(start_time_s)
        end_pose = self._pose_at(end_time_s)

        first = int(np.searchsorted(self._event_t, start_time_s, side="left"))
        last = int(np.searchsorted(self._event_t, end_time_s, side="left"))
        timestamps = np.asarray(self._event_t[first:last], dtype=np.float64)
        coordinates = np.asarray(self._event_xy[first:last], dtype=np.int64)
        polarity = _canonical_polarity(
            np.asarray(self._event_p[first:last]), self.sensor
        )
        if coordinates.size and (
            coordinates[:, 0].min() < 0
            or coordinates[:, 0].max() >= self.width
            or coordinates[:, 1].min() < 0
            or coordinates[:, 1].max() >= self.height
        ):
            raise ValueError("event coordinate lies outside the declared sensor resolution")

        dt_s = self.config.dt_ms / 1000.0
        time_bins = np.floor((timestamps - start_time_s) / dt_s).astype(np.int64)
        if time_bins.size:
            time_bins = np.clip(time_bins, 0, self.config.timesteps - 1)
        event_counts = np.zeros(
            (self.config.timesteps, 2, self.height, self.width), dtype=np.int32
        )
        if len(timestamps):
            np.add.at(
                event_counts,
                (time_bins, polarity, coordinates[:, 1], coordinates[:, 0]),
                1,
            )

        twist = _se3_twist_per_second(
            start_pose, end_pose, self.config.duration_s
        )
        if not np.isfinite(twist).all():
            raise ValueError("derived camera ego-motion contains non-finite values")
        return EVIMO2EgoMotionSample(
            events=torch.from_numpy(event_counts).to(dtype=torch.float32),
            ego_motion=torch.from_numpy(twist.astype(np.float32)),
            start_time_s=start_time_s,
            end_time_s=end_time_s,
            sensor=self.sensor,
            sequence=self.path.name,
            event_count=len(timestamps),
            frame_id=frame_id,
        )
