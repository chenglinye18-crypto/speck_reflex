"""Deterministic EVIMO2 Samsung sequence splits and ego-motion windows."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch import Tensor
from torch.utils.data import Dataset

from .evimo2 import EVIMO2EventConfig, EVIMO2Sensor, EVIMO2Sequence


_REQUIRED_SEQUENCE_FILES = (
    "dataset_events_t.npy",
    "dataset_events_xy.npy",
    "dataset_events_p.npy",
    "dataset_info.npz",
)


@dataclass(frozen=True, slots=True)
class EVIMO2WindowRecord:
    """One reproducible window reference; paths remain relative to data root."""

    sequence: str
    frame_id: int
    end_time_s: float
    ego_motion: tuple[float, float, float, float, float, float]


@dataclass(frozen=True, slots=True)
class EVIMO2EgoMotionIndex:
    train: tuple[EVIMO2WindowRecord, ...]
    validation: tuple[EVIMO2WindowRecord, ...]
    test: tuple[EVIMO2WindowRecord, ...]
    sha256: str

    def to_json_dict(self) -> dict[str, object]:
        return {
            "sha256": self.sha256,
            "train": [asdict(record) for record in self.train],
            "validation": [asdict(record) for record in self.validation],
            "test": [asdict(record) for record in self.test],
        }


@dataclass(frozen=True, slots=True)
class TargetNormalization:
    mean: Tensor
    std: Tensor
    epsilon: float

    @classmethod
    def fit(
        cls, records: Iterable[EVIMO2WindowRecord], *, epsilon: float
    ) -> "TargetNormalization":
        values = torch.tensor([record.ego_motion for record in records], dtype=torch.float64)
        if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] != 6:
            raise ValueError("at least two six-component training targets are required")
        if epsilon <= 0.0:
            raise ValueError("normalization epsilon must be positive")
        mean = values.mean(dim=0)
        std = values.std(dim=0, unbiased=False)
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
            raise ValueError("target normalization statistics must be finite")
        return cls(mean.to(torch.float32), std.to(torch.float32), float(epsilon))

    @property
    def scale(self) -> Tensor:
        return self.std.clamp_min(self.epsilon)

    def normalize(self, value: Tensor) -> Tensor:
        return (value - self.mean.to(value.device)) / self.scale.to(value.device)

    def denormalize(self, value: Tensor) -> Tensor:
        return value * self.scale.to(value.device) + self.mean.to(value.device)

    def state_dict(self) -> dict[str, object]:
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "epsilon": self.epsilon,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> "TargetNormalization":
        return cls(
            mean=torch.tensor(state["mean"], dtype=torch.float32),
            std=torch.tensor(state["std"], dtype=torch.float32),
            epsilon=float(state["epsilon"]),
        )


def _sequence_directories(path: Path) -> tuple[Path, ...]:
    if not path.is_dir():
        raise FileNotFoundError(f"EVIMO2 split directory not found: {path}")
    sequences = tuple(
        candidate
        for candidate in sorted(path.iterdir())
        if candidate.is_dir()
        and all((candidate / filename).is_file() for filename in _REQUIRED_SEQUENCE_FILES)
    )
    if not sequences:
        raise FileNotFoundError(f"no complete EVIMO2 sequences found under: {path}")
    return sequences


def _records_for_sequences(
    root: Path,
    sequences: Iterable[Path],
    *,
    event_config: EVIMO2EventConfig,
    frame_stride: int,
) -> tuple[EVIMO2WindowRecord, ...]:
    records: list[EVIMO2WindowRecord] = []
    for path in sequences:
        sequence = EVIMO2Sequence(
            path,
            sensor=EVIMO2Sensor.SAMSUNG_MONO,
            config=event_config,
        )
        for frame_id in sequence.frame_ids[::frame_stride]:
            end_time_s = sequence.frame_time(frame_id)
            if not sequence.can_sample_at_time(end_time_s):
                continue
            target = sequence.ego_motion_at_time(end_time_s)
            records.append(
                EVIMO2WindowRecord(
                    sequence=path.relative_to(root).as_posix(),
                    frame_id=frame_id,
                    end_time_s=end_time_s,
                    ego_motion=tuple(float(value) for value in target.tolist()),  # type: ignore[arg-type]
                )
            )
    if not records:
        raise ValueError("no valid EVIMO2 windows remain after coverage checks")
    return tuple(records)


def _index_digest(splits: dict[str, tuple[EVIMO2WindowRecord, ...]]) -> str:
    digest = hashlib.sha256()
    for split_name in ("train", "validation", "test"):
        digest.update(split_name.encode())
        for record in splits[split_name]:
            digest.update(
                json.dumps(asdict(record), sort_keys=True, separators=(",", ":")).encode()
            )
    return digest.hexdigest()


def build_ego_motion_index(
    data_root: str | Path,
    *,
    event_config: EVIMO2EventConfig,
    frame_stride: int,
    validation_fraction: float,
    seed: int,
) -> EVIMO2EgoMotionIndex:
    """Split upstream train sequences into train/validation; keep eval as test."""

    root = Path(data_root).expanduser().resolve()
    if frame_stride <= 0:
        raise ValueError("frame_stride must be positive")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must lie between zero and one")

    upstream_train = list(_sequence_directories(root / "sfm" / "train"))
    upstream_test = _sequence_directories(root / "sfm" / "eval")
    if len(upstream_train) < 2:
        raise ValueError("at least two upstream training sequences are required")
    random.Random(seed).shuffle(upstream_train)
    validation_count = max(1, min(len(upstream_train) - 1, round(len(upstream_train) * validation_fraction)))
    validation_sequences = tuple(sorted(upstream_train[:validation_count]))
    train_sequences = tuple(sorted(upstream_train[validation_count:]))

    splits = {
        "train": _records_for_sequences(
            root, train_sequences, event_config=event_config, frame_stride=frame_stride
        ),
        "validation": _records_for_sequences(
            root,
            validation_sequences,
            event_config=event_config,
            frame_stride=frame_stride,
        ),
        "test": _records_for_sequences(
            root, upstream_test, event_config=event_config, frame_stride=frame_stride
        ),
    }
    return EVIMO2EgoMotionIndex(
        train=splits["train"],
        validation=splits["validation"],
        test=splits["test"],
        sha256=_index_digest(splits),
    )


class EVIMO2EgoMotionDataset(Dataset[dict[str, object]]):
    """Lazy read-only window dataset backed by memory-mapped official arrays."""

    def __init__(
        self,
        data_root: str | Path,
        records: Iterable[EVIMO2WindowRecord],
        *,
        event_config: EVIMO2EventConfig,
    ) -> None:
        self.root = Path(data_root).expanduser().resolve()
        self.records = tuple(records)
        self.event_config = event_config
        self._sequences: dict[str, EVIMO2Sequence] = {}
        if not self.records:
            raise ValueError("EVIMO2EgoMotionDataset requires at least one window")

    def __len__(self) -> int:
        return len(self.records)

    def _sequence(self, relative_path: str) -> EVIMO2Sequence:
        if relative_path not in self._sequences:
            self._sequences[relative_path] = EVIMO2Sequence(
                self.root / relative_path,
                sensor=EVIMO2Sensor.SAMSUNG_MONO,
                config=self.event_config,
            )
        return self._sequences[relative_path]

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index]
        sample = self._sequence(record.sequence).sample_at_frame(record.frame_id)
        return {
            "events": sample.events,
            "ego_motion": sample.ego_motion,
            "sequence": record.sequence,
            "frame_id": record.frame_id,
            "event_count": sample.event_count,
        }

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        state["_sequences"] = {}
        return state
