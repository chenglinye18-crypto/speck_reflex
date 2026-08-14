"""Dataset adapters that preserve canonical platform event semantics."""

from .evimo2 import (
    EVIMO2EgoMotionSample,
    EVIMO2EventConfig,
    EVIMO2Sequence,
    EVIMO2Sensor,
)
from .evimo2_ego_motion import (
    EVIMO2EgoMotionDataset,
    EVIMO2EgoMotionIndex,
    EVIMO2WindowRecord,
    TargetNormalization,
    build_ego_motion_index,
)

__all__ = [
    "EVIMO2EgoMotionSample",
    "EVIMO2EventConfig",
    "EVIMO2Sequence",
    "EVIMO2Sensor",
    "EVIMO2EgoMotionDataset",
    "EVIMO2EgoMotionIndex",
    "EVIMO2WindowRecord",
    "TargetNormalization",
    "build_ego_motion_index",
]
