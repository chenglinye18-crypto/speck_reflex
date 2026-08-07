"""Model contract shared by ANN, SNN, hybrid and hardware-backed models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum

from software.event_processing.event_format import EventWindow


class ReflexDirection(IntEnum):
    UNKNOWN = 0
    LEFT = 1
    FRONT = 2
    RIGHT = 3
    REAR = 4


@dataclass(frozen=True, slots=True)
class ReflexPrediction:
    risk: float
    ttc_ms: int | None
    direction: ReflexDirection
    emergency_stop: bool
    timestamp_us: int

    def __post_init__(self) -> None:
        if isinstance(self.risk, bool) or not isinstance(self.risk, (int, float)):
            raise TypeError("risk must be numeric")
        if not 0.0 <= self.risk <= 1.0:
            raise ValueError("risk must be in [0.0, 1.0]")
        if not isinstance(self.direction, ReflexDirection):
            raise TypeError("direction must be a ReflexDirection")
        if type(self.emergency_stop) is not bool:
            raise TypeError("emergency_stop must be boolean")
        if self.ttc_ms is not None:
            if type(self.ttc_ms) is not int:
                raise TypeError("ttc_ms must be an integer or None")
            if not 0 <= self.ttc_ms < 0xFFFF:
                raise ValueError("ttc_ms must fit uint16; 0xFFFF is reserved for unknown")
        if type(self.timestamp_us) is not int or not 0 <= self.timestamp_us <= 0xFFFFFFFFFFFFFFFF:
            raise ValueError("timestamp_us must fit an unsigned 64-bit value")


class ReflexModel(ABC):
    """Backend-neutral event-window to risk-prediction interface."""

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Stable model/version identifier used in experiment metadata."""

    @abstractmethod
    def predict(self, window: EventWindow) -> ReflexPrediction:
        """Evaluate one bounded event window."""

    @abstractmethod
    def reset(self) -> None:
        """Reset temporal state at an explicit stream boundary."""
