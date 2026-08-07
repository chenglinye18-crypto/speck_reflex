"""Future protocol-neutral robot-risk types. No transport or motor control."""

from dataclasses import dataclass
from enum import Enum, IntEnum


class RiskLevel(IntEnum):
    NORMAL = 0
    CAUTION = 1
    LIMIT_SPEED = 2
    STOP_REQUEST = 3
    EMERGENCY_STOP = 4


class RiskDirection(str, Enum):
    UNKNOWN = "UNKNOWN"
    LEFT = "LEFT"
    FRONT = "FRONT"
    RIGHT = "RIGHT"
    REAR = "REAR"


@dataclass(frozen=True)
class RiskMessage:
    protocol_version: int
    sequence_number: int
    timestamp: int
    risk_level: RiskLevel
    direction: RiskDirection
    confidence: float
    time_to_collision: float | None
    heartbeat: bool
    checksum: int | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
