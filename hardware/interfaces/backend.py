"""Hardware abstraction layer shared by STM32N6, FPGA and neuromorphic targets."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass

from software.event_processing.event_format import Event
from software.models.interface import ReflexPrediction


@dataclass(frozen=True, slots=True)
class BackendCapabilities:
    backend_id: str
    event_protocol_version: int
    reflex_protocol_version: int
    max_sensor_width: int
    max_sensor_height: int

    def __post_init__(self) -> None:
        if not self.backend_id.strip():
            raise ValueError("backend_id must not be empty")
        if self.event_protocol_version < 1 or self.reflex_protocol_version < 1:
            raise ValueError("protocol versions must be positive")
        if not 1 <= self.max_sensor_width <= 0x10000 or not 1 <= self.max_sensor_height <= 0x10000:
            raise ValueError("sensor limits must be in [1, 65536]")


class ReflexHardware(ABC):
    """Lifecycle and data-plane contract; implementations must be explicit."""

    @property
    @abstractmethod
    def capabilities(self) -> BackendCapabilities:
        """Describe compatibility before any hardware access."""

    @abstractmethod
    def initialize(self, config: Mapping[str, object]) -> None:
        """Acquire explicitly approved resources and validate configuration."""

    @abstractmethod
    def send_event(self, event: Event) -> None:
        """Send one canonical event using the selected event transport."""

    @abstractmethod
    def receive_output(self, timeout_ms: int) -> ReflexPrediction | None:
        """Receive one advisory risk output, or None on timeout."""

    @abstractmethod
    def shutdown(self) -> None:
        """Release resources without changing persistent device state."""
