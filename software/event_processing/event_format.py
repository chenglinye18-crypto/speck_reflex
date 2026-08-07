"""Canonical, device-independent event and event-window types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True, slots=True)
class Event:
    """One polarity event using the frozen platform field semantics."""

    x: int
    y: int
    timestamp_us: int
    polarity: int

    def __post_init__(self) -> None:
        if any(type(value) is not int for value in (self.x, self.y, self.timestamp_us, self.polarity)):
            raise TypeError("event fields must be integers")
        if not 0 <= self.x <= 0xFFFF or not 0 <= self.y <= 0xFFFF:
            raise ValueError("x and y must fit unsigned 16-bit coordinates")
        if not 0 <= self.timestamp_us <= 0xFFFFFFFFFFFFFFFF:
            raise ValueError("timestamp_us must fit an unsigned 64-bit value")
        if self.polarity not in (0, 1):
            raise ValueError("polarity must be 0 or 1")


@dataclass(frozen=True, slots=True)
class EventWindow:
    """Ordered events plus the spatial and temporal model-input boundary."""

    events: tuple[Event, ...]
    width: int
    height: int
    start_timestamp_us: int
    end_timestamp_us: int

    def __post_init__(self) -> None:
        if type(self.width) is not int or type(self.height) is not int:
            raise TypeError("width and height must be integers")
        if not 1 <= self.width <= 0x10000 or not 1 <= self.height <= 0x10000:
            raise ValueError("width and height must be in [1, 65536]")
        if type(self.start_timestamp_us) is not int or type(self.end_timestamp_us) is not int:
            raise TypeError("window timestamps must be integers")
        if not 0 <= self.start_timestamp_us <= 0xFFFFFFFFFFFFFFFF or not 0 <= self.end_timestamp_us <= 0xFFFFFFFFFFFFFFFF:
            raise ValueError("window timestamps must fit unsigned 64-bit values")
        if self.start_timestamp_us > self.end_timestamp_us:
            raise ValueError("window start must not follow window end")
        previous = self.start_timestamp_us
        for event in self.events:
            if event.x >= self.width or event.y >= self.height:
                raise ValueError("event coordinate lies outside the window sensor size")
            if not self.start_timestamp_us <= event.timestamp_us <= self.end_timestamp_us:
                raise ValueError("event timestamp lies outside the window")
            if event.timestamp_us < previous:
                raise ValueError("events must be ordered by timestamp_us")
            previous = event.timestamp_us

    @classmethod
    def from_events(
        cls,
        events: Iterable[Event],
        *,
        width: int,
        height: int,
        start_timestamp_us: int,
        end_timestamp_us: int,
    ) -> "EventWindow":
        return cls(tuple(events), width, height, start_timestamp_us, end_timestamp_us)
