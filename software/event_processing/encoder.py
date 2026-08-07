"""Contract for translating sensor/source records into canonical events."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Generic, TypeVar

from .event_format import Event

RawEvent = TypeVar("RawEvent")


class EventEncoder(ABC, Generic[RawEvent]):
    """A stateless format adapter; implementations belong to concrete sources."""

    @abstractmethod
    def encode(self, records: Iterable[RawEvent]) -> Iterable[Event]:
        """Convert source records without inventing events."""
