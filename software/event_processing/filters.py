"""Contract for deterministic event-stream filters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable

from .event_format import Event


class EventFilter(ABC):
    @abstractmethod
    def apply(self, events: Iterable[Event]) -> Iterable[Event]:
        """Return a filtered stream while preserving canonical event semantics."""
