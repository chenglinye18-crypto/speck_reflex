"""Minimal model-agnostic simulation orchestration."""

from dataclasses import dataclass

from software.event_processing.event_format import EventWindow
from software.models.interface import ReflexModel, ReflexPrediction


@dataclass(slots=True)
class SimulationRunner:
    model: ReflexModel

    def run_window(self, window: EventWindow) -> ReflexPrediction:
        return self.model.predict(window)
