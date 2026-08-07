import pytest

from hardware.interfaces import ReflexHardware
from software.event_processing import Event, EventWindow
from software.models import ReflexDirection, ReflexModel, ReflexPrediction
from software.simulation import SimulationRunner
from speck_reflex.runtime import RiskLevel


class FixedModel(ReflexModel):
    @property
    def model_id(self) -> str:
        return "unit-fixed-v1"

    def predict(self, window: EventWindow) -> ReflexPrediction:
        return ReflexPrediction(0.75, 120, ReflexDirection.FRONT, False, window.end_timestamp_us)

    def reset(self) -> None:
        return None


@pytest.mark.unit
def test_canonical_event_window_and_model_contract() -> None:
    events = (Event(1, 2, 100, 1), Event(2, 2, 105, 0))
    window = EventWindow(events, width=4, height=3, start_timestamp_us=90, end_timestamp_us=110)
    prediction = SimulationRunner(FixedModel()).run_window(window)
    assert prediction.risk == 0.75
    assert prediction.ttc_ms == 120
    assert prediction.direction is ReflexDirection.FRONT
    assert prediction.timestamp_us == 110


@pytest.mark.unit
def test_interface_rejects_invalid_semantics() -> None:
    with pytest.raises(TypeError, match="integers"):
        Event(0.5, 0, 0, 1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="polarity"):
        Event(0, 0, 0, -1)
    with pytest.raises(ValueError, match="ordered"):
        EventWindow((Event(0, 0, 2, 1), Event(0, 0, 1, 0)), 1, 1, 0, 3)
    with pytest.raises(ValueError, match="risk"):
        ReflexPrediction(1.01, None, ReflexDirection.UNKNOWN, False, 0)


@pytest.mark.unit
def test_frozen_enums_and_abstract_hardware_boundary() -> None:
    assert {direction.name: int(direction) for direction in ReflexDirection} == {
        "UNKNOWN": 0,
        "LEFT": 1,
        "FRONT": 2,
        "RIGHT": 3,
        "REAR": 4,
    }
    assert int(RiskLevel.EMERGENCY_STOP) == 4
    with pytest.raises(TypeError):
        ReflexHardware()  # type: ignore[abstract]
