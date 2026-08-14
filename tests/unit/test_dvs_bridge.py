from __future__ import annotations

import pytest

from software.dvs_bridge import (
    FROZEN_DT_US,
    FROZEN_TIMESTEPS,
    INPUT_KIND,
    PROTOCOL,
    event_window_from_message,
    window_to_event_bins,
)


def _message(*, duration_us: int = FROZEN_TIMESTEPS * FROZEN_DT_US) -> dict[str, object]:
    return {
        "protocol": PROTOCOL,
        "type": "event_window",
        "input_kind": INPUT_KIND,
        "window": {
            "width": 80,
            "height": 60,
            "start_timestamp_us": 1_000_000,
            "end_timestamp_us": 1_000_000 + duration_us,
            "events": [
                [4, 9, 1_000_000, 0],
                [5, 10, 1_001_000, 1],
                [79, 59, 1_063_999, 1],
            ],
        },
    }


def test_frame_difference_bridge_preserves_off_on_and_time_bins() -> None:
    window = event_window_from_message(_message())
    bins = window_to_event_bins(window)
    assert tuple(bins.shape) == (1, 64, 2, 12, 16)
    assert bins[0, 0, 0, 1, 0].item() == 1.0
    assert bins[0, 1, 1, 2, 1].item() == 1.0
    assert bins[0, 63, 1, 11, 15].item() == 1.0
    assert bins.sum().item() == 3.0


def test_bridge_rejects_wrong_window_duration() -> None:
    message = _message(duration_us=63_000)
    message["window"]["events"] = message["window"]["events"][:2]  # type: ignore[index]
    with pytest.raises(ValueError, match="window duration"):
        window_to_event_bins(event_window_from_message(message))


def test_bridge_rejects_non_proxy_input() -> None:
    message = _message()
    message["input_kind"] = "raw_vendor_events"
    with pytest.raises(ValueError, match="frame_difference_proxy"):
        event_window_from_message(message)
