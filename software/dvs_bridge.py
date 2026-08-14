"""Local-only bridge between a frame-derived DVS proxy and the EVIMO2 model.

The Glarus GUI currently exposes decoded frames, not an asynchronous vendor
event packet.  This module therefore accepts canonical event windows generated
from frame differences by the Windows-side adapter.  It deliberately labels
that input as a proxy and does not claim access to a raw DVS stream.
"""

from __future__ import annotations

import argparse
import json
import socketserver
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from software.datasets.evimo2_ego_motion import TargetNormalization
from software.event_processing.event_format import Event, EventWindow
from software.models.snn.config import SNNMotionConfig
from software.models.snn.motion_backbone import SNNMotionBackbone


PROTOCOL = "dvs-motion-bridge/v1"
INPUT_KIND = "frame_difference_proxy"
FROZEN_TIMESTEPS = 64
FROZEN_DT_US = 1_000
FROZEN_SPATIAL_REDUCTION = 5


def event_window_from_message(message: dict[str, Any]) -> EventWindow:
    """Validate one JSON transport message at the canonical event boundary."""

    if message.get("protocol") != PROTOCOL:
        raise ValueError(f"unsupported protocol: {message.get('protocol')!r}")
    if message.get("type") != "event_window":
        raise ValueError("message type must be 'event_window'")
    if message.get("input_kind") != INPUT_KIND:
        raise ValueError("bridge accepts only frame_difference_proxy input")
    payload = message.get("window")
    if not isinstance(payload, dict):
        raise TypeError("window must be an object")
    raw_events = payload.get("events")
    if not isinstance(raw_events, list):
        raise TypeError("window.events must be a list")
    if len(raw_events) > 200_000:
        raise ValueError("window exceeds the 200000-event transport limit")
    events: list[Event] = []
    for raw_event in raw_events:
        if not isinstance(raw_event, list) or len(raw_event) != 4:
            raise TypeError("each event must be [x, y, timestamp_us, polarity]")
        events.append(Event(*raw_event))
    return EventWindow.from_events(
        events,
        width=payload["width"],
        height=payload["height"],
        start_timestamp_us=payload["start_timestamp_us"],
        end_timestamp_us=payload["end_timestamp_us"],
    )


def window_to_event_bins(window: EventWindow) -> Tensor:
    """Bin canonical events into the frozen EVIMO2 model input layout.

    The input contains 64 one-millisecond bins with OFF/ON channel order and
    five-pixel spatial reduction, exactly as used by the existing baseline.
    """

    expected_duration = FROZEN_TIMESTEPS * FROZEN_DT_US
    duration = window.end_timestamp_us - window.start_timestamp_us
    if duration != expected_duration:
        raise ValueError(
            f"window duration must be {expected_duration} us, got {duration} us"
        )
    if window.width % FROZEN_SPATIAL_REDUCTION or window.height % FROZEN_SPATIAL_REDUCTION:
        raise ValueError("sensor dimensions must be divisible by spatial reduction 5")
    bins = torch.zeros(
        (
            1,
            FROZEN_TIMESTEPS,
            2,
            window.height // FROZEN_SPATIAL_REDUCTION,
            window.width // FROZEN_SPATIAL_REDUCTION,
        ),
        dtype=torch.float32,
    )
    for event in window.events:
        bin_index = min(
            FROZEN_TIMESTEPS - 1,
            (event.timestamp_us - window.start_timestamp_us) // FROZEN_DT_US,
        )
        bins[
            0,
            bin_index,
            event.polarity,
            event.y // FROZEN_SPATIAL_REDUCTION,
            event.x // FROZEN_SPATIAL_REDUCTION,
        ] += 1.0
    return bins


@dataclass(slots=True)
class LiveEgoMotionModel:
    """Read-only checkpoint adapter for advisory camera-local motion output."""

    model: SNNMotionBackbone
    normalizer: TargetNormalization
    device: torch.device
    model_id: str

    @classmethod
    def load(cls, checkpoint_path: str | Path, *, device: str = "auto") -> "LiveEgoMotionModel":
        resolved = Path(checkpoint_path).expanduser().resolve()
        resolved_device = torch.device(
            "cuda" if device == "auto" and torch.cuda.is_available() else device
        )
        checkpoint = torch.load(resolved, map_location=resolved_device, weights_only=True)
        if checkpoint.get("task") != "evimo2_samsung_camera_local_ego_motion":
            raise ValueError("checkpoint is not the EVIMO2 camera-local ego-motion baseline")
        model_cfg = checkpoint.get("config", {}).get("model", {})
        gains = tuple(float(value) for value in model_cfg["layer_gains"])
        model = SNNMotionBackbone(
            SNNMotionConfig(enable_ego_head=True, layer_gains=gains)
        ).to(resolved_device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return cls(
            model=model,
            normalizer=TargetNormalization.from_state_dict(checkpoint["target_normalization"]),
            device=resolved_device,
            model_id=f"evimo2-ego-motion:epoch-{checkpoint['epoch']}",
        )

    def predict(self, window: EventWindow) -> dict[str, object]:
        event_bins = window_to_event_bins(window).to(self.device, non_blocking=True)
        with torch.inference_mode():
            self.model.reset_state()
            output = self.model(event_bins)
            self.model.reset_state()
        if output.ego_motion is None:
            raise RuntimeError("loaded model does not have an ego-motion head")
        normalized = output.ego_motion.mean(dim=1)
        motion = self.normalizer.denormalize(normalized).squeeze(0).to("cpu")
        return {
            "protocol": PROTOCOL,
            "type": "camera_local_motion",
            "model_id": self.model_id,
            "timestamp_us": window.end_timestamp_us,
            "event_count": len(window.events),
            "ego_motion_vw": [float(value) for value in motion.tolist()],
            "advisory_only": True,
            "input_kind": INPUT_KIND,
            "warning": (
                "EVIMO2 Samsung-camera baseline; uncalibrated for Glarus and "
                "not a robot-motion, collision-risk, or safety output."
            ),
        }


class _BridgeRequestHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        for raw_line in self.rfile:
            if len(raw_line) > 16 * 1024 * 1024:
                self._write({"type": "error", "error": "message exceeds 16 MiB"})
                return
            try:
                message = json.loads(raw_line)
                if not isinstance(message, dict):
                    raise TypeError("message must be a JSON object")
                window = event_window_from_message(message)
                response = self.server.model.predict(window)  # type: ignore[attr-defined]
            except Exception as exc:
                response = {"protocol": PROTOCOL, "type": "error", "error": str(exc)}
            self._write(response)

    def _write(self, response: dict[str, object]) -> None:
        self.wfile.write(json.dumps(response, separators=(",", ":")).encode() + b"\n")
        self.wfile.flush()


class DvsMotionBridgeServer(socketserver.ThreadingTCPServer):
    """Loopback TCP server; intentionally independent of hardware SDKs."""

    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, address: tuple[str, int], model: LiveEgoMotionModel):
        super().__init__(address, _BridgeRequestHandler)
        self.model = model


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve the local DVS motion bridge.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()
    model = LiveEgoMotionModel.load(args.checkpoint, device=args.device)
    with DvsMotionBridgeServer((args.host, args.port), model) as server:
        print(
            f"DVS_MOTION_BRIDGE_READY host={args.host} port={args.port} "
            f"device={model.device} model={model.model_id}",
            flush=True,
        )
        server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
