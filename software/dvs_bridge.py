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
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

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
FROZEN_SENSOR_WIDTH = 640
FROZEN_SENSOR_HEIGHT = 480


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
    if (window.width, window.height) != (FROZEN_SENSOR_WIDTH, FROZEN_SENSOR_HEIGHT):
        raise ValueError(
            "sensor dimensions must be 640x480 for the frozen bridge contract, "
            f"got {window.width}x{window.height}"
        )
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


def _motion_response(
    window: EventWindow,
    *,
    model_id: str,
    values: list[float],
    tensor: Tensor,
    diagnostics: dict[str, object],
) -> dict[str, object]:
    return {
        "protocol": PROTOCOL,
        "type": "camera_local_motion",
        "model_id": model_id,
        "timestamp_us": window.end_timestamp_us,
        "event_count": len(window.events),
        "ego_motion_vw": values,
        "advisory_only": True,
        "input_kind": INPUT_KIND,
        "server_diagnostics": {
            "tensor_shape": list(tensor.shape),
            "tensor_dtype": str(tensor.dtype),
            **diagnostics,
        },
    }


class BridgePredictor(Protocol):
    model_id: str
    device: object

    def predict(self, window: EventWindow) -> dict[str, object]: ...


@dataclass(slots=True)
class DummyEgoMotionModel:
    """Interface-only predictor that validates and bins events without a model."""

    model_id: str = "dummy-interface-v0"
    device: str = "dummy"

    def predict(self, window: EventWindow) -> dict[str, object]:
        started = time.monotonic()
        event_bins = window_to_event_bins(window)
        elapsed_ms = (time.monotonic() - started) * 1_000.0
        return _motion_response(
            window,
            model_id=self.model_id,
            values=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            tensor=event_bins,
            diagnostics={"model_inference_ms": elapsed_ms},
        )


@dataclass(slots=True)
class LiveEgoMotionModel:
    """Read-only checkpoint adapter for advisory camera-local motion output."""

    model: SNNMotionBackbone
    normalizer: TargetNormalization
    device: torch.device
    model_id: str

    @property
    def runtime_backend(self) -> dict[str, object]:
        return {
            "lif_backend": getattr(self.model, "lif_implementation", "unknown"),
            "execution_mode": self.model.execution_mode,
            "inference_fast_spike": getattr(self.model, "inference_fast_spike", False),
            "ego_only_path": True,
            "batched_pool": True,
        }

    @classmethod
    def load(cls, checkpoint_path: str | Path, *, device: str = "auto") -> "LiveEgoMotionModel":
        resolved = Path(checkpoint_path).expanduser().resolve()
        resolved_device = torch.device(
            "cuda" if device == "auto" and torch.cuda.is_available()
            else "cpu" if device == "auto"
            else device
        )
        checkpoint = torch.load(resolved, map_location=resolved_device, weights_only=True)
        if checkpoint.get("task") != "evimo2_samsung_camera_local_ego_motion":
            raise ValueError("checkpoint is not the EVIMO2 camera-local ego-motion baseline")
        model_cfg = checkpoint.get("config", {}).get("model", {})
        gains = tuple(float(value) for value in model_cfg["layer_gains"])
        optimized_cuda = resolved_device.type == "cuda"
        model = SNNMotionBackbone(
            SNNMotionConfig(enable_ego_head=True, layer_gains=gains),
            lif_implementation="fused" if optimized_cuda else "reference",
            inference_fast_spike=optimized_cuda,
            execution_mode="stage_major_chunked" if optimized_cuda else "time_major",
        ).to(resolved_device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval()
        return cls(
            model=model,
            normalizer=TargetNormalization.from_state_dict(checkpoint["target_normalization"]),
            device=resolved_device,
            model_id=f"evimo2-ego-motion:epoch-{checkpoint['epoch']}",
        )

    def predict(self, window: EventWindow) -> dict[str, object]:
        binning_started = time.monotonic()
        event_bins_cpu = window_to_event_bins(window)
        event_binning_ms = (time.monotonic() - binning_started) * 1_000.0
        h2d_ms = 0.0
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            h2d_start = torch.cuda.Event(enable_timing=True)
            h2d_end = torch.cuda.Event(enable_timing=True)
            h2d_start.record()
            event_bins = event_bins_cpu.to(self.device, non_blocking=True)
            h2d_end.record()
            h2d_end.synchronize()
            h2d_ms = h2d_start.elapsed_time(h2d_end)
            inference_start = torch.cuda.Event(enable_timing=True)
            inference_end = torch.cuda.Event(enable_timing=True)
            inference_start.record()
        else:
            event_bins = event_bins_cpu.to(self.device)
            started = time.monotonic()
        with torch.inference_mode():
            self.model.reset_state()
            if self.model.execution_mode == "stage_major_chunked":
                normalized = self.model.forward_ego_motion(event_bins)
            else:
                output = self.model(event_bins)
                if output.ego_motion is None:
                    raise RuntimeError("loaded model does not have an ego-motion head")
                normalized = output.ego_motion.mean(dim=1)
            self.model.reset_state()
        if self.device.type == "cuda":
            inference_end.record()
            inference_end.synchronize()
            inference_latency_ms = inference_start.elapsed_time(inference_end)
        else:
            inference_latency_ms = (time.monotonic() - started) * 1_000.0
        motion = self.normalizer.denormalize(normalized).squeeze(0).to("cpu")
        if not torch.isfinite(motion).all():
            raise RuntimeError("model produced non-finite ego-motion output")
        response = _motion_response(
            window,
            model_id=self.model_id,
            values=[float(value) for value in motion.tolist()],
            tensor=event_bins,
            diagnostics={
                "event_binning_ms": event_binning_ms,
                "h2d_ms": h2d_ms,
                "model_inference_ms": inference_latency_ms,
                "runtime_backend": self.runtime_backend,
            },
        )
        response["warning"] = (
            "EVIMO2 Samsung-camera baseline; uncalibrated for Glarus and "
            "not a robot-motion, collision-risk, or safety output."
        )
        return response


@dataclass(slots=True)
class BridgeServerDiagnostics:
    requests_received: int = 0
    responses_sent: int = 0
    _last_log_monotonic: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def begin_request(self) -> int:
        with self._lock:
            self.requests_received += 1
            return self.requests_received

    def finish_request(
        self,
        response: dict[str, object],
        *,
        request_number: int,
        processing_latency_ms: float,
        response_serialization_ms: float,
    ) -> None:
        diagnostics = response.setdefault("server_diagnostics", {})
        if isinstance(diagnostics, dict):
            diagnostics["requests_received"] = request_number
            diagnostics["responses_sent"] = self.responses_sent
            diagnostics["processing_latency_ms"] = processing_latency_ms
            diagnostics["response_serialization_ms"] = response_serialization_ms
            diagnostics["response_type"] = response.get("type")
        now = time.monotonic()
        with self._lock:
            if now - self._last_log_monotonic < 1.0:
                return
            self._last_log_monotonic = now
        event_count = response.get("event_count", "--")
        tensor_shape = diagnostics.get("tensor_shape", "--") if isinstance(diagnostics, dict) else "--"
        inference_ms = (
            diagnostics.get("model_inference_ms", "--")
            if isinstance(diagnostics, dict)
            else "--"
        )
        print(
            "bridge_stats "
            f"requests_received={request_number} event_count={event_count} "
            f"tensor_shape={tensor_shape} model_inference_ms={inference_ms} "
            f"response_type={response.get('type')}",
            flush=True,
        )


class _BridgeRequestHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        for raw_line in self.rfile:
            started = time.monotonic()
            request_number = self.server.diagnostics.begin_request()  # type: ignore[attr-defined]
            if len(raw_line) > 16 * 1024 * 1024:
                response = {
                    "protocol": PROTOCOL,
                    "type": "error",
                    "error": "message exceeds 16 MiB",
                }
                serialization_started = time.monotonic()
                json.dumps(response, separators=(",", ":"))
                self.server.diagnostics.finish_request(  # type: ignore[attr-defined]
                    response,
                    request_number=request_number,
                    processing_latency_ms=(time.monotonic() - started) * 1_000.0,
                    response_serialization_ms=(time.monotonic() - serialization_started) * 1_000.0,
                )
                self._write(response)
                with self.server.diagnostics._lock:  # type: ignore[attr-defined]
                    self.server.diagnostics.responses_sent += 1  # type: ignore[attr-defined]
                return
            try:
                message = json.loads(raw_line)
                if not isinstance(message, dict):
                    raise TypeError("message must be a JSON object")
                window = event_window_from_message(message)
                with self.server.inference_lock:  # type: ignore[attr-defined]
                    response = self.server.model.predict(window)  # type: ignore[attr-defined]
            except Exception as exc:
                response = {"protocol": PROTOCOL, "type": "error", "error": str(exc)}
            serialization_started = time.monotonic()
            json.dumps(response, separators=(",", ":"))
            self.server.diagnostics.finish_request(  # type: ignore[attr-defined]
                response,
                request_number=request_number,
                processing_latency_ms=(time.monotonic() - started) * 1_000.0,
                response_serialization_ms=(time.monotonic() - serialization_started) * 1_000.0,
            )
            self._write(response)
            with self.server.diagnostics._lock:  # type: ignore[attr-defined]
                self.server.diagnostics.responses_sent += 1  # type: ignore[attr-defined]

    def _write(self, response: dict[str, object]) -> None:
        self.wfile.write(json.dumps(response, separators=(",", ":")).encode() + b"\n")
        self.wfile.flush()


class DvsMotionBridgeServer(socketserver.ThreadingTCPServer):
    """Loopback TCP server; intentionally independent of hardware SDKs."""

    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, address: tuple[str, int], model: BridgePredictor):
        super().__init__(address, _BridgeRequestHandler)
        self.model = model
        self.inference_lock = threading.Lock()
        self.diagnostics = BridgeServerDiagnostics()


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve the local DVS motion bridge.")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--dummy", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()
    if args.dummy and args.checkpoint is not None:
        parser.error("--dummy and --checkpoint are mutually exclusive")
    if not args.dummy and args.checkpoint is None:
        parser.error("--checkpoint is required unless --dummy is used")
    model: BridgePredictor = (
        DummyEgoMotionModel()
        if args.dummy
        else LiveEgoMotionModel.load(args.checkpoint, device=args.device)
    )
    with DvsMotionBridgeServer((args.host, args.port), model) as server:
        backend = (
            model.runtime_backend if isinstance(model, LiveEgoMotionModel)
            else {"lif_backend": "dummy", "execution_mode": "dummy"}
        )
        print(
            f"DVS_MOTION_BRIDGE_READY host={args.host} port={args.port} "
            f"device={model.device} model={model.model_id} backend={backend}",
            flush=True,
        )
        server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
