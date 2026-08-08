"""Concise read-only smoke command for one EVIMO2 ego-motion sample."""

from __future__ import annotations

import argparse
from pathlib import Path

from .evimo2 import EVIMO2EventConfig, EVIMO2Sensor, EVIMO2Sequence


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect one read-only EVIMO2 event/ego-motion sample."
    )
    parser.add_argument("sequence", type=Path)
    parser.add_argument(
        "--sensor",
        choices=[sensor.value for sensor in EVIMO2Sensor],
        required=True,
    )
    parser.add_argument("--frame-id", type=int, required=True)
    parser.add_argument("--timesteps", type=int, default=32)
    parser.add_argument("--dt-ms", type=float, default=1.0)
    args = parser.parse_args()
    sequence = EVIMO2Sequence(
        args.sequence,
        sensor=args.sensor,
        config=EVIMO2EventConfig(timesteps=args.timesteps, dt_ms=args.dt_ms),
    )
    sample = sequence.sample_at_frame(args.frame_id)
    print(
        f"sequence={sample.sequence} sensor={sample.sensor.value} "
        f"frame_id={sample.frame_id}"
    )
    print(
        f"window=[{sample.start_time_s:.6f}, {sample.end_time_s:.6f}) "
        f"events={sample.event_count}"
    )
    print(
        f"input_shape={tuple(sample.events.shape)} "
        f"input_sum={int(sample.events.sum().item())}"
    )
    print(f"ego_motion_vw={sample.ego_motion.tolist()}")


if __name__ == "__main__":
    main()
