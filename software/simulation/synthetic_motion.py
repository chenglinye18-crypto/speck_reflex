"""Deterministic geometric event patterns for architecture sanity checks."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from enum import Enum

import torch
from torch import Tensor


class SyntheticMotionCase(str, Enum):
    NO_MOTION = "NO_MOTION"
    GLOBAL_LEFT = "GLOBAL_LEFT"
    GLOBAL_RIGHT = "GLOBAL_RIGHT"
    GLOBAL_UP = "GLOBAL_UP"
    GLOBAL_DOWN = "GLOBAL_DOWN"
    STATIC_BG_MOVING_OBJECT = "STATIC_BG_MOVING_OBJECT"
    MOVING_BG_MOVING_OBJECT = "MOVING_BG_MOVING_OBJECT"
    EXPANSION = "EXPANSION"
    CONTRACTION = "CONTRACTION"


@dataclass(frozen=True, slots=True)
class SyntheticMotionConfig:
    height: int = 96
    width: int = 128
    timesteps: int = 32
    dt_ms: float = 1.0
    motion_start_ms: float = 10.0
    velocity_px_per_ms: float = 0.5
    object_size: tuple[int, int] = (16, 20)
    batch_size: int = 1
    seed: int = 17

    def __post_init__(self) -> None:
        if self.height < 8 or self.width < 8:
            raise ValueError("height and width must each be at least 8")
        if self.timesteps < 2 or self.batch_size < 1:
            raise ValueError("timesteps must be at least 2 and batch_size positive")
        if self.dt_ms <= 0.0 or self.velocity_px_per_ms <= 0.0:
            raise ValueError("dt_ms and velocity_px_per_ms must be positive")
        if self.motion_start_ms < 0.0:
            raise ValueError("motion_start_ms must be non-negative")
        object_height, object_width = self.object_size
        if not 2 <= object_height < self.height or not 2 <= object_width < self.width:
            raise ValueError("object_size must fit strictly inside the image")


@dataclass(frozen=True, slots=True)
class SyntheticMotionMetadata:
    motion_type: str
    global_velocity: tuple[float, float]
    object_velocity: tuple[float, float] | None
    dt_ms: float
    seed: int
    polarity_order: tuple[str, str] = ("OFF", "ON")


@dataclass(slots=True)
class SyntheticMotionSample:
    events: Tensor
    independent_motion_mask: Tensor
    metadata: SyntheticMotionMetadata


def _round_displacement(value: float) -> int:
    """Round half away from zero for a stable subpixel accumulator."""

    return math.floor(value + 0.5) if value >= 0.0 else math.ceil(value - 0.5)


class SyntheticMotionGenerator:
    """Render binary geometry, then convert exact frame changes to OFF/ON bins."""

    def __init__(self, config: SyntheticMotionConfig | None = None) -> None:
        self.config = config or SyntheticMotionConfig()

    def generate(self, motion_case: SyntheticMotionCase | str) -> SyntheticMotionSample:
        case = SyntheticMotionCase(motion_case)
        global_velocity, object_velocity = self._velocities(case)
        frames: list[Tensor] = []
        masks: list[Tensor] = []
        for time_index in range(self.config.timesteps):
            time_ms = time_index * self.config.dt_ms
            frame, mask = self._render(case, time_ms, global_velocity, object_velocity)
            frames.append(frame)
            masks.append(mask)

        event_steps = [torch.zeros(2, self.config.height, self.config.width)]
        for previous, current in zip(frames[:-1], frames[1:], strict=True):
            off_events = (current < previous).to(torch.float32)
            on_events = (current > previous).to(torch.float32)
            event_steps.append(torch.stack((off_events, on_events), dim=0))

        events = torch.stack(event_steps, dim=0).unsqueeze(0)
        independent_mask = torch.stack(masks, dim=0).unsqueeze(0).unsqueeze(2)
        if self.config.batch_size > 1:
            events = events.repeat(self.config.batch_size, 1, 1, 1, 1)
            independent_mask = independent_mask.repeat(
                self.config.batch_size, 1, 1, 1, 1
            )

        return SyntheticMotionSample(
            events=events,
            independent_motion_mask=independent_mask,
            metadata=SyntheticMotionMetadata(
                motion_type=case.value,
                global_velocity=global_velocity,
                object_velocity=object_velocity,
                dt_ms=self.config.dt_ms,
                seed=self.config.seed,
            ),
        )

    def _base_pattern(self) -> Tensor:
        rng = random.Random(self.config.seed)
        spacing = 12
        x_offset = rng.randrange(spacing)
        y_offset = rng.randrange(spacing)
        y = torch.arange(self.config.height).view(-1, 1)
        x = torch.arange(self.config.width).view(1, -1)
        vertical_bands = torch.div(x + x_offset, spacing, rounding_mode="floor")
        horizontal_bands = torch.div(y + y_offset, spacing, rounding_mode="floor")
        return ((vertical_bands + horizontal_bands) % 2).to(torch.float32)

    def _elapsed(self, time_ms: float) -> float:
        return max(0.0, time_ms - self.config.motion_start_ms)

    def _shift(self, frame: Tensor, velocity: tuple[float, float], elapsed_ms: float) -> Tensor:
        dx = _round_displacement(velocity[0] * elapsed_ms)
        dy = _round_displacement(velocity[1] * elapsed_ms)
        return torch.roll(frame, shifts=(dy, dx), dims=(0, 1))

    def _velocities(
        self, case: SyntheticMotionCase
    ) -> tuple[tuple[float, float], tuple[float, float] | None]:
        speed = self.config.velocity_px_per_ms
        directions = {
            SyntheticMotionCase.GLOBAL_LEFT: (-speed, 0.0),
            SyntheticMotionCase.GLOBAL_RIGHT: (speed, 0.0),
            SyntheticMotionCase.GLOBAL_UP: (0.0, -speed),
            SyntheticMotionCase.GLOBAL_DOWN: (0.0, speed),
        }
        if case in directions:
            return directions[case], None
        if case is SyntheticMotionCase.STATIC_BG_MOVING_OBJECT:
            return (0.0, 0.0), (speed, 0.0)
        if case is SyntheticMotionCase.MOVING_BG_MOVING_OBJECT:
            # Background moves left while the object moves right relative to it.
            return (-speed, 0.0), (2.0 * speed, 0.0)
        return (0.0, 0.0), None

    def _render(
        self,
        case: SyntheticMotionCase,
        time_ms: float,
        global_velocity: tuple[float, float],
        object_velocity: tuple[float, float] | None,
    ) -> tuple[Tensor, Tensor]:
        if case is SyntheticMotionCase.NO_MOTION:
            zeros = torch.zeros(self.config.height, self.config.width)
            return zeros, torch.zeros_like(zeros, dtype=torch.bool)
        if case in (SyntheticMotionCase.EXPANSION, SyntheticMotionCase.CONTRACTION):
            return self._render_radial(case, time_ms)

        elapsed = self._elapsed(time_ms)
        background = self._shift(self._base_pattern(), global_velocity, elapsed)
        mask = torch.zeros_like(background, dtype=torch.bool)
        if object_velocity is None:
            return background, mask

        # Object apparent motion is global motion plus its relative velocity.
        apparent_velocity = (
            global_velocity[0] + object_velocity[0],
            global_velocity[1] + object_velocity[1],
        )
        rng = random.Random(self.config.seed)
        initial_x = self.config.width // 3 + rng.randint(-2, 2)
        initial_y = self.config.height // 2 + rng.randint(-2, 2)
        center_x = initial_x + _round_displacement(apparent_velocity[0] * elapsed)
        center_y = initial_y + _round_displacement(apparent_velocity[1] * elapsed)
        object_height, object_width = self.config.object_size
        x0 = max(0, min(self.config.width - object_width, center_x - object_width // 2))
        y0 = max(0, min(self.config.height - object_height, center_y - object_height // 2))
        mask[y0 : y0 + object_height, x0 : x0 + object_width] = True
        frame = background.clone()
        frame[mask] = 1.0 - frame[mask]
        return frame, mask

    def _render_radial(
        self, case: SyntheticMotionCase, time_ms: float
    ) -> tuple[Tensor, Tensor]:
        elapsed = self._elapsed(time_ms)
        delta = _round_displacement(self.config.velocity_px_per_ms * elapsed)
        object_height, object_width = self.config.object_size
        if case is SyntheticMotionCase.EXPANSION:
            half_height = max(1, object_height // 4 + delta)
            half_width = max(1, object_width // 4 + delta)
        else:
            half_height = max(1, object_height // 2 - delta)
            half_width = max(1, object_width // 2 - delta)
        half_height = min(half_height, self.config.height // 2 - 1)
        half_width = min(half_width, self.config.width // 2 - 1)
        center_y, center_x = self.config.height // 2, self.config.width // 2
        frame = torch.zeros(self.config.height, self.config.width)
        frame[
            center_y - half_height : center_y + half_height,
            center_x - half_width : center_x + half_width,
        ] = 1.0
        return frame, torch.zeros_like(frame, dtype=torch.bool)
