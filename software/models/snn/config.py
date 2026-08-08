"""Configuration for the hardware-neutral SNN motion backbone."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SNNMotionConfig:
    """Frozen topology and neuron parameters for SNN Motion Backbone v0.1."""

    input_channels: int = 2
    channels: tuple[int, ...] = (12, 16, 24, 24, 32, 32)
    primitive_channels: int = 16
    tau_fast_ms: float = 10.0
    tau_slow_ms: float = 40.0
    threshold: float = 1.0
    dt_ms: float = 1.0
    fast_ratio: float = 0.5
    enable_ego_head: bool = True
    surrogate: str = "atan"
    initialization: str = "pytorch_default_kaiming_uniform_a_sqrt5"
    layer_gains: tuple[float, ...] = (2.0, 4.0, 4.0, 4.0, 4.0, 4.0, 2.0)

    def __post_init__(self) -> None:
        if self.input_channels <= 0:
            raise ValueError("input_channels must be positive")
        if len(self.channels) != 6 or any(channels < 2 for channels in self.channels):
            raise ValueError("channels must contain six entries, each at least 2")
        if self.primitive_channels < 2:
            raise ValueError("primitive_channels must be at least 2")
        if self.tau_fast_ms <= 0.0 or self.tau_slow_ms <= 0.0:
            raise ValueError("LIF time constants must be positive")
        if self.dt_ms <= 0.0:
            raise ValueError("dt_ms must be positive")
        if self.threshold <= 0.0:
            raise ValueError("threshold must be positive")
        if not 0.0 < self.fast_ratio < 1.0:
            raise ValueError("fast_ratio must lie strictly between 0 and 1")
        if self.surrogate != "atan":
            raise ValueError("SNN Motion Backbone v0.1 supports only the 'atan' surrogate")
        if self.initialization != "pytorch_default_kaiming_uniform_a_sqrt5":
            raise ValueError("unsupported initialization")
        if len(self.layer_gains) != 7 or any(gain <= 0.0 for gain in self.layer_gains):
            raise ValueError("layer_gains must contain seven positive values")
