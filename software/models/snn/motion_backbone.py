"""Event-native SNN Motion Backbone v0.1."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import SNNMotionConfig
from .layers import ConvLIFBlock
from .neurons import LIF


@dataclass(slots=True)
class SNNMotionOutput:
    """Named batched temporal outputs from the motion primitive representation."""

    primitive_spikes: Tensor
    local_logits: Tensor
    global_embedding: Tensor
    ego_motion: Tensor | None


class SNNMotionBackbone(nn.Module):
    """Fully convolutional, stateful event-motion feature extractor."""

    _KERNELS = (5, 3, 3, 5, 5, 3)
    _STRIDES = (2, 1, 2, 1, 1, 1)
    _PADDINGS = (2, 1, 1, 2, 2, 1)

    def __init__(self, config: SNNMotionConfig | None = None) -> None:
        super().__init__()
        self.config = config or SNNMotionConfig()

        stage_inputs = (self.config.input_channels, *self.config.channels[:-1])
        self.stages = nn.ModuleList(
            self._make_block(in_channels, out_channels, kernel, stride, padding)
            for in_channels, out_channels, kernel, stride, padding in zip(
                stage_inputs,
                self.config.channels,
                self._KERNELS,
                self._STRIDES,
                self._PADDINGS,
                strict=True,
            )
        )
        self.primitive_bottleneck = self._make_block(
            self.config.channels[-1],
            self.config.primitive_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.local_motion_head = nn.Conv2d(
            self.config.primitive_channels, 2, kernel_size=1, bias=False
        )
        embedding_features = self.config.primitive_channels * 2 * 2
        self.ego_motion_head = (
            nn.Linear(embedding_features, 6)
            if self.config.enable_ego_head
            else None
        )

    def _make_block(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
    ) -> ConvLIFBlock:
        return ConvLIFBlock(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            tau_fast_ms=self.config.tau_fast_ms,
            tau_slow_ms=self.config.tau_slow_ms,
            dt_ms=self.config.dt_ms,
            threshold=self.config.threshold,
            fast_ratio=self.config.fast_ratio,
            surrogate=self.config.surrogate,
        )

    def forward(self, event_bins: Tensor) -> SNNMotionOutput:
        """Process ``[B, T, C, H, W]`` OFF/ON bins without resetting state."""

        if event_bins.ndim != 5:
            raise ValueError("event_bins must have shape [B, T, C, H, W]")
        if event_bins.shape[2] != self.config.input_channels:
            raise ValueError(
                f"expected {self.config.input_channels} input channels in OFF/ON order, "
                f"got {event_bins.shape[2]}"
            )
        if event_bins.shape[1] < 1:
            raise ValueError("event_bins must contain at least one time step")
        if not event_bins.is_floating_point():
            raise TypeError("event_bins must be a floating-point count or occupancy tensor")

        primitive_steps: list[Tensor] = []
        local_steps: list[Tensor] = []
        embedding_steps: list[Tensor] = []
        ego_steps: list[Tensor] | None = [] if self.ego_motion_head is not None else None

        for time_index in range(event_bins.shape[1]):
            spikes = event_bins[:, time_index]
            for stage in self.stages:
                spikes = stage(spikes)
            primitive = self.primitive_bottleneck(spikes)
            local_logits = self.local_motion_head(primitive)
            embedding = F.adaptive_avg_pool2d(primitive, output_size=(2, 2)).flatten(1)

            primitive_steps.append(primitive)
            local_steps.append(local_logits)
            embedding_steps.append(embedding)
            if ego_steps is not None and self.ego_motion_head is not None:
                ego_steps.append(self.ego_motion_head(embedding))

        return SNNMotionOutput(
            primitive_spikes=torch.stack(primitive_steps, dim=1),
            local_logits=torch.stack(local_steps, dim=1),
            global_embedding=torch.stack(embedding_steps, dim=1),
            ego_motion=torch.stack(ego_steps, dim=1) if ego_steps is not None else None,
        )

    def reset_state(self) -> None:
        """Clear every neuron state at a new-sequence boundary."""

        for neuron in self.modules():
            if isinstance(neuron, LIF):
                neuron.reset_state()

    def detach_state(self) -> None:
        """Preserve membrane values while cutting the truncated-BPTT graph."""

        for neuron in self.modules():
            if isinstance(neuron, LIF):
                neuron.detach_state()

    def membrane_states(self) -> tuple[Tensor, ...]:
        """Return initialized membrane states for diagnostics and tests."""

        return tuple(
            neuron.membrane_state
            for neuron in self.modules()
            if isinstance(neuron, LIF) and neuron.membrane_state is not None
        )

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
