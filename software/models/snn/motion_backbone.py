"""Event-native SNN Motion Backbone v0.1."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import SNNMotionConfig
from .layers import ConvLIFBlock
from .neurons import FusedMultiTimescaleLIF, LIF


@dataclass(slots=True)
class SNNMotionOutput:
    """Named batched temporal outputs from the motion primitive representation."""

    primitive_spikes: Tensor
    local_logits: Tensor
    global_embedding: Tensor
    ego_motion: Tensor | None


@dataclass(frozen=True, slots=True)
class LayerSpikeStatistics:
    """Incrementally accumulated activity for one spiking layer."""

    total_spikes: int
    spikes_per_timestep: float
    spikes_per_neuron_per_timestep: float
    fast_spikes: int
    slow_spikes: int


@dataclass(frozen=True, slots=True)
class SNNMotionStatistics:
    """Architecture-level spike counts; no task performance is implied."""

    layers: dict[str, LayerSpikeStatistics]


@dataclass(slots=True)
class SNNMotionRun:
    """Normal model output accompanied by transient instrumentation results."""

    output: SNNMotionOutput
    statistics: SNNMotionStatistics


@dataclass(slots=True)
class _SpikeAccumulator:
    total_spikes: int = 0
    neuron_updates: int = 0
    fast_spikes: int = 0
    slow_spikes: int = 0

    def update(self, spikes: Tensor, fast_channels: int) -> None:
        detached = spikes.detach()
        self.total_spikes += int(torch.count_nonzero(detached).item())
        self.neuron_updates += detached.numel()
        self.fast_spikes += int(
            torch.count_nonzero(detached[:, :fast_channels]).item()
        )
        self.slow_spikes += int(
            torch.count_nonzero(detached[:, fast_channels:]).item()
        )


class SNNMotionBackbone(nn.Module):
    """Fully convolutional, stateful event-motion feature extractor."""

    _KERNELS = (5, 3, 3, 5, 5, 3)
    _STRIDES = (2, 1, 2, 1, 1, 1)
    _PADDINGS = (2, 1, 1, 2, 2, 1)
    # COMPATIBILITY_PATCH: torch 2.10.0+cu128 / CUDA 12.8 / RTX 4060 Laptop,
    # seed 20260814. This shape-specific schedule preserves the frozen B=1 Conv
    # results bitwise while still batching safe stages.
    _EXACT_TEMPORAL_BATCH_SIZES = (64, 8, 64, 32, 32, 64, 1)

    def __init__(
        self,
        config: SNNMotionConfig | None = None,
        *,
        lif_implementation: str = "reference",
        inference_fast_spike: bool = False,
        compiled_lif_mode: str = "none",
        first_step_specialization: bool = False,
        lif_step_primitive: str = "mul_add",
        execution_mode: str = "time_major",
    ) -> None:
        super().__init__()
        self.config = config or SNNMotionConfig()
        if lif_implementation not in {"reference", "fused"}:
            raise ValueError("lif_implementation must be 'reference' or 'fused'")
        self.lif_implementation = lif_implementation
        self.inference_fast_spike = inference_fast_spike
        self.compiled_lif_mode = compiled_lif_mode
        self.first_step_specialization = first_step_specialization
        self.lif_step_primitive = lif_step_primitive
        if execution_mode not in {"time_major", "stage_major", "stage_major_chunked"}:
            raise ValueError(
                "execution_mode must be 'time_major', 'stage_major', or 'stage_major_chunked'"
            )
        self.execution_mode = execution_mode

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
        self.fold_layer_gains(self.config.layer_gains)

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
            lif_implementation=self.lif_implementation,
            inference_fast_spike=self.inference_fast_spike,
            compiled_lif_mode=self.compiled_lif_mode,
            first_step_specialization=self.first_step_specialization,
            lif_step_primitive=self.lif_step_primitive,
        )

    def forward(self, event_bins: Tensor) -> SNNMotionOutput:
        """Process ``[B, T, C, H, W]`` OFF/ON bins without resetting state."""

        self._validate_event_bins(event_bins)
        if self.execution_mode in {"stage_major", "stage_major_chunked"}:
            self._validate_stage_major_inference(event_bins)
            return self._forward_stage_major(event_bins)
        return self._forward_time_major(event_bins)

    def _validate_event_bins(self, event_bins: Tensor) -> None:
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

    @staticmethod
    def _validate_stage_major_inference(event_bins: Tensor) -> None:
        if torch.is_grad_enabled():
            raise RuntimeError(
                "stage_major execution is inference-only; use torch.inference_mode()"
            )
        if event_bins.device.type != "cuda":
            raise RuntimeError("stage_major execution is CUDA-only")

    def _forward_time_major(self, event_bins: Tensor) -> SNNMotionOutput:
        """Trusted reference schedule: complete all stages one timestep at a time."""

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

    def _forward_stage_major(self, event_bins: Tensor) -> SNNMotionOutput:
        """Batch each stateless Conv over time while preserving sequential LIF state."""

        primitives = self._forward_stage_major_primitives(event_bins)

        local_steps: list[Tensor] = []
        embedding_steps: list[Tensor] = []
        ego_steps: list[Tensor] | None = [] if self.ego_motion_head is not None else None
        for time_index in range(event_bins.shape[1]):
            primitive = primitives[:, time_index]
            local_steps.append(self.local_motion_head(primitive))
            embedding = F.adaptive_avg_pool2d(primitive, output_size=(2, 2)).flatten(1)
            embedding_steps.append(embedding)
            if ego_steps is not None and self.ego_motion_head is not None:
                ego_steps.append(self.ego_motion_head(embedding))

        return SNNMotionOutput(
            primitive_spikes=primitives,
            local_logits=torch.stack(local_steps, dim=1),
            global_embedding=torch.stack(embedding_steps, dim=1),
            ego_motion=torch.stack(ego_steps, dim=1) if ego_steps is not None else None,
        )

    def _forward_stage_major_primitives(self, event_bins: Tensor) -> Tensor:
        """Return the unchanged O2 primitive sequence without running any heads."""

        spikes = event_bins
        blocks = (*self.stages, self.primitive_bottleneck)
        chunk_sizes = (
            self._EXACT_TEMPORAL_BATCH_SIZES
            if self.execution_mode == "stage_major_chunked"
            else (None,) * len(blocks)
        )
        for block, chunk_size in zip(blocks, chunk_sizes, strict=True):
            spikes = block.forward_sequence(spikes, temporal_batch_size=chunk_size)
        return spikes

    def forward_ego_motion(
        self,
        event_bins: Tensor,
        *,
        temporal_pool: bool = True,
        temporal_head: bool = False,
        mean_before_head: bool = False,
    ) -> Tensor:
        """Return the live bridge's normalized ``[B, 6]`` ego-motion output.

        This CUDA inference-only path skips the unused local head. Diagnostic
        flags expose O3 candidates; the exact production default batches only
        adaptive pooling and retains the trusted per-timestep Linear calls.
        """

        self._validate_event_bins(event_bins)
        if self.execution_mode != "stage_major_chunked":
            raise RuntimeError(
                "forward_ego_motion requires exact stage_major_chunked execution"
            )
        self._validate_stage_major_inference(event_bins)
        if self.ego_motion_head is None:
            raise RuntimeError("model does not have an ego-motion head")
        if mean_before_head and temporal_head:
            raise ValueError("mean_before_head and temporal_head are mutually exclusive")

        primitives = self._forward_stage_major_primitives(event_bins)
        batch, timesteps = primitives.shape[:2]
        if temporal_pool:
            merged = primitives.reshape(batch * timesteps, *primitives.shape[2:])
            pooled = F.adaptive_avg_pool2d(merged, output_size=(2, 2)).flatten(1)
            embeddings = pooled.reshape(batch, timesteps, -1)
        else:
            embeddings = torch.stack(
                [
                    F.adaptive_avg_pool2d(primitives[:, index], output_size=(2, 2))
                    .flatten(1)
                    for index in range(timesteps)
                ],
                dim=1,
            )

        if mean_before_head:
            return self.ego_motion_head(embeddings.mean(dim=1))
        if temporal_head:
            merged_embeddings = embeddings.reshape(batch * timesteps, -1)
            ego_sequence = self.ego_motion_head(merged_embeddings).reshape(
                batch, timesteps, 6
            )
        else:
            ego_sequence = torch.stack(
                [self.ego_motion_head(embeddings[:, index]) for index in range(timesteps)],
                dim=1,
            )
        return ego_sequence.mean(dim=1)

    def reset_state(self) -> None:
        """Clear every neuron state at a new-sequence boundary."""

        for neuron in self.modules():
            if isinstance(neuron, (LIF, FusedMultiTimescaleLIF)):
                neuron.reset_state()

    def forward_with_stats(self, event_bins: Tensor) -> SNNMotionRun:
        """Run normal forward while incrementally counting layer spikes.

        Temporary hooks retain only scalar counters and are removed before this
        method returns. The normal ``forward`` output and topology are unchanged.
        """

        layer_names = tuple(f"S{index}" for index in range(1, 7)) + ("primitive",)
        blocks = tuple(self.stages) + (self.primitive_bottleneck,)
        accumulators = {name: _SpikeAccumulator() for name in layer_names}
        handles: list[torch.utils.hooks.RemovableHandle] = []

        def make_hook(
            name: str, fast_channels: int
        ) -> Callable[[nn.Module, tuple[Tensor, ...], Tensor], None]:
            def hook(_module: nn.Module, _inputs: tuple[Tensor, ...], output: Tensor) -> None:
                accumulators[name].update(output, fast_channels)

            return hook

        try:
            for name, block in zip(layer_names, blocks, strict=True):
                handles.append(
                    block.neurons.register_forward_hook(
                        make_hook(name, block.neurons.fast_channels)
                    )
                )
            output = self.forward(event_bins)
        finally:
            for handle in handles:
                handle.remove()

        timesteps = event_bins.shape[1]
        statistics = {
            name: LayerSpikeStatistics(
                total_spikes=accumulator.total_spikes,
                spikes_per_timestep=accumulator.total_spikes / timesteps,
                spikes_per_neuron_per_timestep=(
                    accumulator.total_spikes / accumulator.neuron_updates
                    if accumulator.neuron_updates
                    else 0.0
                ),
                fast_spikes=accumulator.fast_spikes,
                slow_spikes=accumulator.slow_spikes,
            )
            for name, accumulator in accumulators.items()
        }
        return SNNMotionRun(output, SNNMotionStatistics(statistics))

    def detach_state(self) -> None:
        """Preserve membrane values while cutting the truncated-BPTT graph."""

        for neuron in self.modules():
            if isinstance(neuron, (LIF, FusedMultiTimescaleLIF)):
                neuron.detach_state()

    def membrane_states(self) -> tuple[Tensor, ...]:
        """Return initialized membrane states for diagnostics and tests."""

        return tuple(
            neuron.membrane_state
            for neuron in self.modules()
            if isinstance(neuron, (LIF, FusedMultiTimescaleLIF))
            and neuron.membrane_state is not None
        )

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def synaptic_convolutions(self) -> tuple[nn.Conv2d, ...]:
        """Return S1--S6 and primitive convolutions in frozen topology order."""

        return tuple(stage.conv for stage in self.stages) + (
            self.primitive_bottleneck.conv,
        )

    def fold_layer_gains(self, gains: tuple[float, ...]) -> None:
        """Multiply static gains into weights without adding forward operations."""

        convolutions = self.synaptic_convolutions()
        if len(gains) != len(convolutions):
            raise ValueError(f"expected {len(convolutions)} gains, got {len(gains)}")
        with torch.no_grad():
            for convolution, gain in zip(convolutions, gains, strict=True):
                if gain <= 0.0 or not math.isfinite(gain):
                    raise ValueError("all layer gains must be positive and finite")
                convolution.weight.mul_(gain)

    def forward_with_diagnostics(self, event_bins: Tensor):
        """Run sampled membrane/current diagnostics without changing ``forward``."""

        from .diagnostics import collect_snn_diagnostics

        return collect_snn_diagnostics(self, event_bins)
