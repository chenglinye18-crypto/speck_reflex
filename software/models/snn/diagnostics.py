"""Sampled numerical diagnostics for SNN excitability calibration."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .motion_backbone import SNNMotionOutput, SNNMotionStatistics

if TYPE_CHECKING:
    from .motion_backbone import SNNMotionBackbone


@dataclass(frozen=True, slots=True)
class DistributionStatistics:
    mean: float
    std: float
    mean_abs: float
    abs_max: float
    p50: float
    p90: float
    p95: float
    p99: float
    p99_9: float


@dataclass(frozen=True, slots=True)
class MembraneStatistics:
    distribution: DistributionStatistics
    ratio_to_threshold: DistributionStatistics
    fraction_above_half_threshold: float
    fraction_above_0_8_threshold: float
    fraction_at_or_above_threshold: float


@dataclass(frozen=True, slots=True)
class SignedCurrentDiagnostics:
    positive: DistributionStatistics
    negative: DistributionStatistics
    net: DistributionStatistics
    cancellation_fraction: float


@dataclass(frozen=True, slots=True)
class LayerNumericalDiagnostics:
    synaptic_current: DistributionStatistics
    membrane: MembraneStatistics
    fast_membrane: MembraneStatistics
    slow_membrane: MembraneStatistics
    fast_firing_fraction: float
    slow_firing_fraction: float
    signed_current: SignedCurrentDiagnostics | None


@dataclass(frozen=True, slots=True)
class SNNNumericalDiagnostics:
    layers: dict[str, LayerNumericalDiagnostics]


@dataclass(slots=True)
class SNNDiagnosticRun:
    output: SNNMotionOutput
    spike_statistics: SNNMotionStatistics
    numerical_diagnostics: SNNNumericalDiagnostics


@dataclass(slots=True)
class _DistributionAccumulator:
    max_samples: int = 65_536
    expected_updates: int = 32
    count: int = 0
    total: float = 0.0
    total_sq: float = 0.0
    total_abs: float = 0.0
    abs_max: float = 0.0
    samples: list[Tensor] = field(default_factory=list)

    def update(self, values: Tensor) -> None:
        flat = values.detach().to(dtype=torch.float32).reshape(-1)
        if flat.numel() == 0:
            return
        self.count += flat.numel()
        self.total += float(flat.sum().item())
        self.total_sq += float(flat.square().sum().item())
        self.total_abs += float(flat.abs().sum().item())
        self.abs_max = max(self.abs_max, float(flat.abs().max().item()))

        per_update = max(1, self.max_samples // max(1, self.expected_updates))
        remaining = self.max_samples - sum(sample.numel() for sample in self.samples)
        take = min(per_update, remaining)
        if take > 0:
            stride = max(1, math.ceil(flat.numel() / take))
            self.samples.append(flat[::stride][:take].cpu())

    def finalize(self) -> DistributionStatistics:
        if self.count == 0:
            return DistributionStatistics(*(0.0 for _ in range(9)))
        mean = self.total / self.count
        variance = max(0.0, self.total_sq / self.count - mean * mean)
        sample = torch.cat(self.samples) if self.samples else torch.zeros(1)
        quantiles = torch.quantile(
            sample, torch.tensor((0.5, 0.9, 0.95, 0.99, 0.999))
        ).tolist()
        return DistributionStatistics(
            mean=mean,
            std=math.sqrt(variance),
            mean_abs=self.total_abs / self.count,
            abs_max=self.abs_max,
            p50=float(quantiles[0]),
            p90=float(quantiles[1]),
            p95=float(quantiles[2]),
            p99=float(quantiles[3]),
            p99_9=float(quantiles[4]),
        )


@dataclass(slots=True)
class _MembraneAccumulator:
    threshold: float
    distribution: _DistributionAccumulator
    above_half: int = 0
    above_0_8: int = 0
    at_or_above: int = 0

    def update(self, membrane: Tensor) -> None:
        detached = membrane.detach()
        self.distribution.update(detached)
        self.above_half += int(
            torch.count_nonzero(detached > 0.5 * self.threshold).item()
        )
        self.above_0_8 += int(
            torch.count_nonzero(detached > 0.8 * self.threshold).item()
        )
        self.at_or_above += int(
            torch.count_nonzero(detached >= self.threshold).item()
        )

    def finalize(self) -> MembraneStatistics:
        distribution = self.distribution.finalize()
        count = self.distribution.count
        return MembraneStatistics(
            distribution=distribution,
            ratio_to_threshold=_scale_distribution(distribution, self.threshold),
            fraction_above_half_threshold=self.above_half / count if count else 0.0,
            fraction_above_0_8_threshold=self.above_0_8 / count if count else 0.0,
            fraction_at_or_above_threshold=self.at_or_above / count if count else 0.0,
        )


@dataclass(slots=True)
class _LayerAccumulator:
    synaptic: _DistributionAccumulator
    membrane: _MembraneAccumulator
    fast_membrane: _MembraneAccumulator
    slow_membrane: _MembraneAccumulator
    positive: _DistributionAccumulator | None = None
    negative: _DistributionAccumulator | None = None


def _scale_distribution(
    statistics: DistributionStatistics, scale: float
) -> DistributionStatistics:
    return DistributionStatistics(
        mean=statistics.mean / scale,
        std=statistics.std / scale,
        mean_abs=statistics.mean_abs / scale,
        abs_max=statistics.abs_max / scale,
        p50=statistics.p50 / scale,
        p90=statistics.p90 / scale,
        p95=statistics.p95 / scale,
        p99=statistics.p99 / scale,
        p99_9=statistics.p99_9 / scale,
    )


def _signed_current(
    accumulator: _LayerAccumulator,
) -> SignedCurrentDiagnostics | None:
    if accumulator.positive is None or accumulator.negative is None:
        return None
    positive = accumulator.positive.finalize()
    negative = accumulator.negative.finalize()
    net = accumulator.synaptic.finalize()
    gross = positive.mean_abs + negative.mean_abs
    cancellation = 1.0 - net.mean_abs / gross if gross > 0.0 else 0.0
    return SignedCurrentDiagnostics(
        positive=positive,
        negative=negative,
        net=net,
        cancellation_fraction=max(0.0, min(1.0, cancellation)),
    )


def collect_snn_diagnostics(
    model: SNNMotionBackbone, event_bins: Tensor
) -> SNNDiagnosticRun:
    """Collect scalar/quantile summaries with bounded deterministic sampling."""

    names = tuple(f"S{index}" for index in range(1, 7)) + ("primitive",)
    blocks = tuple(model.stages) + (model.primitive_bottleneck,)
    threshold = model.config.threshold
    expected_updates = event_bins.shape[1]

    def distribution() -> _DistributionAccumulator:
        return _DistributionAccumulator(expected_updates=expected_updates)

    accumulators = {
        name: _LayerAccumulator(
            synaptic=distribution(),
            membrane=_MembraneAccumulator(threshold, distribution()),
            fast_membrane=_MembraneAccumulator(threshold, distribution()),
            slow_membrane=_MembraneAccumulator(threshold, distribution()),
            positive=distribution() if name in ("S1", "S2") else None,
            negative=distribution() if name in ("S1", "S2") else None,
        )
        for name in names
    }
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def make_conv_hook(name: str, convolution: nn.Conv2d):
        def hook(_module: nn.Module, inputs: tuple[Tensor, ...], output: Tensor) -> None:
            layer = accumulators[name]
            layer.synaptic.update(output)
            if layer.positive is not None and layer.negative is not None:
                spikes = inputs[0]
                kwargs = {
                    "bias": None,
                    "stride": convolution.stride,
                    "padding": convolution.padding,
                    "dilation": convolution.dilation,
                    "groups": convolution.groups,
                }
                layer.positive.update(
                    F.conv2d(spikes, convolution.weight.clamp_min(0.0), **kwargs)
                )
                layer.negative.update(
                    F.conv2d(spikes, convolution.weight.clamp_max(0.0), **kwargs)
                )

        return hook

    def make_neuron_pre_hook(name: str, block):
        def hook(_module: nn.Module, inputs: tuple[Tensor, ...]) -> None:
            current = inputs[0]
            fast_current, slow_current = torch.split(
                current,
                (block.neurons.fast_channels, block.neurons.slow_channels),
                dim=1,
            )
            fast_previous = block.neurons.fast_lif.membrane_state
            slow_previous = block.neurons.slow_lif.membrane_state
            if fast_previous is None:
                fast_previous = torch.zeros_like(fast_current)
            if slow_previous is None:
                slow_previous = torch.zeros_like(slow_current)
            fast_membrane = (
                block.neurons.fast_lif.alpha * fast_previous + fast_current
            )
            slow_membrane = (
                block.neurons.slow_lif.alpha * slow_previous + slow_current
            )
            layer = accumulators[name]
            layer.fast_membrane.update(fast_membrane)
            layer.slow_membrane.update(slow_membrane)
            layer.membrane.update(torch.cat((fast_membrane, slow_membrane), dim=1))

        return hook

    try:
        for name, block in zip(names, blocks, strict=True):
            handles.append(block.conv.register_forward_hook(make_conv_hook(name, block.conv)))
            handles.append(
                block.neurons.register_forward_pre_hook(make_neuron_pre_hook(name, block))
            )
        run = model.forward_with_stats(event_bins)
    finally:
        for handle in handles:
            handle.remove()

    layer_diagnostics: dict[str, LayerNumericalDiagnostics] = {}
    for name, block in zip(names, blocks, strict=True):
        accumulator = accumulators[name]
        spikes = run.statistics.layers[name]
        fast_count = accumulator.fast_membrane.distribution.count
        slow_count = accumulator.slow_membrane.distribution.count
        layer_diagnostics[name] = LayerNumericalDiagnostics(
            synaptic_current=accumulator.synaptic.finalize(),
            membrane=accumulator.membrane.finalize(),
            fast_membrane=accumulator.fast_membrane.finalize(),
            slow_membrane=accumulator.slow_membrane.finalize(),
            fast_firing_fraction=spikes.fast_spikes / fast_count if fast_count else 0.0,
            slow_firing_fraction=spikes.slow_spikes / slow_count if slow_count else 0.0,
            signed_current=_signed_current(accumulator),
        )
    return SNNDiagnosticRun(
        output=run.output,
        spike_statistics=run.statistics,
        numerical_diagnostics=SNNNumericalDiagnostics(layer_diagnostics),
    )
