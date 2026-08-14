"""Local shared-synapse layers used by the SNN motion backbone."""

from __future__ import annotations

from torch import Tensor, nn

from .neurons import FusedMultiTimescaleLIF, MultiTimescaleLIF


class ConvLIFBlock(nn.Module):
    """One hardware-friendly local convolution followed by stateful LIF."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        stride: int,
        padding: int,
        tau_fast_ms: float,
        tau_slow_ms: float,
        dt_ms: float,
        threshold: float,
        fast_ratio: float,
        surrogate: str,
        lif_implementation: str = "reference",
        inference_fast_spike: bool = False,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        lif_class = {
            "reference": MultiTimescaleLIF,
            "fused": FusedMultiTimescaleLIF,
        }.get(lif_implementation)
        if lif_class is None:
            raise ValueError("lif_implementation must be 'reference' or 'fused'")
        self.neurons = lif_class(
            out_channels,
            tau_fast_ms=tau_fast_ms,
            tau_slow_ms=tau_slow_ms,
            dt_ms=dt_ms,
            threshold=threshold,
            fast_ratio=fast_ratio,
            surrogate=surrogate,
            inference_fast_spike=inference_fast_spike,
        )

    def forward(self, spikes: Tensor) -> Tensor:
        return self.neurons(self.conv(spikes))

    def reset_state(self) -> None:
        self.neurons.reset_state()

    def detach_state(self) -> None:
        self.neurons.detach_state()
