"""Small PyTorch LIF primitives with explicit stream-state management."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class _ATanSpike(torch.autograd.Function):
    """Heaviside forward with a smooth arctangent surrogate derivative."""

    @staticmethod
    def forward(ctx: object, membrane_delta: Tensor) -> Tensor:
        ctx.save_for_backward(membrane_delta)  # type: ignore[attr-defined]
        return (membrane_delta >= 0.0).to(membrane_delta.dtype)

    @staticmethod
    def backward(ctx: object, grad_output: Tensor) -> tuple[Tensor]:
        (membrane_delta,) = ctx.saved_tensors  # type: ignore[attr-defined]
        derivative = 1.0 / (1.0 + (math.pi * membrane_delta).square())
        return (grad_output * derivative,)


def surrogate_spike(membrane_delta: Tensor, surrogate: str = "atan") -> Tensor:
    if surrogate != "atan":
        raise ValueError(f"unsupported surrogate: {surrogate!r}")
    return _ATanSpike.apply(membrane_delta)


class LIF(nn.Module):
    """Stateful leaky integrate-and-fire neuron with subtractive reset."""

    def __init__(
        self,
        *,
        tau_ms: float,
        dt_ms: float,
        threshold: float,
        surrogate: str = "atan",
        inference_fast_spike: bool = False,
    ) -> None:
        super().__init__()
        if tau_ms <= 0.0 or dt_ms <= 0.0 or threshold <= 0.0:
            raise ValueError("tau_ms, dt_ms, and threshold must be positive")
        if surrogate != "atan":
            raise ValueError("only the 'atan' surrogate is supported")

        self.tau_ms = float(tau_ms)
        self.dt_ms = float(dt_ms)
        self.threshold = float(threshold)
        self.surrogate = surrogate
        self.inference_fast_spike = inference_fast_spike
        self.register_buffer("alpha", torch.tensor(math.exp(-dt_ms / tau_ms)))
        self.register_buffer("_membrane", None, persistent=False)

    @property
    def membrane_state(self) -> Tensor | None:
        """Current membrane tensor, or ``None`` at a sequence boundary."""

        return self._membrane

    def forward(self, synaptic_input: Tensor) -> Tensor:
        if self._membrane is None:
            previous = torch.zeros_like(synaptic_input)
        else:
            if self._membrane.shape != synaptic_input.shape:
                raise RuntimeError(
                    "LIF state shape differs from input; call reset_state() at the new sequence boundary"
                )
            previous = self._membrane

        membrane = self.alpha * previous + synaptic_input
        membrane_delta = membrane - self.threshold
        spikes = (
            (membrane_delta >= 0.0).to(membrane.dtype)
            if self.inference_fast_spike and not torch.is_grad_enabled()
            else surrogate_spike(membrane_delta, self.surrogate)
        )
        self._membrane = membrane - self.threshold * spikes
        return spikes

    def reset_state(self) -> None:
        """Remove the membrane state so the next call starts from zero."""

        self._membrane = None

    def detach_state(self) -> None:
        """Keep the membrane values while severing their autograd history."""

        if self._membrane is not None:
            self._membrane = self._membrane.detach()

    def extra_repr(self) -> str:
        return (
            f"tau_ms={self.tau_ms}, dt_ms={self.dt_ms}, "
            f"threshold={self.threshold}, surrogate={self.surrogate!r}"
        )


class MultiTimescaleLIF(nn.Module):
    """Split shared synaptic channels between fast and slow LIF neurons."""

    def __init__(
        self,
        channels: int,
        *,
        tau_fast_ms: float,
        tau_slow_ms: float,
        dt_ms: float,
        threshold: float,
        fast_ratio: float = 0.5,
        surrogate: str = "atan",
        inference_fast_spike: bool = False,
    ) -> None:
        super().__init__()
        if channels < 2:
            raise ValueError("MultiTimescaleLIF requires at least two channels")
        if not 0.0 < fast_ratio < 1.0:
            raise ValueError("fast_ratio must lie strictly between 0 and 1")

        # Stable odd-channel rule: floor goes to fast, the remainder to slow.
        self.fast_channels = max(1, min(channels - 1, int(channels * fast_ratio)))
        self.slow_channels = channels - self.fast_channels
        self.fast_lif = LIF(
            tau_ms=tau_fast_ms,
            dt_ms=dt_ms,
            threshold=threshold,
            surrogate=surrogate,
            inference_fast_spike=inference_fast_spike,
        )
        self.slow_lif = LIF(
            tau_ms=tau_slow_ms,
            dt_ms=dt_ms,
            threshold=threshold,
            surrogate=surrogate,
            inference_fast_spike=inference_fast_spike,
        )

    def forward(self, synaptic_input: Tensor) -> Tensor:
        if synaptic_input.ndim != 4:
            raise ValueError("MultiTimescaleLIF expects [B, C, H, W]")
        expected_channels = self.fast_channels + self.slow_channels
        if synaptic_input.shape[1] != expected_channels:
            raise ValueError(
                f"expected {expected_channels} channels, got {synaptic_input.shape[1]}"
            )
        fast, slow = torch.split(
            synaptic_input, (self.fast_channels, self.slow_channels), dim=1
        )
        return torch.cat((self.fast_lif(fast), self.slow_lif(slow)), dim=1)

    def reset_state(self) -> None:
        self.fast_lif.reset_state()
        self.slow_lif.reset_state()

    def detach_state(self) -> None:
        self.fast_lif.detach_state()
        self.slow_lif.detach_state()


class FusedMultiTimescaleLIF(nn.Module):
    """Inference-compatible channel-wise equivalent of ``MultiTimescaleLIF``.

    The reference implementation above remains the default.  This class stores
    one membrane tensor and applies the two decay factors channel-wise, avoiding
    the reference path's split, two independent LIF updates, and cat.
    """

    def __init__(
        self,
        channels: int,
        *,
        tau_fast_ms: float,
        tau_slow_ms: float,
        dt_ms: float,
        threshold: float,
        fast_ratio: float = 0.5,
        surrogate: str = "atan",
        inference_fast_spike: bool = False,
    ) -> None:
        super().__init__()
        if channels < 2:
            raise ValueError("FusedMultiTimescaleLIF requires at least two channels")
        if not 0.0 < fast_ratio < 1.0:
            raise ValueError("fast_ratio must lie strictly between 0 and 1")
        if tau_fast_ms <= 0.0 or tau_slow_ms <= 0.0 or dt_ms <= 0.0 or threshold <= 0.0:
            raise ValueError("time constants, dt_ms, and threshold must be positive")
        if surrogate != "atan":
            raise ValueError("only the 'atan' surrogate is supported")

        self.fast_channels = max(1, min(channels - 1, int(channels * fast_ratio)))
        self.slow_channels = channels - self.fast_channels
        self.threshold = float(threshold)
        self.surrogate = surrogate
        self.inference_fast_spike = inference_fast_spike
        alpha = torch.tensor(
            [math.exp(-dt_ms / tau_fast_ms)] * self.fast_channels
            + [math.exp(-dt_ms / tau_slow_ms)] * self.slow_channels
        ).view(1, channels, 1, 1)
        # Alpha is derived entirely from frozen config values.  Keeping it out
        # of state_dict lets an existing reference checkpoint load strictly.
        self.register_buffer("alpha", alpha, persistent=False)
        self.register_buffer("_membrane", None, persistent=False)

    @property
    def membrane_state(self) -> Tensor | None:
        """Current combined [B, C, H, W] membrane, or ``None`` after reset."""

        return self._membrane

    def forward(self, synaptic_input: Tensor) -> Tensor:
        if synaptic_input.ndim != 4:
            raise ValueError("FusedMultiTimescaleLIF expects [B, C, H, W]")
        expected_channels = self.fast_channels + self.slow_channels
        if synaptic_input.shape[1] != expected_channels:
            raise ValueError(f"expected {expected_channels} channels, got {synaptic_input.shape[1]}")
        if self._membrane is None:
            previous = torch.zeros_like(synaptic_input)
        else:
            if self._membrane.shape != synaptic_input.shape:
                raise RuntimeError(
                    "LIF state shape differs from input; call reset_state() at the new sequence boundary"
                )
            previous = self._membrane

        membrane = self.alpha * previous + synaptic_input
        membrane_delta = membrane - self.threshold
        spikes = (
            (membrane_delta >= 0.0).to(membrane.dtype)
            if self.inference_fast_spike and not torch.is_grad_enabled()
            else surrogate_spike(membrane_delta, self.surrogate)
        )
        self._membrane = membrane - self.threshold * spikes
        return spikes

    def reset_state(self) -> None:
        self._membrane = None

    def detach_state(self) -> None:
        if self._membrane is not None:
            self._membrane = self._membrane.detach()

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Tensor],
        prefix: str,
        local_metadata: dict[str, object],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        """Consume alpha buffers emitted by a reference checkpoint.

        They are fixed functions of the frozen configuration and are recreated
        above; all trainable Conv/head weights retain their checkpoint names.
        """

        state_dict.pop(prefix + "fast_lif.alpha", None)
        state_dict.pop(prefix + "slow_lif.alpha", None)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )
