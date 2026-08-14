from __future__ import annotations

import torch

from software.models.snn import (
    FusedMultiTimescaleLIF,
    MultiTimescaleLIF,
    SNNMotionBackbone,
)


def make_neurons(channels: int = 5) -> tuple[MultiTimescaleLIF, FusedMultiTimescaleLIF]:
    kwargs = dict(
        tau_fast_ms=10.0,
        tau_slow_ms=40.0,
        dt_ms=1.0,
        threshold=1.0,
        fast_ratio=0.5,
    )
    return MultiTimescaleLIF(channels, **kwargs), FusedMultiTimescaleLIF(channels, **kwargs)


def combined_reference_membrane(neurons: MultiTimescaleLIF) -> torch.Tensor:
    return torch.cat((neurons.fast_lif.membrane_state, neurons.slow_lif.membrane_state), dim=1)


def test_fused_matches_reference_single_and_multi_step_with_odd_channels() -> None:
    reference, fused = make_neurons()
    assert (fused.fast_channels, fused.slow_channels) == (2, 3)
    generator = torch.Generator().manual_seed(123)
    with torch.inference_mode():
        for _ in range(8):
            synaptic_input = torch.randn(2, 5, 7, 9, generator=generator)
            assert torch.equal(reference(synaptic_input), fused(synaptic_input))
            assert torch.equal(combined_reference_membrane(reference), fused.membrane_state)


def test_reference_inference_fast_spike_is_bitwise_equivalent() -> None:
    kwargs = dict(
        tau_fast_ms=10.0,
        tau_slow_ms=40.0,
        dt_ms=1.0,
        threshold=1.0,
        fast_ratio=0.5,
    )
    slow = MultiTimescaleLIF(6, **kwargs)
    fast = MultiTimescaleLIF(6, **kwargs)
    fast.fast_lif.inference_fast_spike = True
    fast.slow_lif.inference_fast_spike = True
    generator = torch.Generator().manual_seed(9)
    with torch.inference_mode():
        for _ in range(4):
            synaptic_input = torch.randn(1, 6, 8, 8, generator=generator)
            assert torch.equal(slow(synaptic_input), fast(synaptic_input))
            assert torch.equal(combined_reference_membrane(slow), combined_reference_membrane(fast))


def test_fused_model_t64_outputs_and_state_semantics_match_reference() -> None:
    torch.manual_seed(41)
    reference = SNNMotionBackbone()
    fused = SNNMotionBackbone(lif_implementation="fused")
    fused.load_state_dict(reference.state_dict(), strict=True)
    event_bins = torch.poisson(
        torch.full((1, 64, 2, 16, 16), 0.08), generator=torch.Generator().manual_seed(77)
    )
    assert fused.membrane_states() == ()
    with torch.inference_mode():
        reference_output = reference(event_bins)
        fused_output = fused(event_bins)
    for name in ("primitive_spikes", "local_logits", "global_embedding", "ego_motion"):
        assert torch.equal(getattr(reference_output, name), getattr(fused_output, name))
    reference_blocks = (*reference.stages, reference.primitive_bottleneck)
    fused_blocks = (*fused.stages, fused.primitive_bottleneck)
    for reference_block, fused_block in zip(reference_blocks, fused_blocks, strict=True):
        assert torch.equal(
            combined_reference_membrane(reference_block.neurons), fused_block.neurons.membrane_state
        )
    fused.reset_state()
    assert fused.membrane_states() == ()


def test_fused_reset_and_detach_preserve_state_semantics() -> None:
    _, fused = make_neurons(channels=6)
    synaptic_input = torch.randn(1, 6, 4, 4, requires_grad=True)
    fused(synaptic_input)
    before = fused.membrane_state.clone()
    assert fused.membrane_state.grad_fn is not None
    fused.detach_state()
    assert torch.equal(before, fused.membrane_state)
    assert fused.membrane_state.grad_fn is None
    assert not fused.membrane_state.requires_grad
    fused.reset_state()
    assert fused.membrane_state is None
