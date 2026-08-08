from __future__ import annotations

import pytest
import torch

from software.models.snn import (
    LIF,
    MultiTimescaleLIF,
    SNNMotionBackbone,
    SNNMotionConfig,
)


@pytest.fixture(scope="module")
def reference_output():
    torch.manual_seed(17)
    model = SNNMotionBackbone()
    event_bins = (torch.rand(2, 8, 2, 96, 128) < 0.01).float()
    with torch.no_grad():
        output = model(event_bins)
    return model, output


@pytest.mark.unit
def test_forward_accepts_batched_time_sequence(reference_output) -> None:
    _, output = reference_output
    assert output.primitive_spikes.shape[:2] == (2, 8)
    assert output.local_logits.shape[:2] == (2, 8)
    assert output.global_embedding.shape[:2] == (2, 8)


@pytest.mark.unit
def test_reference_output_shapes(reference_output) -> None:
    _, output = reference_output
    assert output.primitive_spikes.shape == (2, 8, 16, 24, 32)
    assert output.local_logits.shape == (2, 8, 2, 24, 32)
    assert output.global_embedding.shape == (2, 8, 64)
    assert output.ego_motion is not None
    assert output.ego_motion.shape == (2, 8, 6)


@pytest.mark.unit
def test_reset_state_clears_all_membranes() -> None:
    model = SNNMotionBackbone(SNNMotionConfig(enable_ego_head=False))
    model(torch.ones(1, 1, 2, 16, 16))
    assert model.membrane_states()
    model.reset_state()
    assert model.membrane_states() == ()


@pytest.mark.unit
def test_consecutive_forward_calls_continue_state() -> None:
    config = SNNMotionConfig(threshold=1_000_000.0, enable_ego_head=False)
    model = SNNMotionBackbone(config)
    with torch.no_grad():
        model.stages[0].conv.weight.fill_(0.01)
        event_bins = torch.ones(1, 1, 2, 16, 16)
        model(event_bins)
        first = model.stages[0].neurons.fast_lif.membrane_state.clone()
        model(event_bins)
        second = model.stages[0].neurons.fast_lif.membrane_state.clone()
    assert torch.all(second > first)


@pytest.mark.unit
def test_detach_state_keeps_values_and_cuts_graph() -> None:
    model = SNNMotionBackbone(SNNMotionConfig(enable_ego_head=False))
    model(torch.rand(1, 1, 2, 16, 16, requires_grad=True))
    before = tuple(state.clone() for state in model.membrane_states())
    assert any(state.grad_fn is not None for state in model.membrane_states())

    model.detach_state()
    after = model.membrane_states()
    assert len(before) == len(after)
    assert all(torch.equal(old, new) for old, new in zip(before, after, strict=True))
    assert all(state.grad_fn is None and not state.requires_grad for state in after)


@pytest.mark.unit
def test_zero_input_after_reset_produces_no_spikes() -> None:
    model = SNNMotionBackbone(SNNMotionConfig(enable_ego_head=False))
    model.reset_state()
    with torch.no_grad():
        output = model(torch.zeros(1, 3, 2, 16, 16))
    assert torch.count_nonzero(output.primitive_spikes) == 0
    assert torch.count_nonzero(output.local_logits) == 0
    assert torch.count_nonzero(output.global_embedding) == 0
    assert output.ego_motion is None


@pytest.mark.unit
def test_first_stage_response_remains_spatially_local() -> None:
    model = SNNMotionBackbone(SNNMotionConfig(threshold=0.5, enable_ego_head=False))
    first_stage = model.stages[0]
    with torch.no_grad():
        first_stage.conv.weight.zero_()
        first_stage.conv.weight[:, 1, 2, 2] = 1.0
        one_event = torch.zeros(1, 2, 17, 17)
        one_event[0, 1, 8, 8] = 1.0
        response = first_stage(one_event)

    active_spatial = torch.nonzero(response[0].sum(dim=0), as_tuple=False)
    assert active_spatial.tolist() == [[4, 4]]


@pytest.mark.unit
def test_odd_channel_partition_is_stable() -> None:
    neurons = MultiTimescaleLIF(
        5,
        tau_fast_ms=10.0,
        tau_slow_ms=40.0,
        dt_ms=1.0,
        threshold=1.0,
        fast_ratio=0.5,
    )
    assert neurons.fast_channels == 2
    assert neurons.slow_channels == 3


@pytest.mark.unit
def test_atan_surrogate_propagates_gradient() -> None:
    neuron = LIF(tau_ms=10.0, dt_ms=1.0, threshold=1.0)
    synaptic_input = torch.tensor([0.9], requires_grad=True)
    neuron(synaptic_input).sum().backward()
    assert synaptic_input.grad is not None
    assert torch.isfinite(synaptic_input.grad).all()
    assert torch.count_nonzero(synaptic_input.grad) == 1
