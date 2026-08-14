from __future__ import annotations

import pytest
import torch

from software.models.snn import SNNMotionBackbone
from software.models.snn.layers import ConvLIFBlock


def make_block() -> ConvLIFBlock:
    block = ConvLIFBlock(
        2,
        4,
        kernel_size=3,
        stride=1,
        padding=1,
        tau_fast_ms=10.0,
        tau_slow_ms=40.0,
        dt_ms=1.0,
        threshold=1.0,
        fast_ratio=0.5,
        surrogate="atan",
        lif_implementation="fused",
        inference_fast_spike=True,
    )
    with torch.no_grad():
        block.conv.weight.fill_(0.125)
    return block


@pytest.mark.parametrize("timesteps", (1, 8, 64))
def test_temporal_batched_conv_matches_loop_for_controlled_input(timesteps: int) -> None:
    block = make_block()
    inputs = torch.randint(0, 2, (1, timesteps, 2, 12, 16), dtype=torch.float32)
    with torch.inference_mode():
        reference = torch.stack([block.conv(inputs[:, index]) for index in range(timesteps)], 1)
        merged = inputs.reshape(timesteps, 2, 12, 16)
        raw = block.conv(merged)
        batched = raw.reshape(1, timesteps, *raw.shape[1:])
    assert torch.equal(reference, batched)


@pytest.mark.parametrize("timesteps", (1, 8))
def test_conv_lif_sequence_matches_time_major_for_controlled_input(timesteps: int) -> None:
    time_major = make_block()
    stage_major = make_block()
    stage_major.load_state_dict(time_major.state_dict(), strict=True)
    inputs = torch.randint(0, 2, (1, timesteps, 2, 12, 16), dtype=torch.float32)
    with torch.inference_mode():
        reference = torch.stack([time_major(inputs[:, index]) for index in range(timesteps)], 1)
        actual = stage_major.forward_sequence(inputs)
    assert actual.shape == reference.shape
    assert torch.equal(reference, actual)
    assert torch.equal(time_major.neurons.membrane_state, stage_major.neurons.membrane_state)


@pytest.mark.parametrize("execution_mode", ("stage_major", "stage_major_chunked"))
@pytest.mark.skipif(not torch.cuda.is_available(), reason="stage-major is CUDA-only")
def test_stage_major_t64_full_model_state_reset_and_checkpoint_names(
    execution_mode: str,
) -> None:
    torch.manual_seed(81)
    time_major = SNNMotionBackbone(
        lif_implementation="fused", inference_fast_spike=True
    ).cuda().eval()
    stage_major = SNNMotionBackbone(
        lif_implementation="fused",
        inference_fast_spike=True,
        execution_mode=execution_mode,
    ).cuda().eval()
    stage_major.load_state_dict(time_major.state_dict(), strict=True)
    with torch.no_grad():
        for convolution in time_major.synaptic_convolutions():
            convolution.weight.fill_(0.125)
        stage_major.load_state_dict(time_major.state_dict(), strict=True)
    inputs = torch.randint(
        0, 2, (1, 64, 2, 16, 16), dtype=torch.float32, device="cuda"
    )
    with torch.inference_mode():
        expected = time_major(inputs)
        actual = stage_major(inputs)
    for name in ("primitive_spikes", "local_logits", "global_embedding", "ego_motion"):
        assert torch.equal(getattr(expected, name), getattr(actual, name))
    assert all(
        torch.equal(expected_state, actual_state)
        for expected_state, actual_state in zip(
            time_major.membrane_states(), stage_major.membrane_states(), strict=True
        )
    )
    stage_major.reset_state()
    assert stage_major.membrane_states() == ()


def test_stage_major_is_inference_only_and_default_remains_time_major() -> None:
    assert SNNMotionBackbone().execution_mode == "time_major"
    model = SNNMotionBackbone(execution_mode="stage_major")
    with pytest.raises(RuntimeError, match="inference-only"):
        model(torch.zeros(1, 1, 2, 16, 16))
    with torch.inference_mode(), pytest.raises(RuntimeError, match="CUDA-only"):
        model(torch.zeros(1, 1, 2, 16, 16))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="stage-major is CUDA-only")
def test_chunked_stage_major_preserves_state_carry_and_stats_hooks() -> None:
    torch.manual_seed(82)
    time_major = SNNMotionBackbone(
        lif_implementation="fused", inference_fast_spike=True
    ).cuda().eval()
    stage_major = SNNMotionBackbone(
        lif_implementation="fused",
        inference_fast_spike=True,
        execution_mode="stage_major_chunked",
    ).cuda().eval()
    stage_major.load_state_dict(time_major.state_dict(), strict=True)
    inputs = torch.randint(
        0, 2, (1, 2, 2, 16, 16), dtype=torch.float32, device="cuda"
    )
    with torch.inference_mode():
        expected_first = time_major.forward_with_stats(inputs)
        actual_first = stage_major.forward_with_stats(inputs)
        expected_second = time_major(inputs)
        actual_second = stage_major(inputs)
    assert torch.equal(
        expected_first.output.primitive_spikes,
        actual_first.output.primitive_spikes,
    )
    assert expected_first.statistics == actual_first.statistics
    assert torch.equal(expected_second.primitive_spikes, actual_second.primitive_spikes)
    assert all(
        torch.equal(expected_state, actual_state)
        for expected_state, actual_state in zip(
            time_major.membrane_states(), stage_major.membrane_states(), strict=True
        )
    )
    stage_major.detach_state()
    assert all(state.grad_fn is None for state in stage_major.membrane_states())
