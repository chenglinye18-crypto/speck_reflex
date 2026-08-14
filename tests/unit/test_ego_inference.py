from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from software.models.snn import SNNMotionBackbone


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="optimized ego inference is CUDA-only"
)


def make_models() -> tuple[SNNMotionBackbone, SNNMotionBackbone]:
    full = SNNMotionBackbone(
        lif_implementation="fused",
        inference_fast_spike=True,
        execution_mode="stage_major_chunked",
    ).cuda().eval()
    ego = SNNMotionBackbone(
        lif_implementation="fused",
        inference_fast_spike=True,
        execution_mode="stage_major_chunked",
    ).cuda().eval()
    with torch.no_grad():
        for convolution in full.synaptic_convolutions():
            convolution.weight.fill_(0.125)
    ego.load_state_dict(full.state_dict(), strict=True)
    return full, ego


def test_ego_only_exactly_matches_full_forward_and_skips_local_head() -> None:
    torch.manual_seed(91)
    full, ego = make_models()
    inputs = torch.randint(
        0, 2, (1, 64, 2, 16, 16), dtype=torch.float32, device="cuda"
    )
    local_calls = 0
    primitive_steps: list[torch.Tensor] = []

    def local_hook(*_args: object) -> None:
        nonlocal local_calls
        local_calls += 1

    def primitive_hook(
        _module: torch.nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        primitive_steps.append(output)

    local_handle = ego.local_motion_head.register_forward_hook(local_hook)
    primitive_handle = ego.primitive_bottleneck.neurons.register_forward_hook(
        primitive_hook
    )
    try:
        with torch.inference_mode():
            expected_output = full(inputs)
            actual_ego = ego.forward_ego_motion(inputs)
    finally:
        local_handle.remove()
        primitive_handle.remove()

    assert expected_output.ego_motion is not None
    assert torch.equal(expected_output.ego_motion.mean(dim=1), actual_ego)
    assert torch.equal(
        expected_output.primitive_spikes,
        torch.stack(primitive_steps, dim=1),
    )
    assert local_calls == 0
    assert all(
        torch.equal(expected_state, actual_state)
        for expected_state, actual_state in zip(
            full.membrane_states(), ego.membrane_states(), strict=True
        )
    )
    ego.reset_state()
    assert ego.membrane_states() == ()


def test_temporal_pool_is_exact_and_rejected_head_variants_are_numerical() -> None:
    torch.manual_seed(92)
    model = SNNMotionBackbone().cuda().eval()
    primitives = torch.randn(
        1, 64, model.config.primitive_channels, 3, 4, device="cuda"
    )
    with torch.inference_mode():
        loop_pool = torch.stack(
            [
                F.adaptive_avg_pool2d(primitives[:, index], (2, 2)).flatten(1)
                for index in range(64)
            ],
            dim=1,
        )
        merged = primitives.reshape(64, model.config.primitive_channels, 3, 4)
        batch_pool = F.adaptive_avg_pool2d(merged, (2, 2)).flatten(1).reshape(
            1, 64, -1
        )
        loop_head = torch.stack(
            [model.ego_motion_head(loop_pool[:, index]) for index in range(64)],
            dim=1,
        )
        batch_head = model.ego_motion_head(loop_pool.reshape(64, -1)).reshape(
            1, 64, 6
        )
        mean_before = model.ego_motion_head(loop_pool.mean(dim=1))

    assert torch.equal(loop_pool, batch_pool)
    assert torch.allclose(loop_head, batch_head, rtol=0.0, atol=1e-6)
    assert torch.allclose(
        loop_head.mean(dim=1), mean_before, rtol=0.0, atol=1e-6
    )


def test_ego_inference_requires_exact_cuda_backend() -> None:
    model = SNNMotionBackbone()
    with torch.inference_mode(), pytest.raises(RuntimeError, match="stage_major_chunked"):
        model.forward_ego_motion(torch.zeros(1, 1, 2, 16, 16))


def test_o2_exact_schedule_is_unchanged() -> None:
    assert SNNMotionBackbone._EXACT_TEMPORAL_BATCH_SIZES == (
        64,
        8,
        64,
        32,
        32,
        64,
        1,
    )
