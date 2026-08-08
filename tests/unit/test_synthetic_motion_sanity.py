from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from software.models.snn import SNNMotionBackbone, SNNMotionConfig
from software.simulation.snn_sanity import (
    check_early_layer_locality,
    run_fast_slow_sanity,
    temporal_shift_sanity,
)
from software.simulation.synthetic_motion import (
    SyntheticMotionCase,
    SyntheticMotionConfig,
    SyntheticMotionGenerator,
)


@pytest.fixture
def small_config() -> SyntheticMotionConfig:
    return SyntheticMotionConfig(
        height=32,
        width=48,
        timesteps=16,
        motion_start_ms=3.0,
        object_size=(8, 10),
        seed=17,
    )


@pytest.mark.unit
def test_no_motion_is_exactly_zero(small_config) -> None:
    sample = SyntheticMotionGenerator(small_config).generate(SyntheticMotionCase.NO_MOTION)
    assert sample.events.shape == (1, 16, 2, 32, 48)
    assert torch.count_nonzero(sample.events) == 0


@pytest.mark.unit
def test_global_motion_generates_off_and_on_events(small_config) -> None:
    sample = SyntheticMotionGenerator(small_config).generate(SyntheticMotionCase.GLOBAL_LEFT)
    assert torch.count_nonzero(sample.events[:, :, 0]) > 0
    assert torch.count_nonzero(sample.events[:, :, 1]) > 0


@pytest.mark.unit
def test_moving_object_mask_tracks_geometric_position(small_config) -> None:
    config = replace(small_config, velocity_px_per_ms=1.0)
    sample = SyntheticMotionGenerator(config).generate(
        SyntheticMotionCase.STATIC_BG_MOVING_OBJECT
    )
    before = torch.nonzero(sample.independent_motion_mask[0, 3, 0], as_tuple=False)
    after = torch.nonzero(sample.independent_motion_mask[0, 8, 0], as_tuple=False)
    assert len(before) == len(after) == 80
    assert float(after[:, 1].float().mean() - before[:, 1].float().mean()) == 5.0
    assert float(after[:, 0].float().mean() - before[:, 0].float().mean()) == 0.0


@pytest.mark.unit
def test_synthetic_generation_is_deterministic(small_config) -> None:
    first = SyntheticMotionGenerator(small_config).generate(
        SyntheticMotionCase.MOVING_BG_MOVING_OBJECT
    )
    second = SyntheticMotionGenerator(small_config).generate(
        SyntheticMotionCase.MOVING_BG_MOVING_OBJECT
    )
    assert torch.equal(first.events, second.events)
    assert torch.equal(first.independent_motion_mask, second.independent_motion_mask)
    assert first.metadata == second.metadata


@pytest.mark.unit
def test_opposite_translations_have_different_spatial_polarity_patterns(small_config) -> None:
    generator = SyntheticMotionGenerator(small_config)
    left = generator.generate(SyntheticMotionCase.GLOBAL_LEFT).events
    right = generator.generate(SyntheticMotionCase.GLOBAL_RIGHT).events
    assert not torch.equal(left, right)


@pytest.mark.unit
def test_geometric_speed_changes_event_activity(small_config) -> None:
    slow = SyntheticMotionGenerator(
        replace(small_config, velocity_px_per_ms=0.25)
    ).generate(SyntheticMotionCase.GLOBAL_LEFT)
    fast = SyntheticMotionGenerator(
        replace(small_config, velocity_px_per_ms=1.0)
    ).generate(SyntheticMotionCase.GLOBAL_LEFT)
    slow_count = int(slow.events.sum().item())
    fast_count = int(fast.events.sum().item())
    assert slow_count > 0
    assert fast_count > slow_count


@pytest.mark.unit
def test_fast_slow_lif_decay_is_distinct() -> None:
    report = run_fast_slow_sanity()
    assert report.alpha_fast < report.alpha_slow
    assert report.decay_slow_residual > report.decay_fast_residual
    assert report.passed


@pytest.mark.unit
def test_stats_interface_preserves_normal_forward(small_config) -> None:
    torch.manual_seed(17)
    model = SNNMotionBackbone(SNNMotionConfig(enable_ego_head=False))
    events = SyntheticMotionGenerator(small_config).generate(
        SyntheticMotionCase.GLOBAL_RIGHT
    ).events
    model.reset_state()
    with torch.no_grad():
        normal = model(events)
    model.reset_state()
    with torch.no_grad():
        instrumented = model.forward_with_stats(events)
    assert torch.equal(normal.primitive_spikes, instrumented.output.primitive_spikes)
    assert torch.equal(normal.local_logits, instrumented.output.local_logits)
    assert torch.equal(normal.global_embedding, instrumented.output.global_embedding)
    assert tuple(instrumented.statistics.layers) == (
        "S1",
        "S2",
        "S3",
        "S4",
        "S5",
        "S6",
        "primitive",
    )
    for layer in instrumented.statistics.layers.values():
        assert layer.spikes_per_timestep == layer.total_spikes / small_config.timesteps
        assert 0.0 <= layer.spikes_per_neuron_per_timestep <= 1.0
        assert layer.fast_spikes + layer.slow_spikes == layer.total_spikes


@pytest.mark.unit
def test_reset_prevents_state_leakage_between_cases(small_config) -> None:
    torch.manual_seed(17)
    model = SNNMotionBackbone(SNNMotionConfig(enable_ego_head=False))
    generator = SyntheticMotionGenerator(small_config)
    target = generator.generate(SyntheticMotionCase.GLOBAL_LEFT).events
    perturbation = generator.generate(SyntheticMotionCase.EXPANSION).events

    model.reset_state()
    with torch.no_grad():
        first = model(target).primitive_spikes
        model(perturbation)
    model.reset_state()
    with torch.no_grad():
        repeated = model(target).primitive_spikes
    assert torch.equal(first, repeated)


@pytest.mark.unit
def test_local_object_early_support_respects_receptive_field(small_config) -> None:
    config = replace(
        small_config, timesteps=12, motion_start_ms=2.0, velocity_px_per_ms=1.0
    )
    events = SyntheticMotionGenerator(config).generate(
        SyntheticMotionCase.STATIC_BG_MOVING_OBJECT
    ).events
    model = SNNMotionBackbone(
        SNNMotionConfig(threshold=0.1, enable_ego_head=False)
    )
    with torch.no_grad():
        model.stages[0].conv.weight.fill_(0.25)
        model.stages[1].conv.weight.fill_(0.25)
    report = check_early_layer_locality(model, events)
    assert report.s1_active
    assert report.s2_active
    assert report.passed


@pytest.mark.unit
def test_one_bin_temporal_shift_has_bounded_sensitivity() -> None:
    torch.manual_seed(17)
    model = SNNMotionBackbone(SNNMotionConfig(enable_ego_head=False))
    config = SyntheticMotionConfig(seed=17)
    report = temporal_shift_sanity(model, config)
    assert report.input_relative_difference < 0.1
    assert report.layer_relative_difference <= 0.5
    assert report.primitive_relative_difference <= 0.5
    assert report.passed
