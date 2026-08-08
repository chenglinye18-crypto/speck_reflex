from __future__ import annotations

import copy
import math

import pytest
import torch

from software.models.snn import SNNMotionBackbone, SNNMotionConfig
from software.simulation.snn_calibration import (
    LAYER_NAMES,
    UNIT_GAINS,
    CalibrationSettings,
    calibrate_layer_gains,
    calibration_batch,
    calibration_samples,
    gradient_sanity,
    temporary_runtime_gains,
    weight_statistics,
)
from software.simulation.snn_sanity import (
    check_early_layer_locality,
    run_case,
    temporal_shift_sanity,
)
from software.simulation.synthetic_motion import (
    SyntheticMotionCase,
    SyntheticMotionConfig,
    SyntheticMotionGenerator,
)


@pytest.fixture(scope="module")
def calibrated_artifacts():
    torch.manual_seed(17)
    config = SyntheticMotionConfig(seed=17)
    events = calibration_batch(config)
    model = SNNMotionBackbone(SNNMotionConfig(layer_gains=UNIT_GAINS))
    model.reset_state()
    with torch.no_grad():
        before = model.forward_with_diagnostics(events)
    result = calibrate_layer_gains(model, events)
    model.reset_state()
    with torch.no_grad():
        after = model.forward_with_diagnostics(events)
    return config, events, model, before, after, result


@pytest.mark.unit
def test_actual_initialization_and_weight_statistics_are_finite() -> None:
    torch.manual_seed(17)
    model = SNNMotionBackbone(SNNMotionConfig(layer_gains=UNIT_GAINS))
    assert model.config.initialization == "pytorch_default_kaiming_uniform_a_sqrt5"
    statistics = weight_statistics(model)
    assert len(statistics) == 7
    assert all(item.fan_in > 0 and item.fan_out > 0 for item in statistics)
    assert all(
        math.isfinite(value)
        for item in statistics
        for value in (item.mean, item.std, item.abs_max)
    )


@pytest.mark.unit
def test_membrane_diagnostics_are_finite(calibrated_artifacts) -> None:
    _, _, _, before, after, _ = calibrated_artifacts
    for run in (before, after):
        for layer in run.numerical_diagnostics.layers.values():
            distributions = (
                layer.synaptic_current,
                layer.membrane.distribution,
                layer.membrane.ratio_to_threshold,
            )
            assert all(
                math.isfinite(value)
                for distribution in distributions
                for value in (
                    distribution.mean,
                    distribution.std,
                    distribution.abs_max,
                    distribution.p50,
                    distribution.p90,
                    distribution.p95,
                    distribution.p99,
                    distribution.p99_9,
                )
            )


@pytest.mark.unit
def test_fast_slow_diagnostics_are_separate(calibrated_artifacts) -> None:
    _, _, _, _, after, _ = calibrated_artifacts
    s1 = after.numerical_diagnostics.layers["S1"]
    assert s1.fast_membrane.ratio_to_threshold.p99 != s1.slow_membrane.ratio_to_threshold.p99
    assert math.isfinite(s1.fast_firing_fraction)
    assert math.isfinite(s1.slow_firing_fraction)
    assert 0.0 <= s1.fast_firing_fraction <= 1.0
    assert 0.0 <= s1.slow_firing_fraction <= 1.0


@pytest.mark.unit
def test_calibration_is_deterministic(calibrated_artifacts) -> None:
    _, events, _, _, _, first = calibrated_artifacts
    torch.manual_seed(17)
    second_model = SNNMotionBackbone(SNNMotionConfig(layer_gains=UNIT_GAINS))
    second = calibrate_layer_gains(second_model, events)
    assert first.gains == second.gains
    assert first.final_firing_fractions == second.final_firing_fractions


@pytest.mark.unit
def test_no_motion_remains_silent_after_calibration(calibrated_artifacts) -> None:
    config, _, model, _, _, _ = calibrated_artifacts
    sample = SyntheticMotionGenerator(config).generate(SyntheticMotionCase.NO_MOTION)
    result = run_case(model, "NO_MOTION", sample)
    assert result.input_events == 0
    assert all(
        layer.total_spikes == 0 for layer in result.statistics.layers.values()
    )


@pytest.mark.unit
def test_calibrated_activity_reaches_primitive(calibrated_artifacts) -> None:
    _, _, _, _, after, result = calibrated_artifacts
    assert result.final_firing_fractions["primitive"] > 0.0
    assert after.spike_statistics.layers["primitive"].total_spikes > 0


@pytest.mark.unit
def test_calibrated_firing_stays_below_explosion_limit(calibrated_artifacts) -> None:
    _, _, _, _, _, result = calibrated_artifacts
    assert max(result.final_firing_fractions.values()) < CalibrationSettings().overactive_fraction
    assert all(step.firing_fraction < CalibrationSettings().failure_fraction for step in result.steps)


@pytest.mark.unit
def test_calibrated_spatial_locality_is_preserved(calibrated_artifacts) -> None:
    config, _, model, _, _, _ = calibrated_artifacts
    sample = SyntheticMotionGenerator(config).generate(
        SyntheticMotionCase.STATIC_BG_MOVING_OBJECT
    )
    report = check_early_layer_locality(model, sample.events)
    assert report.s1_active and report.s2_active
    assert report.passed


@pytest.mark.unit
def test_calibrated_temporal_shift_sanity_is_preserved(calibrated_artifacts) -> None:
    config, _, model, _, _, _ = calibrated_artifacts
    report = temporal_shift_sanity(model, config)
    assert report.passed
    assert report.layer_relative_difference < 0.5
    assert report.primitive_relative_difference < 0.5


@pytest.mark.unit
def test_gradient_sanity_reaches_every_synaptic_convolution(calibrated_artifacts) -> None:
    config, _, model, _, _, _ = calibrated_artifacts
    events = calibration_samples(config)[1][1].events
    norms = gradient_sanity(model, events)
    assert tuple(norms) == LAYER_NAMES
    assert all(math.isfinite(norm) and norm > 0.0 for norm in norms.values())


@pytest.mark.unit
def test_calibration_gains_fold_into_weights() -> None:
    gains = SNNMotionConfig().layer_gains
    torch.manual_seed(17)
    unit_model = SNNMotionBackbone(SNNMotionConfig(layer_gains=UNIT_GAINS))
    torch.manual_seed(17)
    folded_model = SNNMotionBackbone(SNNMotionConfig(layer_gains=UNIT_GAINS))
    folded_model.fold_layer_gains(gains)
    for unit, folded, gain in zip(
        unit_model.synaptic_convolutions(),
        folded_model.synaptic_convolutions(),
        gains,
        strict=True,
    ):
        assert torch.allclose(folded.weight, unit.weight * gain)


@pytest.mark.unit
def test_runtime_gain_reference_and_folded_output_match() -> None:
    gains = SNNMotionConfig().layer_gains
    torch.manual_seed(17)
    runtime_model = SNNMotionBackbone(SNNMotionConfig(layer_gains=UNIT_GAINS))
    folded_model = copy.deepcopy(runtime_model)
    folded_model.fold_layer_gains(gains)
    events = calibration_samples(SyntheticMotionConfig(seed=17))[1][1].events

    runtime_model.reset_state()
    with torch.no_grad(), temporary_runtime_gains(runtime_model, gains):
        runtime = runtime_model(events)
    folded_model.reset_state()
    with torch.no_grad():
        folded = folded_model(events)
    assert torch.allclose(runtime.primitive_spikes, folded.primitive_spikes)
    assert torch.allclose(runtime.local_logits, folded.local_logits, atol=1e-6, rtol=1e-6)
    assert torch.allclose(
        runtime.global_embedding, folded.global_embedding, atol=1e-6, rtol=1e-6
    )
