"""Deterministic synthetic excitability calibration for SNN Motion v0.1."""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass, replace
from typing import Iterator

import torch
from torch import Tensor, nn

from software.models.snn import SNNMotionBackbone, SNNMotionConfig

from .snn_sanity import (
    check_early_layer_locality,
    run_case,
    run_fast_slow_sanity,
    temporal_shift_sanity,
)
from .synthetic_motion import (
    SyntheticMotionCase,
    SyntheticMotionConfig,
    SyntheticMotionGenerator,
    SyntheticMotionSample,
)

UNIT_GAINS = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
LAYER_NAMES = ("S1", "S2", "S3", "S4", "S5", "S6", "primitive")


@dataclass(frozen=True, slots=True)
class CalibrationSettings:
    hidden_target: tuple[float, float] = (0.001, 0.02)
    primitive_target: tuple[float, float] = (0.0005, 0.01)
    overactive_fraction: float = 0.10
    failure_fraction: float = 0.25
    min_gain: float = 0.25
    max_gain: float = 16.0


@dataclass(frozen=True, slots=True)
class CalibrationStep:
    layer: str
    gain: float
    firing_fraction: float
    status: str


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    gains: tuple[float, ...]
    final_firing_fractions: dict[str, float]
    steps: tuple[CalibrationStep, ...]


@dataclass(frozen=True, slots=True)
class WeightLayerStatistics:
    layer: str
    fan_in: int
    fan_out: int
    mean: float
    std: float
    abs_max: float


@dataclass(frozen=True, slots=True)
class InputMagnitudeStatistics:
    nonzero_fraction: float
    mean_active_magnitude: float
    max_magnitude: float
    events_per_active_pixel: float


class CalibrationError(RuntimeError):
    pass


def calibration_samples(
    config: SyntheticMotionConfig | None = None,
) -> list[tuple[str, SyntheticMotionSample]]:
    """Return the frozen eight-case activity suite, excluding NO_MOTION."""

    config = config or SyntheticMotionConfig(seed=17)
    specifications = (
        ("GLOBAL_LEFT_SLOW", SyntheticMotionCase.GLOBAL_LEFT, 0.25),
        ("GLOBAL_LEFT_FAST", SyntheticMotionCase.GLOBAL_LEFT, 1.0),
        ("GLOBAL_RIGHT", SyntheticMotionCase.GLOBAL_RIGHT, 0.5),
        ("GLOBAL_UP", SyntheticMotionCase.GLOBAL_UP, 0.5),
        ("STATIC_BG_MOVING_OBJECT", SyntheticMotionCase.STATIC_BG_MOVING_OBJECT, 1.0),
        ("MOVING_BG_MOVING_OBJECT", SyntheticMotionCase.MOVING_BG_MOVING_OBJECT, 0.5),
        ("EXPANSION", SyntheticMotionCase.EXPANSION, 0.5),
        ("CONTRACTION", SyntheticMotionCase.CONTRACTION, 0.5),
    )
    return [
        (
            name,
            SyntheticMotionGenerator(
                replace(config, velocity_px_per_ms=velocity)
            ).generate(case),
        )
        for name, case, velocity in specifications
    ]


def calibration_batch(config: SyntheticMotionConfig | None = None) -> Tensor:
    return torch.cat(
        [sample.events for _, sample in calibration_samples(config)], dim=0
    )


def input_magnitude_statistics(events: Tensor) -> InputMagnitudeStatistics:
    active = events.ne(0)
    active_count = int(torch.count_nonzero(active).item())
    active_pixels = int(torch.count_nonzero(active.any(dim=(1, 2))).item())
    magnitudes = events[active].abs()
    return InputMagnitudeStatistics(
        nonzero_fraction=active_count / events.numel(),
        mean_active_magnitude=float(magnitudes.mean().item()) if active_count else 0.0,
        max_magnitude=float(events.abs().max().item()),
        events_per_active_pixel=active_count / active_pixels if active_pixels else 0.0,
    )


def weight_statistics(model: SNNMotionBackbone) -> tuple[WeightLayerStatistics, ...]:
    result: list[WeightLayerStatistics] = []
    for name, convolution in zip(
        LAYER_NAMES, model.synaptic_convolutions(), strict=True
    ):
        kernel_elements = convolution.kernel_size[0] * convolution.kernel_size[1]
        fan_in = convolution.in_channels * kernel_elements // convolution.groups
        fan_out = convolution.out_channels * kernel_elements // convolution.groups
        weight = convolution.weight.detach()
        result.append(
            WeightLayerStatistics(
                layer=name,
                fan_in=fan_in,
                fan_out=fan_out,
                mean=float(weight.mean().item()),
                std=float(weight.std(unbiased=False).item()),
                abs_max=float(weight.abs().max().item()),
            )
        )
    return tuple(result)


def _check_explosion(run, settings: CalibrationSettings) -> None:
    for name, statistics in run.statistics.layers.items():
        if statistics.spikes_per_neuron_per_timestep > settings.failure_fraction:
            raise CalibrationError(
                f"{name} exceeded calibration failure limit "
                f"({statistics.spikes_per_neuron_per_timestep:.3%})"
            )


def calibrate_layer_gains(
    model: SNNMotionBackbone,
    events: Tensor,
    settings: CalibrationSettings | None = None,
) -> CalibrationResult:
    """Sequentially search powers-of-two gains and fold them into each Conv."""

    settings = settings or CalibrationSettings()
    convolutions = model.synaptic_convolutions()
    base_weights = tuple(layer.weight.detach().clone() for layer in convolutions)
    selected: list[float] = []
    steps: list[CalibrationStep] = []

    for layer_index, (name, convolution) in enumerate(
        zip(LAYER_NAMES, convolutions, strict=True)
    ):
        target = (
            settings.primitive_target if name == "primitive" else settings.hidden_target
        )
        candidate = 1.0
        direction: float | None = None
        chosen: float | None = None
        while settings.min_gain <= candidate <= settings.max_gain:
            with torch.no_grad():
                convolution.weight.copy_(base_weights[layer_index] * candidate)
            model.reset_state()
            with torch.no_grad():
                run = model.forward_with_stats(events)
            _check_explosion(run, settings)
            firing = run.statistics.layers[name].spikes_per_neuron_per_timestep
            status = (
                "OVERACTIVE"
                if firing > settings.overactive_fraction
                else "TARGET"
                if target[0] <= firing <= target[1]
                else "LOW"
                if firing < target[0]
                else "HIGH"
            )
            steps.append(CalibrationStep(name, candidate, firing, status))
            if status == "TARGET":
                chosen = candidate
                break
            next_direction = 2.0 if status == "LOW" else 0.5
            if direction is not None and next_direction != direction:
                break
            direction = next_direction
            candidate *= next_direction

        if chosen is None:
            raise CalibrationError(
                f"{name} did not enter target range within gains "
                f"[{settings.min_gain}, {settings.max_gain}]"
            )
        selected.append(chosen)
        with torch.no_grad():
            convolution.weight.copy_(base_weights[layer_index] * chosen)

    model.reset_state()
    with torch.no_grad():
        final_run = model.forward_with_stats(events)
    _check_explosion(final_run, settings)
    final_firing = {
        name: final_run.statistics.layers[name].spikes_per_neuron_per_timestep
        for name in LAYER_NAMES
    }
    return CalibrationResult(tuple(selected), final_firing, tuple(steps))


@contextlib.contextmanager
def temporary_runtime_gains(
    model: SNNMotionBackbone, gains: tuple[float, ...]
) -> Iterator[None]:
    """Reference-only output scaling used to verify static gain folding."""

    convolutions = model.synaptic_convolutions()
    if len(gains) != len(convolutions):
        raise ValueError("gain count does not match synaptic layers")
    handles = [
        convolution.register_forward_hook(
            lambda _module, _inputs, output, gain=gain: output * gain
        )
        for convolution, gain in zip(convolutions, gains, strict=True)
    ]
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def gradient_sanity(
    model: SNNMotionBackbone, events: Tensor
) -> dict[str, float]:
    """Backpropagate a dummy scalar through primitive/local outputs once."""

    model.reset_state()
    model.zero_grad(set_to_none=True)
    output = model(events)
    loss = output.primitive_spikes.mean() + output.local_logits.square().mean()
    loss.backward()
    norms: dict[str, float] = {}
    for name, convolution in zip(
        LAYER_NAMES, model.synaptic_convolutions(), strict=True
    ):
        gradient = convolution.weight.grad
        norms[name] = float(gradient.norm().item()) if gradient is not None else 0.0
    return norms


def _print_diagnostic_table(run) -> None:
    print("Synaptic current distribution:")
    print("Layer        mean       std   abs_max       p50       p90       p95       p99     p99.9")
    for name in LAYER_NAMES:
        layer = run.numerical_diagnostics.layers[name]
        current = layer.synaptic_current
        print(
            f"{name:<10} {current.mean:>9.6f} {current.std:>9.6f} {current.abs_max:>9.6f} "
            f"{current.p50:>9.6f} {current.p90:>9.6f} {current.p95:>9.6f} "
            f"{current.p99:>9.6f} {current.p99_9:>9.6f}"
        )
    print("Membrane / threshold distribution (pre-reset):")
    print("Layer        mean       std   abs_max       p50       p90       p95       p99     p99.9")
    for name in LAYER_NAMES:
        ratio = run.numerical_diagnostics.layers[name].membrane.ratio_to_threshold
        print(
            f"{name:<10} {ratio.mean:>9.6f} {ratio.std:>9.6f} {ratio.abs_max:>9.6f} "
            f"{ratio.p50:>9.6f} {ratio.p90:>9.6f} {ratio.p95:>9.6f} "
            f"{ratio.p99:>9.6f} {ratio.p99_9:>9.6f}"
        )
    print("Threshold proximity and timescale split:")
    print("Layer       >0.5theta >0.8theta  >=theta    firing  fast_p99 slow_p99 fast_fire slow_fire")
    for name in LAYER_NAMES:
        layer = run.numerical_diagnostics.layers[name]
        membrane = layer.membrane
        firing = run.spike_statistics.layers[name].spikes_per_neuron_per_timestep
        print(
            f"{name:<10} {membrane.fraction_above_half_threshold:>9.6f} "
            f"{membrane.fraction_above_0_8_threshold:>9.6f} "
            f"{membrane.fraction_at_or_above_threshold:>9.6f} {firing:>9.6f} "
            f"{layer.fast_membrane.ratio_to_threshold.p99:>9.6f} "
            f"{layer.slow_membrane.ratio_to_threshold.p99:>9.6f} "
            f"{layer.fast_firing_fraction:>9.6f} {layer.slow_firing_fraction:>9.6f}"
        )


def main() -> None:
    torch.manual_seed(17)
    synthetic_config = SyntheticMotionConfig(seed=17)
    events = calibration_batch(synthetic_config)
    base_config = SNNMotionConfig(layer_gains=UNIT_GAINS)
    model = SNNMotionBackbone(base_config)

    print("SNN_EXCITABILITY_CALIBRATION_V0.1")
    print("mode=initialization_sanity_random_weights_binary_synthetic_events")
    print(f"initialization={base_config.initialization}")
    print("\nInitial weight statistics:")
    print("Layer fan_in fan_out mean std abs_max")
    for item in weight_statistics(model):
        print(
            f"{item.layer:<9} {item.fan_in:>6} {item.fan_out:>7} "
            f"{item.mean:> .6f} {item.std:.6f} {item.abs_max:.6f}"
        )
    input_stats = input_magnitude_statistics(events)
    print(
        "\nInput: "
        f"nonzero_fraction={input_stats.nonzero_fraction:.6f} "
        f"active_mean={input_stats.mean_active_magnitude:.6f} "
        f"max={input_stats.max_magnitude:.6f} "
        f"events_per_active_pixel={input_stats.events_per_active_pixel:.6f}"
    )

    model.reset_state()
    with torch.no_grad():
        before = model.forward_with_diagnostics(events)
    print("\nBefore calibration:")
    _print_diagnostic_table(before)
    print("\nSigned-current cancellation:")
    for name in ("S1", "S2"):
        signed = before.numerical_diagnostics.layers[name].signed_current
        assert signed is not None
        print(
            f"{name}: positive_mean_abs={signed.positive.mean_abs:.6f} "
            f"negative_mean_abs={signed.negative.mean_abs:.6f} "
            f"net_mean_abs={signed.net.mean_abs:.6f} "
            f"cancellation={signed.cancellation_fraction:.3%}"
        )

    result = calibrate_layer_gains(model, events)
    print(f"\nCalibrated gains: {result.gains}")
    model.reset_state()
    with torch.no_grad():
        after = model.forward_with_diagnostics(events)
    print("\nAfter calibration:")
    _print_diagnostic_table(after)

    print("\nPer-case calibrated spike counts:")
    print("Case                         Events      S1      S2      S3      S4      S5      S6 Primitive")
    for name, sample in calibration_samples(synthetic_config):
        case = run_case(model, name, sample)
        counts = [
            case.statistics.layers[layer].total_spikes for layer in LAYER_NAMES
        ]
        print(
            f"{name:<28} {case.input_events:>7} "
            + " ".join(f"{count:>7}" for count in counts[:-1])
            + f" {counts[-1]:>9}"
        )

    zero_sample = SyntheticMotionGenerator(synthetic_config).generate(
        SyntheticMotionCase.NO_MOTION
    )
    zero_result = run_case(model, "NO_MOTION", zero_sample)
    zero_spikes = sum(
        layer.total_spikes for layer in zero_result.statistics.layers.values()
    )
    temporal = temporal_shift_sanity(model, synthetic_config)
    locality_events = SyntheticMotionGenerator(
        replace(synthetic_config, velocity_px_per_ms=1.0)
    ).generate(SyntheticMotionCase.STATIC_BG_MOVING_OBJECT).events
    locality = check_early_layer_locality(model, locality_events)
    fast_slow = run_fast_slow_sanity()
    # Use the reference fast-motion case because it provides presynaptic
    # activity at every calibrated layer; this remains a one-step dummy loss.
    gradient_events = calibration_samples(synthetic_config)[1][1].events
    gradients = gradient_sanity(model, gradient_events)

    print(f"\nNO_MOTION zero spike check: {'PASS' if zero_spikes == 0 else 'FAIL'}")
    print(
        "Temporal shift: "
        f"{'PASS' if temporal.passed else 'FAIL'} "
        f"input={temporal.input_relative_difference:.4f} "
        f"layers={temporal.layer_relative_difference:.4f} "
        f"primitive={temporal.primitive_relative_difference:.4f}"
    )
    print(
        f"Spatial locality: {'PASS' if locality.passed else 'FAIL'} "
        f"S1_active={locality.s1_active} S2_active={locality.s2_active}"
    )
    print(f"Fast/slow dynamics unchanged: {'PASS' if fast_slow.passed else 'FAIL'}")
    print("Gradient norms:")
    for name, norm in gradients.items():
        print(f"  {name}: {norm:.8e}")

    maximum_firing = max(result.final_firing_fractions.values())
    if not (
        result.gains == SNNMotionConfig().layer_gains
        and zero_spikes == 0
        and maximum_firing <= CalibrationSettings().overactive_fraction
        and result.final_firing_fractions["primitive"] > 0.0
        and temporal.passed
        and locality.passed
        and fast_slow.passed
        and all(math.isfinite(norm) and norm > 0.0 for norm in gradients.values())
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
