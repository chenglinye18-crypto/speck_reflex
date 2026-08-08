"""Synthetic architecture and temporal-dynamics checks for the random SNN."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import torch
from torch import Tensor

from software.models.snn import (
    MultiTimescaleLIF,
    SNNMotionBackbone,
    SNNMotionStatistics,
)

from .synthetic_motion import (
    SyntheticMotionCase,
    SyntheticMotionConfig,
    SyntheticMotionGenerator,
    SyntheticMotionSample,
)


@dataclass(frozen=True, slots=True)
class SanityCaseResult:
    name: str
    input_events: int
    statistics: SNNMotionStatistics
    global_embedding_norm: float
    local_logits_mean: float
    local_logits_std: float


@dataclass(frozen=True, slots=True)
class FastSlowPatternResult:
    fast_residual: float
    slow_residual: float
    fast_spikes: int
    slow_spikes: int


@dataclass(frozen=True, slots=True)
class FastSlowSanityReport:
    alpha_fast: float
    alpha_slow: float
    decay_fast_residual: float
    decay_slow_residual: float
    patterns: dict[str, FastSlowPatternResult]
    passed: bool


@dataclass(frozen=True, slots=True)
class TemporalShiftReport:
    input_relative_difference: float
    layer_relative_difference: float
    primitive_relative_difference: float
    passed: bool


@dataclass(frozen=True, slots=True)
class SpatialLocalityReport:
    input_bounds: tuple[int, int, int, int]
    theoretical_s1_bounds: tuple[int, int, int, int]
    observed_s1_bounds: tuple[int, int, int, int] | None
    theoretical_s2_bounds: tuple[int, int, int, int]
    observed_s2_bounds: tuple[int, int, int, int] | None
    s1_active: bool
    s2_active: bool
    passed: bool


def run_case(
    model: SNNMotionBackbone, name: str, sample: SyntheticMotionSample
) -> SanityCaseResult:
    """Reset at the case boundary and collect only numerical activity summaries."""

    model.reset_state()
    with torch.no_grad():
        run = model.forward_with_stats(sample.events)
    output = run.output
    return SanityCaseResult(
        name=name,
        input_events=int(sample.events.sum().item()),
        statistics=run.statistics,
        global_embedding_norm=float(
            output.global_embedding.norm(dim=-1).mean().item()
        ),
        local_logits_mean=float(output.local_logits.mean().item()),
        local_logits_std=float(output.local_logits.std(unbiased=False).item()),
    )


def run_fast_slow_sanity() -> FastSlowSanityReport:
    """Numerically probe decay and three simple temporal input patterns."""

    patterns = {
        "short_burst": (0.6, 0.6, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0),
        "low_frequency": (0.6, 0.0, 0.6, 0.0, 0.6, 0.0, 0.6, 0.0),
        "separated_bursts": (0.6, 0.6, 0.0, 0.0, 0.0, 0.0, 0.6, 0.6),
    }
    pattern_results: dict[str, FastSlowPatternResult] = {}
    for name, values in patterns.items():
        neurons = MultiTimescaleLIF(
            2,
            tau_fast_ms=10.0,
            tau_slow_ms=40.0,
            dt_ms=1.0,
            threshold=1.0,
            fast_ratio=0.5,
        )
        fast_spikes = 0
        slow_spikes = 0
        with torch.no_grad():
            for value in values:
                current = torch.full((1, 2, 1, 1), value)
                spikes = neurons(current)
                fast_spikes += int(spikes[:, :1].sum().item())
                slow_spikes += int(spikes[:, 1:].sum().item())
        pattern_results[name] = FastSlowPatternResult(
            fast_residual=float(neurons.fast_lif.membrane_state.item()),
            slow_residual=float(neurons.slow_lif.membrane_state.item()),
            fast_spikes=fast_spikes,
            slow_spikes=slow_spikes,
        )

    decay_probe = MultiTimescaleLIF(
        2,
        tau_fast_ms=10.0,
        tau_slow_ms=40.0,
        dt_ms=1.0,
        threshold=10.0,
        fast_ratio=0.5,
    )
    with torch.no_grad():
        decay_probe(torch.full((1, 2, 1, 1), 0.5))
        for _ in range(10):
            decay_probe(torch.zeros(1, 2, 1, 1))
    alpha_fast = float(decay_probe.fast_lif.alpha.item())
    alpha_slow = float(decay_probe.slow_lif.alpha.item())
    fast_residual = float(decay_probe.fast_lif.membrane_state.item())
    slow_residual = float(decay_probe.slow_lif.membrane_state.item())
    return FastSlowSanityReport(
        alpha_fast=alpha_fast,
        alpha_slow=alpha_slow,
        decay_fast_residual=fast_residual,
        decay_slow_residual=slow_residual,
        patterns=pattern_results,
        passed=alpha_fast < alpha_slow and fast_residual < slow_residual,
    )


def _relative_difference(first: int, second: int) -> float:
    return abs(first - second) / max(first, second, 1)


def temporal_shift_sanity(
    model: SNNMotionBackbone, config: SyntheticMotionConfig
) -> TemporalShiftReport:
    # Use the fast geometric case so the random v0.1 network has measurable S1
    # activity; this is still an architecture probe, not a direction metric.
    config = replace(config, velocity_px_per_ms=1.0)
    first = SyntheticMotionGenerator(
        replace(config, motion_start_ms=10.0)
    ).generate(SyntheticMotionCase.GLOBAL_LEFT)
    second = SyntheticMotionGenerator(
        replace(config, motion_start_ms=11.0)
    ).generate(SyntheticMotionCase.GLOBAL_LEFT)
    result_a = run_case(model, "shift_10ms", first)
    result_b = run_case(model, "shift_11ms", second)
    layer_a = sum(
        result_a.statistics.layers[f"S{index}"].total_spikes
        for index in range(1, 7)
    )
    layer_b = sum(
        result_b.statistics.layers[f"S{index}"].total_spikes
        for index in range(1, 7)
    )
    primitive_a = result_a.statistics.layers["primitive"].total_spikes
    primitive_b = result_b.statistics.layers["primitive"].total_spikes
    input_difference = _relative_difference(result_a.input_events, result_b.input_events)
    layer_difference = _relative_difference(layer_a, layer_b)
    primitive_difference = _relative_difference(primitive_a, primitive_b)
    return TemporalShiftReport(
        input_relative_difference=input_difference,
        layer_relative_difference=layer_difference,
        primitive_relative_difference=primitive_difference,
        passed=(
            input_difference <= 0.35
            and layer_difference <= 0.50
            and primitive_difference <= 0.50
        ),
    )


def _support_bounds(activity: Tensor) -> tuple[int, int, int, int] | None:
    coordinates = torch.nonzero(activity, as_tuple=False)
    if coordinates.numel() == 0:
        return None
    y = coordinates[:, 0]
    x = coordinates[:, 1]
    return int(y.min()), int(y.max()), int(x.min()), int(x.max())


def _conv_support_bounds(
    bounds: tuple[int, int, int, int],
    input_shape: tuple[int, int],
    *,
    kernel: int,
    stride: int,
    padding: int,
) -> tuple[tuple[int, int, int, int], tuple[int, int]]:
    y0, y1, x0, x1 = bounds
    input_height, input_width = input_shape
    output_height = math.floor((input_height + 2 * padding - kernel) / stride) + 1
    output_width = math.floor((input_width + 2 * padding - kernel) / stride) + 1
    out_y0 = max(0, math.ceil((y0 + padding - (kernel - 1)) / stride))
    out_y1 = min(output_height - 1, math.floor((y1 + padding) / stride))
    out_x0 = max(0, math.ceil((x0 + padding - (kernel - 1)) / stride))
    out_x1 = min(output_width - 1, math.floor((x1 + padding) / stride))
    return (out_y0, out_y1, out_x0, out_x1), (output_height, output_width)


def _is_contained(
    observed: tuple[int, int, int, int] | None,
    expected: tuple[int, int, int, int],
) -> bool:
    if observed is None:
        return True
    return (
        expected[0] <= observed[0] <= observed[1] <= expected[1]
        and expected[2] <= observed[2] <= observed[3] <= expected[3]
    )


def check_early_layer_locality(
    model: SNNMotionBackbone, events: Tensor
) -> SpatialLocalityReport:
    """Compare accumulated S1/S2 support with exact convolution fan-out bounds."""

    input_activity = events.ne(0).any(dim=(0, 1, 2))
    input_bounds = _support_bounds(input_activity)
    if input_bounds is None:
        raise ValueError("spatial locality requires at least one input event")

    observed: dict[str, tuple[int, int, int, int] | None] = {"S1": None, "S2": None}

    def make_hook(name: str):
        def hook(_module, _inputs, output: Tensor) -> None:
            current = _support_bounds(output.detach().ne(0).any(dim=(0, 1)))
            previous = observed[name]
            if current is None:
                return
            if previous is None:
                observed[name] = current
            else:
                observed[name] = (
                    min(previous[0], current[0]),
                    max(previous[1], current[1]),
                    min(previous[2], current[2]),
                    max(previous[3], current[3]),
                )

        return hook

    handles = [
        model.stages[0].register_forward_hook(make_hook("S1")),
        model.stages[1].register_forward_hook(make_hook("S2")),
    ]
    model.reset_state()
    try:
        with torch.no_grad():
            model(events)
    finally:
        for handle in handles:
            handle.remove()

    s1_expected, s1_shape = _conv_support_bounds(
        input_bounds,
        (events.shape[-2], events.shape[-1]),
        kernel=5,
        stride=2,
        padding=2,
    )
    s2_expected, _ = _conv_support_bounds(
        s1_expected,
        s1_shape,
        kernel=3,
        stride=1,
        padding=1,
    )
    return SpatialLocalityReport(
        input_bounds=input_bounds,
        theoretical_s1_bounds=s1_expected,
        observed_s1_bounds=observed["S1"],
        theoretical_s2_bounds=s2_expected,
        observed_s2_bounds=observed["S2"],
        s1_active=observed["S1"] is not None,
        s2_active=observed["S2"] is not None,
        passed=_is_contained(observed["S1"], s1_expected)
        and _is_contained(observed["S2"], s2_expected),
    )


def determinism_sanity(
    model: SNNMotionBackbone, generator: SyntheticMotionGenerator
) -> bool:
    first = generator.generate(SyntheticMotionCase.MOVING_BG_MOVING_OBJECT)
    second = generator.generate(SyntheticMotionCase.MOVING_BG_MOVING_OBJECT)
    if not torch.equal(first.events, second.events) or not torch.equal(
        first.independent_motion_mask, second.independent_motion_mask
    ):
        return False
    model.reset_state()
    with torch.no_grad():
        first_output = model(first.events)
    model.reset_state()
    with torch.no_grad():
        second_output = model(second.events)
    return all(
        torch.equal(first_tensor, second_tensor)
        for first_tensor, second_tensor in (
            (first_output.primitive_spikes, second_output.primitive_spikes),
            (first_output.local_logits, second_output.local_logits),
            (first_output.global_embedding, second_output.global_embedding),
        )
    )


def _case_samples(config: SyntheticMotionConfig) -> list[tuple[str, SyntheticMotionSample]]:
    cases = [
        ("NO_MOTION", SyntheticMotionCase.NO_MOTION, 0.5),
        ("GLOBAL_LEFT_SLOW", SyntheticMotionCase.GLOBAL_LEFT, 0.25),
        ("GLOBAL_LEFT_FAST", SyntheticMotionCase.GLOBAL_LEFT, 1.0),
        ("GLOBAL_RIGHT", SyntheticMotionCase.GLOBAL_RIGHT, 0.5),
        ("STATIC_BG_MOVING_OBJECT", SyntheticMotionCase.STATIC_BG_MOVING_OBJECT, 1.0),
        ("MOVING_BG_MOVING_OBJECT", SyntheticMotionCase.MOVING_BG_MOVING_OBJECT, 0.5),
        ("EXPANSION", SyntheticMotionCase.EXPANSION, 0.5),
        ("CONTRACTION", SyntheticMotionCase.CONTRACTION, 0.5),
    ]
    return [
        (
            name,
            SyntheticMotionGenerator(
                replace(config, velocity_px_per_ms=velocity)
            ).generate(case),
        )
        for name, case, velocity in cases
    ]


def main() -> None:
    torch.manual_seed(17)
    config = SyntheticMotionConfig(seed=17)
    model = SNNMotionBackbone()
    results = [run_case(model, name, sample) for name, sample in _case_samples(config)]

    print("SYNTHETIC_MOTION_SANITY_V0.1")
    print("mode=architecture_temporal_dynamics_random_weights")
    header = "Case                         Events      S1      S2      S3      S4      S5      S6 Primitive"
    print(header)
    print("-" * len(header))
    for result in results:
        counts = [
            result.statistics.layers[f"S{index}"].total_spikes
            for index in range(1, 7)
        ]
        primitive = result.statistics.layers["primitive"].total_spikes
        print(
            f"{result.name:<28} {result.input_events:>7} "
            + " ".join(f"{count:>7}" for count in counts)
            + f" {primitive:>9}"
        )

    print("\nReadout/activity summaries (not task metrics):")
    for result in results:
        s1 = result.statistics.layers["S1"]
        primitive = result.statistics.layers["primitive"]
        fast = sum(layer.fast_spikes for layer in result.statistics.layers.values())
        slow = sum(layer.slow_spikes for layer in result.statistics.layers.values())
        print(
            f"{result.name}: S1_firing={s1.spikes_per_neuron_per_timestep:.6f} "
            f"primitive_firing={primitive.spikes_per_neuron_per_timestep:.6f} "
            f"fast/slow={fast}/{slow} global_norm={result.global_embedding_norm:.6f} "
            f"local_mean/std={result.local_logits_mean:.6f}/{result.local_logits_std:.6f}"
        )

    fast_slow = run_fast_slow_sanity()
    print(
        "\nFast/slow LIF decay check: "
        f"{'PASS' if fast_slow.passed else 'FAIL'} "
        f"alpha={fast_slow.alpha_fast:.6f}/{fast_slow.alpha_slow:.6f} "
        f"residual_10ms={fast_slow.decay_fast_residual:.6f}/{fast_slow.decay_slow_residual:.6f}"
    )
    for name, pattern in fast_slow.patterns.items():
        print(
            f"  {name}: membrane={pattern.fast_residual:.6f}/{pattern.slow_residual:.6f} "
            f"spikes={pattern.fast_spikes}/{pattern.slow_spikes}"
        )

    temporal = temporal_shift_sanity(model, config)
    print(
        "Temporal shift sensitivity: "
        f"{'PASS' if temporal.passed else 'FAIL'} "
        f"input={temporal.input_relative_difference:.4f} "
        f"layers={temporal.layer_relative_difference:.4f} "
        f"primitive={temporal.primitive_relative_difference:.4f}"
    )
    locality_sample = SyntheticMotionGenerator(
        replace(config, velocity_px_per_ms=1.0)
    ).generate(SyntheticMotionCase.STATIC_BG_MOVING_OBJECT)
    locality = check_early_layer_locality(model, locality_sample.events)
    print(
        "Spatial locality check: "
        f"{'PASS' if locality.passed else 'FAIL'} "
        f"S1_active={locality.s1_active} S2_active={locality.s2_active}"
    )
    deterministic = determinism_sanity(model, SyntheticMotionGenerator(config))
    print(f"Determinism: {'PASS' if deterministic else 'FAIL'}")

    no_motion = results[0]
    no_motion_spikes = sum(
        layer.total_spikes for layer in no_motion.statistics.layers.values()
    )
    if not (
        no_motion.input_events == 0
        and no_motion_spikes == 0
        and fast_slow.passed
        and temporal.passed
        and locality.passed
        and deterministic
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
