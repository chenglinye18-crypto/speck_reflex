#!/usr/bin/env python3
"""CUDA runtime comparison for SNNMotionBackbone O2/O3 inference paths.

This is a measurement-only tool.  It does not change the model topology,
weights, or neuron equations.  The per-stage path mirrors ``forward`` in this
file so CUDA events can be placed between its existing operations; its totals
are diagnostic and must not replace the normal-forward measurement.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable

import torch
from torch import Tensor
from torch.nn import functional as F

# This script is deliberately runnable from an uninstalled checkout.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from software.dvs_bridge import LiveEgoMotionModel
from software.models.snn.motion_backbone import SNNMotionBackbone


STAGE_NAMES = ("S1", "S2", "S3", "S4", "S5", "S6", "primitive")
HEAD_NAMES = ("local_motion_head", "adaptive_pool", "ego_motion_head", "stack_output")


def percentile(values: list[float], fraction: float) -> float:
    """Linear percentile without a NumPy dependency."""

    ordered = sorted(values)
    if not ordered:
        raise ValueError("no samples")
    position = (len(ordered) - 1) * fraction
    lower, upper = int(position), min(int(position) + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "mean": statistics.fmean(values),
        "min": min(values),
        "max": max(values),
    }


def fmt_summary(values: list[float]) -> str:
    summary = summarize(values)
    return " ".join(f"{key}={value:.3f}" for key, value in summary.items())


def cuda_elapsed_ms(device: torch.device, operation: Callable[[], object]) -> float:
    """Time GPU work on the current stream using CUDA events."""

    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    operation()
    end.record()
    torch.cuda.synchronize(device)
    return start.elapsed_time(end)


def reset_elapsed_ms(model: torch.nn.Module) -> float:
    """``reset_state`` itself is host-only pointer clearing, so use CPU timing."""

    started = time.perf_counter_ns()
    model.reset_state()
    return (time.perf_counter_ns() - started) / 1_000_000.0


def make_cpu_input(timesteps: int) -> Tensor:
    # Fixed, low-rate non-negative counts.  Generated once outside all benchmarks.
    generator = torch.Generator(device="cpu").manual_seed(20260814)
    return torch.poisson(
        torch.full((1, timesteps, 2, 96, 128), 0.08), generator=generator
    ).contiguous()


def warmup(model: torch.nn.Module, event_bins: Tensor, iterations: int, device: torch.device) -> None:
    with torch.inference_mode():
        for _ in range(iterations):
            model.reset_state()
            model(event_bins)
            model.reset_state()
    torch.cuda.synchronize(device)


def measure_baseline(
    model: torch.nn.Module, cpu_input: Tensor, device: torch.device, runs: int
) -> dict[str, list[float]]:
    """Measure each ordinary inference-path boundary without changing forward."""

    measurements: dict[str, list[float]] = defaultdict(list)
    with torch.inference_mode():
        for _ in range(runs):
            torch.cuda.synchronize(device)
            wall_start = time.perf_counter()
            copy_start, copy_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            copy_start.record()
            gpu_input = cpu_input.to(device)
            copy_end.record()
            measurements["reset_before"].append(reset_elapsed_ms(model))
            forward_start, forward_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            forward_start.record()
            model(gpu_input)
            forward_end.record()
            measurements["reset_after"].append(reset_elapsed_ms(model))
            torch.cuda.synchronize(device)
            measurements["H2D"].append(copy_start.elapsed_time(copy_end))
            measurements["forward"].append(forward_start.elapsed_time(forward_end))
            measurements["total"].append((time.perf_counter() - wall_start) * 1_000.0)
    return measurements


def normal_forward_samples(
    model: torch.nn.Module, event_bins: Tensor, device: torch.device, runs: int
) -> list[float]:
    samples: list[float] = []
    with torch.inference_mode():
        for _ in range(runs):
            model.reset_state()
            samples.append(cuda_elapsed_ms(device, lambda: model(event_bins)))
            model.reset_state()
    return samples


def equivalence_summary(
    reference: SNNMotionBackbone,
    candidate: SNNMotionBackbone,
    event_bins: Tensor,
    device: torch.device,
) -> None:
    """Print exact checkpoint-level output and final-state comparisons."""

    reference.reset_state()
    candidate.reset_state()
    with torch.inference_mode():
        expected = reference(event_bins)
        actual = candidate(event_bins)
    torch.cuda.synchronize(device)
    print(f"\nEquivalence: {candidate.execution_mode} vs time_major")
    for name in ("primitive_spikes", "local_logits", "global_embedding", "ego_motion"):
        expected_tensor = getattr(expected, name)
        actual_tensor = getattr(actual, name)
        if expected_tensor is None or actual_tensor is None:
            print(f"{name}: exact={expected_tensor is actual_tensor}")
            continue
        difference = (expected_tensor - actual_tensor).abs()
        mismatch = int(torch.count_nonzero(difference).item())
        differing = torch.nonzero(difference, as_tuple=False)
        first = differing[0].tolist() if mismatch else None
        print(
            f"{name}: exact={torch.equal(expected_tensor, actual_tensor)} "
            f"max_abs={difference.max().item():.9g} "
            f"mean_abs={difference.mean().item():.9g} mismatch={mismatch} "
            f"first_index={first}"
        )
    for name, expected_state, actual_state in zip(
        STAGE_NAMES,
        reference.membrane_states(),
        candidate.membrane_states(),
        strict=True,
    ):
        difference = (expected_state - actual_state).abs()
        print(
            f"state {name}: exact={torch.equal(expected_state, actual_state)} "
            f"max_abs={difference.max().item():.9g} "
            f"mismatch={int(torch.count_nonzero(difference).item())}"
        )
    reference.reset_state()
    candidate.reset_state()


def diagnostic_stage_once(model: torch.nn.Module, event_bins: Tensor, device: torch.device) -> dict[str, float]:
    """Run forward-equivalent operations with events at every existing boundary."""

    events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = defaultdict(list)

    def timed(name: str, operation: Callable[[], Tensor]) -> Tensor:
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        result = operation()
        end.record()
        events[name].append((start, end))
        return result

    model.reset_state()
    primitive_steps: list[Tensor] = []
    local_steps: list[Tensor] = []
    embedding_steps: list[Tensor] = []
    ego_steps: list[Tensor] = []
    with torch.inference_mode():
        for time_index in range(event_bins.shape[1]):
            spikes = event_bins[:, time_index]
            for name, stage in zip(STAGE_NAMES[:6], model.stages, strict=True):
                spikes = timed(name, lambda stage=stage, spikes=spikes: stage(spikes))
            primitive = timed("primitive", lambda: model.primitive_bottleneck(spikes))
            local_logits = timed("local_motion_head", lambda: model.local_motion_head(primitive))
            embedding = timed(
                "adaptive_pool", lambda: F.adaptive_avg_pool2d(primitive, output_size=(2, 2)).flatten(1)
            )
            primitive_steps.append(primitive)
            local_steps.append(local_logits)
            embedding_steps.append(embedding)
            if model.ego_motion_head is not None:
                ego_steps.append(timed("ego_motion_head", lambda: model.ego_motion_head(embedding)))

        # Match the four output constructions in SNNMotionBackbone.forward.
        stack_start, stack_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        stack_start.record()
        stacked_outputs = (
            torch.stack(primitive_steps, 1),
            torch.stack(local_steps, 1),
            torch.stack(embedding_steps, 1),
            torch.stack(ego_steps, 1),
        )
        stack_end.record()
        events["stack_output"].append((stack_start, stack_end))
    torch.cuda.synchronize(device)
    model.reset_state()
    return {name: sum(start.elapsed_time(end) for start, end in pairs) for name, pairs in events.items()}


def conv_lif_once(model: torch.nn.Module, event_bins: Tensor, device: torch.device) -> dict[str, float]:
    """Forward-equivalent block decomposition into conv and LIF calls."""

    events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = defaultdict(list)

    def timed(name: str, operation: Callable[[], Tensor]) -> Tensor:
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        result = operation()
        end.record()
        events[name].append((start, end))
        return result

    blocks = tuple(model.stages) + (model.primitive_bottleneck,)
    model.reset_state()
    with torch.inference_mode():
        for time_index in range(event_bins.shape[1]):
            spikes = event_bins[:, time_index]
            for block in blocks:
                current = timed("Conv", lambda block=block, spikes=spikes: block.conv(spikes))
                spikes = timed("LIF", lambda block=block, current=current: block.neurons(current))
            primitive = spikes
            # Execute the remaining existing operations so this follows normal state/output work.
            local_logits = timed("other", lambda: model.local_motion_head(primitive))
            embedding = timed("other", lambda: F.adaptive_avg_pool2d(primitive, (2, 2)).flatten(1))
            if model.ego_motion_head is not None:
                timed("other", lambda: model.ego_motion_head(embedding))
    torch.cuda.synchronize(device)
    model.reset_state()
    return {name: sum(start.elapsed_time(end) for start, end in pairs) for name, pairs in events.items()}


def aggregate_diagnostic(
    fn: Callable[[], dict[str, float]], runs: int
) -> dict[str, list[float]]:
    values: dict[str, list[float]] = defaultdict(list)
    for _ in range(runs):
        for name, elapsed in fn().items():
            values[name].append(elapsed)
    return values


def profiler_summary(
    label: str, model: torch.nn.Module, event_bins: Tensor, device: torch.device, runs: int
) -> dict[str, int]:
    """Print concise profiler counters; deliberately no large trace file."""

    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    model.reset_state()
    with torch.inference_mode(), torch.profiler.profile(activities=activities, profile_memory=True) as prof:
        for _ in range(runs):
            model.reset_state()
            model(event_bins)
            model.reset_state()
    torch.cuda.synchronize(device)
    cuda_events = sum(
        1 for event in prof.events() if event.device_type == torch.autograd.DeviceType.CUDA
    )
    op_counts = {event.key: event.count for event in prof.key_averages()}
    print(f"\nProfiler Evidence: {label} (instrumented short run; no trace file written)")
    print(f"CUDA kernel/activity events: {cuda_events}")
    print(
        "op calls: "
        f"conv2d={op_counts.get('aten::conv2d', 0)} "
        f"cudnn_convolution={op_counts.get('aten::cudnn_convolution', 0)} "
        f"cat={op_counts.get('aten::cat', 0)} "
        f"split={op_counts.get('aten::split_with_sizes', 0) + op_counts.get('aten::split', 0)} "
        f"mul={op_counts.get('aten::mul', 0)} add={op_counts.get('aten::add', 0)} "
        f"sub={op_counts.get('aten::sub', 0)} ge={op_counts.get('aten::ge', 0)} "
        f"copy_={op_counts.get('aten::copy_', 0)} "
        f"clone={op_counts.get('aten::clone', 0)} "
        f"contiguous={op_counts.get('aten::contiguous', 0)} "
        f"reshape={op_counts.get('aten::reshape', 0)}"
    )
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20))
    return {
        "cuda_activities": cuda_events,
        "conv2d": op_counts.get("aten::conv2d", 0),
        "cudnn_convolution": op_counts.get("aten::cudnn_convolution", 0),
        "cat": op_counts.get("aten::cat", 0),
        "split": op_counts.get("aten::split_with_sizes", 0) + op_counts.get("aten::split", 0),
        "mul": op_counts.get("aten::mul", 0),
        "add": op_counts.get("aten::add", 0),
        "sub": op_counts.get("aten::sub", 0),
        "ge": op_counts.get("aten::ge", 0),
        "copy_": op_counts.get("aten::copy_", 0),
        "clone": op_counts.get("aten::clone", 0),
        "contiguous": op_counts.get("aten::contiguous", 0),
        "reshape": op_counts.get("aten::reshape", 0),
    }


def print_table(title: str, measurements: dict[str, list[float]], ordered_names: Iterable[str]) -> None:
    print(f"\n{title}")
    print(f"{'Metric':<24} {'p50 ms':>10} {'p95 ms':>10} {'mean ms':>10} {'min ms':>10} {'max ms':>10}")
    for name in ordered_names:
        values = measurements[name]
        summary = summarize(values)
        print(f"{name:<24} {summary['p50']:10.3f} {summary['p95']:10.3f} {summary['mean']:10.3f} {summary['min']:10.3f} {summary['max']:10.3f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--profiler-runs", type=int, default=2)
    parser.add_argument("--experiment", choices=("o2", "o3"), default="o3")
    return parser.parse_args()


def make_variant(
    reference: SNNMotionBackbone,
    *,
    lif_implementation: str,
    inference_fast_spike: bool,
    compiled_lif_mode: str = "none",
    first_step_specialization: bool = False,
    lif_step_primitive: str = "mul_add",
    execution_mode: str = "time_major",
    device: torch.device,
) -> SNNMotionBackbone:
    """Build a runtime-only variant from exactly the loaded checkpoint state."""

    variant = SNNMotionBackbone(
        reference.config,
        lif_implementation=lif_implementation,
        inference_fast_spike=inference_fast_spike,
        compiled_lif_mode=compiled_lif_mode,
        first_step_specialization=first_step_specialization,
        lif_step_primitive=lif_step_primitive,
        execution_mode=execution_mode,
    ).to(device)
    variant.load_state_dict(reference.state_dict(), strict=True)
    variant.eval()
    return variant


def temporal_stage_profile(
    model: SNNMotionBackbone,
    event_bins: Tensor,
    device: torch.device,
    *,
    stage_major: bool,
) -> tuple[dict[str, float], dict[str, bool]]:
    """Measure stage Conv and sequential LIF scan without per-stage synchronizations."""

    blocks = (*model.stages, model.primitive_bottleneck)
    names = ("S1", "S2", "S3", "S4", "S5", "S6", "primitive")
    events: dict[str, tuple[torch.cuda.Event, torch.cuda.Event]] = {}
    layout: dict[str, bool] = {}
    spikes = event_bins
    model.reset_state()
    with torch.inference_mode():
        if stage_major:
            chunk_sizes = (
                model._EXACT_TEMPORAL_BATCH_SIZES
                if model.execution_mode == "stage_major_chunked"
                else (event_bins.shape[1],) * len(blocks)
            )
            for name, block, configured_chunk in zip(
                names, blocks, chunk_sizes, strict=True
            ):
                layout[name] = spikes.is_contiguous()
                batch, timesteps, channels, height, width = spikes.shape
                conv_pairs: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
                lif_pairs: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
                spike_steps: list[Tensor] = []
                chunk = min(configured_chunk, timesteps)
                for start in range(0, timesteps, chunk):
                    temporal_chunk = spikes[:, start : start + chunk]
                    chunk_timesteps = temporal_chunk.shape[1]
                    conv_start, conv_end = torch.cuda.Event(True), torch.cuda.Event(True)
                    conv_start.record()
                    merged = temporal_chunk.reshape(
                        batch * chunk_timesteps, channels, height, width
                    )
                    currents = block.conv(merged)
                    conv_end.record()
                    conv_pairs.append((conv_start, conv_end))
                    current_sequence = currents.reshape(
                        batch, chunk_timesteps, *currents.shape[1:]
                    )
                    lif_start, lif_end = torch.cuda.Event(True), torch.cuda.Event(True)
                    lif_start.record()
                    spike_steps.extend(
                        block.neurons(current_sequence[:, index])
                        for index in range(chunk_timesteps)
                    )
                    lif_end.record()
                    lif_pairs.append((lif_start, lif_end))
                spikes = torch.stack(spike_steps, 1)
                events[name + ":conv"] = conv_pairs  # type: ignore[assignment]
                events[name + ":lif"] = lif_pairs  # type: ignore[assignment]
        else:
            stage_events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = defaultdict(list)
            output_steps: list[list[Tensor]] = [[] for _ in blocks]
            for time_index in range(event_bins.shape[1]):
                current_spikes = event_bins[:, time_index]
                for block_index, (name, block) in enumerate(zip(names, blocks, strict=True)):
                    conv_start, conv_end = torch.cuda.Event(True), torch.cuda.Event(True)
                    conv_start.record()
                    currents = block.conv(current_spikes)
                    conv_end.record()
                    lif_start, lif_end = torch.cuda.Event(True), torch.cuda.Event(True)
                    lif_start.record()
                    current_spikes = block.neurons(currents)
                    lif_end.record()
                    stage_events[name + ":conv"].append((conv_start, conv_end))
                    stage_events[name + ":lif"].append((lif_start, lif_end))
                    output_steps[block_index].append(current_spikes)
            for key, pairs in stage_events.items():
                # Keep individual pairs; elapsed values are summed after synchronization.
                events[key] = pairs  # type: ignore[assignment]
    torch.cuda.synchronize(device)
    elapsed: dict[str, float] = {}
    for key, value in events.items():
        if isinstance(value, list):
            elapsed[key] = sum(start.elapsed_time(end) for start, end in value)
        else:
            elapsed[key] = value[0].elapsed_time(value[1])
    model.reset_state()
    return elapsed, layout


def peak_forward_memory_mb(
    model: SNNMotionBackbone, event_bins: Tensor, device: torch.device
) -> tuple[float, float]:
    """Return absolute and incremental allocated-memory peaks for one forward."""

    model.reset_state()
    torch.cuda.synchronize(device)
    baseline = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        output = model(event_bins)
    torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_allocated(device)
    del output
    model.reset_state()
    return peak / 2**20, (peak - baseline) / 2**20


def operation_samples(
    model: SNNMotionBackbone,
    operation: Callable[[], Tensor],
    device: torch.device,
    *,
    warmup_runs: int,
    measured_runs: int,
) -> list[float]:
    """Benchmark a reset-isolated model operation with CUDA events."""

    with torch.inference_mode():
        for _ in range(warmup_runs):
            model.reset_state()
            operation()
            model.reset_state()
    torch.cuda.synchronize(device)
    samples: list[float] = []
    with torch.inference_mode():
        for _ in range(measured_runs):
            model.reset_state()
            samples.append(cuda_elapsed_ms(device, operation))
            model.reset_state()
    return samples


def peak_operation_memory_mb(
    model: SNNMotionBackbone,
    operation: Callable[[], Tensor],
    device: torch.device,
) -> tuple[float, float]:
    model.reset_state()
    torch.cuda.synchronize(device)
    baseline = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        output = operation()
    torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_allocated(device)
    del output
    model.reset_state()
    return peak / 2**20, (peak - baseline) / 2**20


def profile_operation(
    label: str,
    model: SNNMotionBackbone,
    operation: Callable[[], Tensor],
    device: torch.device,
    runs: int,
) -> dict[str, int]:
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    model.reset_state()
    with torch.inference_mode(), torch.profiler.profile(activities=activities) as prof:
        for _ in range(runs):
            model.reset_state()
            operation()
            model.reset_state()
    torch.cuda.synchronize(device)
    cuda_events = sum(
        1 for event in prof.events() if event.device_type == torch.autograd.DeviceType.CUDA
    )
    counts = {event.key: event.count for event in prof.key_averages()}
    summary = {
        "cuda_activities": cuda_events,
        "conv2d": counts.get("aten::conv2d", 0),
        "adaptive_pool": counts.get("aten::adaptive_avg_pool2d", 0),
        "linear": counts.get("aten::linear", 0),
        "copy_": counts.get("aten::copy_", 0),
        "reshape": counts.get("aten::reshape", 0),
        "clone": counts.get("aten::clone", 0),
        "contiguous": counts.get("aten::contiguous", 0),
    }
    print(f"Profiler {label}: " + " ".join(f"{key}={value}" for key, value in summary.items()))
    return summary


def run_o3(args: argparse.Namespace, device: torch.device) -> None:
    """Profile exact ego-only inference and the rejected numerical candidates."""

    loaded = LiveEgoMotionModel.load(args.checkpoint, device=args.device)
    model = make_variant(
        loaded.model,
        lif_implementation="fused",
        inference_fast_spike=True,
        execution_mode="stage_major_chunked",
        device=device,
    )
    event_bins = make_cpu_input(64).to(device)
    print("Environment")
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"torch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"checkpoint: {args.checkpoint.resolve()}")
    print("input: [1, 64, 2, 96, 128], float32; fixed Poisson count tensor (rate=0.08)")

    model.reset_state()
    with torch.inference_mode():
        full_output = model(event_bins)
    torch.cuda.synchronize(device)
    if full_output.ego_motion is None:
        raise RuntimeError("checkpoint does not have an ego-motion head")
    reference_ego = full_output.ego_motion.mean(dim=1)
    reference_states = tuple(state.clone() for state in model.membrane_states())
    primitives = full_output.primitive_spikes

    with torch.inference_mode():
        loop_pool = torch.stack(
            [
                F.adaptive_avg_pool2d(primitives[:, index], (2, 2)).flatten(1)
                for index in range(primitives.shape[1])
            ],
            1,
        )
        batch, timesteps = primitives.shape[:2]
        merged = primitives.reshape(batch * timesteps, *primitives.shape[2:])
        batch_pool = F.adaptive_avg_pool2d(merged, (2, 2)).flatten(1).reshape(
            batch, timesteps, -1
        )
        loop_head = torch.stack(
            [model.ego_motion_head(loop_pool[:, index]) for index in range(timesteps)],
            1,
        )
        batch_head = model.ego_motion_head(loop_pool.reshape(batch * timesteps, -1)).reshape(
            batch, timesteps, 6
        )
        mean_before = model.ego_motion_head(loop_pool.mean(1))
    print("\nHead Micro-Equivalence")
    for label, expected, actual in (
        ("O3-B pool", loop_pool, batch_pool),
        ("O3-C linear sequence", loop_head, batch_head),
        ("O3-C final mean", reference_ego, batch_head.mean(1)),
        ("O3-D mean-before-linear", reference_ego, mean_before),
    ):
        difference = (expected - actual).abs()
        print(
            f"{label}: exact={torch.equal(expected, actual)} "
            f"max_abs={difference.max().item():.9g} "
            f"mean_abs={difference.mean().item():.9g} "
            f"different={int(torch.count_nonzero(difference).item())}"
        )

    candidates: dict[str, Callable[[], Tensor]] = {
        "O3-A skip local": lambda: model.forward_ego_motion(
            event_bins, temporal_pool=False
        ),
        "O3-B batch pool": lambda: model.forward_ego_motion(event_bins),
        "O3-C batch linear": lambda: model.forward_ego_motion(
            event_bins, temporal_pool=False, temporal_head=True
        ),
        "O3-D mean before": lambda: model.forward_ego_motion(
            event_bins, temporal_pool=False, mean_before_head=True
        ),
    }
    print("\nFull Equivalence")
    for label, operation in candidates.items():
        captured_primitives: list[Tensor] = []
        handle = model.primitive_bottleneck.neurons.register_forward_hook(
            lambda _module, _inputs, output: captured_primitives.append(output)
        )
        model.reset_state()
        try:
            with torch.inference_mode():
                candidate_ego = operation()
        finally:
            handle.remove()
        torch.cuda.synchronize(device)
        difference = (reference_ego - candidate_ego).abs()
        candidate_primitives = torch.stack(captured_primitives, 1)
        states_exact = all(
            torch.equal(expected, actual)
            for expected, actual in zip(
                reference_states, model.membrane_states(), strict=True
            )
        )
        print(
            f"{label}: ego_exact={torch.equal(reference_ego, candidate_ego)} "
            f"max_abs={difference.max().item():.9g} "
            f"mean_abs={difference.mean().item():.9g} "
            f"primitive_exact={torch.equal(primitives, candidate_primitives)} "
            f"states_exact={states_exact}"
        )
    model.reset_state()

    operations: dict[str, Callable[[], Tensor]] = {
        "O2 full": lambda: model(event_bins).ego_motion.mean(1),  # type: ignore[union-attr]
        **candidates,
    }
    results: dict[str, dict[str, float]] = {}
    print("\nO3 Performance (CUDA events)")
    print(
        f"{'Variant':<24} {'p50 ms':>10} {'p95 ms':>10} {'mean ms':>10} "
        f"{'min ms':>10} {'max ms':>10} {'speedup':>10}"
    )
    for label, operation in operations.items():
        results[label] = summarize(
            operation_samples(
                model,
                operation,
                device,
                warmup_runs=args.warmup,
                measured_runs=args.runs,
            )
        )
    baseline = results["O2 full"]["p50"]
    for label, result in results.items():
        print(
            f"{label:<24} {result['p50']:10.3f} {result['p95']:10.3f} "
            f"{result['mean']:10.3f} {result['min']:10.3f} {result['max']:10.3f} "
            f"{baseline / result['p50']:9.3f}x"
        )
    print("O3-best = O3-B batch pool (exact)")

    for label in ("O2 full", "O3-B batch pool"):
        absolute, incremental = peak_operation_memory_mb(
            model, operations[label], device
        )
        print(
            f"Memory {label}: absolute_peak_mb={absolute:.3f} "
            f"incremental_forward_mb={incremental:.3f}"
        )
    profile_operation(
        "O2 full", model, operations["O2 full"], device, args.profiler_runs
    )
    profile_operation(
        "O3-best", model, operations["O3-B batch pool"], device, args.profiler_runs
    )


def main() -> None:
    args = parse_args()
    if args.device != "cuda" or not torch.cuda.is_available():
        raise SystemExit("This baseline profiler requires an available CUDA device (--device cuda).")
    if args.warmup < 1 or args.runs < 2 or args.profiler_runs < 1:
        raise SystemExit("--warmup >= 1, --runs >= 2, and --profiler-runs >= 1 are required.")

    device = torch.device(args.device)
    if args.experiment == "o3":
        run_o3(args, device)
        return
    loaded = LiveEgoMotionModel.load(args.checkpoint, device=args.device)
    reference = loaded.model
    variants = {
        "R3 time-major": make_variant(
            reference,
            lif_implementation="fused",
            inference_fast_spike=True,
            device=device,
        ),
        "O2 stage-major": make_variant(
            reference,
            lif_implementation="fused",
            inference_fast_spike=True,
            execution_mode="stage_major",
            device=device,
        ),
        "O2 exact chunked": make_variant(
            reference,
            lif_implementation="fused",
            inference_fast_spike=True,
            execution_mode="stage_major_chunked",
            device=device,
        ),
    }
    print("Environment")
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"torch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"checkpoint: {args.checkpoint.resolve()}")
    print("input: [1, 64, 2, 96, 128], float32; fixed Poisson count tensor (rate=0.08)")

    cpu_input = make_cpu_input(64)
    gpu_input = cpu_input.to(device)
    equivalence_summary(
        variants["R3 time-major"], variants["O2 stage-major"], gpu_input, device
    )
    equivalence_summary(
        variants["R3 time-major"], variants["O2 exact chunked"], gpu_input, device
    )
    variant_results: dict[str, dict[str, float]] = {}
    print("\nO2 Performance (normal forward; CUDA events)")
    print(
        f"{'Variant':<28} {'p50 ms':>10} {'p95 ms':>10} {'mean ms':>10} "
        f"{'min ms':>10} {'max ms':>10} {'speedup':>10}"
    )
    for label, model in variants.items():
        warmup(model, gpu_input, args.warmup, device)
        samples = normal_forward_samples(model, gpu_input, device, args.runs)
        variant_results[label] = summarize(samples)
    baseline_p50 = variant_results["R3 time-major"]["p50"]
    for label, summary in variant_results.items():
        print(
            f"{label:<28} {summary['p50']:10.3f} {summary['p95']:10.3f} "
            f"{summary['mean']:10.3f} {summary['min']:10.3f} {summary['max']:10.3f} "
            f"{baseline_p50 / summary['p50']:9.3f}x"
        )
    print("\nT Scaling (fixed B=1,C=2,H=96,W=128)")
    print(f"{'Variant':<16} {'T':>4} {'p50 ms':>12} {'p95 ms':>12} {'ms/timestep':>14}")
    for timesteps in (1, 8, 32, 64):
        event_bins = make_cpu_input(timesteps).to(device)
        for label, model in variants.items():
            warmup(model, event_bins, args.warmup, device)
            summary = summarize(normal_forward_samples(model, event_bins, device, args.runs))
            print(
                f"{label:<16} {timesteps:4d} {summary['p50']:12.3f} "
                f"{summary['p95']:12.3f} {summary['p50'] / timesteps:14.3f}"
            )

    diagnostic_runs = min(20, args.runs)
    print(f"\nPer-Stage Runtime ({diagnostic_runs} instrumented diagnostic runs, p50)")
    print(
        f"{'Stage':<12} {'R3 conv':>10} {'R3 LIF':>10} {'O2 conv':>10} "
        f"{'O2 LIF':>10} {'total speedup':>14}"
    )
    r3_profiles: dict[str, list[float]] = defaultdict(list)
    o2_profiles: dict[str, list[float]] = defaultdict(list)
    layout: dict[str, bool] = {}
    for _ in range(diagnostic_runs):
        r3_sample, _ = temporal_stage_profile(
            variants["R3 time-major"], gpu_input, device, stage_major=False
        )
        o2_sample, layout = temporal_stage_profile(
            variants["O2 exact chunked"], gpu_input, device, stage_major=True
        )
        for key, value in r3_sample.items():
            r3_profiles[key].append(value)
        for key, value in o2_sample.items():
            o2_profiles[key].append(value)
    for name in ("S1", "S2", "S3", "S4", "S5", "S6", "primitive"):
        r3_conv = summarize(r3_profiles[name + ":conv"])["p50"]
        r3_lif = summarize(r3_profiles[name + ":lif"])["p50"]
        o2_conv = summarize(o2_profiles[name + ":conv"])["p50"]
        o2_lif = summarize(o2_profiles[name + ":lif"])["p50"]
        print(
            f"{name:<12} {r3_conv:10.3f} {r3_lif:10.3f} "
            f"{o2_conv:10.3f} {o2_lif:10.3f} "
            f"{(r3_conv + r3_lif) / (o2_conv + o2_lif):14.3f}x"
        )
    print("stage-major input contiguity: " + " ".join(f"{key}={value}" for key, value in layout.items()))

    for label, model in variants.items():
        absolute, incremental = peak_forward_memory_mb(model, gpu_input, device)
        print(f"Memory {label}: absolute_peak_mb={absolute:.3f} incremental_forward_mb={incremental:.3f}")
    profiler_summary("R3 time-major", variants["R3 time-major"], gpu_input, device, args.profiler_runs)
    profiler_summary("O2 stage-major", variants["O2 stage-major"], gpu_input, device, args.profiler_runs)
    profiler_summary(
        "O2 exact chunked", variants["O2 exact chunked"], gpu_input, device, args.profiler_runs
    )


if __name__ == "__main__":
    main()
