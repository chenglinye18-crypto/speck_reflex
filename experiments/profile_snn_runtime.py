#!/usr/bin/env python3
"""CUDA runtime comparison for reference and fused SNNMotionBackbone LIFs.

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
        f"cat={op_counts.get('aten::cat', 0)} "
        f"split={op_counts.get('aten::split_with_sizes', 0) + op_counts.get('aten::split', 0)} "
        f"mul={op_counts.get('aten::mul', 0)} add={op_counts.get('aten::add', 0)} "
        f"sub={op_counts.get('aten::sub', 0)} ge={op_counts.get('aten::ge', 0)} "
        f"copy_={op_counts.get('aten::copy_', 0)}"
    )
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20))
    return {
        "cuda_activities": cuda_events,
        "cat": op_counts.get("aten::cat", 0),
        "split": op_counts.get("aten::split_with_sizes", 0) + op_counts.get("aten::split", 0),
        "mul": op_counts.get("aten::mul", 0),
        "add": op_counts.get("aten::add", 0),
        "sub": op_counts.get("aten::sub", 0),
        "ge": op_counts.get("aten::ge", 0),
        "copy_": op_counts.get("aten::copy_", 0),
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
    return parser.parse_args()


def make_variant(
    reference: SNNMotionBackbone,
    *,
    lif_implementation: str,
    inference_fast_spike: bool,
    compiled_lif_mode: str = "none",
    first_step_specialization: bool = False,
    lif_step_primitive: str = "mul_add",
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
    ).to(device)
    variant.load_state_dict(reference.state_dict(), strict=True)
    variant.eval()
    return variant


def main() -> None:
    args = parse_args()
    if args.device != "cuda" or not torch.cuda.is_available():
        raise SystemExit("This baseline profiler requires an available CUDA device (--device cuda).")
    if args.warmup < 1 or args.runs < 2 or args.profiler_runs < 1:
        raise SystemExit("--warmup >= 1, --runs >= 2, and --profiler-runs >= 1 are required.")

    device = torch.device(args.device)
    loaded = LiveEgoMotionModel.load(args.checkpoint, device=args.device)
    reference = loaded.model
    variants = {
        "R0 reference": reference,
        "R1 reference + fast spike": make_variant(
            reference, lif_implementation="reference", inference_fast_spike=True, device=device
        ),
        "R2 fused": make_variant(
            reference, lif_implementation="fused", inference_fast_spike=False, device=device
        ),
        "R3 fused + fast spike": make_variant(
            reference, lif_implementation="fused", inference_fast_spike=True, device=device
        ),
        "R4a compiled default": make_variant(
            reference,
            lif_implementation="fused",
            inference_fast_spike=True,
            compiled_lif_mode="default",
            device=device,
        ),
        "R4b compiled reduce-overhead": make_variant(
            reference,
            lif_implementation="fused",
            inference_fast_spike=True,
            compiled_lif_mode="reduce-overhead",
            device=device,
        ),
        "R5 R4a + first step": make_variant(
            reference,
            lif_implementation="fused",
            inference_fast_spike=True,
            compiled_lif_mode="default",
            first_step_specialization=True,
            device=device,
        ),
        "R7 R4a + addcmul": make_variant(
            reference,
            lif_implementation="fused",
            inference_fast_spike=True,
            compiled_lif_mode="default",
            lif_step_primitive="addcmul",
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
    variant_results: dict[str, dict[str, float]] = {}
    print("\nPerformance (normal forward; CUDA events)")
    print(
        f"{'Variant':<28} {'p50 ms':>10} {'p95 ms':>10} {'mean ms':>10} "
        f"{'min ms':>10} {'max ms':>10} {'speedup':>10}"
    )
    compile_timings: dict[str, float] = {}
    rejected_variants: dict[str, str] = {}
    for label, model in variants.items():
        try:
            if label.startswith("R4") or label.startswith("R5") or label.startswith("R7"):
                torch._dynamo.reset()
                model.reset_state()
                started = time.perf_counter()
                with torch.inference_mode():
                    model(gpu_input)
                torch.cuda.synchronize(device)
                compile_timings[label] = time.perf_counter() - started
            warmup(model, gpu_input, args.warmup, device)
            samples = normal_forward_samples(model, gpu_input, device, args.runs)
            variant_results[label] = summarize(samples)
        except RuntimeError as error:
            rejected_variants[label] = str(error).splitlines()[0]
            torch.cuda.synchronize(device)
    baseline_p50 = variant_results["R0 reference"]["p50"]
    for label, summary in variant_results.items():
        print(
            f"{label:<28} {summary['p50']:10.3f} {summary['p95']:10.3f} "
            f"{summary['mean']:10.3f} {summary['min']:10.3f} {summary['max']:10.3f} "
            f"{baseline_p50 / summary['p50']:9.3f}x"
        )
    for label, reason in rejected_variants.items():
        print(f"{label:<28} REJECTED: {reason}")
    for label, elapsed_s in compile_timings.items():
        print(f"{label} first_compile_plus_forward_s={elapsed_s:.3f}")

    optimized = variants["R3 fused + fast spike"]
    print("\nT Scaling: R3 fused + fast spike (fixed B=1,C=2,H=96,W=128)")
    print(f"{'T':>4} {'forward p50 ms':>16} {'forward p95 ms':>16} {'ms/timestep p50':>18}")
    for timesteps in (1, 8, 32, 64):
        event_bins = make_cpu_input(timesteps).to(device)
        warmup(optimized, event_bins, args.warmup, device)
        samples = normal_forward_samples(optimized, event_bins, device, args.runs)
        summary = summarize(samples)
        print(f"{timesteps:4d} {summary['p50']:16.3f} {summary['p95']:16.3f} {summary['p50'] / timesteps:18.3f}")
    profiler_summary("R3 fused + fast spike", optimized, gpu_input, device, args.profiler_runs)
    profiler_summary("R4a compiled default", variants["R4a compiled default"], gpu_input, device, args.profiler_runs)


if __name__ == "__main__":
    main()
