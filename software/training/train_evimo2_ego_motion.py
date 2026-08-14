"""Train the frozen EVIMO2 Samsung camera ego-motion baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import yaml
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm

from software.datasets import (
    EVIMO2EgoMotionDataset,
    EVIMO2EgoMotionIndex,
    EVIMO2EventConfig,
    EVIMO2WindowRecord,
    TargetNormalization,
    build_ego_motion_index,
)
from software.models.snn import SNNMotionBackbone, SNNMotionConfig


ROOT = Path(__file__).resolve().parents[2]


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _record_from_json(value: dict[str, Any]) -> EVIMO2WindowRecord:
    motion = tuple(float(item) for item in value["ego_motion"])
    if len(motion) != 6:
        raise ValueError("cached index target must have six components")
    return EVIMO2WindowRecord(
        sequence=str(value["sequence"]),
        frame_id=int(value["frame_id"]),
        end_time_s=float(value["end_time_s"]),
        ego_motion=motion,  # type: ignore[arg-type]
    )


def _index_spec(
    *, event_config: EVIMO2EventConfig, frame_stride: int, validation_fraction: float, seed: int
) -> dict[str, object]:
    return {
        "timesteps": event_config.timesteps,
        "dt_ms": event_config.dt_ms,
        "spatial_reduction": event_config.spatial_reduction,
        "frame_stride": frame_stride,
        "validation_fraction": validation_fraction,
        "seed": seed,
        "split_policy": "upstream_train_sequence_split_and_upstream_eval_test",
    }


def _load_or_build_index(
    data_root: Path,
    output_path: Path,
    *,
    event_config: EVIMO2EventConfig,
    frame_stride: int,
    validation_fraction: float,
    seed: int,
    rebuild: bool,
) -> EVIMO2EgoMotionIndex:
    spec = _index_spec(
        event_config=event_config,
        frame_stride=frame_stride,
        validation_fraction=validation_fraction,
        seed=seed,
    )
    if output_path.is_file() and not rebuild:
        cached = json.loads(output_path.read_text())
        if cached.get("spec") != spec:
            raise ValueError(
                f"cached index settings differ: {output_path}; use --rebuild-index"
            )
        index = EVIMO2EgoMotionIndex(
            train=tuple(_record_from_json(item) for item in cached["train"]),
            validation=tuple(_record_from_json(item) for item in cached["validation"]),
            test=tuple(_record_from_json(item) for item in cached["test"]),
            sha256=str(cached["sha256"]),
        )
        return index

    index = build_ego_motion_index(
        data_root,
        event_config=event_config,
        frame_stride=frame_stride,
        validation_fraction=validation_fraction,
        seed=seed,
    )
    payload = index.to_json_dict()
    payload["spec"] = spec
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    return index


def _limit(
    records: tuple[EVIMO2WindowRecord, ...], maximum: int | None
) -> tuple[EVIMO2WindowRecord, ...]:
    if maximum is None:
        return records
    if maximum <= 0:
        raise ValueError("window limits must be positive")
    return records[:maximum]


class DeterministicEpochSampler(Sampler[int]):
    """Epoch-addressable shuffle order that is stable across resume."""

    def __init__(self, size: int, *, seed: int) -> None:
        if size <= 0:
            raise ValueError("sampler size must be positive")
        self.size = size
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        if epoch <= 0:
            raise ValueError("epoch must be positive")
        self.epoch = epoch

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        return iter(torch.randperm(self.size, generator=generator).tolist())

    def __len__(self) -> int:
        return self.size


def _effective_index_digest(
    train: Iterable[EVIMO2WindowRecord],
    validation: Iterable[EVIMO2WindowRecord],
    test: Iterable[EVIMO2WindowRecord],
) -> str:
    digest = hashlib.sha256()
    for name, records in (
        ("train", train),
        ("validation", validation),
        ("test", test),
    ):
        digest.update(name.encode())
        for record in records:
            digest.update(
                json.dumps(asdict(record), sort_keys=True, separators=(",", ":")).encode()
            )
    return digest.hexdigest()


def _make_loader(
    root: Path,
    records: tuple[EVIMO2WindowRecord, ...],
    event_config: EVIMO2EventConfig,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
    pin_memory: bool,
) -> DataLoader:
    dataset = EVIMO2EgoMotionDataset(root, records, event_config=event_config)
    sampler = DeterministicEpochSampler(len(dataset), seed=seed) if shuffle else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )


def window_prediction(model: SNNMotionBackbone, events: Tensor) -> Tensor:
    """Return one normalized six-DoF prediction per independent window."""

    output = model(events)
    if output.ego_motion is None:
        raise RuntimeError("ego-motion head must be enabled for this baseline")
    return output.ego_motion.mean(dim=1)


def _activity_audit(
    model: SNNMotionBackbone,
    loader: DataLoader,
    *,
    device: torch.device,
    maximum_windows: int,
    warning_fraction: float,
    failure_fraction: float,
) -> dict[str, object]:
    model.eval()
    active_counts: list[Tensor] = []
    layer_fractions: dict[str, list[float]] = {}
    nonzero = total_cells = total_events = reported_events = 0
    checked = 0
    with torch.no_grad():
        for batch in loader:
            for batch_index in range(batch["events"].shape[0]):
                events_cpu = batch["events"][batch_index : batch_index + 1]
                active = events_cpu[events_cpu > 0]
                if active.numel():
                    active_counts.append(active.to(torch.float64))
                nonzero += int(torch.count_nonzero(events_cpu))
                total_cells += events_cpu.numel()
                total_events += int(events_cpu.sum().item())
                reported_events += int(batch["event_count"][batch_index])
                model.reset_state()
                run = model.forward_with_stats(events_cpu.to(device, non_blocking=True))
                model.reset_state()
                for name, statistics in run.statistics.layers.items():
                    layer_fractions.setdefault(name, []).append(
                        statistics.spikes_per_neuron_per_timestep
                    )
                checked += 1
                if checked >= maximum_windows:
                    break
            if checked >= maximum_windows:
                break
    if checked == 0:
        raise ValueError("activity audit received no windows")
    if total_events != reported_events:
        raise RuntimeError("spatial reduction did not preserve total event count")

    values = torch.cat(active_counts) if active_counts else torch.zeros(1, dtype=torch.float64)
    quantiles = {
        label: float(torch.quantile(values, quantile).item())
        for label, quantile in (
            ("p50", 0.5),
            ("p90", 0.9),
            ("p95", 0.95),
            ("p99", 0.99),
            ("p99_9", 0.999),
        )
    }
    averaged_layers = {
        name: sum(fractions) / len(fractions)
        for name, fractions in layer_fractions.items()
    }
    overactive = [name for name, value in averaged_layers.items() if value > warning_fraction]
    failed = [name for name, value in averaged_layers.items() if value > failure_fraction]

    zero = torch.zeros_like(events_cpu, device=device)
    model.reset_state()
    zero_run = model.forward_with_stats(zero)
    model.reset_state()
    zero_silent = all(stat.total_spikes == 0 for stat in zero_run.statistics.layers.values())
    if not zero_silent:
        raise RuntimeError("zero event input produced spontaneous spikes")

    report: dict[str, object] = {
        "windows": checked,
        "input_nonzero_fraction": nonzero / total_cells,
        "mean_active_event_count": float(values.mean().item()),
        "max_active_event_count": float(values.max().item()),
        "event_count_quantiles": quantiles,
        "total_events": total_events,
        "event_count_preserved": True,
        "layer_firing_fractions": averaged_layers,
        "overactive_layers": overactive,
        "failed_layers": failed,
        "zero_input_silent": zero_silent,
    }
    return report


class _PhysicalMetrics:
    def __init__(self) -> None:
        self.count = 0
        self.component_absolute_sum = torch.zeros(6, dtype=torch.float64)
        self.translation_norm_sum = 0.0
        self.rotation_norm_sum = 0.0

    def update(self, prediction: Tensor, target: Tensor) -> None:
        error = (prediction.detach() - target.detach()).to("cpu", torch.float64)
        self.count += error.shape[0]
        self.component_absolute_sum += error.abs().sum(dim=0)
        self.translation_norm_sum += float(torch.linalg.vector_norm(error[:, :3], dim=1).sum())
        self.rotation_norm_sum += float(torch.linalg.vector_norm(error[:, 3:], dim=1).sum())

    def result(self) -> dict[str, object]:
        if self.count == 0:
            raise ValueError("cannot report metrics without samples")
        return {
            "samples": self.count,
            "component_mae": (self.component_absolute_sum / self.count).tolist(),
            "translation_vector_mae_m_per_s": self.translation_norm_sum / self.count,
            "rotation_vector_mae_rad_per_s": self.rotation_norm_sum / self.count,
        }


def _evaluate(
    model: SNNMotionBackbone,
    loader: DataLoader,
    normalizer: TargetNormalization,
    *,
    device: torch.device,
    beta: float,
) -> tuple[float, dict[str, object]]:
    model.eval()
    loss_sum = 0.0
    sample_count = 0
    metrics = _PhysicalMetrics()
    with torch.no_grad():
        for batch in loader:
            events = batch["events"].to(device, non_blocking=True)
            target = batch["ego_motion"].to(device, non_blocking=True)
            model.reset_state()
            normalized_prediction = window_prediction(model, events)
            normalized_target = normalizer.normalize(target)
            loss = F.smooth_l1_loss(
                normalized_prediction, normalized_target, beta=beta, reduction="mean"
            )
            model.reset_state()
            prediction = normalizer.denormalize(normalized_prediction)
            metrics.update(prediction, target)
            loss_sum += float(loss) * target.shape[0]
            sample_count += target.shape[0]
    return loss_sum / sample_count, metrics.result()


def _checkpoint_payload(
    *,
    model: SNNMotionBackbone,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_validation_loss: float,
    normalizer: TargetNormalization,
    config: dict[str, Any],
    index_hash: str,
) -> dict[str, object]:
    return {
        "format_version": 1,
        "task": "evimo2_samsung_camera_local_ego_motion",
        "epoch": epoch,
        "best_validation_loss": best_validation_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "target_normalization": normalizer.state_dict(),
        "config": config,
        "index_sha256": index_hash,
        "torch_version": str(torch.__version__),
    }


def _atomic_torch_save(payload: dict[str, object], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def _train_epoch(
    model: SNNMotionBackbone,
    loader: DataLoader,
    normalizer: TargetNormalization,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    beta: float,
    accumulation_steps: int,
    gradient_clip_norm: float,
    epoch: int,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_sum = 0.0
    sample_count = 0
    progress = tqdm(loader, desc=f"epoch {epoch}", unit="batch")
    for batch_index, batch in enumerate(progress, start=1):
        events = batch["events"].to(device, non_blocking=True)
        target = batch["ego_motion"].to(device, non_blocking=True)
        model.reset_state()
        prediction = window_prediction(model, events)
        normalized_target = normalizer.normalize(target)
        loss = F.smooth_l1_loss(prediction, normalized_target, beta=beta, reduction="mean")
        if not torch.isfinite(loss):
            raise RuntimeError("training loss became non-finite")
        (loss / accumulation_steps).backward()
        model.reset_state()
        should_step = batch_index % accumulation_steps == 0 or batch_index == len(loader)
        if should_step:
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            if not torch.isfinite(grad_norm):
                raise RuntimeError("gradient norm became non-finite")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        loss_sum += float(loss.detach()) * target.shape[0]
        sample_count += target.shape[0]
        progress.set_postfix(loss=f"{loss_sum / sample_count:.5f}")
    return loss_sum / sample_count


def _device_from_config(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _load_config(path: Path) -> dict[str, Any]:
    with path.open() as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("training config must be a mapping")
    return config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/evimo2_ego_motion_v0.1.yaml"
    )
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-train-windows", type=int)
    parser.add_argument("--max-validation-windows", type=int)
    parser.add_argument("--max-test-windows", type=int)
    parser.add_argument("--audit-windows", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--rebuild-index", action="store_true")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()

    config = _load_config(args.config.resolve())
    data_cfg = config["data"]
    train_cfg = config["training"]
    audit_cfg = config["activity_audit"]
    seed = int(config["seed"])
    _seed_everything(seed)

    data_root = Path(str(data_cfg["root"])).expanduser().resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (ROOT / str(config["output_dir"])).resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    event_config = EVIMO2EventConfig(
        timesteps=int(data_cfg["timesteps"]),
        dt_ms=float(data_cfg["dt_ms"]),
        spatial_reduction=int(data_cfg["spatial_reduction"]),
    )
    if (event_config.timesteps, event_config.dt_ms, event_config.spatial_reduction) != (64, 1.0, 5):
        raise ValueError("v0.1 frozen input requires 64 timesteps, 1 ms bins, and 5x reduction")

    index_path = output_dir / "window_index.json"
    print("Building/loading deterministic sequence-level index...")
    index = _load_or_build_index(
        data_root,
        index_path,
        event_config=event_config,
        frame_stride=int(data_cfg["frame_stride"]),
        validation_fraction=float(data_cfg["validation_fraction"]),
        seed=seed,
        rebuild=args.rebuild_index,
    )
    train_records = _limit(index.train, args.max_train_windows)
    validation_records = _limit(index.validation, args.max_validation_windows)
    test_records = _limit(index.test, args.max_test_windows)
    effective_index_hash = _effective_index_digest(
        train_records, validation_records, test_records
    )
    print(
        f"source_index={index.sha256} run_index={effective_index_hash} "
        f"train={len(train_records)} "
        f"validation={len(validation_records)} test={len(test_records)}"
    )

    normalizer = TargetNormalization.fit(
        train_records, epsilon=float(train_cfg["normalization_epsilon"])
    )
    device = _device_from_config(str(train_cfg["device"]))
    pin_memory = device.type == "cuda"
    batch_size = int(args.batch_size or train_cfg["batch_size"])
    num_workers = (
        int(args.num_workers)
        if args.num_workers is not None
        else int(train_cfg["num_workers"])
    )
    accumulation_steps = int(
        args.gradient_accumulation_steps
        or train_cfg["gradient_accumulation_steps"]
    )
    if batch_size <= 0 or num_workers < 0 or accumulation_steps <= 0:
        raise ValueError("batch size and accumulation must be positive; workers non-negative")
    train_cfg["batch_size"] = batch_size
    train_cfg["num_workers"] = num_workers
    train_cfg["gradient_accumulation_steps"] = accumulation_steps
    common_loader = {
        "event_config": event_config,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "seed": seed,
        "pin_memory": pin_memory,
    }
    train_loader = _make_loader(data_root, train_records, shuffle=True, **common_loader)
    validation_loader = _make_loader(
        data_root, validation_records, shuffle=False, **common_loader
    )
    test_loader = _make_loader(data_root, test_records, shuffle=False, **common_loader)

    model_gains = tuple(float(value) for value in config["model"]["layer_gains"])
    model = SNNMotionBackbone(
        SNNMotionConfig(enable_ego_head=True, layer_gains=model_gains)
    ).to(device)
    model.local_motion_head.requires_grad_(False)
    print(
        f"device={device} parameters={model.parameter_count()} seed={seed} "
        f"layer_gains={model_gains} batch_size={batch_size} "
        f"accumulation={accumulation_steps} workers={num_workers}"
    )
    audit_loader = _make_loader(
        data_root,
        train_records[: max(1, int(args.audit_windows or audit_cfg["windows"]))],
        event_config,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        seed=seed,
        pin_memory=pin_memory,
    )
    audit = _activity_audit(
        model,
        audit_loader,
        device=device,
        maximum_windows=int(args.audit_windows or audit_cfg["windows"]),
        warning_fraction=float(audit_cfg["warning_firing_fraction"]),
        failure_fraction=float(audit_cfg["failure_firing_fraction"]),
    )
    (output_dir / "preflight_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print("activity_audit=" + json.dumps(audit, separators=(",", ":")))
    if audit["failed_layers"]:
        raise RuntimeError(
            "measured-data activity audit blocked training; "
            f"layers above {float(audit_cfg['failure_firing_fraction']):.0%}: "
            f"{audit['failed_layers']}"
        )
    if args.preflight_only:
        print("EVIMO2_EGO_MOTION_PREFLIGHT_PASSED")
        return 0

    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    start_epoch = 1
    best_validation_loss = math.inf
    last_path = output_dir / "last.pt"
    best_path = output_dir / "best.pt"
    resume_path = args.resume
    if resume_path is None and bool(train_cfg["auto_resume"]) and last_path.is_file():
        resume_path = last_path
    if resume_path is not None:
        checkpoint = torch.load(resume_path, map_location=device, weights_only=True)
        if checkpoint["index_sha256"] != effective_index_hash:
            raise ValueError("checkpoint dataset index does not match current index")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        saved_normalizer = TargetNormalization.from_state_dict(
            checkpoint["target_normalization"]
        )
        if not torch.equal(saved_normalizer.mean, normalizer.mean) or not torch.equal(
            saved_normalizer.std, normalizer.std
        ):
            raise ValueError("checkpoint target normalization does not match current split")
        start_epoch = int(checkpoint["epoch"]) + 1
        best_validation_loss = float(checkpoint["best_validation_loss"])
        print(f"resumed={resume_path} next_epoch={start_epoch}")

    epochs = int(args.epochs or train_cfg["epochs"])
    history: list[dict[str, object]] = []
    started = time.monotonic()
    for epoch in range(start_epoch, epochs + 1):
        if isinstance(train_loader.sampler, DeterministicEpochSampler):
            train_loader.sampler.set_epoch(epoch)
        training_loss = _train_epoch(
            model,
            train_loader,
            normalizer,
            optimizer,
            device=device,
            beta=float(train_cfg["smooth_l1_beta"]),
            accumulation_steps=accumulation_steps,
            gradient_clip_norm=float(train_cfg["gradient_clip_norm"]),
            epoch=epoch,
        )
        validation_loss, validation_metrics = _evaluate(
            model,
            validation_loader,
            normalizer,
            device=device,
            beta=float(train_cfg["smooth_l1_beta"]),
        )
        improved = validation_loss < best_validation_loss
        if improved:
            best_validation_loss = validation_loss
        payload = _checkpoint_payload(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_validation_loss=best_validation_loss,
            normalizer=normalizer,
            config=config,
            index_hash=effective_index_hash,
        )
        _atomic_torch_save(payload, last_path)
        if improved:
            _atomic_torch_save(payload, best_path)
        epoch_result = {
            "epoch": epoch,
            "training_loss": training_loss,
            "validation_loss": validation_loss,
            "validation_metrics": validation_metrics,
        }
        history.append(epoch_result)
        print(json.dumps(epoch_result, separators=(",", ":")))

    if not best_path.is_file():
        raise RuntimeError("no best checkpoint is available for final evaluation")
    best_checkpoint = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    test_loss, test_metrics = _evaluate(
        model,
        test_loader,
        normalizer,
        device=device,
        beta=float(train_cfg["smooth_l1_beta"]),
    )
    result = {
        "status": "EVIMO2_EGO_MOTION_TRAINING_COMPLETED",
        "task": "camera_local_ego_motion_not_robot_odometry",
        "source_index_sha256": index.sha256,
        "index_sha256": effective_index_hash,
        "seed": seed,
        "epochs_requested": epochs,
        "elapsed_seconds_this_run": time.monotonic() - started,
        "best_validation_loss": float(best_checkpoint["best_validation_loss"]),
        "test_normalized_smooth_l1": test_loss,
        "test_metrics": test_metrics,
        "target_normalization": normalizer.state_dict(),
        "best_checkpoint": _display_path(best_path),
        "best_checkpoint_sha256": _sha256(best_path),
        "history_this_run": history,
        "local_motion_head_trained": False,
        "robot_motion_claimed": False,
        "hardware_accessed": False,
    }
    (output_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
