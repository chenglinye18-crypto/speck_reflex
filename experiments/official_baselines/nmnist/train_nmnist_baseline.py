#!/usr/bin/env python3
"""Fallback training for the official quick-start BPTT topology.

The functional demo normally uses the licensed NIR checkpoint pinned in the
Sinabs submodule. This trainer exists for reproducible retraining, not as an
automatic download or a claimed reproduction of unknown upstream training.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, random_split
from tonic.datasets import NMNIST
from tonic.transforms import ToFrame
from tqdm import tqdm

from speck_reflex.official_baseline.model import build_random_quick_start_model, reset_states, set_seed
from speck_reflex.official_baseline.results import sha256_file
from speck_reflex.environment import collect_environment


def evaluate(model, loader, batch_size: int, device: torch.device) -> float:
    correct = total = 0
    model.eval()
    with torch.no_grad():
        for frames, labels in loader:
            reset_states(model)
            frames = frames.reshape(-1, 2, 34, 34).float().to(device)
            labels = labels.long().to(device)
            counts = model(frames).reshape(batch_size, 100, -1).sum(1)
            correct += int((counts.argmax(1) == labels).sum())
            total += len(labels)
    return correct / total


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/nmnist_baseline.yaml")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-train-samples", type=int, help="optional controlled-size run")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    seed = int(cfg["seed"])
    set_seed(seed)
    train_cfg = cfg["training_fallback"]
    batch_size = int(train_cfg["batch_size"])
    data_root = ROOT / cfg["data_root"]
    transform = ToFrame(sensor_size=NMNIST.sensor_size, n_time_bins=100)
    full_train = NMNIST(save_to=str(data_root), train=True, transform=transform)
    if args.max_train_samples:
        full_train, _ = random_split(full_train, [min(args.max_train_samples, len(full_train)), max(0, len(full_train) - args.max_train_samples)], generator=torch.Generator().manual_seed(seed))
    validation_size = max(1, int(len(full_train) * float(train_cfg["validation_fraction"])))
    train_set, validation_set = random_split(full_train, [len(full_train) - validation_size, validation_size], generator=torch.Generator().manual_seed(seed))
    test_set = NMNIST(save_to=str(data_root), train=False, transform=transform)
    loaders = {
        "train": DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True),
        "validation": DataLoader(validation_set, batch_size=batch_size, shuffle=False, drop_last=True),
        "test": DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=True),
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_random_quick_start_model(batch_size=batch_size).to(device)
    if args.resume:
        # COMPATIBILITY_PATCH: weights_only prevents arbitrary checkpoint object loading.
        model.load_state_dict(torch.load(args.resume, map_location=device, weights_only=True))
    optimizer = torch.optim.SGD(model.parameters(), lr=float(train_cfg["learning_rate"]))
    criterion = nn.CrossEntropyLoss()
    checkpoint_dir = Path(__file__).parent / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_accuracy = -1.0
    history = []
    started = time.monotonic()
    for epoch in range(1, int(train_cfg["epochs"]) + 1):
        model.train()
        loss_total = 0.0
        for frames, labels in tqdm(loaders["train"], desc=f"epoch {epoch}"):
            reset_states(model)
            optimizer.zero_grad()
            frames = frames.reshape(-1, 2, 34, 34).float().to(device)
            labels = labels.long().to(device)
            counts = model(frames).reshape(batch_size, 100, -1).sum(1)
            loss = criterion(counts, labels)
            loss.backward()
            optimizer.step()
            loss_total += float(loss.detach())
        validation_accuracy = evaluate(model, loaders["validation"], batch_size, device)
        history.append({"epoch": epoch, "training_loss_sum": loss_total, "validation_accuracy": validation_accuracy})
        torch.save(model.state_dict(), checkpoint_dir / "last.pt")
        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            torch.save(model.state_dict(), checkpoint_dir / "best.pt")
    model.load_state_dict(torch.load(checkpoint_dir / "best.pt", map_location=device, weights_only=True))
    test_accuracy = evaluate(model, loaders["test"], batch_size, device)
    metadata = {
        "source_topology": "Sinabs v3.1.3 nmnist_quick_start.ipynb BPTT model",
        "seed": seed,
        "epochs": int(train_cfg["epochs"]),
        "optimizer": "SGD",
        "learning_rate": float(train_cfg["learning_rate"]),
        "split": {"train": len(train_set), "validation": len(validation_set), "test": len(test_set)},
        "best_validation_accuracy": best_accuracy,
        "test_accuracy": test_accuracy,
        "elapsed_seconds": time.monotonic() - started,
        "best_sha256": sha256_file(checkpoint_dir / "best.pt"),
        "software_versions": collect_environment(ROOT),
        "history": history,
    }
    (checkpoint_dir / "training_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
