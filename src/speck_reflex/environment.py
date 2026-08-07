"""Read-only environment inspection; physical device discovery is opt-in elsewhere."""

from __future__ import annotations

import importlib.metadata
import sys
from pathlib import Path

import torch


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def collect_environment(project_root: Path) -> dict[str, object]:
    prefix = Path(sys.prefix).resolve()
    expected_venv = (project_root / ".venv").resolve()
    return {
        "python": sys.version.split()[0],
        "in_project_venv": prefix == expected_venv,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "sinabs": package_version("sinabs"),
        "samna": package_version("samna"),
        "tonic": package_version("tonic"),
        "numpy": package_version("numpy"),
        "hardware_access_mode": "disabled",
    }
