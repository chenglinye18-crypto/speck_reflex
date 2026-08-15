"""Compatibility helpers for running the unmodified SpikeMS model.

The model and neuron implementation remain in ``third_party/SpikeMS``.  This
module only builds the CUDA extension from the official SLAYER source and
works around an import-only OpenCV dependency in SpikeMS's unused loss module.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import types
from typing import Any

import torch
from torch.utils.cpp_extension import load


SLAYER_REPOSITORY = "https://github.com/bamsumit/slayerPytorch.git"
SLAYER_COMMIT = "01beeeb6a181546d6c6830382ce6086bfc587836"


def verify_slayer_source(source_root: Path) -> None:
    source_root = source_root.resolve()
    kernel = source_root / "src" / "cuda" / "slayerKernels.cu"
    if not kernel.is_file():
        raise FileNotFoundError(f"Missing official SLAYER kernel: {kernel}")
    commit = subprocess.check_output(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != SLAYER_COMMIT:
        raise RuntimeError(f"SLAYER source is {commit}; expected {SLAYER_COMMIT}")
    if subprocess.check_output(
        ["git", "-C", str(source_root), "status", "--porcelain"], text=True
    ).strip():
        raise RuntimeError(f"SLAYER source has local modifications: {source_root}")


def load_official_slayer_cuda(source_root: Path, build_root: Path) -> Any:
    """Compile/load the official CUDA kernels without changing their source."""

    verify_slayer_source(source_root)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the official SLAYER forward path")
    capability = torch.cuda.get_device_capability(0)
    # Directly invoking a virtualenv's Python does not prepend its bin directory
    # to PATH. torch's extension loader discovers ninja through PATH.
    environment_bin = Path(sys.executable).parent
    if shutil.which("ninja") is None and (environment_bin / "ninja").is_file():
        os.environ["PATH"] = f"{environment_bin}:{os.environ.get('PATH', '')}"
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", f"{capability[0]}.{capability[1]}")
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(build_root.resolve()))
    module = load(
        name="slayerCuda",
        sources=[str(source_root.resolve() / "src" / "cuda" / "slayerKernels.cu")],
        extra_cuda_cflags=["-O2", "--use_fast_math"],
        verbose=False,
    )
    sys.modules["slayerCuda"] = module
    return module


def import_spikems_model(spikems_root: Path):
    """Import the official model, with an explicit unused-cv2 import shim."""

    spikems_root = spikems_root.resolve()
    if str(spikems_root) not in sys.path:
        sys.path.insert(0, str(spikems_root))
    try:
        importlib.import_module("cv2")
    except ModuleNotFoundError:
        shim = types.ModuleType("cv2")
        shim.__dict__["SPIKEMS_IMPORT_ONLY_SHIM"] = True
        sys.modules["cv2"] = shim
    return importlib.import_module("model.utils")


def build_official_model(spikems_root: Path, checkpoint_path: Path, device: torch.device):
    """Construct the official network and strictly load its official state dict."""

    model_utils = import_spikems_model(spikems_root)
    simulation = {"Ts": 1, "tSample": 100, "tStartLoss": 50}
    data = {
        "height": 260,
        "width": 346,
        "height_c": 128,
        "width_c": 128,
        "k": 1,
        "minEvents": 30,
        "start": 5,
        "timePerMask": 0.001,
    }
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = model_utils.getNetwork("unetRNN6Layer_noBlock", simulation, data)
    load_result = model.load_state_dict(checkpoint["state_dict"], strict=True)
    model = model.to(device=device, dtype=torch.float32).eval()
    return model, checkpoint, load_result
