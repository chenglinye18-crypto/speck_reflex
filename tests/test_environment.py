from __future__ import annotations

import platform
import sys

import samna
import sinabs
import torch
from sinabs.backend.dynapcnn import DynapcnnNetwork
from sinabs.backend.dynapcnn.io import get_device_map


def test_python_version() -> None:
    assert sys.version_info >= (3, 10)


def test_versions() -> None:
    print("Python       :", sys.version.split()[0])
    print("Platform     :", platform.platform())
    print("Kernel       :", platform.release())
    print("Torch        :", torch.__version__)
    print("Torch CUDA   :", torch.version.cuda)
    print("CUDA avail   :", torch.cuda.is_available())
    print("Sinabs       :", getattr(sinabs, "__version__", "unknown"))
    print("Samna        :", getattr(samna, "__version__", "unknown"))

    assert torch.__version__.startswith("2.10.0")
    assert torch.version.cuda == "12.8"
    assert torch.cuda.is_available()


def test_cuda_operation() -> None:
    device = torch.device("cuda:0")

    torch.manual_seed(17)
    a = torch.randn((1024, 1024), device=device)
    b = torch.randn((1024, 1024), device=device)
    c = a @ b

    torch.cuda.synchronize()

    assert c.is_cuda
    assert c.shape == (1024, 1024)
    assert bool(torch.isfinite(c).all())


def test_dynapcnn_import() -> None:
    assert DynapcnnNetwork is not None


def test_device_enumeration() -> None:
    devices = get_device_map()
    print("Speck devices:", devices)

    # No Speck board is connected yet, so {} is normal.
    assert isinstance(devices, dict)
