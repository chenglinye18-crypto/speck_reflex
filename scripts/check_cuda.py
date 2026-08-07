from __future__ import annotations

import time

import torch


def main() -> None:
    print("Torch:", torch.__version__)
    print("Torch CUDA:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")

    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(device)

    print("GPU:", props.name)
    print("Capability:", f"{props.major}.{props.minor}")
    print("VRAM GiB:", f"{props.total_memory / 1024**3:.2f}")

    torch.manual_seed(17)
    torch.cuda.manual_seed_all(17)

    a = torch.randn((4096, 4096), device=device)
    b = torch.randn((4096, 4096), device=device)

    # Warm-up
    _ = a @ b
    torch.cuda.synchronize()

    start = time.perf_counter()
    c = a @ b
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    assert c.is_cuda
    assert c.shape == (4096, 4096)
    assert bool(torch.isfinite(c).all())

    print("Elapsed seconds:", elapsed)
    print(
        "Allocated MiB:",
        torch.cuda.memory_allocated(device) / 1024**2,
    )
    print(
        "Reserved MiB:",
        torch.cuda.memory_reserved(device) / 1024**2,
    )
    print("PYTORCH_CUDA_TEST_PASSED")


if __name__ == "__main__":
    main()
