# Dependency installation

PyTorch is intentionally absent from the normal requirement files. Install the fixed CUDA 12.8 wheels from the PyTorch `cu128` index first, then install `core.txt` and optionally `dev.txt`. See `scripts/bootstrap_wsl_cuda.sh`.

`environment-cu128-reference.txt` is a development-machine freeze for audit only; it is not a portable installation lock file.
