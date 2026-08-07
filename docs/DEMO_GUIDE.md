# Demo guide

## New machine

Clone with submodules, run `bash scripts/bootstrap_wsl_cuda.sh` to inspect prerequisites, then explicitly rerun with `--install`. The bootstrap never installs drivers/toolkits or binds USB.

## Existing environment

Run `source scripts/activate.sh`, `make doctor`, then `make demo-smoke` or `make demo-nmnist`.

The smoke marker `PIPELINE_SMOKE_ONLY` means the deployment chain executed; zero spikes are allowed. The functional marker appears only when all three paths have nonzero output, configuration is valid, and an actual subset metric is recorded. Prediction disagreement is expected to remain visible because discretization and simulator behavior can differ.

Common failures: missing `.venv` means bootstrap is needed; missing submodule checkpoint means run `make submodules`; missing N-MNIST test data triggers Tonic's official test download; CUDA absence skips GPU tests but does not fake success. No current command needs a board.
