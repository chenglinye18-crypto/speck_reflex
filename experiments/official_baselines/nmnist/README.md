# N-MNIST official baselines

`run_official_nmnist_no_hardware.py` is the random-weight deployment smoke test. `evaluate_nmnist_baseline.py` is the functional demo using the NIR checkpoint shipped in the pinned Sinabs v3.1.3 submodule. `train_nmnist_baseline.py` is a fallback reproducer for the quick-start BPTT topology; it is not needed to run the licensed upstream checkpoint.

All paths use host-injected 34×34 N-MNIST events. No script discovers, opens, or writes a physical device. Flash binary generation is disabled by default and is available only in the smoke runner with explicit board-specific CLI arguments.
