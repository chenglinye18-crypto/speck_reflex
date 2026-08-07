# Speck Reflex

Reproducible, no-hardware audit of official SynSense Speck 2f examples against Python 3.10.12,
Torch 2.10.0+cu128, Sinabs 3.1.3, Samna 0.48.6 and Tonic 1.6.0.

The current baseline follows the official Sinabs v3.1.3 N-MNIST quick start. It builds the
official BPTT SNN, performs software inference, constructs `DynapcnnNetwork`, generates and
validates a Speck 2f configuration, runs Specksim, and creates a local flash binary. It never
discovers or opens a device and never writes flash.

```bash
git clone --recurse-submodules https://github.com/chenglinye18-crypto/speck_reflex.git
cd speck_reflex
MPLCONFIGDIR=/tmp/speck-reflex-mpl \
  .venv/bin/python experiments/official_baselines/nmnist/run_official_nmnist_no_hardware.py
```

See [the source audit](docs/OFFICIAL_SOURCE_AUDIT.md),
[porting notes](docs/OFFICIAL_BASELINE_PORTING.md), and
[future hardware steps](docs/NEXT_HARDWARE_STEPS.md).

No robot-avoidance model is included or planned at this stage.
