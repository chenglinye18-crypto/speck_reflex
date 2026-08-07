# Official N-MNIST no-hardware baseline

This workspace ports only the no-hardware sections of the official Sinabs 3.1.3 examples:

- `third_party/synsense/sinabs/docs/speck/notebooks/nmnist_quick_start.ipynb`
- `third_party/synsense/sinabs/docs/speck/specksim.md`
- `third_party/synsense/sinabs/docs/speck/faqs/save_hardware_config_as_binary.md`

Run from the repository root:

```bash
MPLCONFIGDIR=/tmp/speck-reflex-mpl \
  .venv/bin/python experiments/official_baselines/nmnist/run_official_nmnist_no_hardware.py \
  2>&1 | tee logs/official_baselines/nmnist_no_hardware.log
```

The script downloads only the official N-MNIST test split through Tonic. It does not enumerate or
open devices, launch the visualizer, call `DynapcnnNetwork.to(...)`, or write flash.
