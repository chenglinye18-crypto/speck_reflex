# Official source audit

Audit date: 2026-08-07. Third-party sources are retained as unmodified Git submodules.

| Repository | Ref | Commit | Last commit | License |
|---|---|---|---|---|
| `synsense_model_example` | `main` | `3408f37c95a3688d338bbc639d2f74d892db43a5` | 2023-03-06 14:25:29 +08:00 | No LICENSE file found |
| `dvs_tool` | `master` | `9326eed0443cf374a99f6c6b2c34361f8e7d11bf` | 2023-04-12 11:48:19 +08:00 | AGPL-3.0 |
| `docker_speck2f` | `master` | `d3fb39d944977b246393a00132ec5dd83b069bf0` | 2024-04-26 14:46:40 +08:00 | No root LICENSE; bundled label tool is AGPL-3.0 |
| `sinabs` | tag `v3.1.3` | `d84078e0af1bc40f716b61de199880bd8713bd2d` | 2026-02-04 10:39:05 +01:00 | Apache-2.0 |

## Remotes and requirements

- `https://gitlab.com/synsense/synsense_model_example.git`: README requires Ubuntu 16.04+,
  Python 3.6+, Samna, Sinabs and obsolete standalone `sinabs-dynapcnn`. Its smart-door demo
  targets Speck2e and opens hardware, visualizer and power monitoring.
- `https://gitlab.com/synsense/dvs_tool.git`: requires `dv==1.0.10`, NumPy 1.24.2,
  pandas 1.1.3 and PyQt5 5.15.9. It is a GUI labeling/recording tool, not an SNN baseline.
- `https://gitlab.com/synsense/docker_speck2f.git`: pins Python 3.9, Torch 1.12.1,
  Sinabs 2.0 and Samna 0.38.3.0. It was audited only; no container or USB workflow was run.
- `https://github.com/synsense/sinabs.git`: v3.1.3 requires Torch >=1.8 and Samna >=0.33.
  It contains the authoritative N-MNIST quick start and Specksim example used here.

## Installed package and API audit

The installed Sinabs wheel contains no examples or N-MNIST notebook. It does contain the
Dynapcnn backend, Specksim, readout/visualizer support and memory validation.

- `DynapcnnNetwork(snn, input_shape=None, batch_size=None, dvs_input=None, discretize=True, ...)`
- `make_config(layer2core_map='auto', device='speck2fdevkit:0', monitor_layers=None,
  config_modifier=None, chip_layers_ordering=None)`; `chip_layers_ordering` is obsolete.
- `get_device_map() -> Dict` exists but was not called because device enumeration was forbidden.
- `ChipFactory.supported_devices`: `speck2e`, `speck2edevkit`, `speck2fmodule`, `speck2fdevkit`.
- `from_sequential(network, input_shape) -> SpecksimNetwork` exists.
- `samna.speck2f.configuration_to_flash_binary(SpeckConfiguration) -> List[int]` exists.
- `samna.speck2f.validate_configuration(SpeckConfiguration) -> (bool, str)` exists.
- `sinabs.validate_memory_mapping_speck(...)` exists for per-layer checks.

The main baseline is Sinabs v3.1.3 N-MNIST, not the older smart-door repository.
