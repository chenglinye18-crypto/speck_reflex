# Speck Reflex

> This is an independent research reproduction based on public SynSense
> software and documentation. It is not an official SynSense repository.

Speck Reflex reproduces the SynSense Speck 2f no-hardware deployment path and prepares a clearly separated extension boundary for a future DVS robot reflex arc. The current repository has not connected to physical Speck hardware, does not contain a robot-avoidance model, and provides no safety certification or functional-safety guarantee.

Suggested GitHub description: **Unofficial reproduction of SynSense Speck 2f software baselines and development of a robot reflex-arc prototype.**

## Goal and status

| Capability | Status | Evidence/limit |
|---|---|---|
| Official Sinabs v3.1.3 software baseline | Complete | Pinned upstream source and API audit |
| Random-weight deployment smoke test | Complete | Real host-injected N-MNIST events; zero spikes allowed |
| Functional N-MNIST demo | Complete | Official Apache-2.0 NIR checkpoint; actual subset metric |
| DynapCNN mapping and offline Samna config | Complete | No physical device API |
| Specksim comparison | Complete | Differences are retained in result JSON |
| Real Speck 2f and on-chip DVS | Not started | No board connected; USB remains unbound |
| Robot DVS reflex/avoidance model | Not implemented | Requirements and type boundary only |
| STM32 integration and safety validation | Not implemented | STM32 is the final arbiter |

No robot-avoidance model is included in the current baseline. Development of the DVS–Speck robot reflex pipeline will begin after physical hardware, DVS routing and readout validation.

## Intended architecture (not yet implemented)

```text
Ordinary cognitive path:
RGB/other sensors -> Jetson/ROS/vision model -> STM32 -> motors

Fast reflex path:
Speck on-chip DVS -> on-chip SNN -> readout/risk signal
  -> STM32 deterministic safety arbitration -> slow/stop/alarm
```

Jetson handles semantic recognition, person attributes, fall confirmation, and planning. Speck is intended for low-latency detection of rapid approach, sudden intrusion, collision risk, and abnormal motion. STM32 makes the final deterministic safety decision. Speck never directly emits motor PWM. This robot path is not implemented in the current repository.

## Five-minute verification

On an existing prepared environment:

```bash
git submodule update --init --recursive
source scripts/activate.sh
make doctor
make test-fast
make demo-smoke
make demo-nmnist
```

On a new WSL2/Linux machine, first inspect the non-mutating bootstrap plan. The repository does not assume `.venv` exists:

```bash
bash scripts/bootstrap_wsl_cuda.sh
bash scripts/bootstrap_wsl_cuda.sh --install  # only when you choose to install
source scripts/activate.sh
make verify
```

The installer never installs the CUDA Toolkit, an NVIDIA driver, or USB support. PyTorch is installed separately from its official `cu128` index; see [environment setup](docs/ENVIRONMENT_SETUP.md).

## Smoke test versus functional demo

- `make demo-smoke` is the **N-MNIST deployment pipeline smoke test**. It uses seed 17 and deterministic random Xavier weights. It proves data loading, software forward, DynapCNN construction/mapping, offline configuration, and Specksim execution. Zero output spikes are valid. It prints `PIPELINE_SMOKE_ONLY` and makes no accuracy claim.
- `make demo-nmnist` loads `scnn_mnist.nir` from the pinned Sinabs v3.1.3 submodule, verifies SHA256, reports the fixed sample's label/prediction/spike counts for Sinabs, quantized DynapCNN, and Specksim, and computes an actual fixed test-subset metric. Every path must emit nonzero output before it prints `FUNCTIONAL_NMNIST_DEMO_PASSED`.

Both are **host-injected event baselines** using N-MNIST 34×34 events with `dvs_input=false`; neither represents the internal 128×128 DVS path.

## Commands

```text
make doctor       environment diagnostics, no device discovery
make test         all non-hardware, non-slow tests
make test-fast    unit tests without network/GPU/hardware
make test-gpu     independent CUDA tests (skip if unavailable)
make demo-smoke   deterministic random-weight pipeline
make demo-nmnist  official-weight functional demonstration
make verify       all required no-hardware checks in order
make submodules   initialize pinned third-party repositories
```

Flash binary generation is disabled by default. The smoke CLI requires both `--generate-flash-binary` and an explicit board-specific `--io-sel`; no code in this repository writes Flash.

## Layout

- `src/speck_reflex/official_baseline/`: shared official-baseline compatibility code.
- `experiments/official_baselines/nmnist/`: direct smoke, functional evaluation, and fallback training entry points.
- `tests/{unit,integration,gpu,hardware}/`: explicitly marked test tiers.
- `configs/`: baseline parameters and a clearly marked robot design placeholder.
- `third_party/synsense/`: pinned, unmodified upstream submodules.
- `docs/`: audit, reproduction, environment, hardware, safety, and roadmap documentation.

## Reproduction result

The random baseline validates four DynapCNN layers for `speck2fdevkit`; its zero spikes are expected and not a model result. The functional baseline uses the official NIR model, maps five layers for `speck2fmodule`, and records the actual predictions, counts, agreement, and test-subset accuracy in `experiments/official_baselines/nmnist/artifacts/functional_results.json`. Generated results must not be edited to conceal disagreement.

## Next stages

Follow [the hardware validation plan](docs/HARDWARE_VALIDATION_PLAN.md) only after a board is available and explicit hardware access is approved. Robot scope and safety boundaries live in [robot requirements](docs/ROBOT_REFLEX_REQUIREMENTS.md) and [the roadmap](docs/ROADMAP.md).

## Third-party code, license, and safety

The submodules retain their own licenses and are not copied into the main package. See [third-party notices](THIRD_PARTY_NOTICES.md), [source audit](docs/OFFICIAL_SOURCE_AUDIT.md), and [licensing notes](docs/LICENSING.md). **License: pending repository-owner selection.** In particular, the AGPL-3.0 `dvs_tool` redistribution boundary requires separate review.

This research software is not certified for collision avoidance, emergency stopping, medical fall detection, or any other safety-critical use. It must not directly control motors. A separately verified STM32 safety layer remains responsible for final arbitration and fail-safe behavior.
