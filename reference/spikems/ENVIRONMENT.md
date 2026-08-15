# SpikeMS reference environment

The compatibility environment is intentionally outside the repository and reuses the existing
project environment's PyTorch installation without modifying it.

```text
reference environment: /home/speck/.venvs/speck_reflex_spikems_reference
base PyTorch environment: /home/speck/projects/speck_reflex/.venv
official SLAYER source: /home/speck/.cache/spikems_reference/slayerPytorch
JIT extension cache: /home/speck/.cache/torch_extensions/spikems_reference
```

Prepare it with:

```bash
bash scripts/setup_spikems_reference_env.sh
```

The setup script pins the official SLAYER repository to
`01beeeb6a181546d6c6830382ce6086bfc587836`, installs only Ninja in the isolated environment,
and leaves the project `.venv` unchanged.

## Why a compiled extension is required

SpikeMS imports its bundled modified `slayerpytorch`, whose forward path calls:

- `slayerCuda.conv` for causal PSP convolution along the time axis;
- `slayerCuda.getSpikes` for thresholding and refractory-response updates.

These are called in every spiking layer during forward. They are not training-only dependencies,
and the bundled code has no Python fallback. No current `slayerCuda` distribution is available from
PyPI.

The compatibility loader compiles the unmodified official
`src/cuda/slayerKernels.cu` against the installed PyTorch/CUDA toolchain. The only build adaptation is
targeting the current GPU's SM 8.9 instead of the legacy setup script's hard-coded SM 6.0. Both use
optimization level 2 and CUDA fast-math. The CUDA kernel source, SpikeMS network, checkpoint and SRM
neuron code are unchanged.

SpikeMS's package initializer also imports its loss module, which imports OpenCV at module load time.
The model-only validation does not instantiate or call that loss. If OpenCV is absent, the loader
registers an explicitly marked import-only `cv2` shim so the unrelated import does not block network
construction. Any workflow that calls the upstream dataloader or loss must install real OpenCV.

## Validated environment

```text
OS: Ubuntu 22.04.5 LTS
Python: 3.10.12
PyTorch: 2.10.0+cu128
torchvision: 0.25.0+cu128
CUDA toolkit/build: 12.8
GPU: NVIDIA GeForce RTX 4060 Laptop GPU, compute capability 8.9
driver: 577.02
ninja: 1.13.0
```

Run the model gate with:

```bash
export PATH=/home/speck/.venvs/speck_reflex_spikems_reference/bin:$PATH
/home/speck/.venvs/speck_reflex_spikems_reference/bin/python \
  scripts/validate_spikems_model.py
```
