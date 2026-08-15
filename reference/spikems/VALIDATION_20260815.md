# SpikeMS model and EVIMO2 adapter validation — 2026-08-15

## Scope

This validation answers only:

1. whether the official SpikeMS architecture and checkpoint execute one forward on the current GPU;
2. whether one EVIMO2 Motion Segmentation sample can be represented with SpikeMS input and
   foreground-event supervision semantics.

No training, tuning, architecture change, GT-assisted crop or ratio filtering was performed.

Baseline commit: `8d79bdd651ed4f775ffacae28edbbeb40885a146`.

## Dependency audit

`DEPENDENCY_AUDIT=PASS`

- Python-level: PyTorch, NumPy and PyYAML are used by the model path.
- Import-only for this gate: OpenCV is imported by the unused loss module; an explicit empty shim is
  used when OpenCV is absent.
- Runner/dataloader only: StrictYAML, tensorboardX, Pillow, h5py and OpenCV are not required for the
  direct model forward.
- Compiled and forward-critical: `slayerCuda.conv` and `slayerCuda.getSpikes`.
- Official Python fallback: none.
- Compatibility: the unchanged official SLAYER CUDA source at
  `01beeeb6a181546d6c6830382ce6086bfc587836` was compiled for SM 8.9 with PyTorch 2.10/CUDA 12.8.

See [ENVIRONMENT.md](ENVIRONMENT.md) for the reproducible setup and exact boundary.

## Checkpoint structure

`CHECKPOINT_STRUCTURE_GATE=PASS`

- checkpoint epoch: 100
- checkpoint loss: finite float32 tensor, `-0.2687` when displayed to four decimals
- state dict entries: 18
- all tensors: float32, finite, no NaN/Inf
- strict load missing keys: none
- strict load unexpected keys: none
- key conversion: none
- model parameter count: 46,656

Spatial weights match the official topology:

```text
conv1.weight     [16,  2, 3, 3, 1]
conv2.weight     [32, 16, 3, 3, 1]
conv3.weight     [64, 32, 3, 3, 1]
deconv4.0.weight [64, 32, 3, 3, 1]
deconv5.0.weight [32, 16, 3, 3, 1]
deconv6.0.weight [16,  2, 3, 3, 1]
```

The remaining 12 entries are the six layers' registered SRM and refractory kernels.

## Official model construction and synthetic forward

`SPIKEMS_MODEL_FORWARD_GATE=PASS`

- official class: `model.unetRNN6Layer_noBlock.SNN`
- topology: `2 -> 16 -> 32 -> 64 -> 32 -> 16 -> 2`
- device/dtype: CUDA 0 / float32
- marker: `MODEL_INTERFACE_SMOKE_ONLY`
- input shape: `[1, 2, 15, 15, 8]`
- input spikes: 4
- output shape: `[1, 2, 15, 15, 8]`
- output spikes: 0
- output finite: yes
- diagnostic runtime: 144.51 ms for the final recorded run; not a performance result

Zero output is acceptable for an artificial sparse input. The two output channels follow the target
foreground event polarities; they are not foreground/background classes.

## EVIMO2 sequence and alignment

`EVIMO2_ALIGNMENT_GATE=PASS`

```text
path: /home/speck/datasets/evimo2/motion_segmentation/right_camera/imo/eval/
      scene15_dyn_test_05_000000
camera: Prophesee right camera
resolution: 640 x 480
events: 7,402,991
event timestamp range: 0.0166722834 s .. 4.9481115341 s
x range: 0 .. 639
y range: 0 .. 479
stored polarity: 0, 1
mask frames: 295 available masks; metadata contains 299 frame records
mask shape/dtype: [480, 640] / uint16
mask values: background 0; object ids multiplied by 1000
metadata frame cadence: 16.666–16.667 ms
```

Official EVIMO2v2 documentation states that events are stored in seconds, masks contain object ids
multiplied by 1000, and exact irregular mask timestamps are in `dataset_info.npz`'s `meta.frames`.
The selected mask key is derived from that same frame record, not array position or an assumed cadence.

The right/left cameras use the Prophesee convention: stored polarity 0 is negative/OFF and 1 is
positive/ON. Samsung is inverted; the adapter detects the camera path and records the corresponding
channel semantics without changing stored polarity values.

## One-sample adapter

`EVIMO2_SPIKEMS_ADAPTER_GATE=PASS`

- frame index / mask id: 5 / 5
- mask timestamp: 0.100000 s
- physical event window: `[0.090000, 0.110000]` s, 20.0 ms total
- time bins: 100
- physical dt per bin: 0.2 ms
- input and foreground tensor shape: `[2, 480, 640, 100]`
- raw events: 2,064
- polarity 0 / OFF: 830
- polarity 1 / ON: 1,234
- raw foreground events: 1,359
- raw background events: 705
- foreground ratio: 0.658430
- binary input voxels: 2,024
- binary foreground voxels: 1,328
- event-to-voxel collisions: 40
- tensor values: finite, binary
- foreground tensor: verified subset of the input tensor

The 20 ms window preserves SpikeMS preprocessing's documented ±10 ms mask-centered choice while
making the physical mapping explicit. No mask-based crop or ratio filter is used.

The conversion preserves each event's coordinate, stored polarity, window membership and time-bin
assignment. It is not multiplicity-lossless: like upstream SpikeMS, a binary tensor merges events that
land in the same `(polarity, y, x, time_bin)` voxel. This happened 40 times in the selected sample.

Debug visualizations are intentionally ignored by Git:

```text
outputs/spikems_evimo2_adapter/raw_events.png
outputs/spikems_evimo2_adapter/object_mask.png
outputs/spikems_evimo2_adapter/gt_foreground_events.png
```

## One-sample zero-shot

Marker: `EVIMO_TO_EVIMO2_ZERO_SHOT`

- run: yes
- official checkpoint and model: yes
- input events: 2,064 raw / 2,024 binary voxels
- GT foreground: 1,328 binary voxels on the model output support
- prediction shape: `[1, 2, 479, 639, 100]`
- predicted spikes: 2,147
- IoU: 0.002886
- foreground recall: 0.007530
- background leakage: 0.572427
- diagnostic runtime: 324.09 ms; not a performance result
- prediction visualization:
  `outputs/spikems_evimo2_adapter/pred_foreground_events.png`

The output is one row and one column smaller because the official network's three valid stride-2
convolutions and transposed convolutions map 480×640 to 479×639. Metrics use the matching top-left
output support, as the upstream runner does; the input itself was not cropped.

IoU and recall compare binary predicted event voxels with binary GT foreground-event voxels.
Background leakage is predicted spikes outside the selected GT moving-object spatial mask divided by
all predicted spikes. The poor values are a valid zero-shot diagnostic, not a model or adapter tuning
target and not evidence of EVIMO2 performance after training.
