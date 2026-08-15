# SpikeMS EVIMO2 engineering training bring-up — 2026-08-15

## Research question

Can the unchanged SpikeMS architecture, initialized from the official EV-IMO checkpoint, learn
eight fixed EVIMO2 right-camera samples using only the released SLAYER `spikeTime` loss?

This is explicitly the `SPIKEMS_LSPIKE_ENGINEERING_BASELINE`. It is not a reproduction of the
unreleased combined BCE plus spike loss.

## Engineering training definition

- Input: binary EVIMO2 event tensor `[B, 2, H, W, 10]`; channel 0 is negative/OFF and channel 1 is
  positive/ON.
- GT: input event voxels intersected with the EVIMO2 moving-object mask, with polarity and time bin
  preserved. GT is verified to be a subset of input.
- Physical window: 10 ms (`SPIKEMS_PAPER`).
- Physical timestep: 1 ms (`SPIKEMS_PAPER`).
- Runtime input T: 10 (`SPIKEMS_PAPER`).
- Checkpoint kernel construction: the model and loss are constructed with the checkpoint config's
  `tSample=100` so its registered SRM/refractory buffers strict-load unchanged. Runtime duration is
  still determined by the input tensor and is exactly T=10. No network math or buffer is modified.
- Loss: released `slayerpytorch.loss.spikeTime`, unchanged sum reduction and no normalization
  (`SPIKEMS_RELEASED_CODE`).
- Optimizer: fresh Adam state, LR 1e-4, betas `(0.9, 0.999)`, eps 1e-8, weight decay 0, AMSGrad true
  (`SPIKEMS_CHECKPOINT`).
- Initialization: official EV-IMO pretrained checkpoint (`ENGINEERING_CHOICE`).
- Batch size: 1 (`DEBUG_RESOURCE_CHOICE`).
- Spatial strategy: full 480x640 frame; no spatial crop. The 479x639 prediction support is matched
  by a deterministic top-left GT crop with no resize or interpolation.
- Augmentation: none.

## One-sample backward

Debug selection marker: `DEBUG_SAMPLE_SELECTION_USING_GT_STATS`.

```text
sequence: scene15_dyn_test_05_000000 (eval)
frame index: 272
timestamp: 4.550000 s
window: [4.545000, 4.555000] s
raw events: 54,366
foreground events: 30,911
background events: 23,455
OFF / ON events: 22,130 / 32,236
input and full GT: [1, 2, 480, 640, 10]
prediction: [1, 2, 479, 639, 10]
prediction spikes: 2,300
L_spike: 52,540.76171875
parameter tensors with gradient: 6 / 6
parameter tensors with non-zero gradient: 6 / 6
global gradient norm: 128.5105682
maximum per-parameter gradient norm: 89.9168396
all gradients finite: yes
parameter change norm: 0.0211111872
parameter change finite: yes
peak allocated VRAM: 692.046875 MiB
```

```text
SPIKEMS_BACKWARD_GATE=PASS
```

## Eight-sample manifest

The immutable manifest is `overfit8_manifest.json`. All samples are from EVIMO2 Motion Segmentation,
right-camera, `imo/train`.

| # | sequence | frame | time (s) | raw | foreground | background |
|---:|---|---:|---:|---:|---:|---:|
| 0 | scene15_dyn_test_06_000000 | 62 | 1.050000 | 1,106 | 866 | 240 |
| 1 | scene6_dyn_train_03_000000 | 35 | 0.600000 | 1,510 | 686 | 824 |
| 2 | scene15_dyn_test_04_000000 | 166 | 2.783333 | 2,911 | 1,522 | 1,389 |
| 3 | scene13_dyn_test_03_000000 | 15 | 0.266667 | 4,502 | 2,203 | 2,299 |
| 4 | scene14_dyn_test_02_000000 | 57 | 0.966667 | 9,194 | 6,664 | 2,530 |
| 5 | scene13_dyn_test_02_000000 | 122 | 2.050000 | 15,511 | 11,252 | 4,259 |
| 6 | scene13_dyn_test_02_000000 | 766 | 12.783334 | 27,639 | 364 | 27,275 |
| 7 | scene13_dyn_test_01_000000 | 121 | 2.033333 | 33,847 | 24,622 | 9,225 |

Camera velocity was not included because it was not needed for this training-path gate.

## Eight-sample overfit

The final diagnostic run started from the official checkpoint and ran 500 optimizer steps. Exactly
the same eight samples were shuffled with seed 7; no ninth sample or augmentation was introduced.

| Metric | Initial | Final | Change |
|---|---:|---:|---:|
| mean L_spike | 9,360.598465 | 9,324.666229 | ratio 0.996161 |
| mean IoU | 0.0001831 | 0.0028752 | +0.0026921 |
| mean foreground recall | 0.0001838 | 0.0045299 | +0.0043461 |
| mean background leakage | 0.1857062 | 0.4265322 | +0.2408260 (worse) |
| mean predicted spikes | 308.75 | 1,864.375 | +1,555.625 |

Training diagnostics:

```text
optimizer: Adam, LR 1e-4, AMSGrad
steps: 500
batch size: 1
spatial strategy: full frame
runtime: 64.663928 s (model already constructed and samples already loaded)
peak allocated VRAM: 772.323242 MiB
NaN/Inf loss: none
NaN/Inf gradient: none
all-zero gradient: none
```

At 20 steps the mean loss was 9,275.193176, showing a small initial decrease. It stayed nearly flat
thereafter and finished at 99.62% of the initial value. The model first collapsed toward almost no
predicted spikes, then produced increasing sparse predictions after roughly 200 steps. Exact
foreground correspondence remained extremely low and background leakage worsened.

The visual comparison agrees with the metrics: the final predictions contain more sparse event
structure, but they do not visually approach the dense GT foreground-event patterns reliably.

```text
SPIKEMS_OVERFIT8_GATE=FAIL
```

## Minimal debugging performed

1. The checkpoint contains SRM/refractory buffers built for `tSample=100`. The wrapper therefore
   strict-loads the unchanged checkpoint model and feeds it T=10 tensors; it does not rebuild or
   truncate those buffers.
2. A first 20-step diagnostic exposed `0/0` in the reporting-only background-leakage metric when a
   prediction contained zero spikes. The convention was made explicit as leakage 0 in that case;
   recall and prediction spike count separately expose all-zero collapse. Training loss and model
   math were not changed.
3. Full-frame backward fit comfortably, so no crop or GT-assisted crop was added.

## Artifacts

Ignored runtime artifacts:

```text
outputs/spikems_training/backward_gate.json
outputs/spikems_training/overfit8/result.json
outputs/spikems_training/overfit8/training_log.csv
outputs/spikems_training/overfit8/sample_00/{raw_events,gt_foreground_events,pred_before_training,pred_after_training}.png
outputs/spikems_training/overfit8/sample_07/{raw_events,gt_foreground_events,pred_before_training,pred_after_training}.png
```

No trained checkpoint was saved because the overfit gate failed.

## Interpretation

The forward/backward/optimizer training chain is mechanically valid. With the fixed 10 ms definition,
official checkpoint initialization, LR 1e-4 and released L_spike alone, SpikeMS did not clearly
memorize these eight EVIMO2 samples within the allowed 500 steps. This result does not test the
paper's unavailable combined loss and must not be interpreted as a formal EVIMO2 benchmark.

Formal training is not authorized by this result.
