# SpikeMS single-sample full-frame vs crop diagnostic — 2026-08-15

## Research question

Does a fixed foreground-containing 128x128 crop make the unchanged SpikeMS L_spike-only baseline
substantially easier to overfit than the full 480x640 frame?

```text
SPATIAL_SPARSITY_HYPOTHESIS=NOT_SUPPORTED
```

The crop produced a limited improvement, but neither support clearly overfit the one sample after the
same 500 updates. Spatial sparsity may contribute, but the observed difference is too small to call it
the main cause of the previous training failure.

## Fixed sample

The sample was selected only from the existing eight-sample manifest. The initially suggested
`scene13_dyn_test_03_000000`, frame 15 was rejected because its maximum-density crop contained zero
background events.

```text
sequence: scene14_dyn_test_02_000000
split: right_camera / motion_segmentation / imo / train
frame index / mask id: 57 / 57
timestamp: 0.966667 s
window: [0.961667, 0.971667] s
raw events: 9,194
foreground events: 6,664
background events: 2,530
OFF / ON events: 3,790 / 5,404
object mask pixels: 158,705 / 307,200
object mask coverage: 0.51661784
```

Both experiments used T=10, 1 ms/bin, the same polarity mapping, GT construction, official
checkpoint, model, neuron parameters, released `spikeTime` loss, fresh Adam optimizer, LR 1e-4,
AMSGrad, batch size 1, random seed 11, and exactly 500 optimizer steps. Each branch was initialized
independently from the same checkpoint.

## Fixed diagnostic crop

Marker: `DIAGNOSTIC_ONLY_GT_ASSISTED_CROP`.

The foreground-event projection sums both polarity channels and all ten time bins. The row-major
first pixel attaining the maximum projection count is selected, a 128x128 crop is centered there,
and its origin is deterministically clamped at image boundaries.

```text
projection peak: (x=312, y=35), 17 foreground voxels
crop: x0=248, y0=0, width=128, height=128
raw events: 1,057
foreground events: 913
background events: 144
foreground event ratio: 0.86376537
input voxels: 1,040
foreground voxels: 897
background voxels: 143
crop object-mask coverage: 0.58892822
```

The crop coordinate remained fixed for the full experiment.

## Experiment A — full frame

```text
input: [1, 2, 480, 640, 10]
prediction: [1, 2, 479, 639, 10]
input spikes: 9,085
foreground spikes: 6,601
background spikes: 2,484
foreground density: 6,601 / 6,144,000 = 0.0010743815
runtime: 53.578442 s
peak allocated VRAM: 772.323730 MiB
numerical issues: none
```

| Step | L_spike | IoU | Recall | Leakage | Pred spikes | Gradient norm |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 9,784.498047 | 0 | 0 | 0.227273 | 22 | 5.201487 |
| 10 | 9,785.552734 | 0.000150 | 0.000152 | 0.116883 | 77 | 8.717134 |
| 20 | 9,786.508789 | 0.000889 | 0.000910 | 0.050000 | 160 | 8.814646 |
| 50 | 9,780.018555 | 0.001027 | 0.001061 | 0.114035 | 228 | 15.752204 |
| 100 | 9,789.908203 | 0.002082 | 0.002274 | 0.120385 | 623 | 29.960349 |
| 200 | 9,787.490234 | 0.006794 | 0.008034 | 0.122514 | 1,257 | 85.607959 |
| 300 | 9,776.393555 | 0.014510 | 0.018948 | 0.104526 | 2,143 | 98.758497 |
| 400 | 9,739.744141 | 0.015604 | 0.021373 | 0.103488 | 2,580 | 99.146521 |
| 500 | 9,683.076172 | 0.019377 | 0.026982 | 0.108782 | 2,767 | 71.912280 |

Final/initial loss ratio: `0.98963443`.

## Experiment B — 128x128 crop

```text
input: [1, 2, 128, 128, 10]
prediction: [1, 2, 127, 127, 10]
input spikes: 1,040
foreground spikes: 897
background spikes: 143
foreground density: 897 / 327,680 = 0.0027374268
runtime: 6.313442 s
peak allocated VRAM: 43.720703 MiB
numerical issues: none
```

| Step | L_spike | IoU | Recall | Leakage | Pred spikes | Gradient norm |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1,368.896729 | 0 | 0 | 0 | 0 | 0.427328 |
| 10 | 1,368.896729 | 0 | 0 | 0 | 5 | 0.631835 |
| 20 | 1,368.826416 | 0 | 0 | 0 | 14 | 1.161505 |
| 50 | 1,369.845703 | 0.001072 | 0.001121 | 0 | 42 | 3.193053 |
| 100 | 1,370.966919 | 0.006876 | 0.007848 | 0.022556 | 133 | 5.981335 |
| 200 | 1,336.318115 | 0.013462 | 0.015695 | 0.024691 | 162 | 11.178595 |
| 300 | 1,315.833008 | 0.021336 | 0.025785 | 0.038278 | 209 | 11.819223 |
| 400 | 1,297.112549 | 0.025806 | 0.031390 | 0.045249 | 221 | 17.637650 |
| 500 | 1,293.049927 | 0.030499 | 0.036996 | 0.026906 | 223 | 19.749684 |

Final/initial loss ratio: `0.94459275`.

## Primary A/B comparison

| Metric | Full frame | 128x128 crop |
|---|---:|---:|
| Foreground density | 0.00107438 | 0.00273743 |
| Density increase vs full | 1.00x | 2.55x |
| Initial L_spike | 9,784.498047 | 1,368.896729 |
| Final L_spike | 9,683.076172 | 1,293.049927 |
| Loss ratio | 0.989634 | 0.944593 |
| Initial IoU | 0 | 0 |
| Final IoU | 0.019377 | 0.030499 |
| Initial recall | 0 | 0 |
| Final recall | 0.026982 | 0.036996 |
| Initial leakage | 0.227273 | 0 |
| Final leakage | 0.108782 | 0.026906 |
| Initial prediction spikes | 22 | 0 |
| Final prediction spikes | 2,767 | 223 |

The absolute L_spike values cannot be compared directly across supports because the released loss is
an unnormalized sum. Loss ratio and correspondence metrics are the relevant comparisons.

## Per-layer gradient diagnostic

All six trainable layers had finite, non-zero gradients initially and at step 500. The signal was
strongly decoder-heavy in both experiments.

| Layer | Full initial | Full final | Crop initial | Crop final |
|---|---:|---:|---:|---:|
| conv1 | 4.578e-4 | 2.490e-3 | 5.385e-6 | 8.235e-4 |
| conv2 | 1.167e-2 | 6.548e-2 | 1.371e-4 | 1.421e-2 |
| conv3 | 9.683e-2 | 8.173e-1 | 3.068e-3 | 2.226e-1 |
| deconv4 | 5.406e-1 | 6.990 | 4.259e-2 | 2.034 |
| deconv5 | 2.400 | 41.423 | 1.674e-1 | 8.153 |
| deconv6 | 4.582 | 58.361 | 3.909e-1 | 17.871 |

The first and last layer norms differ by more than four orders of magnitude at initialization and
remain separated by roughly four orders after training. This is recorded as a possible future loss /
temporal-dynamics diagnostic; no clipping, optimizer, loss, neuron, or architecture change was made.

## Visual assessment

The crop prediction changes from all-zero to sparse activity and contains a small amount of spatial
correspondence. It does not recover the GT's clear diagonal event structure or either polarity well.
The full-frame result is also sparse and strongly biased toward one displayed polarity.

The fixed crop therefore improves efficiency and metrics modestly, but does not change the outcome
from “cannot clearly overfit one sample” to “can clearly overfit one sample.”

## Interpretation

```text
SPATIAL_SPARSITY_HYPOTHESIS=NOT_SUPPORTED
```

Full-frame spatial sparsity is not supported as the main training obstruction. A foreground crop
helps, so sparsity is likely a secondary factor. The remaining evidence points more strongly toward
the L_spike-only localization signal, the large encoder/decoder gradient imbalance, or the T=10 vs
checkpoint/SLAYER dynamics boundary.

No eight-sample run, formal training, BCE reconstruction, loss normalization, or parameter tuning was
performed.
