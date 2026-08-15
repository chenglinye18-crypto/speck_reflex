# Research Question

On one frozen EVIMO2 sample and one frozen 128x128 crop:

1. Does supervising the continuous final decoder membrane improve foreground
   spatial localization over spike-count BCE?
2. Does an initially gradient-balanced combination with official `spikeTime`
   learn both spatial location and the 5D event sequence?

# Fixed Sample

- Dataset: EVIMO2 Motion Segmentation, `right_camera/imo/train`
- Sequence: `scene14_dyn_test_02_000000`
- Frame: 57, mask timestamp: 0.966667 s
- Window: 0.961667--0.971667 s, 10 bins at 1 ms/bin
- Crop: x0=248, y0=0, width=128, height=128
- Marker: `DIAGNOSTIC_ONLY_GT_ASSISTED_CROP`
- Raw / foreground / background events: 1057 / 913 / 144
- Input: `[1,2,128,128,10]`
- Prediction, aligned GT, and membrane: `[1,2,127,127,10]`
- GT spatial positive / negative pixels: 726 / 15403

# Old Baselines

- L_spike-only, 500 steps: 5D IoU 0.030499, event recall 0.036996,
  leakage 0.026906, and 223 predicted spikes. Spatial metrics were not recorded.
- Spike-count BCE-only, 500 steps: spatial IoU 0.069929, precision 0.069929,
  recall 1.0, F1 0.130717, 10382 active pixels, 5D IoU 0.004438,
  leakage 0.162093, and 48898 predicted spikes.

# Membrane BCE Definition

- Extraction: a forward hook on official `model.deconv6`; the hook output is
  `spikes_mem_6`, before `slayer_conv6.spike`.
- Validation: membrane shape equals prediction shape and `requires_grad=True`;
  no detach is applied.
- Target: `ANY(gt_foreground_events over polarity and time)`, `[B,H,W]`.
- Aggregation: `membrane_max = MAX(membrane over polarity and time)`.
- Threshold: official final-layer `theta6=0.22`.
- Logit: `membrane_max - theta6`.
- Loss: PyTorch `binary_cross_entropy_with_logits`, mean over BHW, with a
  scalar `pos_weight`.
- Marker: `MEMBRANE_SPATIAL_BCE_V2`, provenance `ENGINEERING_CHOICE`.

V1 supervised already-thresholded spike counts through an exponential mapping.
V2 gives the spatial loss access to the continuous response before the final
spike threshold.

# Positive-Weight Screen

All cases independently reload the official checkpoint and train for 100 steps.

| pos_weight | BCE | IoU | Precision | Recall | F1 | Pred pixels | Pred spikes |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.000000 | 0.599088 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 |
| 4.606110 | 0.728734 | 0.031128 | 0.347826 | 0.033058 | 0.060377 | 69 | 214 |
| 21.216253 | 1.061439 | 0.161142 | 0.171737 | 0.723140 | 0.277557 | 3057 | 10523 |

`SELECTED_POS_WEIGHT=21.21625344352617` because it has the highest spatial F1
and is neither an all-zero nor full-frame prediction.

# Stage A — Membrane BCE-only

## Initial and Final

| Metric | Initial | Final |
|---|---:|---:|
| L_space | 1.335326 | 0.801396 |
| Spatial IoU | 0.000000 | 0.289790 |
| Spatial precision | 0.000000 | 0.299954 |
| Spatial recall | 0.000000 | 0.895317 |
| Spatial F1 | 0.000000 | 0.449361 |
| Predicted active pixels | 0 | 2167 |
| 5D event IoU | 0.000000 | 0.020893 |
| Event precision | 0.000000 | 0.022654 |
| Event recall | 0.000000 | 0.211883 |
| Background leakage | 0.000000 | 0.010907 |
| Predicted spikes | 0 | 8343 |

## Learning Curve

| Step | BCE | Spatial IoU | Precision | Recall | F1 | Pred pixels | 5D IoU | Pred spikes | Grad norm |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1.335326 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0.000000 | 0 | 0.002196 |
| 10 | 1.335325 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0.000000 | 0 | 0.002377 |
| 20 | 1.334765 | 0.004115 | 0.500000 | 0.004132 | 0.008197 | 6 | 0.001109 | 11 | 0.004787 |
| 50 | 1.195262 | 0.116792 | 0.127790 | 0.575758 | 0.209157 | 3271 | 0.007606 | 10766 | 0.122327 |
| 100 | 1.061404 | 0.162004 | 0.172561 | 0.725895 | 0.278836 | 3054 | 0.012063 | 10518 | 0.124447 |
| 200 | 0.943337 | 0.203824 | 0.213214 | 0.822314 | 0.338627 | 2800 | 0.015781 | 10308 | 0.125908 |
| 300 | 0.875740 | 0.247710 | 0.258413 | 0.856749 | 0.397064 | 2407 | 0.018221 | 9111 | 0.098360 |
| 400 | 0.826729 | 0.265844 | 0.274894 | 0.889807 | 0.420026 | 2350 | 0.019476 | 9263 | 0.071399 |
| 500 | 0.801396 | 0.289790 | 0.299954 | 0.895317 | 0.449361 | 2167 | 0.020893 | 8343 | 0.062369 |

All six trainable tensors have finite, non-zero gradients. Initial-to-final
per-layer norms were: conv1 `3.65e-8 -> 3.17e-6`, conv2 `9.76e-7 -> 7.05e-5`,
conv3 `2.00e-5 -> 9.56e-4`, deconv4 `1.98e-4 -> 5.65e-3`, deconv5
`8.61e-4 -> 1.47e-2`, and deconv6 `2.01e-3 -> 6.03e-2`.

The final binary spatial visualization follows the GT's repeated thin diagonal
structures. It remains thicker and has extra positives, but it is no longer a
large foreground block.

`MEMBRANE_BCE_GATE=PASS`

# Stage B — Combined Spatial + Temporal

## Initial Gradient Balance

- Initial L_space: 1.3353259563
- Initial L_time: 1368.8967285
- Initial G_space: 0.0021961385
- Initial G_time: 0.4273278935
- `lambda_time = G_space / G_time = 0.00513923518`
- Initial weighted temporal loss contribution: 7.0350822
- Definition: `L_total = L_space + lambda_time * L_time`
- Marker: `GRADIENT_BALANCED_COMBINED_V1`, provenance `ENGINEERING_CHOICE`

The scalar temporal contribution is larger than L_space, while the initial
parameter-gradient norms are matched by construction. Lambda is detached and
fixed for all 500 steps.

## Initial and Final

| Metric | Initial | Final |
|---|---:|---:|
| L_total | 8.370408 | 7.957370 |
| L_space | 1.335326 | 0.970406 |
| L_time | 1368.896729 | 1359.533936 |
| lambda_time * L_time | 7.035082 | 6.986964 |
| Spatial IoU | 0.000000 | 0.269173 |
| Spatial precision | 0.000000 | 0.297342 |
| Spatial recall | 0.000000 | 0.739669 |
| Spatial F1 | 0.000000 | 0.424171 |
| Predicted active pixels | 0 | 1806 |
| 5D event IoU | 0.000000 | 0.024100 |
| Event precision | 0.000000 | 0.031016 |
| Event recall | 0.000000 | 0.097534 |
| Background leakage | 0.000000 | 0.012834 |
| Predicted spikes | 0 | 2805 |

## Learning Curve

| Step | L_total | L_space | L_time | Spatial F1 | 5D IoU | Pred spikes | Grad norm |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 8.370408 | 1.335326 | 1368.896729 | 0.000000 | 0.000000 | 0 | 0.004067 |
| 10 | 8.370223 | 1.335141 | 1368.896729 | 0.000000 | 0.000000 | 1 | 0.010110 |
| 20 | 8.374173 | 1.330066 | 1370.652832 | 0.044329 | 0.003135 | 68 | 0.031377 |
| 50 | 8.575696 | 1.261035 | 1423.297485 | 0.241913 | 0.010018 | 1326 | 0.129691 |
| 100 | 8.519051 | 1.178714 | 1428.293457 | 0.345896 | 0.017659 | 2220 | 0.163588 |
| 200 | 8.244184 | 1.097742 | 1390.565430 | 0.389682 | 0.020250 | 2534 | 0.191077 |
| 300 | 8.184513 | 1.052188 | 1387.818359 | 0.390805 | 0.021437 | 2777 | 0.201535 |
| 400 | 8.147006 | 1.011785 | 1388.381958 | 0.408010 | 0.023836 | 2802 | 0.188006 |
| 500 | 7.957370 | 0.970406 | 1359.533936 | 0.424171 | 0.024100 | 2805 | 0.152478 |

At step 500, all six layer gradients remained finite and non-zero. Global norm
was 0.152478. Per-layer norms were conv1 `1.85e-5`, conv2 `4.18e-4`, conv3
`4.41e-3`, deconv4 `2.69e-2`, deconv5 `7.55e-2`, and deconv6 `1.30e-1`.
There was no NaN or Inf. BCE-only and combined runtimes were 5.03 s and 5.15 s;
peak allocated VRAM was 47.16 MiB.

# Comparison Table

| Metric | L_spike only | Old spike-count BCE | Membrane BCE | Combined |
|---|---:|---:|---:|---:|
| Spatial IoU | N/A | 0.069929 | 0.289790 | 0.269173 |
| Spatial precision | N/A | 0.069929 | 0.299954 | 0.297342 |
| Spatial recall | N/A | 1.000000 | 0.895317 | 0.739669 |
| Spatial F1 | N/A | 0.130717 | 0.449361 | 0.424171 |
| 5D event IoU | 0.030499 | 0.004438 | 0.020893 | 0.024100 |
| Event precision | N/A | N/A | 0.022654 | 0.031016 |
| Event recall | 0.036996 | 0.246637 | 0.211883 | 0.097534 |
| Background leakage | 0.026906 | 0.162093 | 0.010907 | 0.012834 |
| Predicted spikes | 223 | 48898 | 8343 | 2805 |

# BCE_COLLAPSE_DIAGNOSIS

The positive weight strongly controls prediction bias: weight 1 produced an
all-background result, sqrt(class ratio) was very conservative, and the full
class ratio produced the best 100-step F1. With the same full class-ratio weight,
the continuous membrane interface reduced active pixels from 10382 to 2167 and
raised precision from 0.0699 to 0.3000. The old collapse was therefore driven
mainly by the thresholded spike-count supervision interface. The large positive
weight increased foreground extent, but did not by itself force a full-image
solution under membrane supervision.

# COMBINED_SINGLE_SAMPLE_GATE

`COMBINED_SINGLE_SAMPLE_GATE=FAIL`

Combined training learned a useful spatial map and reduced spike overproduction.
Its 5D IoU improved only from 0.020893 to 0.024100 versus membrane BCE-only and
remained below L_spike-only's 0.030499. L_time ended almost unchanged at
1359.534 versus 1368.897 initially. This is insufficient evidence that temporal
structure is being learned reliably, so the combined objective is not yet a
usable spatial-plus-temporal training loss.

# Interpretation

Q1: The binary spike-count interface was the main cause of the prior
all-foreground collapse; positive weighting materially changes the foreground
bias and can amplify it.

Q2: Yes. Membrane BCE clearly improves foreground localization and recovers the
thin-line spatial structure on this fixed sample.

Q3: No. The first gradient-balanced combination is spatially useful and modestly
improves event IoU over BCE-only, but it does not yet establish temporal learning.
This matches Case 2: the next experiment should adjust only loss balancing.

# Visualizations

Generated diagnostic files are intentionally ignored by Git under:

- `outputs/spikems_training/membrane_combined/membrane_bce/`
- `outputs/spikems_training/membrane_combined/combined/`

Each directory contains raw events, GT foreground events, GT spatial target,
binary spatial prediction before/after, continuous membrane score before/after,
and predicted foreground events before/after.

# Files Modified

- `reference/spikems/losses.py`
- `scripts/diagnose_spikems_membrane_combined_loss.py`
- `reference/spikems/MEMBRANE_COMBINED_LOSS_DIAGNOSTIC_20260815.md`

SpikeMS submodule: clean.

EVIMO2 raw data: untouched.

# Next Recommended Step

Run one bounded single-sample loss-balancing diagnostic. Keep the model,
membrane BCE, sample, crop, optimizer, and learning rate fixed. Do not start
8-sample or formal training yet.

STOP
