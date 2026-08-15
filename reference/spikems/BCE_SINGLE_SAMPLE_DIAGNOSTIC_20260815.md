# Reconstructed spatial BCE single-sample diagnostic — 2026-08-15

## Research question

Does direct two-dimensional foreground-event-location supervision let the unchanged SpikeMS model
clearly overfit the same fixed EVIMO2 crop?

```text
BCE_LOCALIZATION_HYPOTHESIS=NOT_SUPPORTED
RECONSTRUCTED_BCE_SINGLE_SAMPLE_GATE=FAIL
```

The loss decreased strongly and covered every GT-positive pixel, but the model activated most of the
crop rather than recovering the GT line structure. This is a high-recall, very-low-precision solution,
not successful localization.

## BCE definition

Marker: `RECONSTRUCTED_SPATIAL_BCE_V1`.

Provenance: `ENGINEERING_RECONSTRUCTION`. This is not claimed to reproduce the unpublished SpikeMS
BCE implementation.

For prediction and aligned GT tensors with `[B,2,H,W,T]` layout:

```text
gt_spatial = ANY(gt_foreground_events, over polarity and time)
pred_count = SUM(pred_foreground_events, over polarity and time)
p_fg = eps + (1 - 2*eps) * (1 - exp(-pred_count))
eps = 1e-6

N_pos = 726
N_neg = 15,403
w_pos = N_neg / N_pos = 21.2162534435

L_bce = -mean(w_pos*y*log(p_fg) + (1-y)*log(1-p_fg))
```

The target uses pixels with actual GT foreground DVS events in the 10 ms window, not the complete
object mask. No count clamp, detach, L_spike, Dice, focal term, regularizer, or additional
normalization is used. Spatial prediction uses the fixed threshold `p_fg >= 0.5`.

At the initial checkpoint, a row-major selected GT-positive pixel `(x=14, y=0)` had
`pred_count=0`. Its direct probability-path derivative was finite and non-zero:

```text
dL_bce / d(pred_count) = -1315.40771484375
```

This verifies that the zero-count positive path was not disconnected by clamp or detach.

## Fixed sample and crop

Exactly the previous diagnostic sample and crop were reused without recomputation:

```text
sequence: scene14_dyn_test_02_000000
frame: 57
timestamp: 0.966667 s
window: [0.961667, 0.971667] s
physical window / T: 10 ms / 10
crop: x0=248, y0=0, width=128, height=128
raw events: 1,057
foreground events: 913
background events: 144
OFF / ON events: 421 / 636
input: [1, 2, 128, 128, 10]
model output / BCE support: [1, 2, 127, 127, 10]
```

Marker: `DIAGNOSTIC_ONLY_GT_ASSISTED_CROP`.

The model was independently initialized from the official EV-IMO checkpoint. Optimizer remained
fresh Adam with LR 1e-4, betas `(0.9, 0.999)`, weight decay 0 and AMSGrad. Batch size was one and
seed was 11.

## Initial

```text
L_bce: 13.1936454773

Spatial IoU: 0
Spatial precision: 0
Spatial recall: 0
Spatial F1: 0
GT active pixels: 726
Predicted active pixels: 0

5D event IoU: 0
5D event recall: 0
Background leakage: 0
Prediction spikes: 0

Global gradient norm: 774.9577475
All gradients finite: yes
All six trainable layers non-zero: yes
```

Initial layer gradient norms:

| Layer | Norm |
|---|---:|
| conv1 | 0.00929967 |
| conv2 | 0.25385159 |
| conv3 | 5.95068645 |
| deconv4 | 79.37993622 |
| deconv5 | 300.78533936 |
| deconv6 | 709.75421143 |

## Final

```text
L_bce: 2.8105955124
loss ratio: 0.2130264541

Spatial IoU: 0.06992872
Spatial precision: 0.06992872
Spatial recall: 1.0
Spatial F1: 0.13071660
GT active pixels: 726
Predicted active pixels: 10,382

5D event IoU: 0.00443817
5D event recall: 0.24663677
Background leakage: 0.16209252
Prediction spikes: 48,898

Global gradient norm: 0.14398648
All gradients finite: yes
All six trainable layers non-zero: yes
```

Final layer gradient norms:

| Layer | Norm |
|---|---:|
| conv1 | 1.0252e-5 |
| conv2 | 1.8530e-4 |
| conv3 | 0.00224469 |
| deconv4 | 0.01655349 |
| deconv5 | 0.07161362 |
| deconv6 | 0.12379219 |

## Learning curve

All metrics are evaluated after the corresponding optimizer update, except step 0.

| Step | BCE | Spatial IoU | Precision | Recall | F1 | Pred pixels | Pred spikes | Event IoU | Grad norm |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 13.19365 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 774.96 |
| 10 | 13.03587 | 0.01200 | 0.27273 | 0.01240 | 0.02372 | 33 | 53 | 0.00212 | 10,868.91 |
| 20 | 11.24293 | 0.09964 | 0.21961 | 0.15427 | 0.18123 | 510 | 1,189 | 0.00726 | 28,078.78 |
| 50 | 3.21302 | 0.08693 | 0.08800 | 0.87741 | 0.15995 | 7,239 | 27,639 | 0.00582 | 21,426.49 |
| 100 | 2.79202 | 0.07251 | 0.07262 | 0.98072 | 0.13522 | 9,805 | 44,093 | 0.00456 | 4,367.65 |
| 200 | 2.83536 | 0.07061 | 0.07064 | 0.99311 | 0.13191 | 10,206 | 47,706 | 0.00430 | 49.36 |
| 300 | 2.83863 | 0.07054 | 0.07058 | 0.99311 | 0.13179 | 10,216 | 47,786 | 0.00431 | 67.21 |
| 400 | 2.81542 | 0.07050 | 0.07052 | 0.99587 | 0.13171 | 10,253 | 48,026 | 0.00435 | 1,686.94 |
| 500 | 2.81060 | 0.06993 | 0.06993 | 1.00000 | 0.13072 | 10,382 | 48,898 | 0.00444 | 0.14399 |

The best recorded spatial IoU was about 0.10 at step 20. Continued optimization reduced BCE mainly
by covering nearly all positive pixels while activating a very large background region. There were
no NaN/Inf values, despite large transient gradient norms.

Training runtime was 5.103902 seconds and peak allocated VRAM was 43.760742 MiB.

## Comparison with the saved L_spike-only crop result

| Metric | L_spike-only | BCE-only |
|---|---:|---:|
| Own loss ratio | 0.944593 | 0.213026 |
| Final 5D event IoU | 0.030499 | 0.004438 |
| Final event recall | 0.036996 | 0.246637 |
| Final background leakage | 0.026906 | 0.162093 |
| Final prediction spikes | 223 | 48,898 |

The previous L_spike prediction was sparse and did not recover the GT structure. BCE-only drove much
more activity and reached full 2D recall, but the final spatial projection became a broad filled
region. Only 6.99% of predicted active pixels were correct. The GT consists of thin diagonal event
lines, which the final BCE prediction does not reproduce.

Event-level metrics are secondary for BCE-only because this loss has no direct timing supervision.
They are included only to expose the large spike increase and background leakage.

## Interpretation

```text
BCE_LOCALIZATION_HYPOTHESIS=NOT_SUPPORTED
RECONSTRUCTED_BCE_SINGLE_SAMPLE_GATE=FAIL
```

`RECONSTRUCTED_SPATIAL_BCE_V1` supplies a strong learning signal and avoids the all-background
solution. Under the required fixed positive weight and spike-count probability mapping, it instead
converges toward extensive foreground overprediction. The final probability map is not visually close
to the GT foreground-event lines, so the required localization criterion is not satisfied.

This result does not justify constructing BCE plus lambda times L_spike. No loss/weight/threshold
tuning, combined loss, eight-sample run, or formal training was performed.
