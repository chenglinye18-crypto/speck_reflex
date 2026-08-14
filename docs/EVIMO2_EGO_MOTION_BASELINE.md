# EVIMO2 Samsung Ego-Motion Baseline v0.1

## Status

```text
design: frozen
implementation: complete
training: controlled smoke only; full run not started
performance claims: none
```

This document freezes the first measured-data training baseline for the SNN
Motion Backbone. The data and training pipeline is implemented, but only a
controlled two-window training smoke has run; no functional model is claimed.
The task is camera-local ego-motion regression, not robot-base odometry,
independent-object segmentation, collision prediction, or motor control.

## Scope

Use only the Samsung event camera sequences from EVIMO2v2 Structure from
Motion / Object Recognition for the initial baseline:

```text
EVIMO2v2 Structure-from-Motion / samsung_mono
```

The raw dataset remains outside this repository under
`~/datasets/evimo2/`. Dataset provenance, polarity handling, and the current
adapter contract are defined in [EVIMO2_DATASET.md](EVIMO2_DATASET.md).

## Frozen data flow

```text
64 ms raw Samsung DVS event window
        |
        v
1 ms temporal bins and canonical OFF/ON polarity
        |
        v
event-coordinate 5x spatial reduction
        |
        v
[B, 64, 2, 96, 128] event counts
        |
        v
existing calibrated Multi-timescale LIF SNN
        |
        v
primitive spikes [B, 64, 16, 24, 32]
        |
        v
per-timestep adaptive average pooling to 2x2
        |
        v
global embedding [B, 64, 64]
        |
        v
temporal mean [B, 64]
        |
        v
Linear(64, 6)
        |
        v
normalized camera-local SE(3) twist [B, 6]
```

The backbone topology, LIF equation, threshold, time constants, and local head
are unchanged by this baseline. The measured-data input audit found that the
earlier binary-synthetic gain profile over-activated S5 and S6, so this
baseline explicitly freezes a Samsung SFM initialization profile:

```text
S1, S2, S3, S4, S5, S6, primitive
0.75, 2.5, 3.0, 3.0, 3.0, 3.0, 2.0
```

These static gains are folded into the initialized convolution weights. They
do not add runtime multiplication or normalization and do not change the
network topology. The generic backbone default remains unchanged.

## Input contract

One raw event is `(x, y, timestamp, polarity)`. The model input uses:

```text
shape:       [B, T, P, H, W] = [B, 64, 2, 96, 128]
T:           64 bins
bin width:   1 ms
duration:    64 ms
channel 0:   OFF
channel 1:   ON
value:       integer event count represented as float32
```

Samsung's native `640 x 480` coordinates are reduced exactly by:

```python
x_reduced = x // 5
y_reduced = y // 5
events[t_bin, canonical_polarity, y_reduced, x_reduced] += 1
```

This is event-domain `5 x 5` sum aggregation. It must be performed while
binning, without constructing and resizing a dense
`[64, 2, 480, 640]` tensor. Bilinear interpolation, polarity modification,
timestamp rewriting, silent clipping, and conversion through RGB frames are
not allowed. Original EVIMO2 files remain unchanged.

## Model readout

The existing backbone produces:

```text
primitive_spikes: [B,64,16,24,32]
global_embedding: [B,64,64]
ego_motion:       [B,64,6]
```

The frozen window readout is:

```python
window_embedding = output.global_embedding.mean(dim=1)
prediction = model.ego_motion_head(window_embedding)
```

For the existing shared linear head, this is mathematically equivalent to
`output.ego_motion.mean(dim=1)`. The training implementation should use one
form consistently and test their equivalence. The local motion head is not
supervised in this phase and must not be described as trained.

## Target contract

For window endpoints `t0` and `t1`, the target is:

```text
xi = log(inv(T_wc(t0)) @ T_wc(t1)) / (t1 - t0)
xi = [vx, vy, vz, wx, wy, wz]
```

Units and axes:

```text
vx, vy, vz: metres per second
wx, wy, wz: radians per second
axes:        camera coordinate system at the window start
shape:       [B, 6]
```

This is the constant-twist equivalent of the endpoint relative pose over the
window. It is not an arithmetic average of instantaneous velocity and is not
robot chassis motion. A validated camera-to-robot extrinsic transform is
required before robot-frame interpretation.

## Dataset split and window sampling

Train, validation, and test partitions must be separated by complete EVIMO2
sequence. Randomly splitting overlapping windows is prohibited because it
would leak nearly identical temporal content across partitions.

Window-index generation must be deterministic and record:

- sequence identifier and upstream split;
- window start and end timestamps;
- ground-truth frame identifier when applicable;
- event count and sensor identifier;
- sampling stride, seed, and rejected-window reason.

Only windows fully covered by both the event stream and camera trajectory are
valid. Dataset normalization statistics must be derived from the training
partition only.

## State boundary

Every training sample is an independent 64 ms window. The model state is
therefore cleared before every batch forward pass:

```python
model.reset_state()
output = model(events)
```

Membrane state must not leak between shuffled windows, sequences, validation,
or test samples. Stateful continuous-stream inference is a later, separate
protocol and is not part of this baseline.

## Normalization and loss

Compute a mean and standard deviation independently for all six target
components using the training partition only:

```text
z_target = (target - train_mean) / max(train_std, epsilon)
loss = SmoothL1(z_prediction, z_target)
```

The implementation must freeze and save:

- the six-component training mean and standard deviation;
- the numerical `epsilon` and SmoothL1 `beta`;
- the coordinate convention and physical units;
- the dataset index hash and split definition.

Validation and test predictions must be converted back to physical units.
Report at least per-component MAE, translation-vector MAE, and rotation-vector
MAE in addition to normalized loss. A single normalized loss is not a
sufficient performance report.

## Mandatory pre-training input audit

The earlier SNN excitability gains were calibrated with binary synthetic
events. Spatially aggregated EVIMO2 cells can contain counts greater than one.
Before optimization, run deterministic measured-data diagnostics and report:

- input nonzero fraction and mean active count;
- event-count p50, p90, p95, p99, p99.9, and maximum;
- S1--S6 and primitive firing fractions;
- fast/slow firing fractions and membrane distributions;
- zero-event-window behavior and finite-value checks.

No-motion input must remain silent. Any hidden-layer firing fraction above
10% is flagged `OVERACTIVE`; above 25% blocks training. Do not silently clip,
normalize, or binarize counts to make this audit pass. If the real input
distribution is incompatible with the calibrated gains, record the evidence
and revisit the encoding/calibration policy explicitly.

The initial measured-data calibration used seed 17 and four deterministic SFM
training windows. Before calibration, S5 and S6 fired at approximately 27.31%
and 32.97%. With the frozen Samsung profile, the measured firing fractions
were approximately:

```text
S1 1.81%   S2 1.62%   S3 1.31%   S4 1.08%
S5 1.01%   S6 0.99%   primitive 0.32%
```

The executable preflight repeats this audit on the configured window count
before every new or resumed training run. These figures establish a numerical
initialization point only; they are not task-performance results.

## Initial training boundary

The first implementation may train only:

- the existing SNN backbone and primitive bottleneck through surrogate
  gradients;
- the existing global ego-motion linear head.

It must not add or claim:

- independent-object motion training;
- local segmentation supervision;
- robot-base velocity or pose estimation;
- optical flow, depth, TTC, collision, or fall detection;
- quantization or STM32, FPGA, Speck, or motor deployment.

The Safety MCU remains the final deterministic arbiter; this model never
directly controls PWM.

## Acceptance gates

Training may start only after all of the following pass:

1. Direct event-domain reduction produces `[64,2,96,128]` and preserves the
   total event count and canonical polarity.
2. Sequence-level splits are deterministic and disjoint.
3. Target normalization uses training data only and round-trips to physical
   units.
4. Every independent window resets SNN state.
5. Real-input activity reaches the primitive layer without firing explosion.
6. Loss and all gradients are finite, including the S1 convolution gradient.
7. Checkpoints contain model state, optimizer state, normalization metadata,
   split/index identity, seed, and software versions.
8. Evaluation reports physical-unit errors and makes no robot-motion or
   independent-object capability claim.

## Reproducibility constants

The first implementation must record rather than infer these values:

```text
network initialization seed: 17
dataset sampling seed:        17
temporal bins:                64
dt:                           1 ms
window duration:              64 ms
input polarity:               channel 0 OFF, channel 1 ON
native resolution:            480 x 640
model resolution:             96 x 128
spatial reduction:            exact 5 x event-coordinate aggregation
state policy:                 reset per independent window
EVIMO2 layer gains:           0.75, 2.5, 3, 3, 3, 3, 2
```

Optimizer, learning rate, batch size, sampling stride, normalization epsilon,
SmoothL1 beta, and epoch budget are intentionally not guessed here. They must
be selected and recorded when the measured-data audit and training
implementation are completed.
