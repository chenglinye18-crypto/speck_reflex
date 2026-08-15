# Research Question

Forward-only tests of two suspected causes of SpikeMS's late-window predictions:

1. Does activity begin progressively later through the six-layer SNN?
2. Does official `spikeTime()` penalize an incorrect spike less near the end of
   a short window?

No training, backward pass, optimizer step, loss modification, or data change
was performed.

# Fixed Sample

- EVIMO2 Motion Segmentation `right_camera/imo/train`
- Sequence `scene14_dyn_test_02_000000`, frame 57
- Window 0.961667--0.971667 s, T=10, 1 ms/bin
- Crop x0=248, y0=0, 128x128
- Raw / foreground / background events: 1057 / 913 / 144
- Official SpikeMS model and official EV-IMO checkpoint

# Experiment A — Layer Temporal Propagation

The official model's own MetaTensors were read after one forward pass. Counts
are binary spike voxels summed over batch, channel, height, and width.

| Layer | t0 | t1 | t2 | t3 | t4 | t5 | t6 | t7 | t8 | t9 | First spike |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| input | 99 | 117 | 123 | 133 | 91 | 98 | 106 | 92 | 93 | 88 | 0 |
| conv1 | 0 | 811 | 1213 | 1515 | 1715 | 1568 | 1685 | 1712 | 1602 | 1681 | 1 |
| conv2 | 0 | 0 | 2377 | 3825 | 5122 | 5955 | 6235 | 6643 | 6889 | 6902 | 2 |
| conv3 | 0 | 0 | 0 | 1959 | 3258 | 3894 | 4280 | 4647 | 4890 | 5033 | 3 |
| deconv4 | 0 | 0 | 0 | 0 | 2658 | 6109 | 6722 | 7550 | 8098 | 8333 | 4 |
| deconv5 | 0 | 0 | 0 | 0 | 0 | 8757 | 13301 | 14291 | 15990 | 16196 | 5 |
| deconv6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | none |

The first-spike sequence is exactly `0,1,2,3,4,5,none`. Each responding layer
adds one timestep of startup delay. The final layer does not reach threshold
anywhere within this 10-bin crop at official initialization.

The input contains 1040 occupied spike voxels versus 1057 raw events because
multiple raw events can map to the same polarity/pixel/time voxel.

`LAYER_TEMPORAL_DELAY=YES`

# Experiment B — spikeTime Window-End Bias

Synthetic prediction and GT shape: `[1,2,127,127,10]`. GT is all zero. Prediction
contains exactly one erroneous spike at fixed batch 0, polarity 0, y=63, x=63.
Only its timestep changes.

| Error spike timestep | Official spikeTime loss |
|---:|---:|
| 0 | 1.83265793 |
| 1 | 1.82342505 |
| 2 | 1.80359507 |
| 3 | 1.76232505 |
| 4 | 1.67990470 |
| 5 | 1.52432013 |
| 6 | 1.25364959 |
| 7 | 0.83978522 |
| 8 | 0.33978522 |
| 9 | 0.00000000 |

Loss decreases monotonically toward the end and is exactly zero at t9.

Official `spikeTime()` computes:

```text
error = PSP(prediction - GT)
loss = 0.5 * sum(error^2) * Ts
```

The loss PSP kernel has length 16 (`Ts=1`, configured `tSample=100`):

```text
[0.0,
 0.82436061,
 1.0,
 0.90979600,
 0.73575890,
 0.55782539,
 0.40600586,
 0.28729749,
 0.19914827,
 0.13588822,
 0.09157819,
 0.06109948,
 0.04042768,
 0.02656402,
 0.01735127,
 0.01127579]
```

This is a causal kernel whose first value is zero. An error spike at t9 would
produce its first non-zero PSP response at t10, outside the evaluated tensor, so
the returned loss is zero. Earlier error spikes retain progressively more of
their PSP tail inside the window and receive progressively larger penalties.

`SPIKETIME_END_BIAS=YES`

# Decision

```text
A. LAYER_TEMPORAL_DELAY = YES
B. SPIKETIME_END_BIAS = YES
```

The current training definition combines same-index `prediction[t]` versus
`GT[t]` supervision with a 10 ms window. It has a structural problem:

- the six-layer SNN consumes most of the short window before its deepest output
  can respond;
- the temporal loss increasingly discounts late mistakes and assigns zero loss
  to an erroneous final-bin spike.

The next priority should be one bounded test of:

```text
warm-up/context + valid loss region
```

Provide earlier events as context so deep layers can warm up, then calculate
loss only over a clearly defined valid region with enough temporal support for
the PSP error tail. Do not begin 8-sample or formal training before validating
that definition.

# Files

- `scripts/diagnose_spikems_temporal_window.py`
- `reference/spikems/TEMPORAL_WINDOW_DIAGNOSTIC_20260815.md`
- Ignored result:
  `outputs/spikems_training/temporal_window_diagnostic/result.json`

SpikeMS submodule: clean.

EVIMO2 raw data: untouched.

STOP
