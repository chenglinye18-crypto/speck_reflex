# Research Question

Does the official SpikeMS checkpoint show the same layer-by-layer delay and
late-window prediction concentration on its original EVIMO data domain?

No training, backward pass, loss change, network change, or EVIMO2 modification
was performed.

# Dataset

- Official source: EVIMO DAVIS346C evaluation wall NPZ archive
- Sequence retained: `eval/wall/npz/seq_00.npz`
- Resolution: 260x346 (height x width)
- Local path:
  `/home/speck/datasets/evimo1/reference/eval/wall/npz/seq_00.npz`
- Extracted sequence size: 396 MiB
- Five samples: frame IDs 26, 27, 28, 29, and 30

The old SpikeMS preformatted Google Drive link now returns 404. The experiment
therefore uses the official EVIMO NPZ and a small compatibility adapter that
reproduces the released SpikeMS preprocessing operations. The SpikeMS submodule
itself remains unchanged.

# Official Preprocessing Audit

| Item | Released behavior |
|---|---|
| Sensor | DAVIS346C, 346x260 |
| Event row | timestamp, x, y, polarity |
| Polarity | released EVIMO values 0/1 are used directly as channels |
| `valid_time=0.01` | center time minus 10 ms through center time plus 10 ms |
| Physical sample window | 20 ms total |
| Mask alignment | `frames[i]['cam']['ts']` and `depth_mask_<frame id>.png` |
| Mask | object ID > 0, followed by one 5x5 dilation |
| GT events | input spike tensor intersected with the dilated object mask |
| Preprocessing `min_events=1000` | code applies the threshold to raw mask value sum |
| Dataloader `minEvents=30` | code applies it to boolean mask pixel count |
| Background filtering | full-frame background/foreground spike ratio |
| Ratio used here | <=1.5, matching the README's recommended test setting |
| Crop | GT-assisted 128x128 crop centered on the densest foreground-event pixel |
| Temporal bin mapping | `trunc((T-1)*(event_t-start)/(stop-start))` |

The adapter also reproduces the released dataloader's `events_idx[next]-1`
off-by-one behavior, which omits the final stored event of each sample.

Important upstream ambiguity: the physical window recovered from preprocessing
is 20 ms, while the released config/checkpoint uses `tSample=100`. Per this
experiment's requested definition, the physical 20 ms window is evaluated with
T=20 (1 ms/bin). This is not claimed to resolve the public T=100 mismatch.

# Selected Samples

The first five frames passing the upstream mask and full-frame ratio filters were
used. Their official 20 ms GT-assisted crop is frozen for both window variants.

| Frame ID | Mask time (s) | Crop x0,y0 | 20 ms background/foreground ratio |
|---:|---:|---:|---:|
| 26 | 0.784958 | 87,100 | 1.1503 |
| 27 | 0.809939 | 101,94 | 0.8278 |
| 28 | 0.834976 | 101,73 | 0.6712 |
| 29 | 0.865063 | 181,114 | 0.6409 |
| 30 | 0.889970 | 183,84 | 0.6825 |

# Reference Inference Metrics

## 10 ms / T=10

| Frame | Spatial IoU | Spatial F1 | Event IoU | Event recall | GT spikes | Pred spikes |
|---:|---:|---:|---:|---:|---:|---:|
| 26 | 0.371094 | 0.541311 | 0.021359 | 0.038036 | 12462 | 10204 |
| 27 | 0.380149 | 0.550881 | 0.016305 | 0.028488 | 13690 | 10619 |
| 28 | 0.424796 | 0.596290 | 0.018144 | 0.032814 | 10788 | 9077 |
| 29 | 0.394924 | 0.566230 | 0.021694 | 0.040480 | 17342 | 15719 |
| 30 | 0.456441 | 0.626789 | 0.021651 | 0.042111 | 20090 | 19830 |
| Mean | 0.405481 | 0.576300 | 0.019831 | 0.036386 | 14874.4 | 13089.8 |

## Official physical window: 20 ms / T=20

| Frame | Spatial IoU | Spatial F1 | Event IoU | Event recall | GT spikes | Pred spikes |
|---:|---:|---:|---:|---:|---:|---:|
| 26 | 0.727051 | 0.841957 | 0.050372 | 0.314535 | 25091 | 139475 |
| 27 | 0.752001 | 0.858448 | 0.052051 | 0.301660 | 27594 | 140650 |
| 28 | 0.772835 | 0.871863 | 0.049571 | 0.298926 | 21875 | 116576 |
| 29 | 0.724603 | 0.840313 | 0.051537 | 0.278447 | 34520 | 161598 |
| 30 | 0.768580 | 0.869149 | 0.058942 | 0.303053 | 39894 | 177313 |
| Mean | 0.749014 | 0.856346 | 0.052495 | 0.299324 | 29794.8 | 147122.4 |

The 20 ms foreground prediction has strong spatial correspondence with GT. The
model therefore produces a reasonable foreground region in its original data
domain. Exact event-time correspondence remains low and prediction spike count
is about five times GT on average.

`EVIMO_REFERENCE_INFERENCE=PASS`

# Layer Temporal Propagation

All five samples and both windows have the same first-spike sequence:

| Layer | input | conv1 | conv2 | conv3 | deconv4 | deconv5 | deconv6 |
|---|---:|---:|---:|---:|---:|---:|---:|
| First timestep | 0 | 1 | 2 | 3 | 4 | 5 | 6 |

The following tables show frame 26. Full per-layer, per-timestep counts for all
five frames are stored in the ignored `result.json` named below.

## Frame 26: 10 ms / T=10

| Layer | t0 | t1 | t2 | t3 | t4 | t5 | t6 | t7 | t8 | t9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| input | 1475 | 1504 | 1688 | 1614 | 1535 | 1597 | 1444 | 1422 | 1430 | 0 |
| conv1 | 0 | 9092 | 11045 | 14214 | 15663 | 16916 | 17788 | 17811 | 17896 | 17989 |
| conv2 | 0 | 0 | 11770 | 14264 | 16392 | 17840 | 18780 | 19500 | 19780 | 20009 |
| conv3 | 0 | 0 | 0 | 4571 | 6414 | 7161 | 7753 | 8123 | 8356 | 8597 |
| deconv4 | 0 | 0 | 0 | 0 | 3582 | 7858 | 8083 | 9112 | 9765 | 10090 |
| deconv5 | 0 | 0 | 0 | 0 | 0 | 9774 | 13757 | 15290 | 16960 | 17640 |
| deconv6 | 0 | 0 | 0 | 0 | 0 | 0 | 52 | 1928 | 3165 | 5059 |

## Frame 26: 20 ms / T=20

| Layer | t0 | t1 | t2 | t3 | t4 | t5 | t6 | t7 | t8 | t9 | t10 | t11 | t12 | t13 | t14 | t15 | t16 | t17 | t18 | t19 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| input | 1541 | 1606 | 1529 | 1440 | 1363 | 1397 | 1463 | 1594 | 1528 | 1443 | 1514 | 1382 | 1351 | 1368 | 1419 | 1499 | 1589 | 1500 | 1514 | 0 |
| conv1 | 0 | 9623 | 12240 | 14418 | 15311 | 15837 | 16455 | 16746 | 17618 | 18102 | 18418 | 18594 | 18380 | 17981 | 17898 | 17761 | 18064 | 18473 | 18996 | 18993 |
| conv2 | 0 | 0 | 12795 | 15506 | 17266 | 18248 | 18853 | 19268 | 19541 | 19808 | 20128 | 20289 | 20600 | 20620 | 20556 | 20651 | 20552 | 20502 | 20701 | 20855 |
| conv3 | 0 | 0 | 0 | 4844 | 6740 | 7294 | 7924 | 8221 | 8405 | 8579 | 8672 | 8811 | 8827 | 8885 | 8943 | 8940 | 8997 | 9027 | 9028 | 9070 |
| deconv4 | 0 | 0 | 0 | 0 | 3728 | 8022 | 8029 | 9315 | 9743 | 10266 | 10479 | 10754 | 10924 | 11026 | 11128 | 11167 | 11190 | 11252 | 11303 | 11401 |
| deconv5 | 0 | 0 | 0 | 0 | 0 | 10122 | 13568 | 15159 | 16937 | 17548 | 18492 | 18798 | 19231 | 19403 | 19533 | 19631 | 19709 | 19787 | 19796 | 19839 |
| deconv6 | 0 | 0 | 0 | 0 | 0 | 0 | 38 | 1840 | 3517 | 5716 | 7741 | 9419 | 10866 | 12002 | 12989 | 13853 | 14617 | 15169 | 15690 | 16018 |

`TEMPORAL_DELAY_10MS=YES`

`TEMPORAL_DELAY_OFFICIAL_WINDOW=YES`

In 10 ms, deconv6 has only t6--t9 available. Across the five samples, 99.63% of
prediction spikes occur in the last three bins and 50.11% occur at t9 alone.

In 20 ms, deconv6 still begins at t6, but it has fourteen output timesteps. The
spatial prediction recovers strongly. Prediction activity remains late-biased:
91.53% occurs in the second half, 33.15% in the last three bins, and 11.37% at
t19 alone.

# Additional Preprocessing Finding

The released `(T-1)` temporal scaling followed by integer truncation leaves the
last input and GT bin empty in every tested sample. At the same time, model
prediction is largest in that final bin. Combined with the separately verified
`spikeTime()` zero penalty for a final-bin error, this is a concrete temporal
supervision mismatch in the public path.

# Interpretation

Original EVIMO with a 20 ms physical window clearly improves spatial inference
over 10 ms. This confirms that EVIMO2's earlier 10 ms adaptation made the depth
latency problem substantially worse.

The original EVIMO domain still shows exact one-bin-per-layer startup delay,
monotonically growing late output, empty final-bin GT, and low 5D event IoU.
Therefore the temporal problem cannot be attributed only to EVIMO2 adaptation.
It is also present in the public SpikeMS temporal/preprocessing definition.

# Gates

```text
EVIMO_REFERENCE_INFERENCE = PASS
TEMPORAL_DELAY_10MS = YES
TEMPORAL_DELAY_OFFICIAL_WINDOW = YES
```

# Next Recommended Step

Use the original EVIMO 20 ms setup for one forward-only definition test of:

```text
earlier context / warm-up
+
explicit valid prediction region
+
loss tail support beyond the final supervised timestep
```

First verify that GT and prediction refer to the same physical time after the
six-layer delay. Do not train, alter the network, or return to 8-sample training
until this alignment is explicit.

# Files

- `reference/spikems/evimo_reference.py`
- `scripts/validate_spikems_evimo_reference.py`
- `reference/spikems/EVIMO_REFERENCE_INFERENCE_20260815.md`
- Ignored output:
  `outputs/spikems_training/evimo_reference_inference/`

SpikeMS submodule: clean.

EVIMO2: untouched.

STOP
