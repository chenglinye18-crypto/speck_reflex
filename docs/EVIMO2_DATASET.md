# EVIMO2v2 dataset contract

## Status and scope

EVIMO2v2 is the first measured event-camera dataset selected for the motion
backbone. The raw dataset is external to this repository and remains under its
upstream license. This repository currently implements only the first adapter
stage:

```text
official event stream
→ canonical OFF/ON event-count tensor
→ camera-local window ego-motion label
```

No EVIMO2 task model has been trained yet. Pixelwise independent-motion labels,
depth training, object-motion training, and robot-base motion are not yet
implemented.

Official sources:

- Dataset: <https://better-flow.github.io/evimo/download_evimo_2.html>
- Ground-truth format: <https://better-flow.github.io/evimo/docs/ground-truth-format.html>
- Dataset paper: <https://arxiv.org/abs/2205.03467>

The upstream dataset is licensed CC BY-SA 4.0. Do not commit or redistribute
the downloaded archives through this repository without satisfying its
attribution and share-alike requirements.

## Local data selection

The development machine keeps the selected data outside the Git checkout:

```text
~/datasets/evimo2/
```

The selected EVIMO2v2 NPZ subsets are:

| Category | Samsung | Prophesee left | Prophesee right | Intended role |
|---|---:|---:|---:|---|
| Motion Segmentation / Object Recognition | yes | yes | yes | Later independent-motion training |
| Motion Segmentation in Low Light | yes | yes | yes | Later noise robustness |
| Structure from Motion / Object Recognition | yes | no | no | Primary ego-motion training |
| Structure from Motion in Low Light | yes | yes | yes | Low-light ego-motion hard cases |

RGB/Flea3, ROS bags, TXT, Rudimentary Motion, and unrequested EVIMO2 subsets
are not part of the current selection.

Raw `.npy`, `.npz`, and archive files must remain outside Git. The repository
contains only adapter code, tests made from tiny generated fixtures, and this
data contract.

## Primary input

### Raw representation

One sequence stores the asynchronous event stream in three index-aligned
arrays:

| File | Shape | Meaning |
|---|---|---|
| `dataset_events_t.npy` | `[N]` | Event timestamps in seconds |
| `dataset_events_xy.npy` | `[N,2]` | `(x,y)` pixel coordinates |
| `dataset_events_p.npy` | `[N]` | Raw sensor polarity, `0` or `1` |

The event at index `i` is:

```python
(x, y, timestamp, raw_polarity) = (
    events_xy[i, 0],
    events_xy[i, 1],
    events_t[i],
    events_p[i],
)
```

Both raw polarity values describe real events. `p=0` does not mean that no
event occurred. Absence of an event is represented by absence of a record.

### Canonical polarity

The project freezes the model-input channel semantics as:

```text
channel 0 = OFF = brightness decrease
channel 1 = ON  = brightness increase
```

EVIMO2 documents Samsung event polarity as inverted relative to Prophesee.
The adapter therefore applies:

```python
Prophesee: canonical_p = raw_p
Samsung:   canonical_p = 1 - raw_p
```

This conversion happens only while reading. Original event files are never
rewritten.

### Temporal binning

The raw stream has no frame rate. The project chooses a simulation timestep:

```text
dt = 1 ms
T = 32
window duration = 32 ms
```

For each raw event:

```python
t_bin = floor((timestamp - window_start) / dt)
events[t_bin, canonical_p, y, x] += 1
```

One adapter sample therefore returns:

```text
[T,P,H,W] = [32,2,480,640]
dtype      = torch.float32
value      = event count in one (time, polarity, y, x) cell
```

A PyTorch `DataLoader` adds the batch dimension:

```text
[B,T,P,H,W]
```

`dt` is a project/model choice. The `discretization=0.01` value in
`dataset_info.npz` is only an upstream event lookup interval and is not the SNN
timestep.

## Primary current output: camera ego-motion label

The adapter uses the camera trajectory in:

```text
dataset_info.npz
└── meta
    └── full_trajectory
        └── cam
```

It interpolates translation linearly, interpolates orientation with quaternion
SLERP, and computes:

```text
log(inv(T_wc_start) @ T_wc_end) / window_duration
```

The resulting window label is:

```text
ego_motion.shape = [6]
ego_motion       = [vx, vy, vz, wx, wy, wz]

vx,vy,vz: metres per second
wx,wy,wz: radians per second
axes:      camera coordinate system at the start of the window
```

This is camera-rig ego-motion, not robot chassis odometry. A future calibrated
camera-to-robot transform will be required before interpreting it as robot-base
motion.

The current adapter sample contract is:

```python
EVIMO2EgoMotionSample(
    events,          # [T,2,H,W], canonical OFF/ON event counts
    ego_motion,      # [6], camera-local average twist
    start_time_s,
    end_time_s,
    sensor,
    sequence,
    event_count,
    frame_id,
)
```

## Other official outputs and labels

| File | Content | Current use |
|---|---|---|
| `dataset_mask.npz` | `[H,W]` object-instance IDs multiplied by 1000 | Audited; not yet used for training |
| `dataset_depth.npz` | `[H,W]` depth in millimetres, with invalid zeros | Audited; not yet used for training |
| `dataset_info.npz` | Intrinsics, distortion, lookup index, timestamps, camera/object poses, IMU | Camera ego-motion label and geometry metadata |
| `dataset_classical.npz` | Conventional images where present | Empty for the selected DVS-only sequences; unused |

`dataset_mask.npz` identifies which object owns a pixel. It does not directly
state whether that object is moving independently. The incorrect shortcut

```python
independent_motion = mask > 0
```

must not be used.

A later audited label builder will combine instance masks with camera and object
poses:

```text
instance ID mask
+ camera pose T_wc(t)
+ object pose T_co(t)
→ object world pose T_wo(t) = T_wc(t) @ T_co(t)
→ independent-motion state
→ pixelwise independent-motion target
```

That label builder is planned, not implemented.

## Model-side outputs

The EVIMO2 adapter produces input and supervision; it does not produce model
predictions. The current SNN Motion Backbone returns:

```text
primitive_spikes: [B,T,16,H/4,W/4]
local_logits:     [B,T, 2,H/4,W/4]
global_embedding: [B,T,64]
ego_motion:       [B,T, 6] or None
```

The frozen first-stage supervision mapping is:

```text
adapter ego_motion [B,6]
→ supervise the temporal mean of the shared global readout [B,6]

future independent-motion mask
→ supervise local_logits
```

The complete 64 ms Samsung Structure-from-Motion baseline contract is frozen
in [EVIMO2_EGO_MOTION_BASELINE.md](EVIMO2_EGO_MOTION_BASELINE.md). Its training
pipeline is implemented and has passed a controlled two-window smoke run. A
full training run has not started and no accuracy claim is made.

## Verified example

The official Samsung sequence
`scene13_dyn_test_00_000000`, frame `55`, produces:

```text
window:      [0.901333, 0.933333) seconds
events:      71,669
input shape: [32,2,480,640]
input sum:   71,669

ego_motion:
[-0.014208, -0.015273, -0.002554,
  0.005683, -0.088631, -0.099412]
```

These are measured event data and a pose-derived label. They are not a trained
model prediction.

## Code and validation

Adapter:

```text
software/datasets/evimo2.py
```

Read-only smoke command:

```bash
python -m software.datasets.inspect_evimo2 \
  ~/datasets/evimo2/motion_segmentation/samsung_mono/imo/eval/scene13_dyn_test_00_000000 \
  --sensor samsung_mono \
  --frame-id 55 \
  --timesteps 32 \
  --dt-ms 1.0
```

Unit tests cover event-count binning, right-open time windows, sensor polarity,
trajectory interpolation, translational and rotational labels, determinism,
read-only behavior, and invalid trajectory ranges.

## Next data work

Before training:

1. Build deterministic sequence-level train/validation/test indices.
2. Audit event density and ego-motion target distributions across SFM splits.
3. Implement the frozen coordinate, normalization, state, and temporal
   supervision policies in
   [EVIMO2_EGO_MOTION_BASELINE.md](EVIMO2_EGO_MOTION_BASELINE.md).
4. Train and report camera ego-motion without calling it robot-base odometry.
5. Only then derive and audit independent-object-motion labels.
