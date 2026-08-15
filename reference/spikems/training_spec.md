# SpikeMS training-definition audit (T0)

Date: 2026-08-15

Scope: official SpikeMS paper, arXiv source, repository at
`c449c83313423d62a23d92df32dd8d3180680a36`, adjacent checkpoint config, model,
modified SLAYER loss, EV-IMO preprocessing/dataloader, and official checkpoint.

This document is an audit result, not an executable training specification.

```text
SPIKEMS_TRAINING_SPEC_GATE=FAIL
```

The official release does not contain enough information to reproduce the loss used to create the
checkpoint faithfully. Per the project gate, no EVIMO2 backward test or overfit run may start.

## Source hierarchy

1. Published [SpikeMS paper](https://arxiv.org/abs/2105.06562) and its arXiv source.
2. Pinned [official SpikeMS repository](https://github.com/prgumd/SpikeMS/tree/c449c83313423d62a23d92df32dd8d3180680a36).
3. `pretrainedModels/EVIMO-pretrained/config/test_config.yaml`, which is a test config adjacent to
   the checkpoint. The checkpoint itself contains no config.
4. Checkpoint optimizer metadata and state dict.

The repository's complete reachable history has only `04455a5` and `c449c83`. Neither revision
contains a training entry point or composite-loss implementation.

## Training-definition comparison

| Field | Paper | Code | Checkpoint / adjacent config |
|---|---|---|---|
| physical window | 10 ms for training | Preprocessor default is mask time +/-10 ms, hence 20 ms per `k=1` sample | UNRESOLVED; physical timestamps are absent |
| number of steps | 10, inferred from 10 ms / 1 ms | `tSample=100` | `tSample=100` in adjacent test config |
| `Ts` | 1 ms physical simulation step | `Ts=1`; unit comment says seconds in one config, but neuron constants and paper support simulation units rather than literal seconds | `Ts=1` in adjacent test config |
| spatial size | EV-IMO uses DAVIS 346, but training crop/size is not stated | full EV-IMO config is H=260, W=346; optional GT-assisted 128x128 crop in `general_config.yaml` | H=260, W=346 in adjacent test config; exact checkpoint training crop is UNRESOLVED |
| crop | UNRESOLVED | optional, centered on the densest GT-foreground pixel | UNRESOLVED |
| loss | `L_bce + lambda * L_spike` | only standalone SLAYER `spikeTime` is executable; no BCE/composite training loss | UNRESOLVED; checkpoint stores only final scalar loss `-0.2687189` |
| loss time steps | UNRESOLVED | `spikeTime` uses all supplied T; `tStartLoss=50` is parsed but never consumed | `tStartLoss=50` in adjacent test config, with no evidence it was applied |
| optimizer | UNRESOLVED in published paper | no training code | Adam state: LR 1e-4, betas (0.9, 0.999), eps 1e-8, weight decay 0, AMSGrad true; six parameter tensors; optimizer step 18,824 |
| batch size | UNRESOLVED in published paper | `batchsize=8` in general test config | `batchsize=8` in adjacent test config; checkpoint training batch size is UNRESOLVED |

The arXiv source contains a commented-out draft sentence mentioning batch size 16, Adam and
`10e-4`, 25 epochs, and 2 ms per timestep. It is not present in the published paper, conflicts with
the checkpoint/config (epoch 100, LR 1e-4, 1 ms paper timestep), and is not treated as a reference
training setting.

## Input definition

### Human-readable

Each event becomes a binary spike at its polarity, image location, and discretized time. Events that
land in the same `(polarity, y, x, time)` voxel collapse to one. The network receives both polarity
channels and predicts two foreground-event polarity channels.

### Exact released code

```text
shape: [B, 2, H, W, T]
T: genconfigs['simulation']['tSample']
time index: (T - 1) * (event_time - start_time) / (stop_time - start_time)
value: binary 1 at [polarity, y, x, time index]
```

The upstream EV-IMO path uses stored polarity values directly as channel indices. It does not name
which channel is ON/OFF. For the selected EVIMO2 right camera, the already validated adapter maps
channel 0 to negative/OFF and channel 1 to positive/ON.

The paper's physical training definition is 10 ms total at a 1 ms simulation step. The released
dataloader instead rescales each selected physical interval to `T=100`, while the supplied EV-IMO
preprocessor constructs a 20 ms interval around each mask by default. Therefore a unique physical
duration per checkpoint time bin cannot be recovered.

## GT foreground definition

### Human-readable

GT foreground events are input events whose pixel falls inside the moving-object mask. They retain
the input event's polarity and time bin. Background events are removed from this target tensor.

### Exact released code

```text
full_mask = bool(depth_mask)
full_mask = dilate(full_mask, 5x5 kernel, one iteration)
gt_foreground_events = (input_spikes AND tiled_full_mask).float()
shape = [B, 2, H, W, T]
```

The target is binary and is a subset of the binary input spike tensor. The released EV-IMO loader
applies one mask across temporal bins for `k=1`. For `k>1`, its loop repeatedly loads the mask id
from the first timeframe, so reliable multi-mask supervision semantics cannot be claimed.

## Loss audit

### Human-readable

The paper describes two terms:

- Spatial/classification term: compare a time-collapsed prediction with foreground/background
  labels.
- Temporal spike term: compare predicted and GT foreground spike trains so event timing matters.

### Published formula

```text
L_total = L_bce + lambda * L_spike
```

The paper says `L_bce` uses temporal spike projections and says `L_spike` derives from a
Van-Rossum-style spike-train distance. GT spikes are produced by masking the input event cloud.

### Executable code that is actually present

`slayerpytorch/src/loss.py::spikeTime(spikeOut, spikeDesired)` computes:

```text
error = PSP(spikeOut - spikeDesired)
L = 0.5 * sum(error ** 2) * Ts
```

Input and target shapes are `[B, C, H, W, T]`; reduction is a sum over every element; all supplied
time steps participate. This is a faithful executable implementation of a standalone temporal spike
loss term.

### Blocking unknowns

- `lambda` is never specified.
- There is no BCE or composite-loss function in the released repository.
- The paper calls output channels positive/negative polarity, while its BCE notation needs
  foreground/background predictions. The mapping is not specified.
- Projection normalization, probability conversion/clamping, numerical epsilon, tensor reduction,
  and batch reduction are not specified.
- `tStartLoss=50` appears in configs but is never referenced by the model, runner, or loss code.
- The checkpoint's saved loss is negative (`-0.2687189`), which is incompatible with directly
  identifying it as the non-negative released `spikeTime` loss. It does not reveal the missing
  implementation.

Choosing BCE details or `lambda` would create a new loss and violate the faithful-baseline rule.

## SNN state semantics

The network consumes the complete T dimension in one forward call. SLAYER applies causal temporal
PSP/refractory convolutions within that tensor, so activity accumulates across time bins inside a
sample.

The model stores no mutable membrane/refractory state between forwards. Intermediate membrane and
spike tensors are local to `forward`. Consequently:

- within one sample: temporal dynamics accumulate over T;
- between samples/forward calls: state starts fresh automatically;
- within a batch: each batch element is independent;
- no external `reset_state()` call exists or is required.

## GT-assisted conditions

### Released reference code

- Optional crop: 128x128 in `general_config.yaml`, centered on the spatial location with the largest
  summed GT foreground-event count. This is GT-assisted.
- Mask preprocessing: 5x5 dilation before foreground-event construction.
- Sample filter: reject when `background_spikes / foreground_spikes > maxBackgroundRatio`.
- Mask-area filter: reject when the number of foreground mask pixels is below `minEvents`; despite
  its name, this counts mask pixels rather than event voxels.
- README testing examples recommend `--crop` and a `maxBackgroundRatio` around 1.5 to 3.

The released training command is absent, so whether and with which values these conditions produced
the official checkpoint is UNRESOLVED.

### Honest EVIMO2 setting

No setting is frozen because T0 failed. Existing adapter behavior remains unchanged: right camera
only, full frame, no GT-assisted crop, no foreground/background-ratio filtering, and explicit
top-left GT alignment to the model's 479x639 output support.

## Gate decision and required next evidence

```text
SPIKEMS_TRAINING_SPEC_GATE=FAIL
SPIKEMS_BACKWARD_GATE=NOT_RUN
SPIKEMS_OVERFIT8_GATE=NOT_RUN
```

Any one of the following could reopen T0:

1. the authors' original training script/config containing BCE, `lambda`, reductions and temporal
   slicing;
2. an authoritative supplementary artifact specifying those values exactly;
3. an explicit decision to run a clearly labeled non-faithful reconstruction loss. That would be a
   new research task, not the official-reference gate requested here.

Per the requested failure rules, the training config, optimizer wrapper, backward script, 8-sample
manifest, and overfit script were not created.
