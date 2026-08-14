# Training

The first measured-data entry point implements the frozen Samsung camera
ego-motion baseline in
[`docs/EVIMO2_EGO_MOTION_BASELINE.md`](../../docs/EVIMO2_EGO_MOTION_BASELINE.md).

Start or automatically resume the configured run with:

```bash
bash scripts/train_evimo2_ego_motion.sh
```

Run only the measured-input activity gate:

```bash
bash scripts/train_evimo2_ego_motion.sh --preflight-only
```

Run a controlled training smoke test without claiming accuracy:

```bash
bash scripts/train_evimo2_ego_motion.sh \
  --epochs 1 \
  --audit-windows 2 \
  --max-train-windows 2 \
  --max-validation-windows 1 \
  --max-test-windows 1 \
  --output-dir /tmp/speck-reflex-evimo2-smoke
```

The default run writes ignored artifacts under
`outputs/evimo2_ego_motion_v0.1/`. `last.pt` is resumed automatically;
`best.pt` is selected by normalized validation SmoothL1. Raw EVIMO2 data stays
outside Git and no hardware API is used.

The output is camera-local SE(3) twist. It is not robot-base odometry, and the
local independent-motion head is not trained in this phase. Existing official
N-MNIST training remains in its historical experiment directory.

Resume is automatic: rerunning the command loads `last.pt` and continues from
the next epoch. Epoch-addressed deterministic sampling prevents a resumed run
from repeating the previous epoch's shuffle order. Runtime throughput options
can be overridden without editing the frozen data/model contract:

```bash
bash scripts/train_evimo2_ego_motion.sh \
  --batch-size 4 \
  --gradient-accumulation-steps 1 \
  --num-workers 2
```

The default `4 x 1` setup preserves the original effective batch size of four
from the slower `1 x 4` gradient-accumulation setup. A 64-window controlled
benchmark took 14.60 seconds versus 45.83 seconds (3.14x faster), with training
losses 0.608553 and 0.608542 respectively. These are throughput checks, not
task-performance results.

`last.pt` is written atomically after each completed epoch. If training is
interrupted during an epoch, rerunning the one-click command resumes from the
last completed epoch; only the unfinished epoch is repeated. Mid-epoch batch
position recovery is intentionally not claimed.
