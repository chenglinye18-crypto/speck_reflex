# Research Question

Can increasing only `lambda_time` preserve the existing membrane-BCE spatial
localization while improving 5D foreground-event IoU?

# Frozen Experiment Definition

- Loss: `L_total = L_space + lambda_time * L_time`
- L_space: unchanged `MEMBRANE_SPATIAL_BCE_V2`
- Positive weight: unchanged, `21.21625344352617`
- L_time: unchanged official `spikeLoss.spikeTime()`
- Model and initialization: official SpikeMS and official EV-IMO checkpoint
- Optimizer: Adam, lr `1e-4`, betas `(0.9,0.999)`, weight decay 0, AMSGrad
- Sample: `scene14_dyn_test_02_000000`, frame 57
- Window: 0.961667--0.971667 s, 10 bins at 1 ms/bin
- Crop: x0=248, y0=0, 128x128, `DIAGNOSTIC_ONLY_GT_ASSISTED_CROP`
- Raw / foreground / background events: 1057 / 913 / 144
- Seed: 11

No BCE, positive weight, model, neuron, optimizer, learning rate, sample, or crop
was changed.

# 100-Step Lambda Screen

The 1x result is reused from commit `43f1660`; 2x, 4x, and 8x are new runs,
each independently initialized from the official checkpoint.

| Multiplier | lambda_time | Spatial F1 | Spatial IoU | Event IoU | Event recall | Leakage | Pred spikes | Eligible |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1x | 0.00513924 | 0.345896 | 0.209114 | 0.017659 | 0.060538 | 0.038739 | 2220 | No |
| 2x | 0.01027848 | 0.300874 | 0.177076 | 0.014155 | 0.038117 | 0.049223 | 1544 | No |
| 4x | 0.02055696 | 0.251142 | 0.143603 | 0.012612 | 0.026906 | 0.056039 | 1035 | No |
| 8x | 0.04111392 | 0.184154 | 0.101415 | 0.009780 | 0.013453 | 0.028818 | 347 | No |

Eligibility requires `Spatial F1 >= 0.35`. None of the four candidates passes.
The reused 1x result is closest at 0.345896. Increasing temporal weight reduces
both spatial quality and 5D event correspondence in this 100-step screen.

# Timestep Spike Counts

Counts use the model's aligned `[2,127,127,10]` output support. The aligned GT
contains 892 spikes; the crop's raw foreground count is 913 because the released
network removes the last spatial row and column.

| Timestep | GT | Prediction 2x | Prediction 4x | Prediction 8x |
|---:|---:|---:|---:|---:|
| 0 | 85 | 0 | 0 | 0 |
| 1 | 103 | 0 | 0 | 0 |
| 2 | 106 | 0 | 0 | 0 |
| 3 | 114 | 0 | 0 | 0 |
| 4 | 78 | 0 | 0 | 0 |
| 5 | 77 | 0 | 0 | 0 |
| 6 | 89 | 0 | 0 | 0 |
| 7 | 85 | 11 | 1 | 0 |
| 8 | 83 | 219 | 120 | 35 |
| 9 | 72 | 1314 | 914 | 312 |

GT activity is distributed across all ten timesteps. Predictions concentrate at
the end of the window, especially timestep 9. Raising lambda suppresses the
number of predicted spikes but does not spread them toward the GT distribution.

# Selection and 500-Step Run

Selection rule:

1. Spatial F1 must be at least 0.35.
2. Among eligible candidates, select the highest 5D event IoU.

`SELECTED_LAMBDA_TIME=NONE`

`STOP_REASON=NO_100_STEP_CANDIDATE_MEETS_SPATIAL_F1_FLOOR`

The 500-step run was not started because no candidate satisfied rule 1. Running
one would require relaxing the user's explicit selection criterion.

# Main Decision

`LAMBDA_TIME_BALANCING_GATE=FAIL`

At 8x, event IoU falls to 0.009780 rather than improving beyond the current
Combined 500-step reference of 0.024100 or the L_spike-only reference of
0.030499. The screen also exposes a temporal concentration problem: predicted
events appear almost entirely in the last bins. Static scaling of `L_time` in
the tested 1x--8x range does not solve the combined-loss problem.

# Files

- `scripts/diagnose_spikems_membrane_combined_loss.py`
- `scripts/diagnose_spikems_lambda_time.py`
- `reference/spikems/LAMBDA_TIME_SCREEN_DIAGNOSTIC_20260815.md`
- Ignored result: `outputs/spikems_training/lambda_time_screen/result.json`

SpikeMS submodule: clean.

EVIMO2 raw data: untouched.

# Next Recommended Step

Stop lambda scanning. Diagnose why official `spikeTime()` training produces
late-window spike concentration before changing architecture or starting an
8-sample experiment.

STOP
