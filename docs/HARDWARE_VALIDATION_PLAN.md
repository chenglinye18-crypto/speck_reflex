# Hardware validation plan (not executed)

No board was connected in this phase. Every hardware step requires explicit approval and `SPECK_ALLOW_HARDWARE_TESTS=1` where applicable.

| Step | Action after board arrival | Acceptance criterion | Rollback/failure response |
|---|---|---|---|
| 1 | Attach only the Speck USB device with Windows `usbipd` under WSL2 | Device visible in WSL with expected VID/PID | Detach in `usbipd`; do not broaden USB forwarding |
| 2 | Install a least-privilege udev rule if needed | Non-root read/write access to only the board | Remove rule and reload udev |
| 3 | Run explicit hardware doctor/device-map query | Current Samna reports the exact device key | Stop; do not guess a legacy device name |
| 4 | Send the host-injected N-MNIST baseline to the board | Same config maps; readout has bounded, repeatable output | Return to offline config and simulator evidence |
| 5 | Run the official internal-DVS visualization | 128×128 sensor events are visible and correctly oriented | Disable routing; retain raw diagnostic log |
| 6 | Configure DVS filter/router to DynapCNN cores | Controlled stimuli reach the intended first core | Restore the last known configuration |
| 7 | Validate readout/GPIO or structured output | Class/risk output matches documented channel mapping | Disable external output; inspect routing |
| 8 | Measure power only after stable inference | Idle/active measurements repeat with stated conditions | Remove measurement path and repeat software checks |
| 9 | Generate and review an offline boot binary | Binary generated for the confirmed board/io selection | Delete binary; never write without separate approval |
| 10 | Test standalone Flash boot under a dedicated procedure | Cold boot is deterministic and recoverable | Use vendor recovery procedure; this repository has no write API |
| 11 | Integrate an STM32 message transport | Sequence, heartbeat, checksum, timeout and fail-safe pass | STM32 ignores input and enters stop/limited safe state |

The future **on-chip DVS baseline** is:

```text
Speck 2f internal 128x128 DVS -> event filter/router -> DynapCNN cores
  -> readout -> GPIO/interrupt or structured output
```

It is distinct from the current 34×34 host-injected N-MNIST path and has not been simulated or claimed as passed. Flash writing, device opening, visualizer execution, recording, and power monitoring remain outside this no-hardware phase.
