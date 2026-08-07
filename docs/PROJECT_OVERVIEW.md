# Project overview

The repository is a robot neuromorphic reflex arc co-design platform, not an SNN-only or vendor-SDK project. Its primary artifacts are stable cross-layer contracts, reproducible benchmarks, and evidence connecting algorithms to timing, power, hardware limits, communication, and Safety MCU behavior.

| Layer | Current state | Next evidence |
|---|---|---|
| Event/model software contracts | Implemented and unit-tested | Real robot event-window adapters and reference risk models |
| Simulation and official baseline | Sinabs N-MNIST software/configuration/Specksim retained | Adapt a robot task without coupling it to Speck APIs |
| Communication protocols | v1 event/reflex semantics and risk frame documented | Golden binary vectors, fuzz/error tests and transport selection |
| STM32N6 | Architecture placeholder only | Board/toolchain selection and deterministic protocol prototype |
| FPGA | Design plan only | Fixed-point specification and block-level golden vectors |
| Neuromorphic hardware | Speck no-hardware baseline only | Physical sensor/routing/readout/power evidence |
| Robot system | Requirements only | Measured dataset, controlled rig and MCU-safe integration |

Terms:

- **Event:** canonical `x, y, timestamp_us, polarity` record.
- **Event window:** ordered events plus explicit sensor geometry and time boundary.
- **Reflex prediction:** backend-neutral risk, TTC, direction, emergency-stop request, and timestamp.
- **Backend:** simulation or hardware implementation behind a common lifecycle/data-plane contract.
- **Safety MCU:** independent final deterministic arbiter; model outputs remain advisory.
- **Host-injected baseline:** host supplies recorded events; it is not an internal sensor path.
