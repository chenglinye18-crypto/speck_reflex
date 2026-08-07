# Roadmap

## Phase 0 — retained software baseline (complete)

Pinned environment, official N-MNIST model, Sinabs software inference, DynapCNN mapping, offline Samna configuration, and Specksim are reproducible without hardware. This is toolchain evidence, not robot capability.

## Phase 1 — platform contracts (current)

Freeze event, model, hardware-backend, reflex-output, and MCU communication semantics. Establish datasets, backend directories, benchmarks, tests, and architecture governance without fake implementations.

Exit: interface unit tests pass; documents agree on units/enums/safety ownership; existing baseline remains green.

## Phase 2 — dataset and simulation reference

Define robot scenes and consent/privacy rules; collect measured DVS data only after sensor validation. Implement simple ANN/SNN/hybrid reference models behind `ReflexModel`; publish actual metrics and failure cases.

## Phase 3 — STM32N6 integration prototype

Validate reflex framing, CRC, heartbeat, freshness, watchdog, bounded queues, timeout and deterministic MCU fallback. Any embedded inference remains a prototype, not a neuromorphic claim.

## Phase 4 — FPGA event accelerator

Implement and verify scheduler, router, synapse memory, LIF update, accumulator and threshold incrementally against golden vectors. Measure timing, resources, event loss and power.

## Phase 5 — physical neuromorphic backend

Validate Speck or another available chip: discovery, host events, internal DVS, routing, readout, power and reset. Add a `ReflexHardware` adapter only after evidence exists.

## Phase 6 — robot closed-loop research

Integrate cognition, independent reflex, Safety MCU and motor controller on a controlled rig. Evaluate end-to-end latency, power, missed risks, false stops, overload and recovery before any field use.

## Phase 7 — assurance work (future, separate)

Hazard analysis, requirements traceability, fault injection, deterministic timing evidence, electrical safety, braking envelopes and relevant standards require a separate qualified process. This research repository alone cannot provide certification.
