# Platform architecture

## Design objective

The platform co-designs an independent, event-driven robot reflex path from sensor timing through model behavior, accelerator implementation, communication, Safety MCU arbitration, and benchmark evidence. Optimization of one layer must not silently change another layer's semantics.

## Layer boundaries

```text
Event source
  -> native sensor adapter
  -> canonical Event stream
  -> deterministic filtering/windowing
  -> ReflexModel or ReflexHardware backend
  -> ReflexPrediction
  -> Reflex Protocol v1
  -> Safety MCU policy and watchdog
  -> motor-controller request
```

- Event adapters own native coordinates, timestamps, wrap, polarity, drops, and calibration.
- Event processing owns deterministic filtering and window formation.
- Models own risk inference but not device transport or motor policy.
- Hardware backends own vendor/toolchain conversion and lifecycle without leaking SDK types upward.
- Protocols own fixed units, enum values, sequence/freshness, and integrity.
- Safety MCU owns the final decision and deterministic fallback.

## Cognition versus reflex

Jetson/ROS handles semantics and planning. The reflex path handles bounded rapid-motion hazards with limited output vocabulary. Either path may inform the MCU, but neither bypasses it. Loss of Jetson must not stop the reflex path; loss of the reflex path must be visible and drive an MCU-defined safe policy.

## Compatibility strategy

The current `src/speck_reflex/official_baseline` package remains a validated Speck/Sinabs compatibility island. New algorithms target `software.models.ReflexModel`. A future Speck adapter will translate canonical events/predictions at the backend boundary after physical hardware validation; no adapter is claimed today.

## Interface change policy

Semantic interfaces use major versions. Breaking coordinate, time-unit, polarity, direction, risk, TTC, CRC, or lifecycle semantics requires a new major version, migration notes, and golden vectors. Backend-specific optional metadata may be extended without changing canonical fields.
