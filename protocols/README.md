# Protocol governance

The canonical semantic interfaces are frozen at version 1. Wire protocols are versioned independently. Breaking field meaning, units, enum values, byte order, or safety behavior requires a new major version plus golden-vector tests. Transport adapters may add framing but may not silently reinterpret fields.

These protocols convey advisory reflex estimates to a Safety MCU. They do not authorize direct motor PWM or replace independent watchdogs, braking limits, or hazard analysis.
