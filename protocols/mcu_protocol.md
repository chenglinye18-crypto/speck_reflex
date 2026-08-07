# Hardware Communication Interface

This document defines transport-independent MCU behavior; UART, SPI, CAN-FD, or Ethernet is not selected yet.

Required channel properties:

- version negotiation and explicit incompatibility failure;
- sequence numbers, freshness windows, CRC, timeout, and duplicate handling;
- startup state is non-operational until health and protocol checks pass;
- periodic heartbeat independent of risk predictions;
- explicit overflow, model fault, sensor fault, and backend-reset status;
- bounded queues with visible drop counters—never silent loss;
- MCU watchdog and motor-safe policy remain independent of the reflex backend;
- shutdown or link loss cannot leave a stale permissive command active.

The MCU consumes `Reflex Output Protocol v1` as advisory evidence, combines it with its own sensors, watchdogs, current limits, motion state, and braking constraints, then issues deterministic commands to the motor controller. No neuromorphic backend directly controls PWM.
