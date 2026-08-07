# Hardware selection

Hardware is selected per research question, not treated as a single permanent dependency.

| Target | Advantages | Limitations | Best first use |
|---|---|---|---|
| STM32N6 | Fast embedded prototype; available MCU ecosystem; validates communication, watchdog and deterministic integration | Not inherently neuromorphic; performance/power may not represent event-driven silicon | Safety-MCU protocol and bounded embedded inference prototype |
| FPGA | Cycle-level control; hardware neuron/router/memory validation; precise instrumentation | RTL/HLS verification and toolchain effort; device/vendor portability; longer development cycle | Golden-vector accelerator equivalence and latency studies |
| Neuromorphic chip | Event-driven execution and potential low power/latency; closest to final research direction | Hardware availability, SDK maturity, constrained operators/memory, device-specific routing/readout | Validated deployment after software and interface baselines |

## Selection gates

Before adding a backend, record availability, license/toolchain, supported event rate and geometry, numeric/model constraints, memory, timestamp semantics, transport, observability, deterministic reset, power measurement method, recovery path, and Safety MCU integration.

No backend becomes the platform architecture. All must implement the same canonical event, prediction, and hardware contracts, with differences reported rather than hidden.
