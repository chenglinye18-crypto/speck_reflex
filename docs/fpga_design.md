# FPGA SNN accelerator design plan

Status: design only. No RTL, HLS kernel, bitstream, timing result, or hardware claim exists.

## Planned data path

```text
event ingress
  -> event scheduler
  -> spike router
  -> synapse-memory lookup
  -> accumulator
  -> LIF neuron state update
  -> threshold/reset
  -> output spike / risk readout
```

Planned building blocks:

- event scheduler with timestamp ordering and bounded queues;
- spike router with explicit multicast/backpressure/drop counters;
- parameterized LIF neuron state and reset behavior;
- synapse memory with documented layout, width, sparsity, and update rules;
- signed accumulator with specified saturation/rounding;
- threshold/readout path compatible with the reflex output interface.

## Verification before implementation

Freeze fixed-point formats, clock/reset domains, throughput target, maximum dimensions, memory budget, overflow policy, and host/MCU transport. Create software golden vectors for every block, then require lint, unit simulation, randomized equivalence, CDC/reset checks, synthesis timing, resource usage, and measured board power. HLS and RTL must use identical protocol vectors.

The first FPGA milestone is not a complete robot model; it is a small, verifiable event-to-spike datapath whose output matches a software reference.
