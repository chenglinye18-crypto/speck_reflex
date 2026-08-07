# Robot Neuromorphic Reflex Arc Co-design Platform

An independent research platform for co-designing low-latency, low-power robot reflex paths across software models, event sensors, embedded processors, FPGA accelerators, neuromorphic chips, communication protocols, and a deterministic Safety MCU.

This is not an SNN-only project, a Speck SDK project, or an official SynSense repository. Speck is one evaluated backend candidate. The long-term platform may support PyTorch simulation, STM32N6 prototypes, FPGA accelerators, SynSense Speck, and other neuromorphic hardware.

## Why robots need a digital reflex arc

A robot's main cognition stack—Jetson, ROS 2, semantic vision, planning, and large learned models—provides rich understanding but can have variable latency, high power demand, and broad failure modes. A reflex path should detect sudden intrusion, rapid approach, collision risk, and abnormal motion through a small, event-driven path that remains available when the cognition computer is busy or offline.

The reflex path does not replace cognition or certified safety logic. It produces bounded advisory risk evidence for a separate Safety MCU, which performs final deterministic arbitration.

## System architecture

```text
Main cognition path
RGB / depth / other sensors
        |
Jetson + ROS 2 + semantic models + planning
        |
        +------------------------------+
                                       v
Event Camera / DVS              Safety MCU              Motor Controller
        |                    final deterministic  ----> braking / speed limits
Event Processing                   arbitration
        |                              ^
Neuromorphic Processor               |
SNN / FPGA / future ASIC             |
        |                              |
Risk Estimation ----------------------+

Fast reflex path: low latency, low power, independent of main cognition.
```

Jetson owns semantic recognition, identity/attributes, confirmed fall understanding, scene reasoning, and planning. The neuromorphic path targets fast motion risk. The Safety MCU combines reflex output with watchdogs, motion state, current/braking limits, and other safety inputs. No model or accelerator directly controls motor PWM.

## Software–hardware co-design

Four versioned boundaries prevent algorithms from being tied to one chip:

1. **Event Input Interface** — canonical `(x, y, timestamp_us, polarity)` events and bounded `EventWindow` objects.
2. **Reflex Model Interface** — any ANN, SNN, or hybrid model maps an event window to risk, TTC, direction, emergency-stop request, and timestamp.
3. **Reflex Output Interface** — a fixed-point, CRC-protected advisory message to the Safety MCU.
4. **Hardware Backend Interface** — lifecycle and data-plane operations shared by simulation adapters, STM32N6, FPGA, Speck, and future targets.

Canonical Python contracts live in `software/` and `hardware/interfaces/`. Wire contracts live in `protocols/`. Vendor types must stop at backend adapters.

See the detailed [platform architecture](docs/architecture.md) and frozen [reflex output protocol](protocols/reflex_protocol.md).

## Current implementation status

| Layer | Status |
|---|---|
| Canonical event/model/backend interfaces | Implemented and unit-tested |
| Protocol v1 semantic and binary layouts | Documented; transport not implemented |
| PyTorch/Sinabs software baseline | Existing official N-MNIST baseline retained |
| Specksim and offline Speck configuration | Existing no-hardware baseline retained |
| STM32N6 firmware/backend | Planned; no driver or hardware connection |
| FPGA accelerator/RTL/HLS | Planned; no implementation |
| Physical Speck/on-chip DVS | Not validated; no board connected |
| Robot collision/TTC model and dataset | Not implemented |
| Safety certification | Not provided |

The retained N-MNIST experiment is a backend/toolchain reference, not a robot reflex model. Its host-injected 34×34 events must not be described as the on-chip DVS path.

## Repository map

```text
software/                hardware-neutral event, model and simulation contracts
  event_processing/      canonical format and adapter/filter interfaces
  models/                ANN / SNN / hybrid model workspaces
  training/ evaluation/  future reproducible pipelines and metrics
hardware/                backend abstraction and hardware-specific plans
  stm32n6/ fpga/ speck/  independent optional backends
  sensors/               DVS and future event-source boundaries
datasets/                manifests only; raw data stays outside Git
protocols/               versioned event, reflex, MCU and ROS 2 contracts
experiments/              collision, TTC, benchmark and retained official baselines
src/speck_reflex/         compatibility package for validated existing baselines
third_party/              pinned, unmodified upstream source
docs/                     architecture, hardware decisions and roadmap
```

## Hardware roadmap

- **STM32N6:** fastest route to embedded timing, communications, watchdog, and MCU integration experiments; useful prototype but not inherently neuromorphic.
- **FPGA:** validates event scheduling, spike routing, LIF neurons, synapse memory, accumulation, and thresholds with measurable cycle/power behavior; longer development cycle.
- **Neuromorphic chips:** intended low-power event-driven direction, including Speck and future devices; availability, SDK constraints, routing, memory, readout, and reproducibility must be evaluated per backend.

See [hardware selection](docs/hardware_selection.md) and the [FPGA design plan](docs/fpga_design.md).
The staged implementation sequence and exit criteria are in the [roadmap](docs/roadmap.md).

## Research benchmark

Backends will receive identical canonical event windows and produce the same reflex schema. Comparisons must report:

- task metrics: missed-risk, false-stop, directional accuracy, TTC error;
- timing: sensor-to-output p50/p95/p99 latency and jitter;
- capacity: sustained event rate, queue depth, event loss, overload recovery;
- efficiency: idle/active power and energy per event/window;
- equivalence: software/quantized/FPGA/neuromorphic output disagreement;
- resilience: cognition-host loss, stale data, corrupted frames, sensor/backend reset.

No benchmark claim is valid unless its dataset provenance, clock boundary, backend version, configuration, seed, and raw results are recorded.

## Existing reproducible baseline

The official Sinabs v3.1.3 N-MNIST work remains available as a compatibility baseline:

```bash
git submodule update --init --recursive
source scripts/activate.sh
make doctor
make test-fast
make demo-smoke    # deterministic random weights; PIPELINE_SMOKE_ONLY
make demo-nmnist   # licensed official NIR checkpoint; functional demo
```

New machines should first run the read-only `bash scripts/bootstrap_wsl_cuda.sh`; `.venv` is never assumed to exist. See the [demo guide](docs/DEMO_GUIDE.md).

## Safety and licensing

This research platform is not certified for collision avoidance, emergency braking, medical fall detection, or any safety-critical function. Reflex predictions are advisory, heartbeat and CRC are necessary but insufficient, and the Safety MCU must fail safely on timeout, corruption, disagreement, or backend loss.

Third-party code retains its own license. The main repository license is still pending owner selection; see [licensing](docs/LICENSING.md) and [third-party notices](THIRD_PARTY_NOTICES.md).
