# Robot Neuromorphic Reflex Arc Co-design Platform instructions

- Treat the pinned Sinabs/Samna source and installed versions as authoritative.
- Treat Speck as one optional backend; do not couple canonical model/event interfaces to vendor SDK types.
- Preserve the frozen Event, Reflex Model, Reflex Output, and Hardware Backend field semantics. Breaking changes require a protocol major version and updated golden tests.
- Never modify files under `third_party/`.
- Do not install the obsolete `sinabs-dynapcnn` package or change the Python environment unless explicitly requested.
- Never bind USB, discover/open hardware automatically, write Flash, or directly control motors.
- Hardware operations require explicit approval and `SPECK_ALLOW_HARDWARE_TESTS=1`.
- STM32 is the final deterministic safety arbiter.
- Mark compatibility changes with `# COMPATIBILITY_PATCH:` and record experimental seed and environment.
- Never describe random-weight smoke tests as functional models, host-injected events as on-chip DVS, or planned robot capabilities as implemented.
- Do not add fake STM32 drivers, placeholder FPGA logic presented as synthesizable, or synthetic data presented as measured data.
- The fast reflex output is advisory; the Safety MCU remains the final deterministic arbiter and no backend directly controls PWM.
- Run relevant tests after changes. Do not commit data, virtual environments, large logs, Flash binaries, or unlicensed weights.
