# Repository instructions

- Treat the pinned Sinabs/Samna source and installed versions as authoritative.
- Never modify files under `third_party/`.
- Do not install the obsolete `sinabs-dynapcnn` package or change the Python environment unless explicitly requested.
- Never bind USB, discover/open hardware automatically, write Flash, or directly control motors.
- Hardware operations require explicit approval and `SPECK_ALLOW_HARDWARE_TESTS=1`.
- STM32 is the final deterministic safety arbiter.
- Mark compatibility changes with `# COMPATIBILITY_PATCH:` and record experimental seed and environment.
- Never describe random-weight smoke tests as functional models, host-injected events as on-chip DVS, or planned robot capabilities as implemented.
- Run relevant tests after changes. Do not commit data, virtual environments, large logs, Flash binaries, or unlicensed weights.
