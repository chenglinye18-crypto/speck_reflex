# SynSense Speck backend — baseline only

Speck is one optional neuromorphic backend, not the platform definition. The repository currently has a no-hardware, host-injected N-MNIST software/configuration baseline. It has not validated a physical board or the on-chip 128×128 DVS path. A future adapter must implement `ReflexHardware` without leaking Samna/Sinabs types into the model interface.
