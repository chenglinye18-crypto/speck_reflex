# Project overview

This independent project has four deliberately separated layers:

| Layer | Current state | Depends on |
|---|---|---|
| Software model | Official random smoke and official-weight N-MNIST functional demo complete | Tonic, Torch, Sinabs |
| Offline deployment/simulation | DynapCNN config and Specksim complete | Sinabs backend, Samna |
| Physical hardware | Not started | Speck 2f, approved USB access, routing/readout validation |
| Robot reflex | Requirements/types only; no model or transport | Hardware evidence, robot data, STM32 contract |

“Host-injected” means recorded 34×34 N-MNIST events enter the deployment/simulator path from the host. “On-chip DVS” means the Speck 2f internal 128×128 sensor is routed on chip; that path is not validated. “Smoke” means structural execution with random weights, not task performance. “Functional demo” means licensed trained weights with actual output and measured test-subset accuracy.
