# Robot DVS reflex requirements (future work)

## 1. Problem definition

During robot motion, when a person or obstacle suddenly enters the path, rapidly approaches the robot, or abnormal motion such as a suspected fall occurs nearby, the system should detect risk promptly and send a risk level, speed-limit request, stop request, or alarm signal to STM32.

## 2. Current robot architecture

The ordinary cognitive chain is sensors → Jetson/ROS/vision → STM32 → motors. Jetson owns semantics, person attributes, fall confirmation, and planning. STM32 owns deterministic safety arbitration.

## 3. Why a separate reflex path

An event-driven Speck path is intended to remain low latency and independent when the host is busy or unavailable. It supplements rather than replaces semantic perception and verified safety logic.

## 4. Target scenarios

Rapid approach, sudden path intrusion, left/center/right spatial risk, collision/TTC level, and unusually violent motion.

## 5. Phase-one scope

Speck should detect rapid approach, sudden intrusion, directional risk, collision/TTC level, and abnormal high motion, and continue independently during host overload or loss.

## 6. Outside phase one

No elderly/child attributes, identity recognition, medical fall confirmation, full scene understanding, path planning, motor PWM control, or functional-safety certification.

## 7. Output protocol

`RiskLevel`: 0 NORMAL, 1 CAUTION, 2 LIMIT_SPEED, 3 STOP_REQUEST, 4 EMERGENCY_STOP. `RiskDirection`: UNKNOWN, LEFT, FRONT, RIGHT, REAR. A future frame includes `protocol_version`, `sequence_number`, `timestamp`, `risk_level`, `direction`, `confidence`, `time_to_collision`, `heartbeat`, and `checksum`. Only neutral types exist today; no serial driver is implemented.

## 8. Safety boundary

STM32 is always the final safety arbiter. Speck never controls motor PWM. Missing/invalid heartbeat, stale sequence, checksum failure, or timeout must produce a separately defined stop-or-limit fail-safe behavior.

## 9. Data collection

After DVS routing is validated, collect consented, versioned on-robot event data for approach speed, direction, lighting, clutter, ego-motion, near misses, normal motion, and difficult negatives. Define labeling and privacy retention before collection.

## 10. Metrics

Measure sensor-to-risk latency, missed-risk rate, false alarm rate, directional accuracy, TTC-bin error, overload independence, heartbeat response, power, and repeatability. Thresholds require a separate hazard analysis.

## 11. Hardware dependencies

Physical Speck 2f, internal DVS routing, readout mapping, approved USB setup, STM32 electrical/transport contract, synchronized timing, and robot stop-distance evidence.

## 12. Risks and unknowns

Domain shift, ego-motion, DVS noise, simulator/hardware mismatch, readout latency, GPIO semantics, timestamp synchronization, dataset licensing/privacy, electrical faults, and lack of safety certification remain unresolved.
