# Reflex Output Protocol v1

This freezes the advisory model-to-Safety-MCU message. The MCU remains the final deterministic arbiter; `emergency_stop` means **request**, never direct motor actuation.

## Semantic message

```text
sequence_number
timestamp_us
risk_score       0.0 .. 1.0
ttc_ms           non-negative or unknown
direction        UNKNOWN / LEFT / FRONT / RIGHT / REAR
risk_level_hint  NORMAL / CAUTION / LIMIT_SPEED / STOP_REQUEST / EMERGENCY_STOP
emergency_stop   boolean request
CRC
```

Frozen enum values:

| Value | Direction | Risk-level hint |
|---:|---|---|
| 0 | UNKNOWN | NORMAL |
| 1 | LEFT | CAUTION |
| 2 | FRONT | LIMIT_SPEED |
| 3 | RIGHT | STOP_REQUEST |
| 4 | REAR | EMERGENCY_STOP |

## Binary frame

The v1 frame is 24 bytes, little-endian:

| Offset | Type | Field |
|---:|---|---|
| 0 | `uint16` | magic `0x5246` (`RF`) |
| 2 | `uint8` | protocol version `1` |
| 3 | `uint8` | flags; bit 0 = emergency-stop request |
| 4 | `uint32` | sequence_number |
| 8 | `uint64` | timestamp_us |
| 16 | `uint16` | risk_score_q15: 0..32767 |
| 18 | `uint16` | ttc_ms; `0xFFFF` means unknown |
| 20 | `uint8` | direction enum |
| 21 | `uint8` | risk-level hint enum |
| 22 | `uint16` | CRC-16/CCITT-FALSE over bytes 0..21 |

`risk_score_q15 = round(clamp(risk_score, 0, 1) * 32767)`. CRC-16/CCITT-FALSE uses polynomial `0x1021`, initial value `0xFFFF`, no reflection, and final XOR `0x0000`.

Receivers must reject bad magic/version/CRC, stale or repeated sequences outside the duplicate policy, invalid enum values, and timestamps outside the configured freshness window. On timeout or invalid input, the MCU applies its independently configured fail-safe policy. Floating-point values never cross this wire boundary.
