# Event Input Protocol v1

## Canonical semantics

Every event has exactly four fields:

```text
x             unsigned integer pixel coordinate
y             unsigned integer pixel coordinate
timestamp_us  unsigned monotonic timestamp in microseconds
polarity      0 or 1
```

Coordinates start at zero. Sensor width/height are session metadata, not inferred from events. Producers must preserve ordering, declare clock origin and wrap behavior, and expose dropped-event/overflow status out of band. Consumers must reject out-of-range coordinates and unordered timestamps within a window.

## Binary record

The transport-neutral v1 record is 16 bytes, little-endian:

| Offset | Type | Field |
|---:|---|---|
| 0 | `uint16` | x |
| 2 | `uint16` | y |
| 4 | `uint64` | timestamp_us |
| 12 | `uint8` | polarity (`0` or `1`) |
| 13 | `uint8` | flags; v1 must send zero |
| 14 | `uint16` | reserved; v1 must send zero |

Packets/streams must separately carry protocol version, sensor geometry, sequence number, event count, and integrity protection. The 16-byte record alone is not a complete safety transport.
