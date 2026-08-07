# Sensor interface boundary

The sensor boundary outputs `(x, y, timestamp_us, polarity)` events. Transport framing, buffering, synchronization, overflow, and dropped-event counters are backend concerns and must remain visible to evaluation.
