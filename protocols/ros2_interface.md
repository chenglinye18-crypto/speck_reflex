# ROS 2 interface — planned

ROS 2 is an observability and cognitive-system integration surface, not the safety-critical reflex transport. Future messages should mirror canonical event-window metadata, reflex predictions, backend health, event drops, and benchmark timestamps. QoS, clock synchronization, namespace, and message definitions remain unselected.

The fast reflex path must continue operating when Jetson/ROS is overloaded, restarting, or unavailable. ROS commands cannot bypass Safety MCU arbitration.
