# Next hardware steps

No Speck 2f was connected during this audit. USB was not bound or enumerated; the code did not
call `get_device_map`, `samna.device.open_device`, `DynapcnnNetwork.to(...)`, a visualizer,
power monitoring, device recording or FlashWrite.

After a Speck 2f arrives, run official examples in this order:

1. Confirm the dynamically reported device key with `get_device_map()`.
2. Run the N-MNIST quick start with explicit `speck2fdevkit:0`, initially without visualizer.
3. Compare hardware output with the saved Sinabs/Dynapcnn/Specksim baseline.
4. Run the official DVS input visualization notebook.
5. Run the official readout-layer example.
6. Run power monitoring after inference and event routing are stable.
7. Test the local configuration binary using the documented board procedure; FlashWrite needs
   separate explicit approval.

Do not start a custom robot-obstacle-avoidance model yet. Preserve the official baseline until
hardware behavior and event routing are verified.
