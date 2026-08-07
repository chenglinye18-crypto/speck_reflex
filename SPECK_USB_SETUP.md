# Speck 2f USB setup for WSL2

Do not bind a USB device until its device name and VID:PID have both been
confirmed as the SynSense Speck board. In particular, do not bind a keyboard,
mouse, storage device, ordinary camera, or any other uncertain device.

## Windows administrator PowerShell

List USB devices:

```powershell
usbipd list
```

Identify the Speck device using both its displayed name and VID:PID. For the
first share, replace `<BUSID>` only with the confirmed Speck bus ID:

```powershell
usbipd bind --busid <BUSID>
```

Attach the confirmed device to Ubuntu 22.04:

```powershell
usbipd attach --wsl Ubuntu-22.04 --busid <BUSID>
```

If the installed usbipd version does not accept the distribution as a
positional argument, inspect the locally installed syntax and use the
equivalent form it documents:

```powershell
usbipd attach --help
```

## WSL verification

```bash
lsusb
source ~/projects/speck_reflex/.venv/bin/activate
python - <<'PY'
from sinabs.backend.dynapcnn.io import get_device_map
print(get_device_map())
PY
```

An empty dictionary before the board is attached is normal.

## SynSense udev rules

After the confirmed Speck device is available in WSL, locate and run the rule
installer, then reload the rules:

```bash
command -v install-synsense-rules
sudo "$(command -v install-synsense-rules)"
sudo udevadm control --reload-rules
sudo udevadm trigger
```

## Operational notes

- Bind only a positively identified Speck device.
- Never bind keyboards, mice, disks, ordinary cameras, or uncertain devices.
- Prefer a direct USB 3.x port and avoid USB hubs.
- A reboot or unplug/replug may require another `usbipd attach`.
- Samna hardware support under WSL must still be validated with the physical
  Speck 2f board.
