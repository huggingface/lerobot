# SO-101 Utility Scripts

Helper scripts for assembling and configuring the SO-101 leader arm.

## Scripts

### `scan_motors.py` — Identify motor IDs on the bus

Scans the motor bus and reports which IDs are present, along with the
corresponding joint name. Useful when motor labels have been lost or you
want to verify which motors have already been configured.

```bash
# Scan for all SO-101 motors (IDs 1–6)
python examples/so101/scan_motors.py --port /dev/tty.usbmodem5B790332241

# Example output:
#   Scanning IDs 1–6 on /dev/tty.usbmodem5B790332241 at 1000000 baud...
#
#   ✓ Found motor ID 1  →  shoulder_pan
#   ✓ Found motor ID 2  →  shoulder_lift
#   ✓ Found motor ID 3  →  elbow_flex
#   ✓ Found motor ID 4  →  wrist_flex
#   ✓ Found motor ID 5  →  wrist_roll
#   ✓ Found motor ID 6  →  gripper
#
#   Found 6 motor(s): [1, 2, 3, 4, 5, 6]
```

Connect one motor at a time to identify a single motor, or daisy-chain
all motors to get a full report.

---

### `set_motor_ids.py` — Configure motor IDs

Sets motor IDs for individual SO-101 leader arm motors. This is useful
when `lerobot-setup-motors` fails partway through (e.g. due to a loose
cable), leaving some motors unconfigured. Since motor IDs are stored in
non-volatile memory, already-configured motors are unaffected — only the
remaining ones need to be set.

```bash
# Configure all motors (same as lerobot-setup-motors, but resumable)
python examples/so101/set_motor_ids.py --port /dev/tty.usbmodem5B790332241

# Configure only specific motors after a partial failure
python examples/so101/set_motor_ids.py \
  --port /dev/tty.usbmodem5B790332241 \
  --motors shoulder_lift shoulder_pan
```

Connect each motor **individually** when prompted — do not daisy-chain
multiple motors during ID assignment, as they all default to ID 1 and
will conflict on the bus.

---

## Finding your port

Connect the URT-2 board to your computer via USB, then run:

```bash
# macOS / Linux
ls /dev/tty.usb*

# or
python -m serial.tools.list_ports
```

## Requirements

Install the feetech extra before using these scripts:

```bash
uv sync --extra feetech
```

> **Note:** Python 3.12 or 3.13 is required. Python 3.14 is not yet
> supported due to a `draccus` incompatibility.

