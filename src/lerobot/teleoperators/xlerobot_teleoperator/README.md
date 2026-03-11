# XLeRobot Teleoperators

An example run with so101 arms: [video](https://youtu.be/OGI-Qtl3s6s)

This package introduces the teleoperation stack for the modular `xlerobot` platform. It includes two composite teleoperators (`xlerobot_default_composite` and `xlerobot_keyboard_composite`) plus reusable sub-teleoperators for LeKiwi/BiWheel base control, mount pan/tilt, bimanual SO-101 leader arms, and single-arm Panthera keyboard control (cartesian).

## Folder structure

```text
src/lerobot/teleoperators/xlerobot_teleoperator/
|-- README.md
|-- __init__.py
|-- configs/
|   |-- base_only.json
|   |-- xlerobot_default_composite_lekiwi.json
|   |-- xlerobot_keyboard_composite_biwheel.json
|   |-- xlerobot_keyboard_composite_panthera_left_ee.json
|   `-- xlerobot_keyboard_composite_panthera_left_ee_with_base.json
|-- default_composite/
|   |-- __init__.py
|   |-- config.py
|   `-- teleop.py
|-- keyboard_composite/
|   |-- __init__.py
|   |-- config.py
|   `-- teleop.py
`-- sub_teleoperators/
    |-- __init__.py
    |-- biwheel_gamepad/
    |   |-- config_biwheel_gamepad.py
    |   `-- teleop_biwheel_gamepad.py
    |-- biwheel_keyboard/
    |   |-- configuration_biwheel_keyboard.py
    |   `-- teleop_biwheel_keyboard.py
    |-- lekiwi_base_gamepad/
    |   |-- config_lekiwi_base_gamepad.py
    |   `-- teleop_lekiwi_base_gamepad.py
    |-- panthera_keyboard_ee/
    |   |-- config_panthera_keyboard_ee.py
    |   `-- teleop_panthera_keyboard_ee.py
    `-- xlerobot_mount_gamepad/
        |-- config.py
        `-- teleop.py
```

## Default configs

Sample teleoperator configuration files live in `src/lerobot/teleoperators/xlerobot_teleoperator/configs/`.
You can use them with `lerobot-teleoperate` via `--teleop.config_file` and override individual fields on the CLI.
The JSON files include a `_comment` field that documents the intended CLI flag.

```bash
lerobot-teleoperate \
  --robot.type=xlerobot \
  --robot.config_file=src/lerobot/robots/xlerobot/configs/xlerobot_biwheel_feetech.json \
  --teleop.type=xlerobot_default_composite \
  --teleop.config_file=src/lerobot/teleoperators/xlerobot_teleoperator/configs/xlerobot_default_composite_lekiwi.json \
  --teleop.base.max_speed_mps=0.6
```

Keyboard base control with the built-in LeRobot keyboard teleop:

```bash
lerobot-teleoperate \
  --robot.type=xlerobot \
  --robot.config_file=src/lerobot/robots/xlerobot/configs/base_only.json \
  --teleop.type=xlerobot_keyboard_composite \
  --teleop.config_file=src/lerobot/teleoperators/xlerobot_teleoperator/configs/xlerobot_keyboard_composite_biwheel.json
```

Panthera arm keyboard teleop (Cartesian EE mapping) with xlerobot:

```bash
lerobot-teleoperate \
  --robot.type=xlerobot \
  --robot.config_file=src/lerobot/robots/xlerobot/configs/xlerobot_biwheel_odrive_panthera_left.json \
  --teleop.type=xlerobot_keyboard_composite \
  --teleop.config_file=src/lerobot/teleoperators/xlerobot_teleoperator/configs/xlerobot_keyboard_composite_panthera_left_ee.json
```
The provided robot config enables Panthera joint impedance + gravity/friction compensation.

For Cartesian arm + keyboard base together, use:

```bash
lerobot-teleoperate \
  --robot.type=xlerobot \
  --robot.config_file=src/lerobot/robots/xlerobot/configs/xlerobot_biwheel_odrive_panthera_left.json \
  --teleop.type=xlerobot_keyboard_composite \
  --teleop.config_file=src/lerobot/teleoperators/xlerobot_teleoperator/configs/xlerobot_keyboard_composite_panthera_left_ee_with_base.json
```

The original CLI-only setup still works too (no config files needed):

```bash
lerobot-teleoperate \
  --robot.type=xlerobot \
  --teleop.type=xlerobot_default_composite \
  --teleop.base_type=lekiwi_base_gamepad \
  --teleop.base='{"joystick_index": 0, "max_speed_mps": 0.6, "deadzone": 0.15, "yaw_speed_deg": 45}'
```

## Remote keyboard teleoperation with Rerun

Use this when `lerobot-teleoperate` runs on Thor and you want to view Rerun on your local machine.

### Linux local machine

Start local viewer:

```bash
rerun
```

Connect to Thor:

```bash
ssh -Y thor@thor
```

Then run teleop on Thor with your local viewer machine IP:

```bash
lerobot-teleoperate \
  --robot.type=xlerobot \
  --robot.config_file=src/lerobot/robots/xlerobot/configs/xlerobot_biwheel_odrive_panthera_left.json \
  --teleop.type=xlerobot_keyboard_composite \
  --teleop.config_file=src/lerobot/teleoperators/xlerobot_teleoperator/configs/xlerobot_keyboard_composite_panthera_left_ee_with_base.json \
  --display_data=true \
  --fps=30 \
  --display_ip=<LOCAL_MACHINE_IP> \
  --display_port=9876
```

### macOS local machine (XQuartz + RECORD + xterm keyboard capture)

Install and start XQuartz:

```bash
brew install --cask xquartz
open -a XQuartz
```

Enable test extensions (includes `RECORD`), then restart XQuartz:

```bash
defaults write org.xquartz.X11 enable_test_extensions -boolean true
killall XQuartz
open -a XQuartz
```

Verify `RECORD` exists:

```bash
/opt/X11/bin/xdpyinfo | grep -i RECORD
```

If this prints nothing, `pynput` keyboard listeners can fail with
`AttributeError: record_create_context`.

Open `xterm` from XQuartz and use it for keyboard teleop control.
Keyboard events for `pynput` are captured from the focused X11 window (`xterm`), not from regular macOS Terminal.

Start Rerun on Mac and keep it running:

```bash
rerun
```

In XQuartz `xterm`, connect to Thor:

```bash
ssh -Y thor@thor
```

On Thor, confirm X11 forwarding is active:

```bash
echo "$DISPLAY"
```

It should be non-empty (for example `localhost:10.0`).

Optional: tune Rerun micro-batching on Thor before starting teleop:

```bash
export RERUN_FLUSH_TICK_SECS=0.03
export RERUN_FLUSH_NUM_BYTES=65536
export RERUN_FLUSH_NUM_ROWS=4096
```

Then run teleop on Thor (direct to Mac IP):

```bash
lerobot-teleoperate \
  --robot.type=xlerobot \
  --robot.config_file=src/lerobot/robots/xlerobot/configs/xlerobot_biwheel_odrive_panthera_left.json \
  --teleop.type=xlerobot_keyboard_composite \
  --teleop.config_file=src/lerobot/teleoperators/xlerobot_teleoperator/configs/xlerobot_keyboard_composite_panthera_left_ee_with_base.json \
  --display_data=true \
  --fps=30 \
  --display_ip=<MAC_LAN_IP> \
  --display_port=9876 \
  --display_compressed_images=True
```

Replace `<MAC_LAN_IP>` with your Mac LAN IP.

## Default composite (`xlerobot_default_composite`)

Defined in `default_composite/teleop.py`, this teleoperator merges:

- A `BiSO101Leader` instance (`arms_config`) for dual-arm control.
- A base gamepad teleop (`base_config`) that can target either the LeKiwi or BiWheel drive via Xbox gamepad axes.
- A mount gamepad teleop (`mount_config`) for pan/tilt camera motion.

Each sub-teleoperator contributes its action/feedback schema, and the composite exposes a single dictionary that policies, recorders, and replayers can consume. Script updates (`lerobot-record`, `lerobot-replay`, `lerobot-teleoperate`) already know about this new teleop type.

## Keyboard composite (`xlerobot_keyboard_composite`)

Defined in `keyboard_composite/teleop.py`, this variant keeps the same composite pattern while swapping the base controller:

- An optional Panthera arm keyboard teleop (`arm_config`) with type `panthera_keyboard_ee`.
- `arm_side` routes arm actions to either `left_` or `right_` action namespace.
- A `KeyboardRoverTeleop` instance (`base_config`) from `lerobot.teleoperators.keyboard`.
- An optional `XLeRobotMountGamepadTeleop` instance (`mount_config`) for pan/tilt.

The keyboard rover output (`linear.vel`, `angular.vel`) is mapped to XLeRobot base keys (`x.vel`, `theta.vel`) so it works with `biwheel_base`/`biwheel_odrive` action interfaces. Panthera arm keyboard output is prefixed (`left_...` or `right_...`) for xlerobot arm routing.

---
