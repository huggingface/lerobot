# XLeRobot Modular Platform

An example run with so101 arms: [video](https://youtu.be/OGI-Qtl3s6s)

`xlerobot` is a configurable mobile manipulator platform composed from optional sub-robots:

- **Arms** with support for dual SO-101/SO-100 follower arms.
- **Mobile base** abstraction with support for `lekiwi_base`, `biwheel_base`, `biwheel_feetech`, and `biwheel_odrive`.
- **Pan/Tilt camera mount** driven by Feetech servos.
- **Multi-camera rig** using the standard camera factory at robot level and arm level.
- **Shared bus or dedicated bus wiring** so Feetech components can share buses while ODrive stays on a dedicated connection.

The key idea is modular composition: any sub-robot can be used, omitted, or swapped depending on the hardware you actually mount. A deployment may use only a base, a base plus mount, dual SO-101 arms, or any other supported combination.

Quick links:

- [Base Options](#base-options)
- [Default Configs](#default-configs)
- [CLI Examples](./CLI_EXAMPLES.md)
- [XLeRobot Teleoperators README](../../teleoperators/xlerobot_teleoperator/README.md)
- [Biwheel Base README](./sub_robots/biwheel_base/README.md)
- [Biwheel Hardware Installation](./sub_robots/biwheel_base/README.md#hardware-installation)
- [Remote Teleoperation](#remote-keyboard-teleoperation-with-rerun)

<p align="center">
  <img
    src="https://github.com/user-attachments/assets/b48a0a41-7422-4f10-8dc6-a66a2fd746ad"
    alt="SO101 arms on XLeRobot"
    style="max-height: 420px; width: auto;"
  />
</p>

## Folder structure

```text
src/lerobot/robots/xlerobot/
|-- README.md
|-- CLI_EXAMPLES.md
|-- __init__.py
|-- config_xlerobot.py
|-- xlerobot.py
|-- configs/
|   |-- base_only.json
|   `-- xlerobot_biwheel_feetech.json
|-- shared_bus_mode/
|   |-- component_assembly.py
|   `-- shared_bus.py
`-- sub_robots/
    |-- __init__.py
    |-- biwheel_base/
    |   |-- README.md
    |   |-- config_biwheel_base.py
    |   |-- biwheel_base.py
    |   |-- biwheel_feetech.py
    |   `-- biwheel_odrive.py
    |-- lekiwi_base/
    |   |-- config.py
    |   `-- lekiwi_base.py
    `-- xlerobot_mount/
        |-- config.py
        `-- xlerobot_mount.py
```

## The robot class

`XLeRobot` orchestrates several sub-robots, each with its own configuration and handshake needs. The class:

- Provides shared bus configs, injects IDs, and enforces that every shared-bus component is routed through its declared `shared_buses` entry.
- Bridges component observations and actions into a single namespace (`left_*`, `right_*`, `x.vel`, `mount_pan.pos`, ...).
- Keeps the newest camera frame around in case a sensor read fails mid-run, which is useful during mobile deployments.
- Provides safe connect, disconnect, and calibration routines that cascade to all mounted components.
- Integrates with `lerobot-record`, `lerobot-replay`, and `lerobot-teleoperate` without custom script glue.

Any component block can be left empty (`{}`) when that sub-robot is not present in a given build.

## Base Options

`xlerobot` keeps `lekiwi_base` as a supported option and also supports the two-wheel `biwheel_*` family:

- `lekiwi_base`: original omniwheel-style mobile base
- `biwheel_feetech`: differential drive using Feetech wheel motors on a shared bus
- `biwheel_odrive`: differential drive using ODrive-controlled wheels on a dedicated connection
- `biwheel_base`: generic alias resolved by `xlerobot`; it uses Feetech by default, or ODrive if `driver: odrive`

For the biwheel variants, both observations and actions use:

- `x.vel`: linear velocity in m/s
- `theta.vel`: yaw velocity in deg/s

The biwheel stack also supports `invert_left_motor`, `invert_right_motor`, and `reverse_front_direction` to fix motor wiring and chassis-front conventions without changing higher-level teleop code.

`biwheel_feetech` can be placed on `shared_buses`.
`biwheel_odrive` must stay off shared buses and manage its own ODrive connection.

See `sub_robots/biwheel_base/README.md` for hardware and setup details.

<p align="center">
  <img
    src="https://raw.githubusercontent.com/Vector-Wangel/XLeRobot/main/hardware/odrive/media/assembled.png"
    alt="Biwheel ODrive base on XLeRobot"
    style="max-height: 420px; width: auto;"
  />
</p>

## Onboard Compute And Remote Operation

`xlerobot` can be made remote by attaching an onboard compute device to the platform and running the robot stack there. Any suitable edge computer can work, including:

- NVIDIA Jetson Thor
- NVIDIA Jetson Orin
- Raspberry Pi

If you do not want a dedicated onboard edge device, a laptop can also be used as the robot computer.

<p align="center">
  <img
    src="https://raw.githubusercontent.com/Vector-Wangel/XLeRobot/main/hardware/thor/media/thor.png"
    alt="Thor onboard compute for XLeRobot"
    style="max-height: 420px; width: auto;"
  />
</p>

Thor can be secured to the basket mesh using:

- [`hardware/thor/thor_holder_corner.step`](https://github.com/Vector-Wangel/XLeRobot/blob/main/hardware/thor/thor_holder_corner.step)
- [`hardware/thor/thor_holder_edge.step`](https://github.com/Vector-Wangel/XLeRobot/blob/main/hardware/thor/thor_holder_edge.step)

In practice, this means `lerobot-teleoperate`, `lerobot-record`, or inference can run on the robot-side machine while visualization or operator input can be forwarded from another machine over the network as needed. See [Remote Keyboard Teleoperation With Rerun](#remote-keyboard-teleoperation-with-rerun) for one supported workflow.

## Configuration example

Use the config examples below directly with `lerobot-teleoperate`, or create an `XLeRobotConfig` instance with equivalent fields. The inline examples use the same JSON object shape as `--robot.config_file`.

The example below keeps `lekiwi_base` as the mobile base. It is only one valid composition, not the only intended layout.

When using shared buses, make sure the physical motor IDs match the logical offsets. In sub-robot configs, motor IDs index from 1. In the `shared_buses` field, the sub-robot IDs are shifted by `motor_id_offset`. For example, if `mount.pan_motor_id` is `1` and the shared-bus entry sets `motor_id_offset` to `6`, the hardware motor ID must be `7`.

XLeRobot's default is to connect the left arm and the mount on the same board, and the right arm and the base on the same board. Refer to the [assembly docs](https://xlerobot.readthedocs.io/en/latest/hardware/getting_started/assemble.html) for more details.

If you prefer to have each sub-robot on a different board, configure each with its own shared bus and you are good to go.

```json
{
  "type": "xlerobot",
  "left_arm": {"id": "xlerobot_arm_left"},
  "right_arm": {"id": "xlerobot_arm_right"},
  "base": {
    "type": "lekiwi_base",
    "base_motor_ids": [2, 1, 3],
    "wheel_radius_m": 0.05,
    "base_radius_m": 0.125
  },
  "mount": {
    "pan_motor_id": 1,
    "tilt_motor_id": 2,
    "motor_model": "sts3215",
    "pan_key": "mount_pan.pos",
    "tilt_key": "mount_tilt.pos",
    "max_pan_speed_dps": 60.0,
    "max_tilt_speed_dps": 45.0,
    "pan_range": [-90.0, 90.0],
    "tilt_range": [-30.0, 60.0]
  },
  "cameras": {
    "top": {
      "type": "opencv",
      "index_or_path": 8,
      "width": 640,
      "height": 480,
      "fps": 30
    }
  },
  "shared_buses": {
    "left_bus": {
      "port": "/dev/ttyACM2",
      "components": [
        {"component": "left_arm"},
        {"component": "mount", "motor_id_offset": 6}
      ]
    },
    "right_bus": {
      "port": "/dev/ttyACM3",
      "components": [
        {"component": "right_arm"},
        {"component": "base", "motor_id_offset": 6}
      ]
    }
  }
}
```

With this config you can drive or record the platform via standard `lerobot-teleoperate`, `lerobot-record`, and `lerobot-replay`.

Customize the base type (`lekiwi_base`, `biwheel_base`, `biwheel_feetech`, or `biwheel_odrive`), arm type (`so101_follower` or `so100_follower`), mount gains, or camera set without editing Python. The config pipeline handles serialization, validation, and processor wiring for you.

## Default configs

Sample robot configuration files live in `src/lerobot/robots/xlerobot/configs/`. You can pass them directly to `lerobot-teleoperate` via `--robot.config_file` and still override values on the CLI. The JSON files include a `_comment` field that documents the intended CLI flag.

The original inline CLI-only configuration style still works too. For longer shell examples, see [CLI_EXAMPLES.md](./CLI_EXAMPLES.md).

```bash
lerobot-teleoperate \
  --robot.type=xlerobot \
  --robot.config_file=src/lerobot/robots/xlerobot/configs/xlerobot_biwheel_feetech.json \
  --teleop.type=xlerobot_default_composite \
  --teleop.config_file=src/lerobot/teleoperators/xlerobot_teleoperator/configs/xlerobot_default_composite_lekiwi.json \
  --teleop.base_type=biwheel_gamepad \
  --teleop.base='{"joystick_index": 0, "max_speed_mps": 0.8, "yaw_speed_deg": 45.0, "deadzone": 0.15}' \
  --robot.shared_buses.left_bus.port=/dev/ttyACM2 \
  --display_data=true
```

### Base-only (ODrive)

If you only want the mobile base, use `base_only.json` as a reference. It disables arms and mount and configures ODrive axis mapping and motor inversion so that `x.vel` drives forward instead of yawing.

For both `biwheel_odrive` and `biwheel_feetech`, you can flip only forward and backward direction by setting `base.reverse_front_direction: true` or `--robot.base.reverse_front_direction=true` on the CLI. This keeps turning sign unchanged.

```bash
lerobot-teleoperate \
  --robot.type=xlerobot \
  --robot.config_file=src/lerobot/robots/xlerobot/configs/base_only.json \
  --teleop.type=xlerobot_default_composite \
  --teleop.config_file=src/lerobot/teleoperators/xlerobot_teleoperator/configs/base_only.json \
  --display_data=true
```

### Base-only (Keyboard teleop)

If you want keyboard control for the biwheel base, use the keyboard composite teleoperator:

```bash
lerobot-teleoperate \
  --robot.type=xlerobot \
  --robot.config_file=src/lerobot/robots/xlerobot/configs/base_only.json \
  --teleop.type=xlerobot_keyboard_composite \
  --teleop.config_file=src/lerobot/teleoperators/xlerobot_teleoperator/configs/xlerobot_keyboard_composite_biwheel.json \
  --display_data=true
```

## Remote keyboard teleoperation with Rerun

Use this when `lerobot-teleoperate` runs on Thor and Rerun runs on your local machine.

### Linux local machine

Start local viewer:

```bash
rerun
```

Connect to Thor:

```bash
ssh -Y thor@thor
```

Run teleop on Thor with your local viewer machine IP:

```bash
lerobot-teleoperate \
  --robot.type=xlerobot \
  --robot.config_file=src/lerobot/robots/xlerobot/configs/base_only.json \
  --teleop.type=xlerobot_keyboard_composite \
  --teleop.config_file=src/lerobot/teleoperators/xlerobot_teleoperator/configs/xlerobot_keyboard_composite_biwheel.json \
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

Enable test extensions and restart XQuartz:

```bash
defaults write org.xquartz.X11 enable_test_extensions -boolean true
killall XQuartz
open -a XQuartz
```

Verify `RECORD` is available:

```bash
/opt/X11/bin/xdpyinfo | grep -i RECORD
```

If `RECORD` is missing, keyboard teleop listeners using `pynput` may fail with `AttributeError: record_create_context`.

Open `xterm` from XQuartz and use it for keyboard teleop control. Keyboard events for `pynput` are captured from the focused X11 window (`xterm`), not from regular macOS Terminal.

Start Rerun on Mac and keep it running:

```bash
rerun
```

Connect to Thor:

```bash
ssh -Y thor@thor
```

On Thor, verify X11 forwarding:

```bash
echo "$DISPLAY"
```

`DISPLAY` must be non-empty before starting `lerobot-teleoperate`.

Optional: tune Rerun micro-batching on Thor before starting teleop:

```bash
export RERUN_FLUSH_TICK_SECS=0.03
export RERUN_FLUSH_NUM_BYTES=65536
export RERUN_FLUSH_NUM_ROWS=4096
```

Then run teleop on Thor:

```bash
lerobot-teleoperate \
  --robot.type=xlerobot \
  --robot.config_file=src/lerobot/robots/xlerobot/configs/base_only.json \
  --teleop.type=xlerobot_keyboard_composite \
  --teleop.config_file=src/lerobot/teleoperators/xlerobot_teleoperator/configs/xlerobot_keyboard_composite_biwheel.json \
  --display_data=true \
  --fps=30 \
  --display_ip=<MAC_LAN_IP> \
  --display_port=9876 \
  --display_compressed_images=True
```

Replace `<MAC_LAN_IP>` with your Mac LAN IP.

## Dedicated buses (non-shared components)

Some drivers, for example ODrive-based bases, cannot share a Feetech bus. In that case, omit the component from `shared_buses` and set `shared_bus: false` in the component config. The component is then expected to manage its own connection.

```json
{
  "type": "xlerobot",
  "base": {
    "type": "biwheel_odrive",
    "shared_bus": false,
    "odrive_serial": "123456789ABC"
  },
  "shared_buses": {
    "left_bus": {
      "port": "/dev/ttyACM2",
      "components": [
        {"component": "left_arm"},
        {"component": "mount", "motor_id_offset": 6}
      ]
    }
  }
}
```
