# XLeRobot Modular Platform

An example run with so101 arms: [video](https://youtu.be/OGI-Qtl3s6s)

`xlerobot` is a configurable mobile manipulator platform composed from optional sub-robots:

- **Arms** with support for dual SO-101/SO-100 follower arms or a single left/right `panthera_arm`.
- **Mobile base** abstraction with support for `lekiwi_base`, `biwheel_base`, `biwheel_feetech`, and `biwheel_odrive`.
- **Pan/Tilt camera mount** driven by Feetech servos.
- **Multi-camera rig** using the standard camera factory (robot-level and arm-attached cameras).
- **Shared bus or dedicated bus wiring** so Feetech components can share buses while ODrive/Panthera stay on dedicated connections.

## Folder structure

```text
src/lerobot/robots/xlerobot/
|-- README.md
|-- __init__.py
|-- config_xlerobot.py
|-- xlerobot.py
|-- configs/
|   |-- base_only.json
|   |-- xlerobot_biwheel_feetech.json
|   `-- xlerobot_biwheel_odrive_panthera_left.json
|-- shared_bus_mode/
|   |-- component_assembly.py
|   `-- shared_bus.py
`-- sub_robots/
    |-- __init__.py
    |-- biwheel_base/
    |   |-- config_biwheel_base.py
    |   |-- biwheel_base.py
    |   |-- biwheel_feetech.py
    |   `-- biwheel_odrive.py
    |-- lekiwi_base/
    |   |-- config.py
    |   `-- lekiwi_base.py
    |-- panthera_arm/
    |   |-- config.py
    |   `-- panthera_arm.py
    `-- xlerobot_mount/
        |-- config.py
        `-- xlerobot_mount.py
```

## The robot class

`XLeRobot` orchestrates several sub-robots, each with its own configuration/handshake needs. The class:

- Provides shared bus configs, injects IDs, and enforces that every component is routed through their declared shared bus (`shared_buses`).
- Bridges component observations and actions into a single namespace (`left_*`, `right_*`, `x.vel`, `mount_pan.pos`, …) for policies and scripts.
- Keeps the newest camera frame around in case a sensor read fails mid-run, which is crucial during mobile deployments.
- Provides safe connect/disconnect/calibration routines that cascade to all mounted components.
- Integrates with updated `lerobot-record`, `lerobot-replay`, and `lerobot-teleoperate` commands. No custom code required to capture trajectories or run inference.

## Configuration example

Use the config examples below directly with `lerobot-teleoperate`, or create an `XLeRobotConfig` instance with equivalent fields.

Note, make sure on the shared buses, you have set the motor ID correctly. In subrobot's configs, the motor IDs index from 1. In the `shared_buses` field, the subrobot's IDs will be shifted by `motor_id_offset`. For example, the `pan_motor_id` for the `mount` will be 1 + 6 = 7. So, you would also need to set the FeeTech motor to be 7 using supported motor programming tool. This ensures the IDs do no collide with the other subrobots on the same bus.

XLeRobot's default is to connect left arm and the mount on the same board, and right arm and the base on the same board. Refer to [doc](https://xlerobot.readthedocs.io/en/latest/hardware/getting_started/assemble.html) for more details.

If you prefer to have each sub robot on a different board, configure each with a shared bus and you are good to go.

```yaml
robot:
  type: xlerobot

  left_arm: {id: xlerobot_arm_left}
  right_arm: {id: xlerobot_arm_right}
  base:
    type: lekiwi_base
    base_motor_ids: [2, 1, 3]
    wheel_radius_m: 0.05
    base_radius_m: 0.125
  mount:
    pan_motor_id: 1
    tilt_motor_id: 2
    motor_model: sts3215
    pan_key: "mount_pan.pos",
    tilt_key: "mount_tilt.pos",
    max_pan_speed_dps: 60.0,
    max_tilt_speed_dps: 45.0,
    pan_range: [-90.0, 90.0],
    tilt_range: [-30.0, 60.0]

  cameras:
    top:
      type: opencv
      index_or_path: 8
      width: 640
      height: 480
      fps: 30

  shared_buses:
    left_bus:
      port: /dev/ttyACM2
      components:
        - {component: left_arm}
        - {component: mount, motor_id_offset: 6}
    right_bus:
      port: /dev/ttyACM3
      components:
        - {component: right_arm}
        - {component: base, motor_id_offset: 6}
```

With this config you can drive/record the platform via standard `lerobot-teleoperate`, `lerobot-record`, and `lerobot-replay`.

Customize the base type (`lekiwi_base`, `biwheel_base`, `biwheel_feetech`, or `biwheel_odrive`), arm type (`so101_follower` or `panthera_arm`), mount gains, or camera set without editing Python. The config pipeline handles serialization, validation, and processor wiring for you.

## Default configs

Sample robot configuration files live in `src/lerobot/robots/xlerobot/configs/`. You can pass them directly to
`lerobot-teleoperate` via `--robot.config_file` and still override values on the CLI. The JSON files include a
`_comment` field that documents the intended CLI flag.

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

If you only want the mobile base, use `base_only.json` as a reference. It disables arms/mount and configures
ODrive axis mapping and motor inversion (for example `invert_left_motor: true` and swapping `axis_left/right`)
so that `x.vel` drives forward instead of yawing.

For both `biwheel_odrive` and `biwheel_feetech`, you can flip only forward/backward direction by setting
`base.reverse_front_direction: true` (or `--robot.base.reverse_front_direction=true` on CLI).
This keeps turning sign unchanged (`theta.vel` still uses the same left/right convention).

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

### Panthera arm + ODrive base (Keyboard teleop, Cartesian EE mapping)

For a single left Panthera arm mounted on xlerobot:

```bash
lerobot-teleoperate \
  --robot.type=xlerobot \
  --robot.config_file=src/lerobot/robots/xlerobot/configs/xlerobot_biwheel_odrive_panthera_left.json \
  --teleop.type=xlerobot_keyboard_composite \
  --teleop.config_file=src/lerobot/teleoperators/xlerobot_teleoperator/configs/xlerobot_keyboard_composite_panthera_left_ee.json \
  --display_data=true
```
Panthera arms use a different motor/driver stack and must stay off `shared_buses` (`shared_bus: false`).
This example uses a single left Panthera arm.
This sample config enables Panthera joint impedance + gravity/friction compensation
(`left_arm.use_joint_impedance: true`).
It now also includes:

- `left_arm.cameras.gripper_camera` (arm-attached camera)
- `cameras.base_camera` (robot-level/base camera)

In observations these appear as `left_gripper_camera` and `base_camera`.
If your device indices differ, update `index_or_path` values in the JSON config before running.

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
  --robot.config_file=src/lerobot/robots/xlerobot/configs/xlerobot_biwheel_odrive_panthera_left.json \
  --teleop.type=xlerobot_keyboard_composite \
  --teleop.config_file=src/lerobot/teleoperators/xlerobot_teleoperator/configs/xlerobot_keyboard_composite_panthera_left_ee_with_base.json \
  --display_data=true \
  --fps=30 \
  --display_ip=<LOCAL_MACHINE_IP> \
  --display_port=9876
```

### macOS local machine (XQuartz + RECORD + xterm keyboard capture)

Install/start XQuartz:

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

If `RECORD` is missing, keyboard teleop listeners using `pynput` may fail with
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

## Dedicated buses (non-shared components)

Some drivers (for example ODrive-based bases) cannot share a Feetech bus. In that case, omit the component from
`shared_buses` and set `shared_bus: false` in the component config. The component is then expected to manage its
own connection (for example via `port` or `odrive_serial`).

```yaml
robot:
  type: xlerobot
  base:
    type: biwheel_odrive
    shared_bus: false
    odrive_serial: "123456789ABC"

  shared_buses:
    left_bus:
      port: /dev/ttyACM2
      components:
        - {component: left_arm}
        - {component: mount, motor_id_offset: 6}
```

# XLeRobot integration based on

- https://github.com/Astera-org/brainbot
- https://github.com/Vector-Wangel/XLeRobot
- https://github.com/bingogome/lerobot

# Example Command Line Run

```bash
lerobot-teleoperate \
    --robot.type=xlerobot \
    --robot.left_arm='{
        "id": "xlerobot_arm_left"
    }' \
    --robot.right_arm='{
        "id": "xlerobot_arm_right"
    }' \
    --robot.base='{
        "type": "lekiwi_base",
        "wheel_radius_m": 0.05,
        "base_radius_m": 0.125
    }' \
    --robot.mount='{
        "pan_motor_id": 1,
        "tilt_motor_id": 2,
        "motor_model": "sts3215",
        "pan_key": "mount_pan.pos",
        "tilt_key": "mount_tilt.pos",
        "max_pan_speed_dps": 60.0,
        "max_tilt_speed_dps": 45.0,
        "pan_range": [-90.0, 90.0],
        "tilt_range": [-30.0, 60.0]
    }' \
    --robot.cameras='{
        "top":   {"type": "opencv", "index_or_path": 8, "width": 640, "height": 480, "fps": 30}
    }' \
    --robot.shared_buses='{
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
    }' \
    --teleop.type=xlerobot_default_composite \
    --teleop.base_type=lekiwi_base_gamepad \
    --teleop.arms='{
        "left_arm_config": {
            "port": "/dev/ttyACM0"
        },
        "right_arm_config": {
            "port": "/dev/ttyACM1"
        },
        "id": "leader"
    }' \
    --teleop.base='{
        "joystick_index": 0,
        "max_speed_mps": 0.8,
        "deadzone": 0.15,
        "yaw_speed_deg": 45
    }' \
    --teleop.mount='{
        "joystick_index": 0,
        "deadzone": 0.15,
        "polling_fps": 50,
        "max_pan_speed_dps": 60.0,
        "max_tilt_speed_dps": 45.0,
        "pan_axis": 3,
        "tilt_axis": 4,
        "invert_pan": false,
        "invert_tilt": true,
        "pan_range": [-90.0, 90.0],
        "tilt_range": [-30.0, 60.0]
    }' \
    --display_data=true
```

Or, if you want to run without arms or mount, you can either use the base-only configs:

```bash
lerobot-teleoperate \
  --robot.type=xlerobot \
  --robot.config_file=src/lerobot/robots/xlerobot/configs/base_only.json \
  --teleop.type=xlerobot_default_composite \
  --teleop.config_file=src/lerobot/teleoperators/xlerobot_teleoperator/configs/base_only.json \
  --display_data=true
```

…or stick to the original CLI-only configuration (no config files):

```bash
lerobot-teleoperate \
  --robot.type=xlerobot \
  --robot.base='{
      "type": "lekiwi_base",
      "wheel_radius_m": 0.05,
      "base_radius_m": 0.125
  }' \
  --robot.mount='{
      "pan_motor_id": 1,
      "tilt_motor_id": 2,
      "motor_model": "sts3215",
      "pan_key": "mount_pan.pos",
      "tilt_key": "mount_tilt.pos",
      "max_pan_speed_dps": 60.0,
      "max_tilt_speed_dps": 45.0,
      "pan_range": [-90.0, 90.0],
      "tilt_range": [-30.0, 60.0]
  }' \
  --robot.cameras='{}' \
  --robot.shared_buses='{
      "base_bus": {
          "port": "/dev/ttyACM4",
          "components": [
              {"component": "base"}
          ]
      },
      "mount_bus": {
          "port": "/dev/ttyACM5",
          "components": [
              {"component": "mount"}
          ]
      }
  }' \
  --teleop.type=xlerobot_default_composite \
  --teleop.base_type=lekiwi_base_gamepad \
  --teleop.base='{
      "joystick_index": 0,
      "max_speed_mps": 0.8,
      "deadzone": 0.15,
      "yaw_speed_deg": 45
  }' \
  --teleop.mount='{
      "joystick_index": 0,
      "deadzone": 0.15,
      "polling_fps": 50,
      "max_pan_speed_dps": 60.0,
      "max_tilt_speed_dps": 45.0,
      "pan_axis": 3,
      "tilt_axis": 4,
      "invert_pan": false,
      "invert_tilt": true,
      "pan_range": [-90.0, 90.0],
      "tilt_range": [-30.0, 60.0]
  }' \
  --display_data=true
```

---
