# XLeRobot Modular Platform

An example run using the bash example in the end of this ReadMe document: [video](https://youtu.be/OGI-Qtl3s6s)

`xlerobot` is a fully mobile manipulator robot by composing:

- **Dual SO-101 follower arms** with shared calibration assets.
- **Mobile base** abstraction with current support for `lekiwi_base` and `biwheel_base`.
- **Pan/Tilt camera mount** driven by Feetech servos.
- **Multi-camera rig** wired through the standard camera factory so training pipelines receive synchronized RGB frames.
- **Shared motor buses** that multiplex multiple components on one serial port, reducing cabling and easing deployment on embedded PCs.

This is the same configuration showcased in the [XLeRobot demo run script](run.sh) and mirrors the hardware described in the linked community projects.

## The robot class

`XLeRobot` orchestrates several sub-robots, each with its own configuration/handshake needs. The class:

- Provides shared bus configs, injects IDs, and enforces that every component is routed through their declared shared bus (`shared_buses`).
- Bridges component observations and actions into a single namespace (`left_*`, `right_*`, `x.vel`, `mount_pan.pos`, …) for policies and scripts.
- Keeps the newest camera frame around in case a sensor read fails mid-run, which is crucial during mobile deployments.
- Provides safe connect/disconnect/calibration routines that cascade to all mounted components.
- Integrates with updated `lerobot-record`, `lerobot-replay`, and `lerobot-teleoperate` commands. No custom code required to capture trajectories or run inference.

## Configuration example

Simply run the demo run script under `./src/lerobot/teleoperators/xlerobot_teleoperator/run.sh`. Or, if needed, you may create an XLeRobotConfig instance by configuring it like below.

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

Customize the base type (`lekiwi_base`, `biwheel_base`, `biwheel_feetech`, or `biwheel_odrive`), mount gains, or camera set without editing Python. The config pipeline handles serialization, validation, and processor wiring for you.

## Default configs

Sample robot configuration files live in `src/lerobot/robots/xlerobot/configs/`. You can pass them directly to
`lerobot-teleoperate` via `--robot.config_file` and still override values on the CLI. The JSON files include a
`_comment` field that documents the intended CLI flag.

```bash
lerobot-teleoperate \
  --robot.type=xlerobot \
  --robot.config_file=src/lerobot/robots/xlerobot/configs/xlerobot_biwheel_feetech.json \
  --teleop.type=xlerobot_default_composite \
  --teleop.config_file=src/lerobot/teleoperators/xlerobot_teleoperator/configs/xlerobot_default_composite_biwheel.json \
  --robot.shared_buses.left_bus.port=/dev/ttyACM2 \
  --display_data=true
```

### Base-only (ODrive)

If you only want the mobile base, use `base_only.json` as a reference. It disables arms/mount and configures
ODrive axis mapping and motor inversion (for example `invert_left_motor: true` and swapping `axis_left/right`)
so that `x.vel` drives forward instead of yawing.

```bash
lerobot-teleoperate \
  --robot.type=xlerobot \
  --robot.config_file=src/lerobot/robots/xlerobot/configs/base_only.json \
  --teleop.type=xlerobot_default_composite \
  --teleop.config_file=src/lerobot/teleoperators/xlerobot_teleoperator/configs/base_only.json \
  --display_data=true
```

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
        "left_arm_port": "/dev/ttyACM0",
        "right_arm_port": "/dev/ttyACM1",
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
