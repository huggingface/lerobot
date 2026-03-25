# XLeRobot CLI Examples

This file collects the inline CLI-only `xlerobot` examples that still work
without JSON config files. The config-file workflow in
[`README.md`](./README.md) is simpler for day-to-day use, but these commands are
still useful when you want to run everything directly from the shell.

## Full Inline Example

This example uses:

- dual SO follower arms
- `lekiwi_base`
- Feetech mount
- shared bus wiring
- `xlerobot_default_composite` teleoperation

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
    "top": {"type": "opencv", "index_or_path": 8, "width": 640, "height": 480, "fps": 30}
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
    }
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

## Base-Only With Config Files

If you only want the base and prefer a short command, the config-file route is
still the simplest:

```bash
lerobot-teleoperate \
  --robot.type=xlerobot \
  --robot.config_file=src/lerobot/robots/xlerobot/configs/base_only.json \
  --teleop.type=xlerobot_default_composite \
  --teleop.config_file=src/lerobot/teleoperators/xlerobot_teleoperator/configs/base_only.json \
  --display_data=true
```

## Base And Mount Inline Example

This inline example omits the arms and keeps only a mobile base plus mount:

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

## Notes

- `lekiwi_base` remains a supported option; these examples keep it intentionally.
- Inline CLI configuration is flexible, but config files are easier to maintain for repeated runs.
- For Panthera-specific setup, use the dedicated config-file examples in [`README.md`](./README.md) and `sub_robots/panthera_arm/README.md`.
