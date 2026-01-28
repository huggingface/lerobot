# XLeRobot Teleoperators

An example run using `./src/lerobot/teleoperators/xlerobot_teleoperator/run.sh` : [video](https://youtu.be/OGI-Qtl3s6s)

This package introduces the teleoperation stack for the modular `xlerobot` platform. It contains a default composite teleoperator plus reusable sub-teleoperators for the mobile base, mount, and bimanual leader arms.

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

The original CLI-only setup still works too (no config files needed):

```bash
lerobot-teleoperate \
  --robot.type=xlerobot \
  --teleop.type=xlerobot_default_composite \
  --teleop.base_type=lekiwi_base_gamepad \
  --teleop.base='{"joystick_index": 0, "max_speed_mps": 0.6, "deadzone": 0.15, "yaw_speed_deg": 45}'
```

## Default composite (`xlerobot_default_composite`)

Defined in `default_composite/teleop.py`, this teleoperator merges:

- A `BiSO101Leader` instance (`arms_config`) for dual-arm control.
- A base gamepad teleop (`base_config`) that can target either the LeKiwi or BiWheel drive via Xbox gamepad axes.
- A mount gamepad teleop (`mount_config`) for pan/tilt camera motion.

Each sub-teleoperator contributes its action/feedback schema, and the composite exposes a single dictionary that policies, recorders, and replayers can consume. Script updates (`lerobot-record`, `lerobot-replay`, `lerobot-teleoperate`) already know about this new teleop type.

## Example launch

`run.sh` shows a full command wiring the new teleoperator with the `xlerobot` robot:

- Change `teleop.base_type` to `biwheel_gamepad` if your shared bus exposes the BiWheel base instead of LeKiwi.
- Skip any sub-teleoperator by omitting its config. The composite automatically removes unused components.
- The sub-teleoperators live under `sub_teleoperators/` if you want to extend joystick mappings, add buttons, or integrate other peripherals.

---
