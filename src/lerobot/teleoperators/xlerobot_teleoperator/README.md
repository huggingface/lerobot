# XLeRobot Teleoperators

An example run using `./src/lerobot/teleoperators/xlerobot_teleoperator/run.sh` : [video](https://drive.google.com/file/d/1Kqvb8zP6Zjkz2CuB5h4jL4ymOBka8ckQ/view?usp=sharing)

This package introduces the teleoperation stack for the modular `xlerobot` platform. It contains a default composite teleoperator plus reusable sub-teleoperators for the mobile base, mount, and bimanual leader arms.

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
