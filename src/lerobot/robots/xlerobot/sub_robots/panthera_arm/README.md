# Panthera Arm

This sub-robot integrates the Panthera arm into `xlerobot`.
The robot/SDK is from [HighTorque-Robotics/Panthera-HT_SDK](https://github.com/HighTorque-Robotics/Panthera-HT_SDK). This integration was tested with the smaller Panthera variant, not the full-sized arm.

[Jump to hardware installation](#hardware-installation)

<p align="center">
  <img
    src="https://raw.githubusercontent.com/Vector-Wangel/XLeRobot/main/hardware/high_torque_robotics/media/panthera.png"
    alt="Panthera arm mounted on XLeRobot"
    style="max-height: 420px; width: auto;"
  />
</p>

## What this wrapper provides

- A `panthera_arm` robot type for `xlerobot`.
- Cartesian end-effector delta actions (`delta_x`, `delta_y`, `delta_z`, `delta_roll`, `delta_pitch`, `delta_yaw`) plus `gripper`.
- Joint-space state for 6 arm joints, gripper state, and end-effector position.
- Optional arm-mounted cameras through the standard LeRobot camera factory.
- Optional joint impedance mode with gravity/friction compensation.

## Files

```text
src/lerobot/robots/xlerobot/sub_robots/panthera_arm/
|-- Panthera_lib/
|-- README.md
|-- __init__.py
|-- config.py
|-- robot_param/
|-- xlerobot-HT_description/
`-- panthera_arm.py
```

## SDK setup

This wrapper imports `Panthera` from
`src/lerobot/robots/xlerobot/sub_robots/panthera_arm/Panthera_lib/`.

The Panthera setup for `xlerobot` has three pieces:

- `Panthera_lib/`: download `hardware/high_torque_robotics/Panthera_lib.zip` from [Vector-Wangel/XLeRobot](https://github.com/Vector-Wangel/XLeRobot) and extract it into `src/lerobot/robots/xlerobot/sub_robots/panthera_arm/Panthera_lib/`.
- `robot_param/`: must be downloaded separately for the smaller XLeRobot-tested Panthera variant.
- `xlerobot-HT_description/`: must be downloaded separately for the smaller XLeRobot-tested Panthera variant.

You still need the Panthera Python runtime dependencies in your environment.
Use the upstream
[`Panthera-HT_SDK/panthera_python`](https://github.com/HighTorque-Robotics/Panthera-HT_SDK/tree/main/panthera_python)
package for:

- the matching `motor_whl/hightorque_robot-*.whl`
- the extra `Panthera_lib` Python dependencies (`pyyaml`, `pin`, `scipy`)

Note that the default assets in the Panthera SDK repository are for the full-sized
Panthera. This integration was tested with the smaller Panthera variant, so use
the asset bundles from
[Vector-Wangel/XLeRobot](https://github.com/Vector-Wangel/XLeRobot):

- `hardware/high_torque_robotics/Panthera_lib.zip`
- `hardware/high_torque_robotics/robot_param.zip`
- `hardware/high_torque_robotics/xlerobot-HT_description.zip`

After downloading and extracting them, place the contents in:

- `src/lerobot/robots/xlerobot/sub_robots/panthera_arm/Panthera_lib/`
- `src/lerobot/robots/xlerobot/sub_robots/panthera_arm/robot_param/`
- `src/lerobot/robots/xlerobot/sub_robots/panthera_arm/xlerobot-HT_description/`

The expected folder layout is:

```text
src/lerobot/robots/xlerobot/sub_robots/panthera_arm/
|-- Panthera_lib/
|-- robot_param/
`-- xlerobot-HT_description/
```

## Configuration

Register this arm as `type: panthera_arm`.
Common configuration fields:

- `config_path`: optional Panthera robot YAML config. If omitted, this wrapper looks for `robot_param/Follower.yaml` in this folder.
- `cameras`: optional arm-attached cameras.
- `xy_step_m`, `vertical_step_m`, `rotation_step_deg`: Cartesian command scaling.
- `joint_velocity`, `max_torque`: direct joint command limits.
- `use_joint_impedance`: enable impedance mode.
- `run_startup_sequence`: optional startup motion sequence on connect.

Example from this repo:

```json
"left_arm": {
  "type": "panthera_arm",
  "shared_bus": false,
  "use_joint_impedance": true,
  "run_startup_sequence": true
}
```

By default this wrapper looks for
`src/lerobot/robots/xlerobot/sub_robots/panthera_arm/robot_param/Follower.yaml`.
Place the downloaded Panthera files there, or set `config_path` to another YAML location.

Panthera uses its own SDK and connection path, so it must NOT be placed on `shared_buses`.

## Example run

Use the existing xlerobot sample config:

```bash
lerobot-teleoperate \
  --robot.type=xlerobot \
  --robot.config_file=src/lerobot/robots/xlerobot/configs/xlerobot_biwheel_odrive_panthera_left.json \
  --teleop.type=xlerobot_keyboard_composite \
  --teleop.config_file=src/lerobot/teleoperators/xlerobot_teleoperator/configs/xlerobot_keyboard_composite_panthera_left_ee.json
```

This sample runs a single left Panthera arm on `xlerobot` with keyboard Cartesian end-effector teleoperation.

## Observations and actions

Observed values include:

- `joint1.pos` ... `joint6.pos`
- `joint1.vel` ... `joint6.vel`
- `joint1.torque` ... `joint6.torque`
- `gripper.pos`
- `gripper.torque`
- `ee.x`, `ee.y`, `ee.z`
- any configured camera streams

Accepted actions are:

- `delta_x`
- `delta_y`
- `delta_z`
- `delta_roll`
- `delta_pitch`
- `delta_yaw`
- `gripper`

## Notes

- IK is solved inside the wrapper before sending Panthera commands.
- Workspace radius and height are clamped by config limits.
- The optional startup sequence is disabled by default in `PantheraArmConfig` to avoid unexpected motion on connect.

## Hardware Installation

### Robot Mount

The Panthera mounting hardware for XLeRobot is in [Vector-Wangel/XLeRobot](https://github.com/Vector-Wangel/XLeRobot),
under `hardware/high_torque_robotics/`.

Use `hardware/high_torque_robotics/ht_holder_single.step` to mount the Panthera arm onto the XLeRobot top basket.
Secure it with M4 screws through the provided hole pattern, which is intended to fasten directly to the IKEA basket mesh.

<p align="center">
  <img
    src="https://raw.githubusercontent.com/Vector-Wangel/XLeRobot/main/hardware/high_torque_robotics/media/base_mount_cad.png"
    alt="Panthera arm base mount CAD"
    style="max-height: 420px; width: auto;"
  />
</p>

<p align="center">
  <img
    src="https://raw.githubusercontent.com/Vector-Wangel/XLeRobot/main/hardware/high_torque_robotics/media/base_mount.png"
    alt="Panthera arm base mount on XLeRobot"
    style="max-height: 420px; width: auto;"
  />
</p>

### Robot Controller Board

Mount the controller board on the bottom side of the basket using four copies of `hardware/high_torque_robotics/ht_board_holder_corner.step`.

After fixing the board in place, route and secure the power, motor, and communication cables. You may use zip ties to do so.

For reference renders and photos, check the `hardware/high_torque_robotics/media/` folder in the XLeRobot repository.

<p align="center">
  <img
    src="https://raw.githubusercontent.com/Vector-Wangel/XLeRobot/main/hardware/high_torque_robotics/media/board_mount_cad.png"
    alt="Panthera controller board mount CAD"
    style="max-height: 420px; width: auto;"
  />
</p>

<p align="center">
  <img
    src="https://raw.githubusercontent.com/Vector-Wangel/XLeRobot/main/hardware/high_torque_robotics/media/board_mount.png"
    alt="Panthera controller board mount on XLeRobot"
    style="max-height: 420px; width: auto;"
  />
</p>

### Wiring Schematic

TODO
