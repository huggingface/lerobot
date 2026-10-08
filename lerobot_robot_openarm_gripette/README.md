# lerobot_robot_openarm_gripette

LeRobot plugin for an OpenArm 7-joint follower (Damiao motors over CAN) fitted with a
[Pollen Gripette](https://github.com/pollen-robotics/grabette/tree/develop/packages/gripette)
gripper. The Gripette runs its own gRPC server on a Pi Zero 2W, so no USB cable is needed: the
arm is driven by the stock `OpenArmFollower` (with the CAN gripper motor removed) and the gripper
is reached over WiFi.

## Install

In the environment where `lerobot` (with the `openarms` extra) is already installed:

```bash
pip install -e lerobot_robot_openarm_gripette
```

LeRobot discovers the package automatically through its `lerobot_robot_` prefix, so
`--robot.type=openarm_gripette_follower` works in every `lerobot-*` CLI.

## Usage

```bash
lerobot-teleoperate \
    --robot.type=openarm_gripette_follower \
    --robot.id=my_follower \
    --robot.arm_config.port=can0 \
    --robot.arm_config.side=right \
    --robot.gripette_addr=172.18.7.107:50051 \
    --teleop.type=openarm_leader \
    --teleop.port=can1 \
    --teleop.id=my_leader
```

## Features

- Arm: `joint_1.pos` … `joint_7.pos` in degrees, exactly as `openarm_follower`.
- Gripper: a single `gripper.pos` in radians in the Gripette frame (0 = open, positive = closing).
  Each motor receives `gripper.pos * gripper_motor_scales[i]`, clipped to
  `[0, gripper_motor_max[i]]` (the server rejects out-of-range goals).
- Camera: `gripette` (configurable via `gripper_camera_name`), an RGB frame resized to
  `gripper_camera_height` x `gripper_camera_width` (defaults match the server's 648x486 stream).
  Disable with `--robot.use_gripper_camera=false`.

The OpenArm leader and OpenArm Mini emit `gripper.pos` in degrees (0 to -65), not Gripette
radians, so their gripper command is clipped to "open" until a mapping is added.

## Tests

```bash
PYTHONPATH=../src:. python -m pytest tests
```

The tests stub `gripette.client` and the CAN controller, so they need neither hardware nor the
`gripette` package.
