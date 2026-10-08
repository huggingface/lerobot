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
- Gripper: `proximal.pos` and `distal.pos` in radians in the Gripette frame (0 = open,
  positive = closing), clipped to `[0, gripper_motor_max]` (the server rejects out-of-range goals).
- Camera: `gripette` (configurable via `gripper_camera_name`), an RGB frame resized to
  `gripper_camera_height` x `gripper_camera_width` (defaults match the server's 648x486 stream).
  Disable with `--robot.use_gripper_camera=false`.

The OpenArm leader and OpenArm Mini emit a single `gripper.pos` in degrees, so teleoperation
drives the arm only until a gripper mapping is added.

## Running a Grabette policy

`examples/run_grabette_policy.py` runs policies trained on handheld Grabette data with Cartesian
pose-relative actions (LeRobot's `feat/cartesian-relative-actions`), e.g.
[CarolinePascal/grabette-pick-and-place-texture-act](https://huggingface.co/CarolinePascal/grabette-pick-and-place-texture-act).
Each 50-step chunk of `oak_l` camera poses is anchored at the arm's current pose and solved with
placo IK on Pollen's OpenArm + Gripette URDF; the predicted (strategy, closure) is decoded to the
two Gripette motors with `gripette.grasp_projection`.

```bash
pip install -e "lerobot_robot_openarm_gripette[policy]"
python lerobot_robot_openarm_gripette/examples/run_grabette_policy.py \
    --port can0 --gripette 172.18.7.107:50051 --dry_run
```

- Start with `--dry_run`: it runs the policy and IK on live camera frames and prints how far each
  chunk would move the camera, without moving anything.
- After connecting, the runner disables only the arm torque. Support the arm, place the Gripette
  where the demonstrations started, and press ENTER. The runner enables torque, holds the measured
  pose, prints its joints and `oak_l` position, then requires a second ENTER before policy execution.
- Set the initial gripper with `--initial_gripper_closure 0` to open it or
  `--initial_gripper_closure 1` to close it. Use `--initial_gripper_strategy` to select the grasp
  strategy; omitted values preserve the measured state.
- The URDF is the right arm, with joints mapped `r_arm_pitch..r_wrist_pitch` -> `joint_1..joint_7`.
  Joint zeros must match the URDF (arm hanging straight down); flip reversed joints with
  `--joint_signs`.
- Match the demonstrations' distance and view of the scene: the policy reproduces demo-scale
  motions relative to wherever the arm starts.
- The run aborts if IK moves any joint more than `--max_joint_step` degrees in one tick.
- Predictions are flushed to a timestamped JSONL file under `outputs/openarm_gripette_policy`.
  Each action records the raw policy output, chunk/action indices, IK and gripper targets, and
  whether it was sent to the robot. Records are also printed live in the terminal. Override the
  directory with `--log_dir`.

## Tests

```bash
PYTHONPATH=../src:. python -m pytest tests
```

The tests stub `gripette.client` and the CAN controller, so they need neither hardware nor the
`gripette` package.
