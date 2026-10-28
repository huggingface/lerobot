# lerobot_robot_openarm_gripette

Grabette policy runners for an OpenArm fitted with a
[Pollen Gripette](https://github.com/pollen-robotics/grabette/tree/develop/packages/gripette).

The robot (`openarm_gripette_follower`) and its camera (`type: gripette`) ship with LeRobot.
Install them with the `gripette` extra — see the [OpenArm guide](../docs/source/openarm.mdx).
This package keeps the policy rollout helpers and re-exports the robot classes so existing imports
keep working.

## Install

In the environment where this checkout of `lerobot` is installed (`pip install -e ".[gripette]"`
plus the `gripette` client from the OpenArm guide):

```bash
pip install -e "lerobot_robot_openarm_gripette[policy]"
```

## Usage

```bash
lerobot-teleoperate \
    --robot.type=openarm_gripette_follower \
    --robot.id=my_follower \
    --robot.port=can0 \
    --robot.side=right \
    --robot.gripette_addr=172.18.7.107:50051 \
    --robot.cameras="{ gripette: {type: gripette, addr: 172.18.7.107:50051, width: 648, height: 486, fps: 10} }" \
    --teleop.type=openarm_leader \
    --teleop.port=can1 \
    --teleop.id=my_leader
```

## Features

- Arm: `joint_1.pos` … `joint_7.pos` in degrees, exactly as `openarm_follower`.
- Gripper: `proximal.pos` and `distal.pos` in radians in the Gripette frame (0 = open,
  positive = closing), clipped to `[0, gripper_motor_max]` (the server rejects out-of-range goals).
- Camera: configure it on `--robot.cameras` as `type: gripette`. It is not part of the gripper.
  Defaults match the server's 648x486 stream at 10 fps.

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
python lerobot_robot_openarm_gripette/examples/run_grabette_rollout.py \
    --port can0 --gripette 172.18.7.107:50051 \
    --fps 10 --n_action_steps 10 --display_data --dry_run
```

- `run_grabette_rollout.py` uses LeRobot's standard synchronous rollout engine and robot processors.
  `--display_data` opens Rerun with the camera, policy commands, and robot state.
- Start with `--dry_run`: it runs the policy and IK on live camera frames without moving anything.
  Remove it only after checking the targets in Rerun.
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
  whether it was sent to the robot. Cartesian, strategy, and closure commands are also printed live
  in the terminal. Override the directory with `--log_dir`. This JSONL logging remains available in
  the legacy `run_grabette_policy.py` runner.

## Tests

```bash
PYTHONPATH=../src:. python -m pytest tests
```

Robot and camera tests live in the LeRobot tree: `tests/robots/test_openarm_gripette_follower.py`
and `tests/cameras/test_gripette_camera.py`.

The tests stub `gripette.client` and the CAN controller, so they need neither hardware nor the
`gripette` package.
