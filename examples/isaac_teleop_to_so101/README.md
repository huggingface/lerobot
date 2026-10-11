# Isaac Teleop → SO-101

Teleoperate an SO-101/SO-100 follower arm — and record LeRobot datasets — with NVIDIA
[Isaac Teleop](https://github.com/NVIDIA/IsaacTeleop). Two input devices ship today:

- **XR (VR) controller** (`--teleop.type=xr_controller`) — the controller's grip pose drives the
  end-effector through a squeeze-to-engage clutch and LeRobot's Cartesian IK pipeline; the analog
  trigger drives the gripper.
- **SO-101 leader arm** (`--teleop.type=so101_leader`) — a back-drivable leader arm mirrored 1:1
  onto the follower via Isaac Teleop's native `so101_leader` plugin (no clutch, no IK).

The full narrative guide (how the clutch works, CloudXR setup, headset pairing, tuning, and
troubleshooting) is in the [LeRobot docs](https://huggingface.co/docs/lerobot/isaac_teleop)
(source: `docs/source/isaac_teleop.mdx`). This README is the canonical install and usage
reference.

## Requirements

- Linux workstation (see NVIDIA's
  [system requirements](https://nvidia.github.io/IsaacTeleop/main/references/requirements.html)
  for supported OS/GPU/headset combinations; `isaacteleop` publishes Linux wheels only).
- An SO-101 (or SO-100) follower arm, calibrated with `lerobot-calibrate`.
- For the XR device: a CloudXR-capable headset (e.g. Quest 3, Pico 4, Apple Vision Pro) on the
  same network.
- For the leader device: a second, back-drivable SO-101 leader arm and the `so101_leader` plugin
  binary built from the Isaac Teleop source tree (see
  [Build from source](https://nvidia.github.io/IsaacTeleop/main/getting_started/build_from_source/index.html)).

## Installation

This example lives in the LeRobot repository and is not part of the `lerobot` pip package, so
work from a source checkout. From the repo root:

```bash
# LeRobot with the extras this example uses:
#   feetech    - SO-101 serial motor bus
#   kinematics - Placo IK solver (XR controller path)
#   dataset    - dataset recording (record.py)
# huggingface_hub >= 1.5 is needed by the automatic URDF fetch (Buckets API).
uv pip install -e ".[feetech,kinematics,dataset]" "huggingface_hub>=1.5"

# Isaac Teleop from public PyPI. `cloudxr` brings the CloudXR runtime bindings;
# `retargeters-lite` is the scipy-based retargeter path that resolves on both
# x86_64 and ARM (the full `retargeters` extra does not resolve on aarch64).
uv pip install "isaacteleop[cloudxr,retargeters-lite]~=1.3.131" "scipy>=1.14"

# Optional, x86_64 only: the full retargeter stack.
uv pip install "isaacteleop[retargeters]~=1.3.131"
```

One-time CloudXR EULA (the auto-launch prompts on stdin and would hang on a headless machine):

```bash
python -m isaacteleop.cloudxr --accept-eula
```

## Usage

Run everything from the repo root with `python -m` so the `examples` package resolves.

### Teleoperate — XR controller

```bash
python -m examples.isaac_teleop_to_so101.teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=so101_follower_arm \
    --teleop.type=xr_controller
```

On startup the script launches the CloudXR runtime (~30 s), prints the workstation IP to enter in
the headset's CloudXR web client, waits for the controllers to stream, slews the arm to a reset
pose (`--reset_to_origin=false` to skip), and then: **hold the squeeze/grip** to engage, move the
controller to drive the arm, pull the trigger to close the gripper. Releasing the squeeze freezes
the arm. The SO-101 URDF is fetched automatically from the `lerobot/robot-urdfs` Hugging Face
bucket into the LeRobot cache on first run.

To customize the reset pose: back-drive the arm to the pose you want, then

```bash
python -m examples.isaac_teleop_to_so101.override_reset_pose --port /dev/ttyACM0 --id so101_follower_arm
```

which writes it to `HF_LEROBOT_HOME/reset_poses/<robot.name>/<robot.id>.json`; runs with the same
`--robot.id` use it automatically.

### Teleoperate — SO-101 leader arm

```bash
python -m examples.isaac_teleop_to_so101.teleoperate \
    --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=so101_follower_arm \
    --teleop.type=so101_leader --teleop.port=/dev/ttyACM1 --teleop.id=so101_leader_arm \
    --launch_plugin=/path/to/IsaacTeleop/install/plugins/so101_leader/so101_leader_plugin
```

The follower is first slewed to the leader's pose over `--align_duration` seconds
(`--align=false` to skip), then mirrors it 1:1. The plugin reuses the serial leader's calibration
(`HF_LEROBOT_CALIBRATION/teleoperators/so_leader/<teleop.id>.json`).

### Record a dataset

`record.py` takes the same `--robot.*`/`--teleop.*`/loop flags plus `lerobot-record`-style
`--dataset.*` flags:

```bash
python -m examples.isaac_teleop_to_so101.record \
    --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=so101_follower_arm \
    --teleop.type=xr_controller \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
    --dataset.repo_id=<hf_user>/<dataset_name> \
    --dataset.single_task="Pick up the cube" \
    --dataset.num_episodes=3 --dataset.episode_time_s=20 --dataset.reset_time_s=5
```

Keyboard shortcuts (terminal-first, so they work over SSH): **Right/n** end episode early,
**Left/r** re-record, **Esc/q** stop after the current episode.

Run either script with `--help` for all flags.

## Layout

```
isaac_teleop/            device library: session lifecycle (base.py), XRController,
                         SO101LeaderArm, Clutch, configs, and the XR→IK processor step
common.py                shared loop infra: device bundles, clutch/IK pipeline wiring,
                         reset/align slews, URDF fetch, keyboard listener
teleoperate.py           teleoperation CLI (device selected via --teleop.type)
record.py                dataset-recording CLI (same device selection + --dataset.*)
override_reset_pose.py   save the current joints as the per-arm reset pose
default.env              CloudXR device-profile overrides passed to the launcher
```
