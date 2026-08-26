# Isaac Teleop → SO-101

Teleoperate an SO-101/SO-100 follower arm — and record LeRobot datasets — with NVIDIA
[Isaac Teleop](https://github.com/NVIDIA/IsaacTeleop). The input device is an **XR (VR)
controller**: its grip pose drives the end-effector through a squeeze-to-engage clutch and
LeRobot's Cartesian IK pipeline, and the analog trigger drives the gripper.

The full narrative guide (how the clutch works, CloudXR setup, headset pairing, tuning, and
troubleshooting) is in the [LeRobot docs](https://huggingface.co/docs/lerobot/isaac_teleop)
(source: `docs/source/isaac_teleop.mdx`). This README is the canonical install and usage
reference.

## Requirements

- Linux workstation (see NVIDIA's
  [system requirements](https://nvidia.github.io/IsaacTeleop/main/references/requirements.html)
  for supported OS/GPU/headset combinations; `isaacteleop` publishes Linux wheels only).
- An SO-101 (or SO-100) follower arm, calibrated with `lerobot-calibrate`.
- A CloudXR-capable headset (e.g. Quest 3, Pico 4, Apple Vision Pro) on the same network.

## Installation

This example lives in the LeRobot repository and is not part of the `lerobot` pip package, so
work from a source checkout. From the repo root:

```bash
# The extras this example uses:
#   isaac-teleop - isaacteleop[cloudxr,retargeters-lite] + kinematics (Placo IK) + scipy
#   feetech      - SO-101 serial motor bus
#   dataset      - dataset recording (record.py)
# huggingface_hub >= 1.5 is needed by the automatic URDF fetch (Buckets API).
uv pip install -e ".[isaac-teleop,feetech,dataset]" "huggingface_hub>=1.5"
```

`isaacteleop` is pinned to the 1.5 line, which is currently a **pre-release** whose wheels
(manylinux only, glibc ≥ 2.35) are published on the NVIDIA index rather than public PyPI, which
carries a source distribution that does not build. `pyproject.toml` declares that index as
`nvidia` and routes only `isaacteleop` to it, so `uv` needs no extra flags; the extra is marked
`sys_platform == 'linux'`, so on macOS/Windows it resolves without `isaacteleop` (and the example
cannot run there). With plain `pip`, which does not read `[tool.uv]`, install it explicitly:

```bash
pip install "isaacteleop[cloudxr,retargeters-lite]~=1.5" \
    --extra-index-url https://pypi.nvidia.com --pre
```

One-time CloudXR EULA (the auto-launch prompts on stdin and would hang on a headless machine):

```bash
python -m isaacteleop.cloudxr --accept-eula
```

## Usage

Run everything from the repo root with `python -m` so the `examples` package resolves.

### Teleoperate

```bash
python -m examples.isaac_teleop_to_so101.teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=so101_follower_arm
```

On startup the script launches the CloudXR runtime (~30 s), prints the workstation IP to enter in
the headset's CloudXR web client, waits for the controllers to stream, slews the arm to a reset
pose (`--reset_to_origin=false` to skip), and then: **hold the squeeze/grip** to engage, move the
controller to drive the arm, pull the trigger to close the gripper. Releasing the squeeze slews the
arm back to the reset pose and re-homes the clutch there, so the next squeeze starts from a known
pose. The SO-101 URDF is fetched automatically from the `lerobot/robot-urdfs` Hugging Face
bucket into the LeRobot cache on first run.

To customize the reset pose: back-drive the arm to the pose you want, then

```bash
python -m examples.isaac_teleop_to_so101.override_reset_pose \
    --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=so101_follower_arm
```

which writes it to `HF_LEROBOT_HOME/reset_poses/<robot.name>/<robot.id>.json`; runs with the same
`--robot.id` use it automatically. It takes the same `--robot.*` arguments as `teleoperate.py`, so
it works for any arm with a profile.

### See the arm, and align to it before engaging

The headset runs Isaac Teleop's `ClutchPreview` — **the same object `Teleop/examples/robot_viz` runs**, not a reimplementation of it. The scene, the drag, the thumbstick tuning, the leader ghost, the safety-harness colours, the phase machine and the engage gate all live in `isaacteleop.viz.robot`; this example builds the graph, hands the preview four objects and drives it once a frame.

Two tools, exactly one drawn at a time:

| phase          | what you see                                                                                                                                                                     |
| -------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **disengaged** | an SO-101 you **drag by hand** — position from the controller, yaw from your wrist, both landing on the jaw. Green when the clutch will latch, grey when it won't.               |
| **engaged**    | the arm vanishes and a **leader gripper appears in your hand**, its jaw driven by your trigger, **amber** while the safety limiter is clamping and **red** while it is refusing. |

The drag preview is not a picture of your SO-101 — its joints are frozen at the library's own home pose and the whole arm is moved as a rigid body. It is where you _aim_ before engaging, and the gate is what stops you engaging badly: `SO101ClutchRetargeter` composes orientation as a delta, so a squeeze taken with your wrist 40° off the pose it latches leaves the arm 40° off your hand for the whole engagement. Green means all three conjuncts hold — wrist inside the band, limiter passing through, and both held for the dwell. A denied squeeze is _owed_, not lost.

**Thumbstick** walks the arm's offset from your hand (a rate, so it holds where you leave it); **B** resets it; **A + stick** trims the yaw. Both print their tuned value on release, in the constant's own form.

```bash
--teleop.robot_twin=false            # no preview, no gate; today's bare teleoperation
--teleop.engage_gate=false           # widen the band to 180 deg (the limiter conjunct stays)
--teleop.engage_enter_deg=20 --teleop.engage_exit_deg=30 --teleop.engage_dwell_s=0.1
--teleop.twin_gl_device=-1           # multi-GPU: the GL context must land on the compositor's card
```

It needs a Linux `isaacteleop` built with `-DBUILD_VIZ=ON`; without one it warns and the loop runs unchanged. The SO-101 scene (upstream MJCF + 17 STLs) is fetched checksum-verified into `~/.cache/isaacteleop/so101-assets/` on first use — `ISAACTELEOP_SO101_ASSETS` points at a pre-populated copy instead. The reBot gets no preview: the arm and ghost are SO-101 geometry, and showing a model of a different robot would be worse than showing none.

#### Two differences from `robot_viz`, both deliberate

**The real arm owns the clutch's home** (`owns_clutch_home=False`). In `robot_viz` the preview is the only arm there is, so it supplies the clutch's home every disengaged frame. Here that would command your SO-101 to wherever you waved the preview, at servo speed, on the engage frame. The home stays where it was: measured forward kinematics after the reset slew.

**The graph runs in the XR anchor frame, and the rebase happens on the way out.** The preview places its arm and ghost through `viz.robot.frames`, which requires the clutch's own frame to be XR; `base_T_anchor` is applied to the commanded pose instead of to the controller. That is _exactly_ equivalent — the clutch's translation is affine in its operands and its rotation is a left-composed delta, so both are equivariant under a rigid change of frame (checked numerically: 0 position error, 2e-16 rotation, over 2000 random SO(3) samples).

One consequence worth knowing: the pose that reaches the IK pipeline is the **rate limiter's** output, not the clutch's raw one. That is what the ghost renders, so commanding anything else would make the ghost a lie. The limits are `robot_viz`'s, and its README is explicit they are chosen for a demo rather than measured against an SO-101.

#### The posture it demands is worth reading before you wear the headset

The wrist the gate asks for follows from the preview's own home pose, and the library logs it at startup — where the tool points, and where the gate wants your hand — warning past 45°. `robot_viz`'s `README.md` has the full derivation and the retune procedure; the constant lives in `viz/robot/so101_ghost.py`.

### Teleoperate — XR controller on another arm

The XR path is not SO-101-specific. Any follower with a `RobotProfile` entry in `common.py` can be
driven from the same loop; the profile carries its URDF, IK frame, reach-derived bounds, reset pose,
gripper endpoints and clutch gain. It assumes the follower reports joints in the same convention
its URDF uses. The reBot B601-RS (RobStride motors, SocketCAN) ships as a profile:

```bash
python -m examples.isaac_teleop_to_so101.teleoperate \
    --robot.type=rebot_b601_follower --robot.motor_family=rs \
    --robot.port=can0 --robot.can_adapter=socketcan \
    --robot.id=rebot_arm
```

Its URDF (plus ~64 MB of meshes) is fetched from
[Seeed-Projects/reBot-Isaacsim](https://github.com/Seeed-Projects/reBot-Isaacsim) into the LeRobot
cache on first run; `REBOT_RS_URDF` points at a local copy instead. Being 6-DOF it tracks commanded
orientation fully (`orientation_weight=1.0`), unlike the 5-DOF SO-101's soft-orientation IK.

A profile may also set `clutch_position_scale`, the controller-to-EE translation gain — the reBot
reaches 0.911 m against the SO-101's 0.545 m, so it runs 1:1 where the device default halves hand
motion. Retune it in the profile, and keep `max_ee_step_m` above the per-frame EE step the new
gain produces — a larger gain means a hand sweep commands proportionally more EE travel per frame.

Before driving a **new** arm under torque, check its profile against the hardware: the reset pose
and bounds are derived from the URDF, and a follower whose joint signs disagree with its URDF
commands a mirrored pose. `rebot_readback.py` checks that against the arm with torque disabled,
and never calls `send_action`.

### Record a dataset

`record.py` takes the same `--robot.*`/`--teleop.*`/loop flags plus `lerobot-record`-style
`--dataset.*` flags:

```bash
python -m examples.isaac_teleop_to_so101.record \
    --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=so101_follower_arm \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
    --dataset.repo_id=<hf_user>/<dataset_name> \
    --dataset.single_task="Pick up the cube" \
    --dataset.num_episodes=3 --dataset.episode_time_s=20 --dataset.reset_time_s=5
```

The clutch is the episode boundary: releasing the squeeze (after engaging it at least once) ends
and saves the episode, then the arm slews back to its reset pose during the reset window and the
clutch is re-homed there. One episode is one uninterrupted squeeze. `--reset_to_origin=false`
disables the slew (startup, declutch and between episodes alike).

Keyboard shortcuts (terminal-first, so they work over SSH): **Right/n** end episode early,
**Left/r** re-record, **Esc/q** stop after the current episode.

Run either script with `--help` for all flags.

## Layout

```
isaac_teleop/            device library: session lifecycle (base.py), XRController (the
                         robot_viz retargeting graph + the preview), configs, and the
                         XR→IK processor step
robot_twin.py            builds the four objects isaacteleop's ClutchPreview binds to
common.py                shared loop infra: device bundle, IK pipeline wiring, reset slew,
                         URDF fetch, twin driving, keyboard listener
teleoperate.py           teleoperation CLI
record.py                dataset-recording CLI (same flags + --dataset.*)
override_reset_pose.py   save the current joints as the per-arm reset pose
default.env              CloudXR device-profile overrides passed to the launcher
```
