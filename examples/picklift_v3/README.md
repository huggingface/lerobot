# PickLift v3 real-data collector

This directory is an isolated LeRobot Dataset v3 collection path. It does not
read or modify legacy Windows episodes and it never writes outside the
explicit `dataset_root`.

The frozen training contract is:

- 20 Hz deterministic training view
- `observation.state`: float32 `[6]`, follower state read before the action
- `action`: float32 `[6]`, the clipped action actually returned by
  `SOFollower.send_action`
- joint order: `shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`,
  `wrist_roll`, `gripper`
- `observation.images.front`: RGB uint8 640x480
- no wrist/handeye feature

The frozen physical placement protocol and per-episode procedure are in
[`PICKLIFT_PROTOCOL.md`](./PICKLIFT_PROTOCOL.md) and
[`OPERATOR_CHECKLIST.md`](./OPERATOR_CHECKLIST.md). Start a balanced schedule
from [`spawn_plan.template.json`](./spawn_plan.template.json).
New formal collection uses `picklift_spawn_v3`: `+X` forward is 20–35 cm,
`+Y` lateral remains ±10 cm, and `Y=0` is the base centerline. Its
`picklift_task_grid_v2` inherits the exact origin and axes from
`picklift_task_grid_v1`; the new
`picklift_red_cube_alignment_v2` proposes `(X=0.25 m, Y=0 m)` as the
canonical-image center reference. That image placement remains pending until
the operator places the physical cube at 25 cm and captures a new canonical
frame. The v2 10–25 cm contract and v1 legacy contract remain immutable in
[`PICKLIFT_PROTOCOL_V2.md`](./PICKLIFT_PROTOCOL_V2.md) and
[`PICKLIFT_PROTOCOL_V1.md`](./PICKLIFT_PROTOCOL_V1.md).

The aligned front profile is
`icspring_front_crop_1280x960_to_640x480_v1`: acquire native 1920×1080 RGB
MJPEG at 30 FPS, center-crop `(x=320, y=60, width=1280, height=960)`, preserve
the lens distortion, and resize with OpenCV `INTER_AREA` to the canonical
640×480 RGB training view. Dataset sampling remains exactly 20 FPS. This
implements the camera geometry that was accepted during real/MuJoCo
alignment; the matching MuJoCo reference uses vertical FOV 47°.
See [`CAMERA_PROFILE.md`](./CAMERA_PROFILE.md) for the exact executable
pipeline and the distinction between the accepted MuJoCo FOV and approximate
physical-lens measurement.

The normal command is deliberately fail-closed. `mode=real` additionally
requires `powered_real_run_ack=I_HAVE_COMPLETED_THE_POWERED_SAFETY_CHECK`.
Do not supply that acknowledgement until both arms are mechanically safe,
12 V has intentionally been enabled, the serial-role mapping has been
confirmed, and the first-motion acceptance is authorized.

The validated default is `direct_absolute`, matching the established
Leader-to-Follower workflow: calibrated Leader joint values are sent directly
to the Follower without a relative-step clamp. The program waits at
`CONTROL_READY`; collection begins only after the operator explicitly confirms
that it may start. `relative_rebase` remains available as an optional mode,
and its startup offset is recorded in provenance. In either mode the training
`action` is the command actually returned by `SOFollower.send_action`.

## Ubuntu operator console

Launch from a terminal on the Ubuntu desktop:

```bash
cd /home/ubuntu24/Teleop/lerobot
./examples/picklift_v3/run_ui.sh /path/to/completed-pilot-config.json
```

The window shows the live canonical front image, collection state, elapsed
time, and frame count. Click **START** (or press `S`/Enter) only after setup is
ready. Click **END EPISODE** (or press `E`/Space) to finish early. `Q`/Escape
quits. After the episode, select **SUCCESS**, **FAILURE**, or **DISCARD**.
The collector never starts merely because the window opened.

## No-recording practice

Double-click **SO-101 真机练习** on the Ubuntu desktop. It starts live
Leader-to-Follower control with the canonical front camera preview and clearly
shows `PRACTICE` and `NO DATA RECORDING`. Click **STOP** or press `E`/Space to
finish; torque is disabled on exit. Practice mode never constructs a
`LeRobotDataset` and does not write episodes.

The launcher reads its machine-local device configuration from
`${XDG_CONFIG_HOME:-$HOME/.config}/lerobot/picklift-practice.json` (or
`$PICKLIFT_PRACTICE_CONFIG`). Create it from
`configs/practice.template.json`. Stable serial/camera paths and calibration
IDs stay in that local file and are not committed.

Create an engineering smoke dataset without opening a camera or serial port:

```bash
uv run python -m examples.picklift_v3.record \
  --config examples/picklift_v3/configs/engineering_smoke.json
```

Validate an existing config without touching devices:

```bash
uv run python -m examples.picklift_v3.record \
  --config examples/picklift_v3/configs/pilot.template.json --validate-only
```

Provenance is stored under `provenance/dataset.json` and
`provenance/episodes/episode_XXXXXX.json`. It includes the spawn protocol,
spawn ID, actual x/y/yaw, and result. Operator values must be pseudonymous IDs;
direct identity information is rejected.
