# Unitree G1 — SONIC encoder/decoder whole-body control

This package runs NVIDIA's **SONIC** encoder/decoder on the Unitree G1, in MuJoCo
simulation or on real hardware, driven by a dense **34-D whole-body command** (the
OpenHLM / pi0.5 action layout). It is a pure-Python/ONNX reimplementation of the
reference-tracking half of the SONIC deploy stack (no `gear_sonic`/torch dependency, and
no motion planner): the encoder compresses a reference motion window into a latent token
and the decoder maps that token + proprioception history into 50 Hz joint-position
targets for the robot's PD controller.

## Controllers

Selected with `--robot.controller=<ClassName>`:

| Controller                     | Purpose                                                       |
| ------------------------------ | ------------------------------------------------------------ |
| `SonicWholeBodyController`     | SONIC encoder/decoder driven by a 34-D OpenHLM/pi0.5 command |
| `GrootLocomotionController`    | GR00T locomotion policy                                       |
| `HolosomaLocomotionController` | Holosoma locomotion policy                                    |

The rest of this document covers the SONIC whole-body path.

Each tick the `SonicWholeBodyController` takes a 34-D command (`wb.0.pos … wb.33.pos`) in the OpenHLM
layout:

```
[L-arm(7), L-grip(1), R-arm(7), R-grip(1), L-leg(6), R-leg(6), waist(3),
 root roll/pitch + yaw-rate(3)]
```

The 29 joint targets become the SONIC encode-mode-0 reference (accumulated into a rolling
50-frame trajectory with finite-difference velocities so the encoder's lookahead sees a
real motion sequence), the root roll/pitch set the anchor orientation, and the two grip
scalars can drive the Dex3 hands (see below). On startup the controller **interpolates**
from the robot's measured pose into the policy's commanded target over ~3 s (no snap).

## Requirements

- `onnxruntime` (CPU) **or** `onnxruntime-gpu` (recommended). Verify with:
  ```bash
  python -c "import onnxruntime as ort; print(ort.get_available_providers())"
  ```
- `mujoco` for simulation (`is_simulation=True`).
- The SONIC encoder/decoder ONNX models download automatically from the
  `nvidia/GEAR-SONIC` Hub repo.

## Running a rollout

Drive the G1 with a 34-D VLA policy (OpenHLM / pi0.5) via `lerobot-rollout`:

```bash
lerobot-rollout \
  --strategy.type=base \
  --policy.path=<pi05_openhlm_dir> \
  --robot.type=unitree_g1 \
  --robot.controller=SonicWholeBodyController \
  --robot.is_simulation=true \
  --robot.publish_hands=true \
  --task="<language instruction>" \
  --duration=45 --device=cuda
```

### Cameras

Image-conditioned policies need camera frames. Two options are available without live
cameras:

- **Black frames**: `--robot.empty_cameras='[base, left_wrist, right_wrist]'`.
- **Replay a recorded episode** as the camera feed:
  ```bash
  --robot.replay_camera_parquet=<episode.parquet> \
  --robot.replay_camera_map='{base: head_image_left, left_wrist: left_wrist_image, right_wrist: right_wrist_image}'
  ```

### Hands (Dex3)

`--robot.publish_hands=true` publishes `rt/dex3/{left,right}/cmd` from the two grip
scalars (`wb.7.pos` left, `wb.15.pos` right). The scalar is interpolated between
`hand_open_grip_value` (default 1.0 = open) and `hand_closed_grip_value` (default 0.0 =
closed) and scaled onto `hand_closed_pose` (7 joints:
`thumb_0, thumb_1, thumb_2, middle_0, middle_1, index_0, index_1`). Flip the signs in
`hand_closed_pose` if the fingers curl the wrong way, or raise `hand_kp` for a firmer
grip.

## Observation state

When the whole-body controller is active the robot exposes a 34-D proprio state
(`wb_state.0.pos … wb_state.33.pos`) in the same OpenHLM layout as the action, which the
rollout aggregates into `observation.state` for the policy.
