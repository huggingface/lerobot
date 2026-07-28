# PickLift physical placement protocol v2

This current protocol is frozen as `picklift_spawn_v2`. It applies to new
Ubuntu Leader-to-Follower formal collection. The former 15–25 cm contract
remains immutable as [`picklift_spawn_v1`](./PICKLIFT_PROTOCOL_V1.md);
historical manifests and datasets are not rewritten.

## Task frame and allowed area

- The task frame is frozen as `picklift_task_grid_v1` and is identical in the
  physical setup and MuJoCo.
- Fix the arm/base rotation center at the mat bottom **30 cm** mark. Its
  projection onto the task plane is the origin `(X=0, Y=0)`.
- `+X forward` follows the physical grid from the robot region into the
  workspace, toward the calibrated red-cube location.
- `+Y lateral` follows the perpendicular grid direction designated `+Y`, with
  the same sign as MuJoCo. The grid line through the origin along `+X` is the
  `Y=0` centerline.
- Measure positions using the centers of the mat's **2 cm squares**.
- `spawn_x_cm` is task-grid forward `X`. Allowed range: **10–25 cm**.
- `spawn_y_cm` is task-grid lateral `Y`. Allowed range: **−10–+10 cm**.
- `spawn_yaw_deg` is the object's measured yaw. Allowed range: **0–90°**.
- Record the actual measured values, not only the planned region center.
- Never infer either axis from camera-image horizontal/vertical or from table
  edges.
- The frozen alignment reference is the red cube center at
  `(X=0.15 m, Y=0 m)`.

## Balanced 3×3 regions

Rows increase along `+X` from near to far; columns increase along `+Y` from
negative to positive lateral positions.
Boundary values are assigned to the higher-numbered bin, except the maximum.

| Region | X forward range (cm) | Y lateral range (cm) |
|---|---:|---:|
| `r1c1` | 10–15 | −10–−3.33 |
| `r1c2` | 10–15 | −3.33–3.33 |
| `r1c3` | 10–15 | 3.33–10 |
| `r2c1` | 15–20 | −10–−3.33 |
| `r2c2` | 15–20 | −3.33–3.33 |
| `r2c3` | 15–20 | 3.33–10 |
| `r3c1` | 20–25 | −10–−3.33 |
| `r3c2` | 20–25 | −3.33–3.33 |
| `r3c3` | 20–25 | 3.33–10 |

Sample the nine regions evenly. Within each selected region, randomize the
actual square-center position and yaw in 0–90°.

## Spawn and retry state machine

1. Select the next balanced region and create a stable `spawn_id`.
2. Place the object, then record `spawn_region`, actual `spawn_x_cm`,
   `spawn_y_cm`, and `spawn_yaw_deg`.
3. Remove the operator's hands/body from the front image.
4. Wait until the object, mat, camera, and robot are stable.
5. Confirm the displayed spawn values, then start recording.
6. End the episode and select exactly one result:
   `success`, `failure`, or `discard`.
7. On `failure` or `discard`, keep the same `spawn_id` and the same actual
   pose for the retry.
8. Advance to a new `spawn_id`/pose only after a successful episode has been
   saved.

Failure episodes may remain as bounded evidence with `result=failure`.
Discard episodes are marked `result=discard` and must be excluded from the
deterministic training view.

## Required episode manifest fields

Every episode provenance record includes:

- `spawn_protocol_version=picklift_spawn_v2`
- `task_frame_id=picklift_task_grid_v1`
- the expanded task-frame origin, centerline, axes, units, measurement rule,
  and known `(0.15 m, 0 m)` red-cube reference
- `spawn_id`
- `spawn_region`
- `spawn_x_cm`, `spawn_y_cm`, `spawn_yaw_deg`
- `result` and the consistent boolean `success`
- the existing task, setup, operator/session, robot, camera, timing,
  termination, dropped-frame, and synchronization provenance
