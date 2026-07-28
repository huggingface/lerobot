# PickLift physical placement protocol v1

This protocol is frozen as `picklift_spawn_v1`. It applies to new Ubuntu
Leader-to-Follower collection only. Historical datasets are not rewritten.

## Coordinate frame and allowed area

- Fix the arm center at the mat bottom **30 cm** mark.
- Measure positions using the centers of the mat's **2 cm squares**.
- `spawn_x_cm` is the mat horizontal coordinate, increasing left-to-right when
  looking forward from the robot. Allowed range: **20–40 cm**.
- `spawn_y_cm` is forward distance from the arm center. Allowed range:
  **15–25 cm**.
- `spawn_yaw_deg` is the object's measured yaw. Allowed range: **0–90°**.
- Record the actual measured values, not only the planned region center.

## Balanced 3×3 regions

Rows increase from near to far; columns increase from left to right.
Boundary values are assigned to the higher-numbered bin, except the maximum.

| Region | x range (cm) | y range (cm) |
|---|---:|---:|
| `r1c1` | 20–26.67 | 15–18.33 |
| `r1c2` | 26.67–33.33 | 15–18.33 |
| `r1c3` | 33.33–40 | 15–18.33 |
| `r2c1` | 20–26.67 | 18.33–21.67 |
| `r2c2` | 26.67–33.33 | 18.33–21.67 |
| `r2c3` | 33.33–40 | 18.33–21.67 |
| `r3c1` | 20–26.67 | 21.67–25 |
| `r3c2` | 26.67–33.33 | 21.67–25 |
| `r3c3` | 33.33–40 | 21.67–25 |

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

- `spawn_protocol_version=picklift_spawn_v1`
- `spawn_id`
- `spawn_region`
- `spawn_x_cm`, `spawn_y_cm`, `spawn_yaw_deg`
- `result` and the consistent boolean `success`
- the existing task, setup, operator/session, robot, camera, timing,
  termination, dropped-frame, and synchronization provenance
