# PickLift physical placement protocol v5

This current protocol is frozen as `picklift_spawn_v5` and
`picklift_collection_v5`. It applies to new Ubuntu Leader-to-Follower formal
collection. The former measured-yaw 12-cell contract remains immutable as
[`picklift_spawn_v4`](./PICKLIFT_PROTOCOL_V4.md); v3, v2, and v1 retain their
historical identities. Historical configs, manifests, episodes, and datasets
are not rewritten.

## Task frame and allowed area

- The task frame is `picklift_task_grid_v2`. Its origin, axes, units,
  centerline, and measurement rule are inherited unchanged from
  `picklift_task_grid_v1`; the version changes only because the alignment
  reference is now a separate versioned object.
- Fix the arm/base rotation center at the mat bottom **30 cm** mark. Its
  projection onto the task plane is the origin `(X=0, Y=0)`.
- `+X forward` follows the physical grid from the robot region into the
  workspace, toward the calibrated red-cube location.
- `+Y lateral` follows the perpendicular grid direction designated `+Y`, with
  the same sign as MuJoCo. The grid line through the origin along `+X` is the
  `Y=0` centerline.
- Use the physical mat's clearly visible **5×5 cm coarse boundary lines** to
  identify each placement cell. The object remains the established 2 cm red
  cube.
- `spawn_x_cm` is task-grid forward `X`. Allowed range: **20–35 cm**.
- `spawn_y_cm` is task-grid lateral `Y`. Allowed range: **−10–+10 cm**.
- `spawn_yaw_deg` is always `null`; the operator does not measure, estimate,
  or enter a numeric yaw.
- Before START, place the cube at any arbitrary orientation intended to be
  within **0–90°**, preferably unlike the immediately previous episode.
- Record actual measured X/Y, but never invent nominal or estimated yaw.
- Never infer either axis from camera-image horizontal/vertical or from table
  edges.
- The alignment-only reference is
  `picklift_red_cube_alignment_v2`: red cube center
  `(X=0.25 m, Y=0 m)`, targeting the center of the canonical 640×480 image.
- That image-center claim is **pending physical confirmation**. It must not be
  inferred from the former 15 cm screenshot.
- `(25 cm, 0 cm)` lies on cell boundaries and is not a randomized episode
  placement or a recommended cell-center point.

## Exact 3×4 coarse-grid regions

Rows increase along `+X` from near to far; columns increase along `+Y` from
negative to positive lateral positions.
Boundary values are assigned to the higher-numbered bin, except the maximum.
Every cell is exactly **5×5 cm**, using these physical coarse-grid edges:

- X edges: `20, 25, 30, 35 cm`
- Y edges: `−10, −5, 0, +5, +10 cm`

| Region | X forward range (cm) | Y lateral range (cm) |
|---|---:|---:|
| `r1c1` | 20–25 | −10–−5 |
| `r1c2` | 20–25 | −5–0 |
| `r1c3` | 20–25 | 0–5 |
| `r1c4` | 20–25 | 5–10 |
| `r2c1` | 25–30 | −10–−5 |
| `r2c2` | 25–30 | −5–0 |
| `r2c3` | 25–30 | 0–5 |
| `r2c4` | 25–30 | 5–10 |
| `r3c1` | 30–35 | −10–−5 |
| `r3c2` | 30–35 | −5–0 |
| `r3c3` | 30–35 | 0–5 |
| `r3c4` | 30–35 | 5–10 |

Sample all twelve cells evenly. Within each selected cell, place the cube
clearly inside the physical coarse boundaries, vary its position, and give it
an arbitrary orientation intended to be within 0–90°. No uniform yaw
distribution is claimed. The first coverage plan supplies a recommended
interior X/Y point only; actual X/Y must still be recorded.

## Spawn and retry state machine

1. Select the next balanced region and create a stable `spawn_id`.
2. Place the object, then record `spawn_region`, actual `spawn_x_cm`, and
   `spawn_y_cm`. Change the orientation without measuring it.
3. Set `yaw_randomization_confirmed=true` only after the arbitrary orientation
   has been placed; this is not an angle annotation.
4. Remove the operator's hands/body from the front image.
5. Wait until the object, mat, camera, and robot are stable.
6. Confirm the displayed spawn values, then start recording.
7. End the episode and select exactly one result:
   `success`, `failure`, or `discard`.
8. On `failure` or `discard`, keep the same `spawn_id` and the same actual
   pose for the retry.
9. Advance to a new `spawn_id`/pose only after a successful episode has been
   saved.

The continuous workflow `picklift_continuous_batch_v1` keeps FAILURE and
DISCARD as bounded attempt provenance but clears their pending frame buffers;
neither enters the deterministic v3 training dataset. Only operator-confirmed
SUCCESS attempts receive dataset episode indices. This stricter training-view
rule does not rewrite earlier v5 pilot datasets.

## Manual success contract

The real Leader–Follower setup has no object-height sensor, bilateral contact
sensor, or automatic success detector. The system cannot announce success at
the moment of grasp. The operator manually ends the episode and annotates the
result.

Select **SUCCESS** only when all conditions are visually satisfied:

1. The red cube is clearly lifted at least approximately 5 cm above its
   initial tabletop height. Aim for 6–8 cm to avoid threshold ambiguity.
2. The cube is visibly held between both gripper fingers, not supported or
   hooked by the table, another robot part, or the environment.
3. This state remains stable for at least 0.5 seconds.
4. The cube has not dropped and is still in the successful pose at END.

After reaching success, hold for 0.5–1 second, manually END, then confirm the
SUCCESS summary. Use FAILURE when task criteria are not met. Use DISCARD only
for recording, configuration, or safety anomalies. Never fabricate
`lift_height_m` or `is_grasped`; both remain null because they are not
measured.

## Required episode manifest fields

Every episode provenance record includes:

- `spawn_protocol_version=picklift_spawn_v5`
- `collection_protocol_version=picklift_collection_v5`
- `task_spec_revision=picklift_taskspec_v2_unmeasured_yaw`
- `task_frame_id=picklift_task_grid_v2`
- `alignment_reference_id=picklift_red_cube_alignment_v2`
- exact X/Y cell edges, 5 cm cell size, and `3×4=12` grid shape
- the expanded task-frame origin, centerline, axes, units, measurement rule,
  plus the separately versioned, pending `(0.25 m, 0 m)` alignment reference
- `spawn_id`
- `spawn_region`
- `spawn_x_cm`, `spawn_y_cm`, and `spawn_yaw_deg=null`
- `yaw_annotation_mode=unmeasured_random`
- `yaw_intended_range_deg=[0,90]`
- `yaw_sampling_method=operator_unmeasured_arbitrary`
- `yaw_distribution_claim=unknown`
- boolean `yaw_randomization_confirmed`
- `success_annotation_source=operator_visual_v1`
- `success_detection_mode=manual_proxy_for_nexus_v1`
- `lift_height_m=null` and `is_grasped=null`
- expanded `success_contract=picklift_manual_success_v1`
- `result` and the consistent boolean `success`
- the existing task, setup, operator/session, robot, camera, timing,
  termination, dropped-frame, and synchronization provenance

The canonical image is the raw evidence of actual object orientation. A later
versioned derived view may estimate yaw/bin with its own method and
confidence, without changing the raw episode. Review yaw coverage after the
12/24-episode pilots; do not claim uniform coverage from this raw annotation.
