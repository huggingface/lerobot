# PickLift Ubuntu collection checklist

## Before the session

- [ ] Arm center fixed at the mat bottom 30 cm mark.
- [ ] Task-grid origin, `+X` arrow, and `Y=0` centerline match the geometry
      inherited from `picklift_task_grid_v1` and the MuJoCo grid.
- [ ] Front camera mount, lighting, mat, and object version match the session
      configuration.
- [ ] Pseudonymous operator/session/task/setup values are filled in.
- [ ] The 3×4/12-cell spawn plan is balanced.

## Before every episode

- [ ] Select the planned `spawn_id` and region.
- [ ] Identify the cell from the physical mat's 5×5 cm coarse boundary lines,
      not from the camera image or table edge.
- [ ] Place the object at `X forward=20–35 cm`,
      `Y lateral=−10–+10 cm`.
- [ ] Record actual x/y; do not measure, estimate, or enter a yaw number.
- [ ] Within the selected cell, turn the cube to any 0–90° orientation;
      whenever practical, avoid mechanically repeating the previous one.
- [ ] Confirm only that yaw was varied; this is not an angle annotation.
- [ ] Confirm the UI shows the same spawn ID, region, X/Y, and
      **Yaw arbitrary 0..90 | no measure**.
- [ ] Roughly match the Leader/Follower gripper openings; the first five joints
      are rebased, while the gripper keeps direct calibrated 0–100 control.
- [ ] Operator hands/body are out of the front image.
- [ ] Robot, object, mat, and camera are stable.
- [ ] For alignment confirmation only: place the red cube at
      `(X=25 cm, Y=0)` and verify it in a new canonical screenshot; do not use
      the old 15 cm image or count this boundary point as an episode.
- [ ] Click **START** only after all checks above pass.

## Manual success check before END

- [ ] Cube visibly lifted at least about 5 cm; aim for 6–8 cm.
- [ ] Cube visibly held between both fingers without external support.
- [ ] Successful pose held continuously for 0.5–1 second.
- [ ] Cube still held when manually clicking **END**.

## After every episode

- [ ] Click **SUCCESS**, **FAILURE**, or **DISCARD**.
- [ ] Wait for the immediate `ACCEPTED` feedback, then `SAVING` and
      `SAVED`/`NOT SAVED`; do not click the result twice.
- [ ] SUCCESS is a manual visual annotation, not an automatic detection.
- [ ] Use FAILURE for unmet task criteria; DISCARD only for recording,
      configuration, or safety anomalies.
- [ ] Verify the attempt manifest contains spawn ID, actual x/y,
      `spawn_yaw_deg=null`, yaw provenance, result, termination reason, and
      `saved_to_training`.
- [ ] If failure/discard: retry without changing spawn ID or pose.
- [ ] If success was saved: mark the spawn complete and advance to the next
      balanced spawn.
- [ ] In continuous mode, wait for `READY / CONNECT`; torque has been disabled
      before the next reset.
