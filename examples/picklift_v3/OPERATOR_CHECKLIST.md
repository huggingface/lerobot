# PickLift Ubuntu collection checklist

## Before the session

- [ ] Arm center fixed at the mat bottom 30 cm mark.
- [ ] Task-grid origin, `+X` arrow, and `Y=0` centerline match
      `picklift_task_grid_v1` and the MuJoCo grid.
- [ ] Front camera mount, lighting, mat, and object version match the session
      configuration.
- [ ] Pseudonymous operator/session/task/setup values are filled in.
- [ ] The 3×3 spawn plan is balanced.

## Before every episode

- [ ] Select the planned `spawn_id` and region.
- [ ] Measure along the grid—not the image or table edge.
- [ ] Place the object at `X forward=10–25 cm`,
      `Y lateral=−10–+10 cm`.
- [ ] Record actual square-center x/y and actual yaw=0–90°.
- [ ] Confirm the UI shows the same spawn ID, region, `Xfwd`, `Ylat`, and yaw.
- [ ] Operator hands/body are out of the front image.
- [ ] Robot, object, mat, and camera are stable.
- [ ] Click **START** only after all checks above pass.

## After every episode

- [ ] Click **SUCCESS**, **FAILURE**, or **DISCARD**.
- [ ] Verify the saved episode manifest contains spawn ID, actual x/y/yaw,
      result, and termination reason.
- [ ] If failure/discard: retry without changing spawn ID or pose.
- [ ] If success was saved: mark the spawn complete and advance to the next
      balanced spawn.
