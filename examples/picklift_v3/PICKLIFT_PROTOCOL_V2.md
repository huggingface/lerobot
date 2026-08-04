# PickLift physical placement protocol v2 (legacy)

This historical contract is frozen as `picklift_spawn_v2`. It remains
available so prior configs and manifests retain their original identity.
New collection uses `picklift_spawn_v3`; old datasets are never rewritten.

## Frozen v2 task frame and range

- `task_frame_id=picklift_task_grid_v1`
- Origin: SO-101 base center projected onto the task plane at the mat bottom
  30 cm mark.
- `+X`: forward along the physical/MuJoCo grid.
- `+Y`: lateral along the perpendicular grid with the MuJoCo sign.
- `Y=0`: base centerline.
- `X forward`: **10–25 cm**.
- `Y lateral`: **−10–+10 cm**.
- Frozen red-cube reference: `(X=0.15 m, Y=0 m)`.

Rows increase along `+X`; columns increase from `−Y` to `+Y`. Failure or
discard retries keep the same `spawn_id` and exact pose. The plan advances
only after a successful episode is saved.
