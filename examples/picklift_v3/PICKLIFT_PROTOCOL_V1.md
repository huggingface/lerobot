# PickLift physical placement protocol v1 (legacy)

This historical contract is frozen as `picklift_spawn_v1`. It remains
available so old manifests retain their original identity. New formal
collection uses `picklift_spawn_v2`; historical datasets are never rewritten.

## Coordinate frame and allowed area

- Arm center at the mat bottom **30 cm** mark.
- Positions measured using the centers of the mat's **2 cm squares**.
- Mat-horizontal `spawn_x_cm`: **20–40 cm**.
- Forward-distance `spawn_y_cm`: **15–25 cm**.
- Object `spawn_yaw_deg`: **0–90°**.

## Legacy balanced 3×3 regions

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

Failure or discard retries keep the same `spawn_id` and exact pose. The plan
advances only after a successful episode is saved.
