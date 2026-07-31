# Task1 paired Eval-v2 minimal replication

This experiment is a fresh-session, 24-trial replication of the frozen
Real24-only versus Real24+QuestSim48-v7 paired real evaluation.

It preserves the exact 12-pose off-center/yaw Eval-v2 set, 15 mm offsets,
checkpoints, success rule, 30-second official-send policy window, frozen ready
pose, camera input, and action contract. The only intentional change is that
the first model at every pose is reversed relative to the first run.

No first-run trial evidence is reused. The first-run result did not select or
modify any pose or checkpoint.

## Fixed models

- `real24_only`: `ba233bd25aad5bb63a8a863a8fe1f8ebfdb804547a5add33520c8e1e93f890fb`
- `questsim48_v7`: `900fee855150b0809c063bb305d91b98f83c1920152f1d8f30a8d284bef3fe03`

Both checkpoints must retain ACT `chunk_size=n_action_steps=67`, front-only
canonical RGB `[3,480,640]`, state `[6]`, action `[6]`, and checkpoint-saved
ImageNet visual normalization.

## Frozen 24-trial replication order

| Trial | Pose | Model |
|---|---|---|
| t01 | r3c3, X=31 cm, Y=+4 cm, yaw 0° | questsim48_v7 |
| t02 | r3c3, X=31 cm, Y=+4 cm, yaw 0° | real24_only |
| t03 | r3c4, X=31 cm, Y=+6 cm, yaw 45° | real24_only |
| t04 | r3c4, X=31 cm, Y=+6 cm, yaw 45° | questsim48_v7 |
| t05 | r2c4, X=26 cm, Y=+9 cm, yaw 0° | questsim48_v7 |
| t06 | r2c4, X=26 cm, Y=+9 cm, yaw 0° | real24_only |
| t07 | r2c1, X=29 cm, Y=-9 cm, yaw 45° | real24_only |
| t08 | r2c1, X=29 cm, Y=-9 cm, yaw 45° | questsim48_v7 |
| t09 | r2c2, X=29 cm, Y=-1 cm, yaw 0° | questsim48_v7 |
| t10 | r2c2, X=29 cm, Y=-1 cm, yaw 0° | real24_only |
| t11 | r2c3, X=26 cm, Y=+1 cm, yaw 45° | real24_only |
| t12 | r2c3, X=26 cm, Y=+1 cm, yaw 45° | questsim48_v7 |
| t13 | r1c2, X=21 cm, Y=-1 cm, yaw 45° | questsim48_v7 |
| t14 | r1c2, X=21 cm, Y=-1 cm, yaw 45° | real24_only |
| t15 | r1c4, X=24 cm, Y=+9 cm, yaw 45° | real24_only |
| t16 | r1c4, X=24 cm, Y=+9 cm, yaw 45° | questsim48_v7 |
| t17 | r1c1, X=21 cm, Y=-9 cm, yaw 0° | questsim48_v7 |
| t18 | r1c1, X=21 cm, Y=-9 cm, yaw 0° | real24_only |
| t19 | r1c3, X=24 cm, Y=+1 cm, yaw 0° | real24_only |
| t20 | r1c3, X=24 cm, Y=+1 cm, yaw 0° | questsim48_v7 |
| t21 | r3c2, X=34 cm, Y=-4 cm, yaw 45° | questsim48_v7 |
| t22 | r3c2, X=34 cm, Y=-4 cm, yaw 45° | real24_only |
| t23 | r3c1, X=34 cm, Y=-6 cm, yaw 0° | real24_only |
| t24 | r3c1, X=34 cm, Y=-6 cm, yaw 0° | questsim48_v7 |

## Unchanged deployment contract

Each trial moves to the same frozen ready pose through the accepted 3-second,
20 Hz linear official-send trajectory, then observes the existing 3° ready
tolerance. ACT is reset after ready arrival and before tick 0.

Every policy window runs for the full 30 seconds without early success stop or
catch-up bursts. `max_relative_target=None`; there is no runner absolute
clamp, 5° step limiter, safe-open command, or added action wrapper.

## Software-only stage

```bash
python paired_evaluator.py --software-dry-run
pytest -q test_paired_evaluator.py
```

This stage does not inspect serial, camera, robot, torque, or 12 V. Hardware
execution remains blocked until a later explicit user GO.

The first future onsite placement is unchanged:
`r3c3`, cube center `X=31 cm`, `Y=+4 cm`, edges parallel to the grid (`0°`).
