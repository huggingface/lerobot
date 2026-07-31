# Task1 Real24-only vs QuestSim48-v7 paired Eval-v2

This directory freezes the software-only preparation for a fresh 24-trial
real-robot engineering comparison. It reuses the accepted official-send
engine and the exact frozen 12-pose off-center/yaw Eval-v2 set. No historical
trial result is reused.

## Fixed models

- `real24_only`: fixed 100k model SHA
  `ba233bd25aad5bb63a8a863a8fe1f8ebfdb804547a5add33520c8e1e93f890fb`
- `questsim48_v7`: fixed 100k model SHA
  `900fee855150b0809c063bb305d91b98f83c1920152f1d8f30a8d284bef3fe03`

Both checkpoints must have ACT `chunk_size=n_action_steps=67`, front-only
canonical RGB `[3,480,640]`, state `[6]`, action `[6]`, and checkpoint-saved
ImageNet visual normalization.

## Fixed 24-trial order

| Trial | Pose | Model |
|---|---|---|
| t01 | r3c3, X=31 cm, Y=+4 cm, yaw 0° | real24_only |
| t02 | r3c3, X=31 cm, Y=+4 cm, yaw 0° | questsim48_v7 |
| t03 | r3c4, X=31 cm, Y=+6 cm, yaw 45° | questsim48_v7 |
| t04 | r3c4, X=31 cm, Y=+6 cm, yaw 45° | real24_only |
| t05 | r2c4, X=26 cm, Y=+9 cm, yaw 0° | real24_only |
| t06 | r2c4, X=26 cm, Y=+9 cm, yaw 0° | questsim48_v7 |
| t07 | r2c1, X=29 cm, Y=-9 cm, yaw 45° | questsim48_v7 |
| t08 | r2c1, X=29 cm, Y=-9 cm, yaw 45° | real24_only |
| t09 | r2c2, X=29 cm, Y=-1 cm, yaw 0° | real24_only |
| t10 | r2c2, X=29 cm, Y=-1 cm, yaw 0° | questsim48_v7 |
| t11 | r2c3, X=26 cm, Y=+1 cm, yaw 45° | questsim48_v7 |
| t12 | r2c3, X=26 cm, Y=+1 cm, yaw 45° | real24_only |
| t13 | r1c2, X=21 cm, Y=-1 cm, yaw 45° | real24_only |
| t14 | r1c2, X=21 cm, Y=-1 cm, yaw 45° | questsim48_v7 |
| t15 | r1c4, X=24 cm, Y=+9 cm, yaw 45° | questsim48_v7 |
| t16 | r1c4, X=24 cm, Y=+9 cm, yaw 45° | real24_only |
| t17 | r1c1, X=21 cm, Y=-9 cm, yaw 0° | real24_only |
| t18 | r1c1, X=21 cm, Y=-9 cm, yaw 0° | questsim48_v7 |
| t19 | r1c3, X=24 cm, Y=+1 cm, yaw 0° | questsim48_v7 |
| t20 | r1c3, X=24 cm, Y=+1 cm, yaw 0° | real24_only |
| t21 | r3c2, X=34 cm, Y=-4 cm, yaw 45° | real24_only |
| t22 | r3c2, X=34 cm, Y=-4 cm, yaw 45° | questsim48_v7 |
| t23 | r3c1, X=34 cm, Y=-6 cm, yaw 0° | questsim48_v7 |
| t24 | r3c1, X=34 cm, Y=-6 cm, yaw 0° | real24_only |

## Deployment contract

Every trial uses the same frozen ready pose. Ready and return use the accepted
three-second 20 Hz linear official-send trajectory, then hold the exact target
until the existing 3° observation tolerance is met. ACT is reset only after
ready arrival and before tick 0. Each policy window runs for the full 30
seconds at per-tick pacing with no catch-up burst and no early success stop.

`max_relative_target=None`; the runner adds no absolute clamp, relative step
limiter, safe-open action, or other action wrapper. Per-tick evidence preserves
observation, raw/requested action, and the return value from the official
follower send path.

## Software-only commands

```bash
python paired_evaluator.py --software-dry-run
pytest -q test_paired_evaluator.py
```

`--execute-hardware` remains gated by an explicit later GO plus the existing
operator-confirmation flag. This software-preparation stage does not inspect
serial, camera, robot, or torque.

The next onsite instruction, after a later explicit hardware authorization,
is: keep the original Real camera/grid setup, open Follower 12 V, then place
the red cube center at `r3c3`, `X=31 cm`, `Y=+4 cm`, with cube edges parallel
to the grid (`0°`).

## Completed result

The fresh paired session completed all 24 scored trials. One original `t09`
window had an operator placement mismatch; that evidence remains preserved
and unscored, and its single linked replacement is the scored row.

Canonical-video review agreed with all 24 operator labels:

- `real24_only`: 0/12 reviewed successes; 12 `missed_grasp`
- `questsim48_v7`: 3/12 reviewed successes (`t10`, `t19`, `t22`);
  9 `missed_grasp`
- paired poses: 9 both-failure, 3 QuestSim48-v7-only

This is a single-session engineering result on the frozen 12-pose set, not a
paper-scale causal or generalization estimate. Large videos and tick evidence
remain under the immutable artifact root and are not committed to Git.
