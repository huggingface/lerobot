# Task1 Real24-only off-center/yaw Eval-v2 pilot

This directory owns the software contract for the 12-trial minimum-difficulty
pilot frozen by research-control commit
`ca62d043d55a70f06c91a9b4ffdfcac16a63d5cd`. It uses the fixed Real24-only
step-100000 checkpoint and copies the exact pose-manifest order; it does not
regenerate positions or select poses from prior successes or failures.

## Frozen protocol

- One model only: Real24-only 100k, model SHA
  `ba233bd25aad5bb63a8a863a8fe1f8ebfdb804547a5add33520c8e1e93f890fb`.
- Twelve named 5 cm cells, one pose per cell, with four quadrants balanced
  3/3/3/3 and yaw 0/45 degrees balanced 6/6.
- The Real camera remains
  `icspring_front_crop_1280x960_to_640x480_v1`. The accepted research
  alignment identity references joint v1 plus Sim camera v6, but does not
  change Real camera bytes.
- Every trial moves to the same frozen Real24 ready pose before and after its
  30-second action window. Arrival tolerance is 3.0 degrees. ACT reset occurs
  after ready arrival and before tick 0.
- The exact official-send engine is reused: `max_relative_target=None`, no
  runner absolute clamp, no five-degree step limiter, and per-tick pacing
  without catch-up bursts.
- Success may occur at any point in the full window and need not persist to the
  final frame. Policy/task failures are not retried. One explicitly linked
  replacement is allowed only for preserved infrastructure-invalid evidence.
- Gripper default opening remains a recorded nuisance. No safe-open command,
  range narrowing, or extra action restriction is added.

## Placement and pre-action evidence

Every trial has one Chinese placement prompt. It names the 5 cm cell, the
near/far and frozen task-grid Y half, and either grid-parallel (0 degrees) or
45-degree orientation. The operator is never asked to type numerical
coordinates.

The first instruction is:

> 请在 r3c3 的5cm格内，把方块中心放在靠机械臂一半、Y正向一半的四分之一区域；方块边与网格线平行（0°）。

The runner extracts canonical video frame 0 as a standalone 640x480 pre-action
PNG. The pinned engine writes this frame before tick-0 inference and before
the first action send. The requested pose remains a nominal manual-placement
instruction, not instrumented measurement truth.

## Software-only verification

From `/home/ubuntu24/Teleop/lerobot`:

```bash
.venv/bin/python -m pytest -q \
  experiments/task1_picklift_real24_offcenter_yaw_eval_v2_pilot_v1/test_evalv2_pilot.py

.venv/bin/python \
  experiments/task1_picklift_real24_offcenter_yaw_eval_v2_pilot_v1/evalv2_pilot.py \
  --software-dry-run --freeze-software-evidence
```

The dry-run uses only fake robot, camera, bus, and policy objects. It does not
enumerate or open serial/camera devices, connect the robot, enable torque,
touch 12 V, or run a policy rollout.

## Hardware gate

Hardware execution is intentionally not part of software preparation. Before
the first future trial, the user must explicitly confirm that the existing
Real camera/grid setup is unchanged, Follower 12 V is on, and the cube matches
the first `r3c3` placement instruction above.

Only after that confirmation may the first command be run:

```bash
.venv/bin/python \
  experiments/task1_picklift_real24_offcenter_yaw_eval_v2_pilot_v1/evalv2_pilot.py \
  --execute-hardware \
  --trial-id evalv2_pilot_r3c3 \
  --operator-confirmed-ready
```
