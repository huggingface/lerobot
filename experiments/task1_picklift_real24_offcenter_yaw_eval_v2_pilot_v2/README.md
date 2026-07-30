# Task1 Real24-only off-center/yaw Eval-v2 pilot v2

This directory owns the revised software contract for the 12-trial
minimum-difficulty pilot frozen by research-control commit
`9d220248f5cff7c9eb78837f2636bf979185f01d`. It uses the fixed Real24-only
step-100000 checkpoint and preserves the predecessor pose order, quadrants,
and yaw assignments. Before any hardware trial, the nominal per-axis offset
was revised from 12.5 mm to 15 mm so every cube center lies on the existing
10 mm task-grid lattice.

## Frozen protocol

- One model only: Real24-only 100k, model SHA
  `ba233bd25aad5bb63a8a863a8fe1f8ebfdb804547a5add33520c8e1e93f890fb`.
- Twelve named 5 cm cells, one pose per cell, with four quadrants balanced
  3/3/3/3 and yaw 0/45 degrees balanced 6/6. Every nominal center is 15 mm
  from its cell center along both axes.
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

Every trial has one Chinese placement prompt with an integer-centimetre
task-grid intersection. For yaw 0 the cube edges remain grid-parallel. For
yaw 45 the center stays on the named intersection while the cube rotates
around that center; edge alignment is not claimed.

The first instruction is:

> 请把红色方块中心放在任务网格坐标 x=31cm、y=+4cm 的交点；方块边与网格线平行（0°）。

The runner extracts canonical video frame 0 as a standalone 640x480 pre-action
PNG. The pinned engine writes this frame before tick-0 inference and before
the first action send. The requested pose remains a nominal manual-placement
instruction, not instrumented measurement truth.

## Software-only verification

From `/home/ubuntu24/Teleop/lerobot`:

```bash
.venv/bin/python -m pytest -q \
  experiments/task1_picklift_real24_offcenter_yaw_eval_v2_pilot_v2/test_evalv2_pilot.py

.venv/bin/python \
  experiments/task1_picklift_real24_offcenter_yaw_eval_v2_pilot_v2/evalv2_pilot.py \
  --software-dry-run --freeze-software-evidence
```

The dry-run uses only fake robot, camera, bus, and policy objects. It does not
enumerate or open serial/camera devices, connect the robot, enable torque,
touch 12 V, or run a policy rollout.

## Hardware gate

Hardware execution is intentionally not part of software preparation. Before
the first future trial, the user must explicitly confirm that the existing
Real camera/grid setup is unchanged, Follower 12 V is on, and the cube matches
the first `x=31 cm, y=+4 cm, yaw=0` placement instruction above.

Only after that confirmation may the first command be run:

```bash
.venv/bin/python \
  experiments/task1_picklift_real24_offcenter_yaw_eval_v2_pilot_v2/evalv2_pilot.py \
  --execute-hardware \
  --trial-id evalv2_pilot_v2_r3c3 \
  --operator-confirmed-ready
```
