# Task1 Eval-v2 pilot continuation: three-second ready/return

This directory owns the continuation contract for trials 5-12 of the
grid-aligned 12-trial pilot. Research-control commit
`e7b628192c9b4d0d1f9ef7f8e9fa9e6417a01874` freezes a three-second
ready/return trajectory after the operator observed that the predecessor's
direct full-target return was too fast. Trials 1-4 and their evidence remain
immutable under the predecessor plan.

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

## Ready/return revision

- The ready target and 3 degree arrival tolerance do not change.
- A non-ready start is interpolated linearly to the frozen target using 60
  requested poses over 3.0 seconds at 20 Hz.
- After interpolation, the exact ready target is held at 20 Hz until the
  existing tolerance is observed.
- Every requested pose still uses official `SO101Follower.send_action`.
- This is a time-parameterized reference trajectory outside the scored
  window. It is not a policy-action clamp or current-relative limiter.
- The fixed checkpoint, pose order, ACT queue reset, tick-0 observation,
  30-second policy window, and success definition are unchanged.

## Placement-invalid evidence

If the operator confirms after a completed window that the cube was placed at
a different pose than the frozen trial, the original video and per-tick
evidence remain immutable and are not scored as a policy failure. The
`mark_placement_invalid.py` helper writes a separate marker bound to the
original evidence SHA. The runner then permits exactly one `--replacement`
using the same frozen pose, model, and order. It refuses a marker created after
an operator success/failure label already exists.

## Placement and pre-action evidence

Every trial has one Chinese placement prompt with an integer-centimetre
task-grid intersection. For yaw 0 the cube edges remain grid-parallel. For
yaw 45 the center stays on the named intersection while the cube rotates
around that center; edge alignment is not claimed.

The first continuation instruction is:

> 请把红色方块中心放在任务网格坐标 x=29cm、y=-1cm 的交点；方块边与网格线平行（0°）。

The runner extracts canonical video frame 0 as a standalone 640x480 pre-action
PNG. The pinned engine writes this frame before tick-0 inference and before
the first action send. The requested pose remains a nominal manual-placement
instruction, not instrumented measurement truth.

## Software-only verification

From `/home/ubuntu24/Teleop/lerobot`:

```bash
.venv/bin/python -m pytest -q \
  experiments/task1_picklift_real24_offcenter_yaw_eval_v2_pilot_v3_return3s/test_evalv2_pilot.py

.venv/bin/python \
  experiments/task1_picklift_real24_offcenter_yaw_eval_v2_pilot_v3_return3s/evalv2_pilot.py \
  --software-dry-run --freeze-software-evidence
```

The dry-run uses only fake robot, camera, bus, and policy objects. It does not
enumerate or open serial/camera devices, connect the robot, enable torque,
touch 12 V, or run a policy rollout.

## Hardware gate

Hardware execution is intentionally not part of software preparation. Before
the first future trial, the user must explicitly confirm that the existing
Real camera/grid setup is unchanged, Follower 12 V is on, and the cube matches
trial 5 at `x=29 cm, y=-1 cm, yaw=0`.

Only after that confirmation may the first command be run:

```bash
.venv/bin/python \
  experiments/task1_picklift_real24_offcenter_yaw_eval_v2_pilot_v3_return3s/evalv2_pilot.py \
  --execute-hardware \
  --trial-id evalv2_pilot_v2_r2c2 \
  --operator-confirmed-ready
```
