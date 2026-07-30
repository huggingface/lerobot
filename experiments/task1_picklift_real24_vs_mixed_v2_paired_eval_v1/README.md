# Task1 Real24-only vs Mixed v2 paired real evaluation

This directory owns a fresh 24-trial matched real-robot comparison. It reuses
the predecessor paired evaluator's final official-send protocol and changes
model B to the controlled Mixed v2 fixed step-100000 checkpoint. Historical
paired trials are not reused.

## Frozen comparison

- Model A: Real24-only fixed100k, model SHA
  `ba233bd25aad5bb63a8a863a8fe1f8ebfdb804547a5add33520c8e1e93f890fb`.
- Model B: Mixed v2 fixed100k, model SHA
  `b7faae880393bdbf5e44ebeaab1f399f732d6ee325be698f999c90eb865cee68`.
- Twelve fixed 5 cm grid centers; both models run once per cell in the
  predecessor's frozen alternating order.
- Every trial moves to the same frozen Real24 ready pose before and after the
  30-second action window. Arrival tolerance is the already validated 3.0
  degrees; the requested target is unchanged.
- ACT reset occurs after ready arrival and before tick 0.
- The execution engine is the exact `34cc7ac` official-send implementation:
  `max_relative_target=None`, no runner absolute clamp, no five-degree step
  limiter, and per-tick pacing without catch-up bursts.
- Success may occur at any point in the full window and need not persist to the
  final frame. Policy/task failures are never retried.

## Software and identity verification

From `/home/ubuntu24/Teleop/lerobot`:

```bash
.venv/bin/python -m pytest -q \
  experiments/task1_picklift_real24_vs_mixed_v2_paired_eval_v1/test_paired_evaluator.py

.venv/bin/python \
  experiments/task1_picklift_real24_vs_mixed_v2_paired_eval_v1/paired_evaluator.py \
  --software-dry-run --freeze-software-evidence

.venv/bin/python \
  experiments/task1_picklift_real24_vs_mixed_v2_paired_eval_v1/preflight_identity.py \
  --freeze
```

The software dry-run uses fake robot, camera, bus, and policies. The identity
snapshot reads only by-id links, frozen files, and device-busy metadata; it
does not open serial/camera devices, enable torque, or send actions.

## Hardware gate

Before `t01`, the user confirms only that Follower 12 V is on, the existing
camera/grid setup is unchanged, and the red block is at the center of `r1c1`.
The single-trial command then uses:

```bash
.venv/bin/python \
  experiments/task1_picklift_real24_vs_mixed_v2_paired_eval_v1/paired_evaluator.py \
  --execute-hardware --trial-id t01 --operator-confirmed-ready
```

The runner enforces the next missing original trial. One explicitly linked
replacement is allowed only for preserved infrastructure-invalid evidence.
Every completed trial records an immutable operator label; canonical-video
review is performed only after all 24 trials are complete.
