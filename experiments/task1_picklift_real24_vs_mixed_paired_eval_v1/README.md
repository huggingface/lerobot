# Task1 Real24-only vs mixed ACT paired real evaluation v1

This directory owns the software-frozen 24-trial matched real comparison. It
does not contain a hardware result. Historical evaluation evidence, both
checkpoints, and all training datasets remain immutable.

## Hardware protocol revision v2

The first authorized `t01` attempt preserved in the evidence root never
entered the policy window: all joints reached the frozen ready target closely,
but `elbow_flex` plateaued at a 2.54945 degree observation error for the full
60-second movement timeout. The operator confirmed there was no obstruction.

Revision v2 changes only the ready-pose observation tolerance from 1.0 to 3.0
degrees. It continues to request the exact same frozen Real-24 ready vector.
Both checkpoints, trial order, policy reset/tick-0 ordering, official action
send path, 30-second window, and post-trial return target remain unchanged.
The invalid v1 artifact is immutable; the rerun is linked as `replacement1`.

## Frozen comparison

- Model A: fixed Real24-only ACT step 100000, model SHA
  `ba233bd25aad5bb63a8a863a8fe1f8ebfdb804547a5add33520c8e1e93f890fb`.
- Model B: fixed Real24+Quest-Sim24 ACT step 100000, model SHA
  `e054e682057f09a4653af00a4580da173d3d1658ef5c34244bdbf3ca1a125de5`.
- Twelve fixed 5 cm grid centers; both models run once per cell in the frozen
  alternating order in `evaluation_plan_ready_tolerance3_v2.json`.
- Every trial moves to the same frozen ready pose before and after the
  30-second action window. ACT reset occurs after ready arrival and before
  tick 0.
- The execution engine is the exact `34cc7ac` official-send implementation:
  `max_relative_target=None`, no runner absolute clamp, no five-degree step
  limiter, and per-tick pacing without catch-up bursts.
- Success may occur at any point in the 30-second window. It does not require
  holding the object through the final frame and does not stop the policy
  window early.

## Software-only verification

Run from the LeRobot repository with its existing virtual environment:

```bash
.venv/bin/python -m pytest -q \
  experiments/task1_picklift_real24_vs_mixed_paired_eval_v1/test_paired_evaluator.py

.venv/bin/python \
  experiments/task1_picklift_real24_vs_mixed_paired_eval_v1/paired_evaluator.py \
  --software-dry-run --freeze-software-evidence
```

The software dry-run hashes both checkpoints and the pinned engine/profile,
then exercises all 24 trials through fake robot, camera, bus, and policies. It
does not inspect `/dev`, import the hardware engine, connect a camera or serial
port, enable torque, or perform a rollout.

## Hardware gate

Stop after software preparation. Before any hardware command, the user must be
physically present, turn on Follower 12 V, confirm the workspace and first-cell
placement, and explicitly authorize execution. The future single-trial command
is:

```bash
.venv/bin/python \
  experiments/task1_picklift_real24_vs_mixed_paired_eval_v1/paired_evaluator.py \
  --execute-hardware --trial-id t01 --operator-confirmed-ready
```

The runner enforces the next missing original trial in frozen order. Policy
failures cannot be retried. One `--replacement` is permitted only when the
preserved original evidence reports an infrastructure/runtime invalidation.

After each future trial, write immutable operator and canonical-video review
labels with `annotate_trial.py`. Labels use the “success at any time” contract
and never modify the trial JSON or video.
