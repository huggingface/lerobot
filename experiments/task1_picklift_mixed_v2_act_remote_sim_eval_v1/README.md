# Task1 Mixed v2 ACT Remote MuJoCo frozen120 diagnostic

This experiment evaluates the fixed controlled Mixed v2 ACT step-100000
checkpoint through the deployed formal Task1 Nexus adapter. It is a
hardware-free Real-to-Sim diagnostic, not a real-robot result and not a paper
performance estimate.

The runner imports the existing Remote adapter for scene, v5 camera/spawn,
reset, clock, and official success semantics. It does not copy or redefine
those components.

## Action and state boundary

- Policy state is checked only as finite `float32[6]` in the frozen SO-101
  order and dataset units (body degrees, gripper `RANGE_0_100` percent).
- No Follower calibration range check or simulation state projection is used.
- ACT postprocessor output is copied unchanged to `requested_action` and
  passed directly to `adapter.apply_action`.
- `max_relative_target=None`; there is no runner absolute calibration clamp,
  relative clamp, or additional action limit.
- The formal Nexus sink remains responsible for converting dataset units to
  simulator units and clipping to its public environment action space.
  Requested action, actual environment action, and its clip mask are recorded.

## Phases

Run with the Remote runtime Python and EGL:

```bash
export MUJOCO_GL=egl
export PYTHONPATH=/home/ubuntu24/Teleop/lerobot/src:/home/ubuntu24/Teleop/lerobot/experiments/task1_picklift_mixed_v2_act_remote_sim_eval_v1:/home/ubuntu24/SO101QuestRemote/robot-host
PY=/home/ubuntu24/SO101QuestRemote-runtime/.venv/bin/python

$PY -m pytest -q \
  experiments/task1_picklift_mixed_v2_act_remote_sim_eval_v1/test_mixed_sim.py

$PY experiments/task1_picklift_mixed_v2_act_remote_sim_eval_v1/run_remote_sim.py \
  --phase gate12 \
  --output-dir /home/ubuntu24/Teleop/artifacts/evaluation/task1_picklift_mixed_v2_act_remote_sim_eval_v1/gate12_mixed_v2_act100k_remote_1dfac5_20260730_v1

$PY experiments/task1_picklift_mixed_v2_act_remote_sim_eval_v1/verify_evidence.py \
  --phase gate12 \
  --output-dir /home/ubuntu24/Teleop/artifacts/evaluation/task1_picklift_mixed_v2_act_remote_sim_eval_v1/gate12_mixed_v2_act100k_remote_1dfac5_20260730_v1
```

Only after gate12 is independently verified 12/12 interface-valid does the
authorized runner proceed to the plan's fixed frozen120 output. Each episode
runs all 600 policy ticks and 1500 environment steps, including after confirmed
success.

The fixed checkpoint loads its own processor. Startup verifies
`use_imagenet_stats=true`, the preprocessor/stat hashes, and exact saved
ImageNet visual mean/std.

The frozen comparison is Mixed v1 `52/120`, whose immutable evidence is
verified before use. Mixed v2 and Mixed v1 share the Remote runner contract,
but this remains a single-seed, small-sample descriptive simulation diagnostic.
It does not isolate a causal mechanism or establish real-robot/paper performance.

## Completed result

- Gate12: `12/12` interface-valid, `5/12` official success.
- Frozen120: `120/120` interface-valid, `45/120 = 37.50%` official success,
  `72,000` policy ticks, and `180,000` environment steps.
- Mixed v1 frozen120: `52/120 = 43.33%`; the descriptive Mixed v2-minus-v1
  difference is `-7/120`, or `-5.83` percentage points.
- All 120 episodes ended through the frozen `max_steps_reached` rule. Mixed v2
  failures were `75 policy_task_failure`; no interface failure occurred.
- Nexus formal-sink clipping affected `30,624/72,000` policy ticks and
  `31,150/432,000` joint values. The runner did not modify requested actions.
- Evidence root:
  `/home/ubuntu24/Teleop/artifacts/evaluation/task1_picklift_mixed_v2_act_remote_sim_eval_v1`.

These figures passed independent per-tick recomputation and immutable hash
verification. They remain Remote-simulation diagnostics only.
