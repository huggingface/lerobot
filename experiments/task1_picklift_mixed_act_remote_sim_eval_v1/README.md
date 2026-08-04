# Task1 mixed ACT Remote MuJoCo diagnostic v1

This experiment evaluates the fixed Real24+Quest-Sim24 ACT step-100000
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
export PYTHONPATH=/home/ubuntu24/Teleop/lerobot/src:/home/ubuntu24/Teleop/lerobot/experiments/task1_picklift_mixed_act_remote_sim_eval_v1:/home/ubuntu24/SO101QuestRemote/robot-host
PY=/home/ubuntu24/SO101QuestRemote-runtime/.venv/bin/python

$PY -m pytest -q \
  experiments/task1_picklift_mixed_act_remote_sim_eval_v1/test_mixed_sim.py

$PY experiments/task1_picklift_mixed_act_remote_sim_eval_v1/run_remote_sim.py \
  --phase gate12 \
  --output-dir /home/ubuntu24/Teleop/artifacts/evaluation/task1_picklift_mixed_act_remote_sim_eval_v1/gate12_mixed_act100k_remote_1dfac5_20260729_v1

$PY experiments/task1_picklift_mixed_act_remote_sim_eval_v1/verify_evidence.py \
  --phase gate12 \
  --output-dir /home/ubuntu24/Teleop/artifacts/evaluation/task1_picklift_mixed_act_remote_sim_eval_v1/gate12_mixed_act100k_remote_1dfac5_20260729_v1
```

Only after gate12 is independently verified 12/12 interface-valid does the
authorized runner proceed to the plan's fixed frozen120 output. Each episode
runs all 600 policy ticks and 1500 environment steps, including after confirmed
success.

Historical Real-only `1/120` evidence used a different action-processing
version. It may be shown as non-attributable diagnostic context only; differences
must not be directly attributed to the mixed dataset or checkpoint.
