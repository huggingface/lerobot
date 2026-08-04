# Task 1 PickLift real-24 ACT baseline v1

This directory freezes the first real-only ACT engineering baseline for the
Task 1 PickLift paper loop. It is not a paper result and it does not authorize
hardware execution.

The run uses only `observation.state` and `observation.images.front` as inputs
and predicts `action`. It does not use the old dataset, old weights, a wrist
camera, or a hand-eye camera.

The old recipe used 100 action steps at 30 FPS, or about 3.33 seconds. This
20 FPS run uses 67 steps, or 3.35 seconds, to preserve the temporal horizon
without a hyperparameter search.

Smoke command:

```bash
cd /home/ubuntu24/Teleop/lerobot
.venv/bin/lerobot-train \
  --config_path experiments/task1_picklift_real24_act_v1/train_config.json \
  --steps=500 \
  --save_freq=500 \
  --log_freq=20 \
  --output_dir=/home/ubuntu24/Teleop/artifacts/training/task1_picklift_real24_act_v1/smoke_500
```

Full command:

```bash
cd /home/ubuntu24/Teleop/lerobot
.venv/bin/lerobot-train \
  --config_path experiments/task1_picklift_real24_act_v1/train_config.json
```

Checkpoint selection is frozen before evaluation: evaluate only the fixed
100,000-step checkpoint. The 20k, 40k, 60k, and 80k checkpoints are retained
for provenance and diagnosis, not selected after seeing real-robot outcomes.

Offline checkpoint validation:

```bash
cd /home/ubuntu24/Teleop/lerobot
.venv/bin/python \
  experiments/task1_picklift_real24_act_v1/validate_checkpoint.py \
  --output /home/ubuntu24/Teleop/artifacts/training/task1_picklift_real24_act_v1/offline_validation_100k.json
```

`deployment_safety.py` freezes the actual Follower calibration-file hash and
the corresponding six per-joint degree ranges. Any calibration drift, wrong
action shape, or non-finite output fails closed. Finite policy outputs are
clipped per joint before they may be handed to a robot runtime. This helper
must remain in the action path during real evaluation; `max_relative_target`
is an additional rate limit, not a replacement for the calibration clamp.

The 12-trial order, fixed checkpoint, setup identity, success definition, and
failure categories are frozen in `evaluation_plan.json`. Hardware execution is
not authorized by these files.

Dry-run the first frozen trial without touching the camera or serial port:

```bash
cd /home/ubuntu24/Teleop/lerobot
.venv/bin/python \
  experiments/task1_picklift_real24_act_v1/evaluate_real.py \
  --spawn-region r1c1 \
  --follower-port /dev/serial/by-id/<CONFIRM_FOLLOWER_PORT>
```

Only after the user explicitly authorizes real evaluation and turns on the
Follower 12 V supply, add `--execute-hardware`. The script still pauses before
connection, latches the current raw Follower position as the initial goal
before verified torque enable, applies the frozen calibration clamp before
every send, retains `max_relative_target=5.0` as a second rate limit, logs raw,
clipped, and actually sent actions, and disables torque in `finally`.

## Hardware-free Real-to-Remote-Sim diagnostic

`sim_policy_inference.py` is the ACT-owned, hardware-free policy endpoint for
the existing Remote adapter. It accepts only a finite floating state `[6]` and
the active Remote canonical `uint8` RGB front frame `[480,640,3]`. It loads only
the fixed 100k checkpoint and its saved processors, then returns raw,
calibration-clipped, and relative-clipped `sent_action` values.

The ACT side does not import or own the Remote environment, scene, camera,
reset, success, or termination implementation. The Remote adapter contract
must be handed off by the owning task before either the 12-episode interface
gate or the frozen 120-episode diagnostic is started.

No-environment checkpoint smoke:

```bash
cd /home/ubuntu24/Teleop/lerobot
.venv/bin/python \
  experiments/task1_picklift_real24_act_v1/fake_sim_policy_smoke.py \
  --output /home/ubuntu24/Teleop/artifacts/training/task1_picklift_real24_act_v1/fake_sim_policy_smoke.json
```

The rollout plan and evidence schema are frozen in
`sim_experiment_manifest.json`. `summarize_sim_results.py` validates episode
JSONL and reports overall and per-cell diagnostic success rates. These results
never replace the 12 real-robot trials and are not paper-effect evidence.
