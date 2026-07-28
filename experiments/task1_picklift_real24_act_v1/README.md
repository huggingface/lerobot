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
