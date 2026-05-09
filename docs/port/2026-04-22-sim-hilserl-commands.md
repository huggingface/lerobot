# sim HIL-SERL — end-to-end cmds (manual / binary / SARM)

branch: feature/reward-models-port
sim env: sim_assembling/AssembleBase-v0 (UR10e + 2F85, 3-obj nesting)
teleop: PS4/DualSense via fork's `gamepad` teleop (pygame joystick, 4D delta-XYZ+gripper — matches env)

## IMPORTANT

- **demo recording for sim uses `python -m lerobot.rl.gym_manipulator --mode=record`, NOT `lerobot-record`.** `lerobot-record` is the CLI for physical robots and has a different schema.
- **two JSON types per scenario**:
  - `sim_assembling_<type>_env.json` — env-only cfg. Used for **recording** (gym_manipulator --mode=record).
  - `sim_assembling_<type>_train.json` — full `TrainRLServerPipelineConfig`. Used for **HIL-SERL actor+learner**.
- actor+learner are two processes reading the SAME `_train.json` (gRPC on 127.0.0.1:50051 by default).
- `num_episodes` key in `lerobot-record` ≠ `num_episodes_to_record` in `gym_manipulator --mode=record`. Don't cross-copy.

## prep (once)

```bash
cd /home/dom-iva/github.com/orel/lerobot/lerobot
uv venv --python 3.12 .venv
uv pip install -e '.[hilserl,test]'
uv pip install -e ../simulator_for_IL_RL ../RC10_control
```

### rendering: interactive (default) vs headless

The sim env JSONs default to `render_mode="all"` — opens an interactive MuJoCo viewer window **and** captures camera frames for the pipeline. You need a display (local machine, or SSH with X11 forwarding).

- **Interactive (default)**: do NOT export `MUJOCO_GL`. X server must be reachable via `$DISPLAY`.
- **Headless** (server / CI / over SSH without X): export `MUJOCO_GL=egl` **and** override `--env.render_mode=rgb_array` (viewer is skipped, cameras still render via EGL).
- `render_mode` options (sim): `"rgb_array"` (offscreen-only), `"human"` (viewer-only, no cameras — not usable for record/train), `"all"` (both, recommended for interactive use).

```bash
# headless override example:
export MUJOCO_GL=egl
# then add: --env.render_mode=rgb_array to the record / actor / learner commands.
```

## sanity tests

```bash
MUJOCO_GL=egl uv run pytest tests/envs/test_sim_assembling.py tests/rl/test_sim_*.py -v
```

## scenario 1 — manual (teleop SUCCESS btn as reward)

no classifier, no offline training. reward = press Triangle during sim rollout.

### 1.1 record demos

```bash
uv run python -m lerobot.rl.gym_manipulator \
    --config_path=src/lerobot/rl/sim_assembling_manual_env.json \
    --mode=record \
    --dataset.repo_id=local/sim_assemble_manual \
    --dataset.task=assemble_nesting \
    --dataset.num_episodes_to_record=20 \
    --dataset.push_to_hub=false \
    --dataset.overwrite=true   # reuse after a prior failed attempt; off by default
```

### 1.2 launch HIL-SERL (two terminals)

```bash
# term A — learner
MUJOCO_GL=egl uv run python -m lerobot.rl.learner \
    --config_path=src/lerobot/rl/sim_assembling_manual_train.json

# term B — actor
MUJOCO_GL=egl uv run python -m lerobot.rl.actor \
    --config_path=src/lerobot/rl/sim_assembling_manual_train.json
```

## scenario 2 — binary image classifier (CNN)

### 2.1 collect demos (success + failure)

```bash
# success (press Triangle at end)
MUJOCO_GL=egl uv run python -m lerobot.rl.gym_manipulator \
    --config_path=src/lerobot/rl/sim_assembling_manual_env.json \
    --mode=record \
    --dataset.repo_id=local/sim_cnn_success \
    --dataset.task=assemble_nesting \
    --dataset.num_episodes_to_record=20

# failure (don't press Triangle; let control_time_s expire)
MUJOCO_GL=egl uv run python -m lerobot.rl.gym_manipulator \
    --config_path=src/lerobot/rl/sim_assembling_manual_env.json \
    --mode=record \
    --dataset.repo_id=local/sim_cnn_failure \
    --dataset.task=assemble_nesting \
    --dataset.num_episodes_to_record=20
```

### 2.2 split classifier dataset into train/val

```bash
uv run lerobot-split-reward-dataset \
    --src-repo-id local/sim_cnn_success \
    --val-stride 4
```

### 2.3 train classifier

```bash
uv run lerobot-train-reward-classifier \
    --policy.type=reward_classifier \
    --policy.num_cameras=2 \
    --policy.model_name=helper2424/resnet10 \
    --dataset.repo_id=local/sim_cnn_success-train \
    --output_dir=outputs/sim_assembling_cnn_reward \
    --job_name=sim_assembling_cnn_reward \
    --steps=2000 --batch_size=32
```

### 2.4 launch HIL-SERL w/ CNN reward

Edit `src/lerobot/rl/sim_assembling_cnn_train.json` to point `env.processor.reward_model.pretrained_path` at your checkpoint, then:

```bash
# term A — learner
MUJOCO_GL=egl uv run python -m lerobot.rl.learner \
    --config_path=src/lerobot/rl/sim_assembling_cnn_train.json \
    --env.processor.reward_model.pretrained_path=outputs/sim_assembling_cnn_reward/checkpoints/last/pretrained_model \
    --dataset.repo_id=local/sim_cnn_success

# term B — actor
MUJOCO_GL=egl uv run python -m lerobot.rl.actor \
    --config_path=src/lerobot/rl/sim_assembling_cnn_train.json \
    --env.processor.reward_model.pretrained_path=outputs/sim_assembling_cnn_reward/checkpoints/last/pretrained_model \
    --dataset.repo_id=local/sim_cnn_success
```

## scenario 3 — SARM

### 3.1 collect demos (success + failure, same pattern as 2.1)

Use `local/sim_sarm_success` + `local/sim_sarm_failure` as repo ids (kept separate from CNN's to avoid cross-contamination).

### 3.2 merge + annotate + split (dense_only subtasks)

```bash
uv run lerobot-prepare-sarm-data \
    --success-repo-id local/sim_sarm_success \
    --failure-repo-id local/sim_sarm_failure \
    --output-repo-id local/sim_sarm_combined \
    --val-stride 4
```

### 3.3 train SARM

```bash
uv run lerobot-train \
    --policy.type=sarm \
    --policy.annotation_mode=single_stage \
    --dataset.repo_id=local/sim_sarm_combined-train \
    --output_dir=outputs/sim_assembling_sarm \
    --job_name=sim_assembling_sarm \
    --steps=5000 --batch_size=16
```

### 3.4 relabel offline buffer (reward_mode MUST match online cfg)

```bash
uv run lerobot-relabel-sarm \
    --src-repo-id local/sim_sarm_success \
    --sarm-checkpoint outputs/sim_assembling_sarm/checkpoints/last/pretrained_model \
    --reward-mode delta \
    --task assemble_nesting \
    --device cuda
```

### 3.5 launch HIL-SERL w/ SARM

```bash
# term A — learner
MUJOCO_GL=egl uv run python -m lerobot.rl.learner \
    --config_path=src/lerobot/rl/sim_assembling_sarm_train.json \
    --env.processor.reward_model.pretrained_path=outputs/sim_assembling_sarm/checkpoints/last/pretrained_model \
    --env.processor.reward_model.stats_dataset_repo_id=local/sim_sarm_success \
    --dataset.repo_id=local/sim_sarm_success_sarm_delta

# term B — actor
MUJOCO_GL=egl uv run python -m lerobot.rl.actor \
    --config_path=src/lerobot/rl/sim_assembling_sarm_train.json \
    --env.processor.reward_model.pretrained_path=outputs/sim_assembling_sarm/checkpoints/last/pretrained_model \
    --env.processor.reward_model.stats_dataset_repo_id=local/sim_sarm_success \
    --dataset.repo_id=local/sim_sarm_success_sarm_delta
```

## realtime vs fast mode

- **Recording** (`sim_assembling_*_env.json`) → `env.realtime=true`. Sim paces to `control_hz` (default 20 Hz) so a human can actually teleoperate. Running faster makes manual control impossible — you'd overshoot every target.
- **HIL-SERL training** (`sim_assembling_*_train.json`) → `env.realtime=false`. Sim runs in `mode='fast'` with no per-step sleep, so the actor collects transitions as fast as compute allows. Measured ~91 steps/s @ 20 Hz control (~4.5× real-time) on dev box. For even more: lower `env.fps` (smaller frame_skip = fewer MuJoCo substeps).
- Override per-run via `--env.realtime=true|false` on the command line.

## teleop notes

- JSONs use `teleop.type=gamepad` (fork's `GamepadTeleop`). 4D delta-XYZ + gripper — matches sim env action space exactly.
- DualSense works unchanged: pygame's joystick layer maps it identically to PS4 on Linux.
- Button map (gym-hil/gamepad convention): **R1** held = intervention, **Triangle/Y** = SUCCESS, **Circle/A** = TERMINATE, **Square/X** = RERECORD, **R2/L2** = close/open gripper.
- Fork also ships `ps4_joystick` teleop (wraps rc10_api), but its delta-mode action is 5D (includes yaw). To use it with the sim env, pass `--env.include_yaw_slot=true` (sim env expands to 5D, dyaw is ignored internally).
- Headless CI (no controller): pass `--env.teleop=null` to skip teleop entirely; reward falls back to env-internal (which is 0 for sim_assembling — useful only for pipeline smoke).

## reward-model gotchas (recap)

- `reward_mode` MUST match between SARM offline relabel and online env cfg (both `delta` OR both `binary` OR both `dense`).
- SARM `task` string MUST match training exactly (empty string is valid if trained that way).
- `stats_dataset_repo_id` MUST point to a dataset with populated `meta.stats` — **never** to a `split_dataset` output (those strip stats).
- `MUJOCO_GL=egl` required on headless boxes.
- `env.realtime=true` keeps real-time pacing; `false` unleashes sim speed.

## quick reference — correct CLI names

| operation | command | key arg for episode count |
|---|---|---|
| record sim demos | `python -m lerobot.rl.gym_manipulator --mode=record` | `--dataset.num_episodes_to_record` |
| record real-robot demos | `lerobot-record` (physical only) | `--dataset.num_episodes` |
| split classifier dataset | `lerobot-split-reward-dataset` | `--val-stride` |
| train classifier | `lerobot-train-reward-classifier` | `--steps` |
| prepare SARM data | `lerobot-prepare-sarm-data` | — |
| train SARM | `lerobot-train --policy.type=sarm` | `--steps` |
| relabel SARM | `lerobot-relabel-sarm` | — |
| HIL-SERL learner | `python -m lerobot.rl.learner` | — |
| HIL-SERL actor | `python -m lerobot.rl.actor` | — |
