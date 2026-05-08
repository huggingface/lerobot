# Space-Bar Progress ACT Results

Date: 2026-05-08

## Summary

Progress-conditioned ACT was trained locally on the derived blind SO101 space-bar dataset.

- Source dataset: `Carsamba/so101_blind_task2`
- Derived local dataset: `outputs/datasets/so101_blind_task2_progress_v1`
- Derived dataset repo id used locally: `local/so101_blind_task2_progress_v1`
- Policy checkpoint: `outputs/train/act_spacebar_progress/checkpoints/last/pretrained_model`
- Training device: RTX 3070 with `torch 2.10.0+cu128`
- Training status: completed 10000 steps
- Final logged loss: `0.104`
- Offline action check: predictions did not collapse to a constant middle pose

Hugging Face push was not performed because this machine is not logged in to HF:
`LocalTokenNotFoundError: Token is required to call the /whoami-v2 endpoint`.

## Dataset Audit

The original and derived datasets passed the headline checks.

| Check | Result |
| --- | --- |
| Episodes | 20 |
| Frames | 6000 |
| Frames per episode | 300 |
| FPS | 30 |
| Robot type | `so_follower` |
| `observation.state` shape | 6 |
| `action` shape | 6 |
| Added feature | `observation.environment_state`, shape 3 |

State/action names:

```text
shoulder_pan.pos
shoulder_lift.pos
elbow_flex.pos
wrist_flex.pos
wrist_roll.pos
gripper.pos
```

Progress feature values were verified:

| Global frame | Episode | Frame | `observation.environment_state` |
| --- | --- | --- | --- |
| 0 | 0 | 0 | `[0.0, 1.0, 1.0]` |
| 149 | 0 | 149 | `[0.49832776, 0.50167227, 1.0]` |
| 150 | 0 | 150 | `[0.50167227, 0.49832776, -1.0]` |
| 299 | 0 | 299 | `[1.0, 0.0, -1.0]` |
| 300 | 1 | 0 | `[0.0, 1.0, 1.0]` |
| 5999 | 19 | 299 | `[1.0, 0.0, -1.0]` |

## Training Command

Important local note: `uv run --extra training` resynced this Windows venv to CPU-only torch. CUDA was fixed by installing CUDA PyTorch explicitly:

```powershell
uv pip install --reinstall-package torch --reinstall-package torchvision `
  "torch==2.10.0+cu128" "torchvision==0.25.0+cu128" `
  --index-url https://download.pytorch.org/whl/cu128
```

The successful training command used `--no-sync`:

```powershell
uv run --no-sync python -m lerobot.scripts.lerobot_train `
  --dataset.repo_id=local/so101_blind_task2_progress_v1 `
  --dataset.root=outputs/datasets/so101_blind_task2_progress_v1 `
  --policy.type=act `
  --policy.chunk_size=60 `
  --policy.n_action_steps=30 `
  --policy.use_vae=false `
  --batch_size=64 `
  --steps=10000 `
  --eval_freq=0 `
  --save_freq=1000 `
  --policy.device=cuda `
  --policy.push_to_hub=false `
  --wandb.enable=false `
  --num_workers=0 `
  --output_dir=outputs/train/act_spacebar_progress `
  --job_name=act_spacebar_progress
```

## Training Loss

| Step | Loss |
| --- | --- |
| 1000 | 0.251 |
| 2000 | 0.204 |
| 3000 | 0.172 |
| 4000 | 0.153 |
| 5000 | 0.139 |
| 6000 | 0.129 |
| 7000 | 0.122 |
| 8000 | 0.114 |
| 9000 | 0.108 |
| 10000 | 0.104 |

Runtime was about 28 minutes 19 seconds after warmup, around 6 steps/sec.

## Offline ACT Check

Command:

```powershell
uv run --no-sync python scripts/project4/offline_act_progress_check.py `
  --policy-path=outputs/train/act_spacebar_progress/checkpoints/last/pretrained_model `
  --dataset-repo-id=local/so101_blind_task2_progress_v1 `
  --dataset-root=outputs/datasets/so101_blind_task2_progress_v1 `
  --episodes 0 5 10 19 `
  --device=cuda `
  --output-dir=outputs/project4_offline_check
```

| Episode | MAE | RMSE | Mean predicted step delta |
| --- | --- | --- | --- |
| 0 | 1.4763 | 2.6681 | 1.0205 |
| 5 | 0.8763 | 1.7672 | 0.8831 |
| 10 | 1.9017 | 3.3792 | 0.8846 |
| 19 | 1.0062 | 1.9813 | 0.8988 |

Predicted action standard deviations by episode:

| Episode | Std by action dimension |
| --- | --- |
| 0 | `[5.8257, 41.4627, 24.1884, 16.6605, 4.0461, 0.1205]` |
| 5 | `[1.5443, 35.7186, 21.7918, 14.1832, 1.2722, 0.0035]` |
| 10 | `[2.9448, 57.3464, 51.3252, 11.2164, 4.2298, 0.0338]` |
| 19 | `[2.6059, 34.5587, 20.4081, 16.5409, 0.2063, 0.0049]` |

Interpretation: the policy is not outputting a constant middle pose. The arm joint dimensions have substantial trajectory variation, and the predicted step deltas are close to the recorded dataset's mean action step delta from audit (`0.8885`).

## Robot Run Command

First replay the original demo before policy control:

```powershell
uv run --no-sync python -m lerobot.scripts.lerobot_replay `
  --robot.type=so101_follower `
  --robot.port=<robot_port> `
  --dataset.repo_id=Carsamba/so101_blind_task2 `
  --episode=0
```

Then run progress ACT:

```powershell
uv run --no-sync python scripts/project4/rollout_progress_act.py `
  --strategy.type=base `
  --policy.path=outputs/train/act_spacebar_progress/checkpoints/last/pretrained_model `
  --training_dataset_repo_id=local/so101_blind_task2_progress_v1 `
  --training_dataset_root=outputs/datasets/so101_blind_task2_progress_v1 `
  --robot.type=so101_follower `
  --robot.port=<robot_port> `
  --duration=10 `
  --fps=30 `
  --task="press the space bar"
```

The rollout script seeds the robot from episode 0 before starting ACT control and injects the online progress feature from elapsed rollout time.

## Push Status

Local git commit message: `Add progress-conditioned ACT rescue workflow`.

Git push target was the current branch: `mlp-bc-policy`.

Push attempt failed because the current GitHub credentials do not have write access:

```text
remote: Permission to ardacinardemirtas/lerobot.git denied to TitanSmash.
fatal: unable to access 'https://github.com/ardacinardemirtas/lerobot.git/': The requested URL returned error: 403
```

`gh` is not installed on this machine, and no writable `TitanSmash/lerobot` fork remote was found.

Model and dataset artifacts are local under `outputs/`, which is gitignored. To use the checkpoint on another machine, either copy `outputs/train/act_spacebar_progress/checkpoints/last/pretrained_model` or log in to Hugging Face and push:

```powershell
huggingface-cli login

uv run --no-sync python scripts/project4/augment_spacebar_progress_dataset.py `
  --output-repo-id=<hf_user>/so101_blind_task2_progress_v1 `
  --output-root=outputs/datasets/so101_blind_task2_progress_v1 `
  --overwrite `
  --push-to-hub

uv run --no-sync python -m lerobot.scripts.lerobot_train `
  --dataset.repo_id=<hf_user>/so101_blind_task2_progress_v1 `
  --policy.type=act `
  --policy.chunk_size=60 `
  --policy.n_action_steps=30 `
  --policy.use_vae=false `
  --batch_size=64 `
  --steps=10000 `
  --eval_freq=0 `
  --save_freq=1000 `
  --policy.device=cuda `
  --policy.push_to_hub=true `
  --policy.repo_id=<hf_user>/act_spacebar_progress_v1 `
  --wandb.enable=false
```
