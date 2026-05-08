# Project 4: Progress-Conditioned ACT Rescue

This folder contains the code path for rescuing the blind SO101 space-bar imitation-learning task.

The source dataset stays immutable: `Carsamba/so101_blind_task2`. The derived dataset adds `observation.environment_state = [progress, 1 - progress, approach_vs_retract]`, so ACT can disambiguate approach, press, and retract phases without camera input.

## 1. Audit the Original Dataset

```powershell
python scripts/project4/audit_spacebar_dataset.py `
  --repo-id=Carsamba/so101_blind_task2
```

This writes:

- `outputs/project4_audit/spacebar_dataset_audit.json`
- `outputs/project4_audit/spacebar_state_action_trajectories.png`

## 2. Create the Progress Dataset

```powershell
python scripts/project4/augment_spacebar_progress_dataset.py `
  --output-repo-id=<hf_user>/so101_blind_task2_progress_v1 `
  --output-root=outputs/datasets/so101_blind_task2_progress_v1 `
  --overwrite
```

Add `--push-to-hub` after the local audit looks good.

## 3. Train ACT on Brev

```powershell
lerobot-train `
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
  --policy.push_to_hub=false `
  --wandb.enable=false
```

## 4. Offline Check the Checkpoint

```powershell
python scripts/project4/offline_act_progress_check.py `
  --policy-path=outputs/train/<run>/checkpoints/last/pretrained_model `
  --dataset-repo-id=<hf_user>/so101_blind_task2_progress_v1 `
  --episodes 0 5 10 19
```

Watch for a falling train loss, non-trivial predicted action standard deviation, and predicted trajectories that follow the recorded press/retract shape instead of collapsing to one middle pose.

## 5. Robot Rollout

Replay the original demo first:

```powershell
lerobot-replay `
  --robot.type=so101_follower `
  --robot.port=<robot_port> `
  --dataset.repo_id=Carsamba/so101_blind_task2 `
  --episode=0
```

Then run ACT with online progress injection:

```powershell
python scripts/project4/rollout_progress_act.py `
  --strategy.type=base `
  --policy.path=outputs/train/<run>/checkpoints/last/pretrained_model `
  --robot.type=so101_follower `
  --robot.port=<robot_port> `
  --duration=10 `
  --fps=30 `
  --task="press the space bar"
```

The rollout script seeds the robot from episode 0 of the training dataset before starting policy control. If your checkpoint folder does not include `train_config.json`, add:

```powershell
--training_dataset_repo_id=<hf_user>/so101_blind_task2_progress_v1
```
