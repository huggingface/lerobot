# LeRobot - Quick Reference Guide

## Mes répo
**Github Zarax:** https://github.com/Zaraxxx/lerobot  
**HuggingFace Hub Zarax:** https://huggingface.co/Zarax/datasets  
**WanDb Zarax:** https://....

## Documentation
**Documentation HuggingFace LeRobot:** https://huggingface.co/docs/lerobot  
**Github LeRobot:** https://github.com/huggingface/lerobot  
**Discord LeRobot:** https://discord.com/invite/ttk5CV6tUw  

## Robot Configuration

**Robot ID:** zarax
**Follower Port:** COM7
**Leader Port:** COM8

## Setup & Environment
### Initial Setup
```bash
cd C:\GitHub\Zaraxxx\lerobot
conda activate lerobot
```
### Hugging Face Login
```bash
huggingface-cli login
```

<br>

## Calibration
### Calibrate Follower (COM7)
```bash
lerobot-calibrate --robot.type=so101_follower --robot.port=COM7 --robot.id=zarax
```
### Calibrate Leader (COM8)
```bash
lerobot-calibrate --teleop.type=so101_leader --teleop.port=COM8 --teleop.id=zarax
```
---
## Camera Configuration
### Find Available Cameras
```bash
lerobot-find-cameras opencv
# or for Intel Realsense cameras:
lerobot-find-cameras realsense
```
<br>

## Teleoperation

### Teleoperation without Camera
```bash
lerobot-teleoperate `
  --robot.type=so101_follower `
  --robot.port=COM7 `
  --robot.id=zarax `
  --teleop.type=so101_leader `
  --teleop.port=COM8 `
  --teleop.id=zarax
```

### Teleoperation with Camera (Recommended)
```bash
# Using config file
lerobot-teleoperate --config_path .\config\teleop\zarax_teleop_config_camdroite.yaml

# Using CLI
lerobot-teleoperate `
  --robot.type=so101_follower `
  --robot.port=COM7 `
  --robot.id=zarax `
  --robot.cameras="{ camera_0: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}" `
  --teleop.type=so101_leader `
  --teleop.port=COM8 `
  --teleop.id=zarax `
  --display_data=true
```

<br>

## Datasets
### Dataset Locations
#### Local Dataset
```
C:\Users\picip\.cache\huggingface\lerobot\Zarax\
```
#### Hub Dataset Location (Hugging Face website)
https://huggingface.co/Zarax/datasets

### Dataset Commands
#### Delete Local Datasets
```bash
python delete_datasets.py --all
# or
python delete_datasets.py --user Zarax
```

#### Recording Datasets
```bash
# Using config file (Recommended)
lerobot-record --config_path .\config\record\zarax_record_config_camdroite.yaml

# Using CLI - Record 5 episodes
lerobot-record `
  --robot.type=so101_follower `
  --robot.port=COM7 `
  --robot.id=zarax `
  --robot.cameras="{ camera_0: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}" `
  --teleop.type=so101_leader `
  --teleop.port=COM8 `
  --teleop.id=zarax `
  --display_data=true `
  --dataset.repo_id=Zarax/zarax-demo `
  --dataset.num_episodes=5 `
  --dataset.single_task="Simple movement test"
```

#### Edit Datasets (Remove Episodes)
```bash
# Delete episode 6 and create a new dataset
lerobot-edit-dataset `
  --repo_id Zarax/zarax-demo `
  --new_repo_id Zarax/zarax-demo-v2 `
  --operation.type delete_episodes `
  --operation.episode_indices "[6]"
```

#### Replay Dataset Episodes
```bash
# Replay a specific episode to verify it
lerobot-replay `
  --dataset.repo_id=Zarax/zarax-demo `
  --episodes [0] `
  --fps 30
```

<br>

## Training

### Weights & Biases (WandB) Setup
```bash
# Create account at https://wandb.ai/signup
# Login to WandB
wandb login
```

### Training with Config Files (Recommended)
```bash
# Full ACT training with GPU (100K steps, ~5h)
lerobot-train --config_path .\config\training\zarax_train_config_act.yaml

# Simplified training (50K steps, faster for testing)
lerobot-train --config_path .\config\training\zarax_train_config_simple.yaml

# CPU training (SLOW! Only for testing without GPU)
lerobot-train --config_path .\config\training\zarax_train_config_cpu.yaml
```

### Training with CLI Parameters
```bash
# Basic training command
lerobot-train `
  --dataset.repo_id=Zarax/zarax-demo `
  --policy.type=act `
  --policy.repo_id=Zarax/act-zarax-v1 `
  --output_dir=outputs/train/act_zarax_v1 `
  --job_name=act_zarax_v1 `
  --policy.device=cuda `
  --wandb.enable=true

# Without WandB
lerobot-train `
  --dataset.repo_id=Zarax/zarax-demo `
  --policy.type=act `
  --policy.repo_id=Zarax/act-zarax-v1 `
  --output_dir=outputs/train/act_zarax_v1 `
  --job_name=act_zarax_v1 `
  --policy.device=cuda `
  --wandb.enable=false
```

### Resume Training from Checkpoint
```bash
# Resume from last checkpoint
lerobot-train `
  --config_path=outputs/train/act_zarax_v1/checkpoints/last/pretrained_model/train_config.json `
  --resume=true
```

### Monitor Training Progress
```bash
# Python script with percentage and time remaining (Recommended!)
python config/monitor_training.py

# For a different training
python config/monitor_training.py --log-path outputs/train/my_training/train.log --total-steps 50000

# Watch logs in real-time (PowerShell)
Get-Content outputs\train\act_zarax_v1\wandb\run-*\files\output.log -Wait -Tail 10

# Monitor GPU usage
nvidia-smi -l 2

# Check WandB dashboard
# Link appears in terminal after training starts
```

### Training Locations
```
Local output: outputs\train\act_zarax_v1\
Checkpoints: outputs\train\act_zarax_v1\checkpoints\
Logs: outputs\train\act_zarax_v1\wandb\
Trained model: https://huggingface.co/Zarax/act-zarax-v1
```

### Upload Checkpoint Manually
```bash
# Upload latest checkpoint
huggingface-cli upload Zarax/act-zarax-v1 `
  outputs/train/act_zarax_v1/checkpoints/last/pretrained_model

# Upload specific checkpoint (e.g., 20K steps)
huggingface-cli upload Zarax/act-zarax-v1 `
  outputs/train/act_zarax_v1/checkpoints/020000/pretrained_model
```

<br>

## Quick Start: Run Your Trained Model

**The simplest way to test your trained model:**
```bash
.\run_model.bat
```

That's it! One command to run your robot with the trained model.

**What the script does:**
1. Cleans up any previous test dataset automatically
2. Runs `lerobot-record` with the evaluation config
3. Robot operates autonomously using your trained ACT model

**Customization:**
- Edit `config/eval/zarax_eval_simple.yaml` to change settings
- Change checkpoint: modify `pretrained_path` in the config
- Change number of test episodes: modify `num_episodes`

<br>

## Testing the Model (Inference & Evaluation)

### ⭐ THE SIMPLE SOLUTION (Recommended)

**Just run this script:**
```bash
.\run_model.bat
```

**That's it!** The script:
- ✅ Automatically cleans up previous test datasets
- ✅ Runs the robot with your trained model
- ✅ Never uploads to HuggingFace
- ✅ Works every time with the same command

**Files:**
- Script: `run_model.bat`
- Config: `config/eval/zarax_eval_simple.yaml`

### Important: Dataset Name Convention

**CRITICAL**: When testing a trained model, the dataset name MUST start with `eval_` (e.g., `Zarax/eval_zarax`). This is required by LeRobot to distinguish evaluation datasets from training datasets.

### Alternative: Manual Config File Method
```bash
# Using config file manually
# IMPORTANT: Use forward slashes on Windows
lerobot-record --config_path ./config/eval/zarax_eval_config.yaml

# If dataset exists, delete it first:
rm -rf "C:\Users\picip\.cache\huggingface\lerobot\Zarax\eval_zarax"
```

### Test with Local Checkpoint (CLI Method 1 - Simplest)
```bash
# Using --policy.path (LeRobot 0.4.4+ syntax) - VERIFIED WORKING
lerobot-record `
  --robot.type=so101_follower `
  --robot.port=COM7 `
  --robot.id=zarax `
  --robot.cameras="{ camera_0: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}" `
  --policy.path=outputs/train/act_zarax_v1/checkpoints/020000/pretrained_model `
  --display_data=true `
  --dataset.repo_id=Zarax/eval_zarax `
  --dataset.num_episodes=1 `
  --dataset.single_task="Evaluation test"
```

### Test with Local Checkpoint (CLI Method 2 - Explicit)
```bash
# Using --policy.type and --policy.pretrained_path - VERIFIED WORKING
lerobot-record `
  --robot.type=so101_follower `
  --robot.port=COM7 `
  --robot.id=zarax `
  --robot.cameras="{ camera_0: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}" `
  --policy.type=act `
  --policy.pretrained_path=outputs/train/act_zarax_v1/checkpoints/020000/pretrained_model `
  --display_data=true `
  --dataset.repo_id=Zarax/eval_zarax `
  --dataset.num_episodes=1 `
  --dataset.single_task="Evaluation test"
```

### Test with Hugging Face Model
```bash
# Using model from Hugging Face Hub
lerobot-record `
  --robot.type=so101_follower `
  --robot.port=COM7 `
  --robot.id=zarax `
  --robot.cameras="{ camera_0: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}" `
  --policy.path=Zarax/act-zarax-v1 `
  --display_data=true `
  --dataset.repo_id=Zarax/eval_zarax `
  --dataset.num_episodes=1 `
  --dataset.single_task="Evaluation test"
```

### What to Expect
- Robot uses camera + AI model (no leader arm needed)
- Will try to reproduce movements from training episodes
- Performance depends on dataset size (9 episodes = limited generalization)
- Start with simple tasks shown during data collection

### Important Notes & Discoveries

**Running the Robot:**
- Use `run_model.bat` for the simplest experience
- Press Ctrl+C to stop the robot
- The robot uses: camera input → AI model → robot actions (no leader arm needed)

**Why There's No "Inference-Only" Mode:**
- `num_episodes: 0` does NOT work (robot connects then exits immediately)
- LeRobot requires `num_episodes >= 1` to actually run the robot
- This is a known limitation: [GitHub Issue #2227](https://github.com/huggingface/lerobot/issues/2227)
- Solution: Accept that episodes are recorded, use `push_to_hub: false` to keep them local

**Policy syntax differences - CRITICAL**:
- **CLI**: Both `--policy.path=...` AND `--policy.type=act --policy.pretrained_path=...` work
- **YAML config files**: ONLY `policy: type: act` + `pretrained_path: ...` works
- `policy.path` does NOT work in YAML files (parser limitation)

**Windows path issues**:
- Use forward slashes in --config_path: `./config/eval/file.yaml`
- NOT backslashes: `.\config\eval\file.yaml` (will fail)
- HuggingFace repo paths with `/` get converted to `\` causing errors
- Solution: Use local checkpoint paths on Windows

**Other notes:**
- **OLD syntax** `--control.policy.path` is from older docs, doesn't work in v0.4.4
- Dataset name MUST start with `eval_` when using a policy
- `lerobot-eval` is for simulation environments only (LIBERO, MetaWorld)
- For real robots, always use `lerobot-record` with policy parameters
