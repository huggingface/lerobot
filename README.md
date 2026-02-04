# piper_lerobot Dataset v3 Format

[Hugging Face Documentation](http://huggingface.co/docs/lerobot)

## 1. Environment Creation

```bash
uv venv
uv sync
```

## 2. Test Cameras

Use `lerobot-find-cameras` to identify device nodes.

Note: Two cameras cannot be connected to the computer via the same hub, otherwise there may be reading issues.

```bash
sudo apt install guvcview    # Install Guvcview
guvcview --device=/dev/video0  # Test wrist camera
guvcview --device=/dev/video2  # Test ground camera
```

## 3. Connect Robot Arm

"3-7.1:1.0" should be changed to your own CAN port number based on the output.

```bash
conda activate lerobot
bash find_all_can_port.sh
bash can_activate.sh can_master 1000000 "1-8.2:1.0"
bash can_activate.sh can_follower 1000000 "1-8.3:1.0"
```

## 3.5 Hardware Teleoperation Setup

Our experimental setup consists of **four Piper robots**, divided into two groups (Left Arm Group and Right Arm Group), each containing one Leader and one Follower.

- **Hardware Connection**: Each pair of Leader and Follower robots are connected via a CAN bus.
- **Teleoperation Principle**: Utilizing the hardware-level teleoperation feature provided by the Piper SDK, the Leader and Follower can communicate directly for control once configured in Master/Slave mode, without requiring PC computation.
- **Data Collection**: The Follower robots are connected to the PC via USB. During data recording, the PC reads the state data from the Follower (Slave) while the operator manually manipulates the Leader to demonstrate actions.
- **Inference & Replay**: For model inference or replay, we **cut the power to the Leader robots**. The PC then sends control signals directly to the Follower robots via USB to execute actions.

For more details on the SDK, please refer to the official documentation and API functions.

## 3.6 Software Teleoperation Setup (4 CAN Ports)

If you don't want to use the SDK's hardware master-slave mode, you can use the `piper_dual_teleop` plugin for **software-level teleoperation**. This requires 4 independent CAN ports:

```bash
# Activate 4 CAN ports
bash can_activate.sh can_left_leader 1000000 "<usb_port1>"
bash can_activate.sh can_left_follower 1000000 "<usb_port2>"
bash can_activate.sh can_right_leader 1000000 "<usb_port3>"
bash can_activate.sh can_right_follower 1000000 "<usb_port4>"
```

**Software Teleoperation Workflow**:

- **During data collection**: Software reads Leader joint positions → writes to Follower joints
- **During inference/replay**: Set `use_teleop=false`, only 2 Follower CAN ports needed

## 4. Teleoperation

```bash
lerobot-teleoperate \
    --robot.type=piper_follower \
    --robot.id=my_follower_arm \
    --teleop.type=piper_leader \
    --teleop.id=my_leader_arm \
    --display_data=true
```

## 5. Login to Hugging Face

Set domestic mirror acceleration (if in China)

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

Add your token to the CLI by running this command:

```bash
hf auth login --token ${HUGGINGFACE_TOKEN} --add-to-git-credential
```

Verify login

```bash
HF_USER=$(hf auth whoami | head -n 1)
echo $HF_USER
```

Upload dataset to Hugging Face

```bash
hf upload jokeru/pick_and_place ~/.cache/huggingface/lerobot/jokeru/pick_and_place \
  --repo-type dataset \
  --revision "v3.0"
```

## 6. Data Collection

Record teleoperation data from the leader arms to the follower robot.

```bash
# Record a new episode
# Ensure cameras are connected.
uv run lerobot-record \
  --robot.type=piper_dual \
  --robot.left_port=can_left \
  --robot.right_port=can_right \
  --robot.read_only=true \
  --robot.cameras='{
      "left": {
        "type": "opencv",
        "index_or_path": "/dev/video12",
        "width": 640,
        "height": 480,
        "fps": 30,
        "rotation": 0
      },
      "right": {
        "type": "opencv",
        "index_or_path": "/dev/video4",
        "width": 640,
        "height": 480,
        "fps": 30,
        "rotation": 0
      },
      "middle": {
        "type": "opencv",
        "index_or_path": "/dev/video6",
        "width": 640,
        "height": 480,
        "fps": 30,
        "rotation": 0
      }
    }' \
  --dataset.repo_id=local/lerobot_new_dataset \
  --dataset.num_episodes=50 \
  --dataset.episode_time_s=30 \
  --dataset.single_task="Dual arm manipulation task." \
  --display_data=true \
  --dataset.push_to_hub=false
```

_Note: Adjust `episode_time_s` to match your task length since you cannot use keyboard shortcuts in headless mode._

### Software Teleop Recording (4 CAN Ports)

Using `piper_dual_teleop` plugin, software reads Leader positions and writes to Followers:

```bash
uv run lerobot-record \
  --robot.type=piper_dual_teleop \
  --robot.left_leader_port=can_left_leader \
  --robot.left_follower_port=can_left_follower \
  --robot.right_leader_port=can_right_leader \
  --robot.right_follower_port=can_right_follower \
  --robot.use_teleop=true \
  --robot.cameras='{"left":{"type":"opencv","index_or_path":"/dev/video4","width":640,"height":480,"fps":30},"right":{"type":"opencv","index_or_path":"/dev/video12","width":640,"height":480,"fps":30},"middle":{"type":"opencv","index_or_path":"/dev/video6","width":640,"height":480,"fps":30}}' \
  --dataset.repo_id=local/dual_teleop_dataset \
  --dataset.num_episodes=50 \
  --dataset.single_task="Dual arm manipulation task." \
  --display_data=true \
  --dataset.push_to_hub=false
```

### Other optional parameters:

```
  --dataset.episode_time_s=60 Duration of each episode recording (default 60 seconds), can be ended early.
  --dataset.reset_time_s=60 Duration to reset the environment after each episode (default 60 seconds).
  --dataset.num_episodes=50 Total number of episodes to record (default 50).
```

Data will be saved to ~/.cache/huggingface/lerobot/jokeru

Use keyboard control during recording

### Control data collection using keyboard shortcuts

Press Right Arrow (→): Stop current event early, or reset time, then switch to the next one.

Press Left Arrow (←): Cancel current event and re-record.

Press ESC: Stop session immediately, encode video, and upload dataset.

### Merge Datasets

```bash
# Merge multiple datasets (requires all dataset features to be identical)
HF_HUB_OFFLINE=1 lerobot-edit-dataset \
  --repo_id jokeru/pick_and_place \
  --operation.type merge \
  --operation.repo_ids "['jokeru/record_apple', 'jokeru/record_banana','jokeru/record_watermelon','jokeru/record_tape']" \
  --push_to_hub false
```

### Delete episodes from Dataset

```
HF_LEROBOT_HOME=$HOME/.cache/huggingface/lerobot uv run lerobot-edit-dataset \
  --repo_id local/lerobot_new_dataset \
  --new_repo_id local/lerobot_new_dataset_filtered \
  --operation.type delete_episodes \
  --operation.episode_indices "[2]"
```

### Output the number episodes

```
HF_LEROBOT_HOME=$HOME/.cache/huggingface/lerobot uv run python -c "from lerobot.datasets.lerobot_dataset import LeRobotDataset; ds = LeRobotDataset('local/lerobot_new_dataset_filtered'); print(f'Episodes: {ds.meta.total_episodes}')"
```

## 7. Dataset Visualization

Verify the recorded data (cameras and joint positions) using Rerun.
This will open a Rerun window where you can inspect `observation/images` and `observation/state`.

```bash
# Visualize Episode 0
uv run lerobot-dataset-viz --repo-id local/lerobot_new_dataset --root ~/.cache/huggingface/lerobot/local/lerobot_new_dataset --episode-index 0
```

### MuJoCo Visualization

Visualize dataset episodes using MuJoCo:

```bash
uv run lerobot-dataset-viz-mujoco --dataset local/lerobot_new_dataset --episode 0
```

Visualize action sequence from NPZ file using MuJoCo:

```bash
uv run python src/lerobot/scripts/visualize_npz_piper.py ~/Downloads/action_chunks.npz --key action
```

## Replaying Episode

```bash
lerobot-replay \
    --robot.type=piper_dual \
    --robot.left_port=can_left \
    --robot.right_port=can_right \
    --dataset.repo_id=local/lerobot_new_dataset \
    --robot.cameras='{"left":{"type":"opencv","index_or_path":"/dev/video10","width":640,"height":480,"fps":30,"rotation":0},"right":{"type":"opencv","index_or_path":"/dev/video4","width":640,"height":480,"fps":30,"rotation":0},"middle":{"type":"opencv","index_or_path":"/dev/video12","width":640,"height":480,"fps":30,"rotation":0}}' \
    --dataset.episode=0
```

### Software Teleop Replay (2 CAN Ports)

```bash
uv run lerobot-replay \
    --robot.type=piper_dual_teleop \
    --robot.left_follower_port=can_left_follower \
    --robot.right_follower_port=can_right_follower \
    --robot.use_teleop=false \
    --robot.cameras='{"left":{"type":"opencv","index_or_path":"/dev/video10","width":640,"height":480,"fps":30},"right":{"type":"opencv","index_or_path":"/dev/video4","width":640,"height":480,"fps":30},"middle":{"type":"opencv","index_or_path":"/dev/video12","width":640,"height":480,"fps":30}}' \
    --dataset.repo_id=local/dual_teleop_dataset \
    --dataset.episode=0
```

## 8. Disable All

```bash
python utils/teleop_disable.py
```

## 9. ACT

### Training

Train an ACT policy on the newly collected dataset.

```bash
# Train command
uv run lerobot-train \
  --policy.type=act \
  --dataset.repo_id=local/lerobot_pick_and_place \
  --output_dir=outputs/train/lerobot_pick_and_place_100 \
  --job_name=act_piper \
  --wandb.mode=offline \
  --policy.push_to_hub=false \
  --policy.chunk_size=100 \
  --steps=100000 \
  --save_freq=10000
```

### Upload model or checkpoints to Hugging Face

Upload model

```bash
hf upload jokeru/pick_and_place ~/.cache/huggingface/lerobot/jokeru/pick_and_place \
  --repo-type model \
  --revision "main"
```

### Testing on Real Robot

As requested, use `lerobot-record` to test the policy (Inference). This will run the policy and record the result.

```bash
# Test policy using lerobot-record
uv run lerobot-record \
  --robot.type=piper_dual \
  --robot.left_port=can_left \
  --robot.right_port=can_right \
  --robot.cameras='{"left":{"type":"opencv","index_or_path":"/dev/video10","width":640,"height":480,"fps":30,"rotation":0},"right":{"type":"opencv","index_or_path":"/dev/video4","width":640,"height":480,"fps":30,"rotation":0},"middle":{"type":"opencv","index_or_path":"/dev/video12","width":640,"height":480,"fps":30,"rotation":0}}' \
  --dataset.repo_id=local/eval_recording_test \
  --dataset.num_episodes=2 \
  --policy.type=act \
  --policy.pretrained_path=/home/droplab/workspace/lerobot_piper/outputs/train/lerobot_pick_and_place_50/checkpoints/last/pretrained_model \
  --dataset.single_task="Dual arm evaluation task" \
  --display_data=true \
  --dataset.push_to_hub=false
```

### Software Teleop Inference (4 CAN → 2 CAN)

Inference with `use_teleop=false`, only 2 Follower CAN ports needed:

```bash
uv run lerobot-record \
  --robot.type=piper_dual_teleop \
  --robot.left_follower_port=can_left_follower \
  --robot.right_follower_port=can_right_follower \
  --robot.use_teleop=false \
  --robot.cameras='{"left":{"type":"opencv","index_or_path":"/dev/video4","width":640,"height":480,"fps":30},"right":{"type":"opencv","index_or_path":"/dev/video12","width":640,"height":480,"fps":30},"middle":{"type":"opencv","index_or_path":"/dev/video6","width":640,"height":480,"fps":30}}' \
  --dataset.repo_id=local/eval_test \
  --dataset.num_episodes=2 \
  --policy.type=act \
  --policy.pretrained_path=<path_to_model> \
  --dataset.single_task="Dual arm evaluation task" \
  --display_data=true \
  --dataset.push_to_hub=false
```

_(Note: You can also use `lerobot-eval` for pure evaluation without recording if desired, but this matches your request to use `lerobot-record`)_

## 10. OpenPi

### Environment Installation

Install lerobot pi dependencies

```bash
pip install -e ".[pi]"
```

### Training

```bash
python src/lerobot/scripts/lerobot_train.py\
    --dataset.repo_id=jokeru/record2 \
    --policy.type=pi05 \
    --output_dir=./outputs/pi05_training \
    --job_name=pi05_training \
    --policy.repo_id=jokeru/pi05 \
    --policy.pretrained_path=lerobot/pi05_libero \
    --policy.compile_model=true \
    --policy.gradient_checkpointing=true \
    --wandb.enable=false \
    --policy.dtype=bfloat16 \
    --steps=3000 \
    --policy.device=cuda \
    --batch_size=32
```

pi05_base or pi05_libero will be downloaded to e.g. ~/.cache/huggingface/hub/models--lerobot--pi05_base

### Multi-GPU Training

Can be tested using tests/training/test_multi_gpu.py

Requires installing pytest dependency first

```bash
pip install pytest
```

```bash
nohup accelerate launch --num_processes=8 \
  src/lerobot/scripts/lerobot_train.py\
    --dataset.repo_id=jokeru/record2 \
    --policy.type=pi05 \
    --output_dir=./outputs/pi05_training \
    --job_name=pi05_training \
    --policy.repo_id=jokeru/pi05 \
    --policy.pretrained_path=lerobot/pi05_libero \
    --policy.compile_model=true \
    --policy.gradient_checkpointing=true \
    --wandb.enable=false \
    --policy.dtype=bfloat16 \
    --steps=3000 \
    --policy.device=cuda \
    --batch_size=32 > outputs/pi05_training.log 2>&1 &
```

### Local Inference

#### RTC

Pretrained

```bash
python examples/rtc/eval_with_real_robot.py \
  --policy.path=lerobot/pi05_base \
  --robot.type=piper_follower \
  --robot.cameras='{
    "wrist": {
      "type": "opencv",
      "index_or_path": "/dev/video0",
      "width": 480,
      "height": 640,
      "fps": 30,
      "rotation": -90
    },
    "ground": {
      "type": "opencv",
      "index_or_path": "/dev/video2",
      "width": 480,
      "height": 640,
      "fps": 30,
      "rotation": 90
    }
  }' \
  --task="Pick up it and put it into the basket." \
  --duration=120 \
  --action_queue_size_to_get_new_actions=30 \
  --fps=50 \
  --rtc.execution_horizon=5 \
  --display_data=true \
  --device=cuda
```

```bash
python examples/rtc/eval_with_real_robot.py \
  --policy.path=jokeru/pi05_pick_and_place \
  --robot.type=piper_follower \
  --robot.cameras='{
    "wrist": {
      "type": "opencv",
      "index_or_path": "/dev/video0",
      "width": 480,
      "height": 640,
      "fps": 30,
      "rotation": -90
    },
    "ground": {
      "type": "opencv",
      "index_or_path": "/dev/video2",
      "width": 480,
      "height": 640,
      "fps": 30,
      "rotation": 90
    }
  }' \
  --task="Pick up it and put it into the basket." \
  --duration=120 \
  --action_queue_size_to_get_new_actions=30 \
  --fps=50 \
  --rtc.execution_horizon=5 \
  --display_data=true \
  --device=cuda
```

## 11. Async Inference (Insufficient local VRAM)

### Installation

```bash
pip install -e ".[async]"
```

### Enable Remote Inference Server

Use CUDA_VISIBLE_DEVICES to set free GPU for inference, otherwise GPU0 is used by default

```bash
CUDA_VISIBLE_DEVICES=1 python -m src.lerobot.async_inference.policy_server \
    --host=127.0.0.1 \
    --port=8080 \
    --fps=30 \
    --inference_latency=0.033 \
    --obs_queue_timeout=1
```

### Establish Port Forwarding if Port Not Open

Establish port forwarding on the client side, forwarding local port 8080 to remote server port 8080 via SSH to access services running on the server

```bash
ssh -L 8080:127.0.0.1:8080 server_username@server_address -N
```

Verify port forwarding is successful

```bash
nc -zv 127.0.0.1 8080
```

### Client Access

```bash
python -m src.lerobot.async_inference.robot_client \
    --server_address=127.0.0.1:8080 \
    --robot.type=piper_follower \
    --robot.cameras='{"wrist": {"type": "opencv", "index_or_path": "/dev/video6", "width": 480, "height": 640, "fps": 30, "rotation": 90}, "ground": {"type": "opencv", "index_or_path": "/dev/video0", "width": 480, "height": 640, "fps": 30, "rotation": -90}}' \
    --task="Pick up the apple and put it into the basket." \
    --policy_type=pi05 \
    --pretrained_name_or_path=jokeru/pi05_apple \
    --policy_device=cuda \
    --actions_per_chunk=50 \
    --chunk_size_threshold=0.5 \
    --aggregate_fn_name=weighted_average \
    --debug_visualize_queue_size=True
```
