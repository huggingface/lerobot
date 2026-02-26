# LeRobot SO101 Bimanual Setup Guide

This guide covers teleoperation and data recording for SO101 bimanual robot arms on macOS.

## Prerequisites

1. Install lerobot following the README installation section
2. Activate the conda environment: `conda activate lerobot`

## Hardware Setup

### USB Hub Configuration (MacBook)

```
left follower + left leader -> hub_1
hub_1 + front camera -> hub_2 -> laptop

right_follower + right leader + hand camera -> hub_3 -> laptop
```

## Step 1: Find Device Ports

Ports change when the computer restarts or USB is unplugged/replugged.

```bash
lerobot-find-port
```

This will list all connected devices with their port paths (e.g., `/dev/tty.usbmodem5AB01799231`).

## Step 2: Find Cameras

```bash
lerobot-find-cameras
```

Check the captured images to identify which index corresponds to which camera:
- Index 0: hand camera
- Index 1: front camera
- Index 2: side camera

## Step 3: Calibration

Calibration files are stored at:
```
~/.cache/huggingface/lerobot/calibration/robots/{robot_type}/{robot_id}.json
~/.cache/huggingface/lerobot/calibration/teleoperators/{teleop_type}/{teleop_id}.json
```

If calibration is needed, it will run automatically on first use.

## Step 4: Test Teleoperation (Single Arm)

Test left arm:
```bash
lerobot-teleoperate \
  --robot.id=left_follower --robot.port=/dev/tty.usbmodem5AB01799231 --robot.type=so101_follower \
  --teleop.id=left_leader --teleop.port=/dev/tty.usbmodem5AB01800861 --teleop.type=so101_leader
```

Test right arm:
```bash
lerobot-teleoperate \
  --robot.id=right_follower --robot.port=/dev/tty.usbmodem5AB01799541 --robot.type=so101_follower \
  --teleop.id=right_leader --teleop.port=/dev/tty.usbmodem5AB01802741 --teleop.type=so101_leader
```

## Step 5: Record Data (Bimanual)

Before recording a new dataset, clear any cached data:
```bash
rm -r ~/.cache/huggingface/lerobot/{HF_USER}/{dataset_name}
```

Record command template:
```bash
lerobot-record \
    --robot.type=bi_so101_follower \
    --robot.left_arm_port={LEFT_FOLLOWER_PORT} \
    --robot.right_arm_port={RIGHT_FOLLOWER_PORT} \
    --robot.id=bimanual_follower \
    --robot.cameras='{
      "hand_cam": {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30},
      "front_cam": {"type": "opencv", "index_or_path": 1, "width": 640, "height": 480, "fps": 30},
      "side_cam": {"type": "opencv", "index_or_path": 2, "width": 640, "height": 480, "fps": 30}
    }' \
    --teleop.type=bi_so101_leader \
    --teleop.left_arm_port={LEFT_LEADER_PORT} \
    --teleop.right_arm_port={RIGHT_LEADER_PORT} \
    --teleop.id=bimanual_leader \
    --dataset.repo_id={HF_USER}/{dataset_name} \
    --dataset.single_task="{task_description}" \
    --dataset.num_episodes={num_episodes} \
    --dataset.episode_time_s={seconds_per_episode} \
    --display_data=true
```

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| `--robot.type` | `so101_follower` (single) or `bi_so101_follower` (bimanual) |
| `--teleop.type` | `so101_leader` (single) or `bi_so101_leader` (bimanual) |
| `--robot.cameras` | JSON config for cameras (type, index, resolution, fps) |
| `--dataset.repo_id` | HuggingFace repo in format `username/dataset_name` |
| `--dataset.single_task` | Description of the task being recorded |
| `--dataset.num_episodes` | Number of episodes to record |
| `--dataset.episode_time_s` | Max time per episode in seconds |
| `--display_data` | Show camera feeds while recording |

## Web UI Development

When implementing a new feature or encountering an unfixed issue in the Web UI, always update `src/lerobot/webui/PROGRESS.md`:
- Add a dated entry under the **Changelog** section describing what was added or what issue was found
- Include which backend/frontend files were affected

## Troubleshooting

### FFmpeg Duplicate Class Warnings
If you see warnings about `AVFFrameReceiver` duplicates, they're harmless. To fix:
```bash
pip uninstall opencv-python
pip install opencv-python-headless
```

### Line Continuation in Shell
Use `\` (backslash) for line continuation, NOT `/` (forward slash).

### Check HuggingFace Login
```bash
huggingface-cli whoami
```
