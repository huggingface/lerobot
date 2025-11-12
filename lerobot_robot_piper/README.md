# LeRobot Piper Integration

This package provides integration between LeRobot and the Piper robot arm.

## Installation

```bash
pip install -e .
```

## Usage

### Teleoperation

```bash
lerobot-teleoperate \
    --robot.type=piper \
    --robot.can_interface=can0 \
    --robot.bitrate=1000000 \
    --robot.include_gripper=true \
    --robot.use_degrees=false \
    --robot.cameras='{"wrist": {"type": "opencv", "index_or_path": 2, "width": 640, "height": 480, "fps": 30, "fourcc": "MJPG"}}' \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM0 \
    --teleop.use_degrees=false
```

### Recording

```bash
lerobot-record \
    --robot.type=piper \
    --robot.can_interface=can0 \
    --robot.bitrate=1000000 \
    --robot.include_gripper=true \
    --robot.use_degrees=false \
    --robot.cameras='{"wrist": {"type": "opencv", "index_or_path": 4, "width": 640, "height": 480, "fps": 30, "fourcc": "MJPG"}}' \
    --dataset.repo_id=local/piper-test \
    --dataset.root=/path/to/datasets \
    --dataset.single_task="pick and place task" \
    --dataset.num_episodes=10 \
    --dataset.episode_time_s=45 \
    --dataset.reset_time_s=10 \
    --dataset.video=true
```

### ACT Policy Deployment (Async Inference)

First start the policy server:

```bash
python -m lerobot.async_inference.policy_server \
    --host=127.0.0.1 \
    --port=8080 \
    --pretrained_name_or_path=/path/to/trained/model \
    --policy_device=cuda
```

Then run the robot client:

```bash
python -m lerobot.async_inference.robot_client \
    --server_address=127.0.0.1:8080 \
    --robot.type=piper \
    --robot.can_interface=can0 \
    --robot.bitrate=1000000 \
    --robot.include_gripper=true \
    --robot.use_degrees=false \
    --robot.cameras='{"wrist": {"type": "opencv", "index_or_path": 2, "width": 640, "height": 480, "fps": 30, "fourcc": "MJPG"}}' \
    --task="pick green stem above the red strawberries" \
    --policy_type=act \
    --actions_per_chunk=200 \
    --chunk_size_threshold=0.5 \
    --aggregate_fn_name=weighted_average
```

## Configuration

The Piper robot supports the following configuration options:

- `can_interface`: CAN interface to use (default: "can0")
- `bitrate`: CAN bitrate (default: 1000000)
- `joint_names`: Names for the 6 joints (automatically set)
- `joint_signs`: Sign flips for joint directions (default: [-1, 1, 1, -1, 1, -1])
- `joint_aliases`: Mapping from teleoperator joint names to Piper joint names
- `include_gripper`: Whether to expose gripper control (default: False)
- `use_degrees`: Whether to use degrees or radians (default: True)
- `cameras`: Dictionary of camera configurations
- `enable_timeout`: Timeout for Piper SDK enable (default: 5.0 seconds)

## Requirements

- LeRobot >= 0.4.0
- python-can
- piper_sdk
