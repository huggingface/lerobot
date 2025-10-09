#!/bin/bash

# LeRobot SO101 Recording Script
# This script runs the lerobot record command with SO101 leader/follower configuration

# Check if HF_USER environment variable is set or provided as argument
if [ -n "$1" ]; then
    HF_USER="$1"
    echo "Using HF_USER from command line argument: $HF_USER"
elif [ -z "$HF_USER" ]; then
    echo "Error: HF_USER environment variable is not set and no username provided."
    echo "Please either:"
    echo "  1. Set environment variable: export HF_USER=your_huggingface_username"
    echo "  2. Or run with username as argument: ./record_so101.sh your_huggingface_username"
    exit 1
else
    echo "Using HF_USER from environment: $HF_USER"
fi

# Run the lerobot record command
python -m lerobot.record \
    --robot.type=bi_so101_follower \
    --robot.left_arm_port=/dev/ttyACM2 \
    --robot.right_arm_port=/dev/ttyACM3 \
    --robot.id=follower \
    --robot.cameras='{
        left: {"type": "opencv", "index_or_path": 14, "width": 640, "height": 480, "fps": 15},
        right: {"type": "opencv", "index_or_path": 12, "width": 640, "height": 480, "fps": 15},
        top: {"type": "opencv", "index_or_path": 10, "width": 640, "height": 480, "fps": 15},
    }' \
    --teleop.type=bi_so101_leader \
    --teleop.left_arm_port=/dev/ttyACM1 \
    --teleop.right_arm_port=/dev/ttyACM0 \
    --teleop.id=leader \
    --display_data=true \
    --dataset.repo_id=${HF_USER}/transfer-the-block \
    --dataset.num_episodes=50 \
    --dataset.single_task="Transfer the blocks" \

echo "Recording completed!"
