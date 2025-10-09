#!/bin/bash

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

lerobot-record  \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=my_awesome_follower_arm \
    --robot.cameras="{ gripper: {type: opencv, index_or_path: 6, width: 640, height: 480, fps: 30}, side: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}}" \
    --display_data=false \
    --dataset.repo_id=${HF_USER}/eval_so101_2 \
    --dataset.single_task="Move the pen" \
    --policy.path=${HF_USER}/act_so101_test