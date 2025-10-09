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
    --robot.type=bi_so101_follower \
    --robot.left_arm_port=/dev/ttyACM2 \
    --robot.right_arm_port=/dev/ttyACM3 \
    --robot.id=follower \
    --robot.cameras='{
        left: {"type": "opencv", "index_or_path": 14, "width": 640, "height": 480, "fps": 15},
        right: {"type": "opencv", "index_or_path": 12, "width": 640, "height": 480, "fps": 15},
        top: {"type": "opencv", "index_or_path": 10, "width": 640, "height": 480, "fps": 15},
        bot: {"type": "opencv", "index_or_path": 4, "width": 640, "height": 480, "fps": 15}
    }' \
    --display_data=false \
    --dataset.repo_id=${HF_USER}/eval_boxing-the-blocks-1 \
    --dataset.single_task="Stack the blocks" \
    --policy.path=${HF_USER}/act_so101_bi_boxing_block_1