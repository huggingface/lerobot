#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lerobot

lerobot-record \
    --robot.type=nero_follower \
    --robot.channel=can0 \
    --robot.interface=socketcan \
    --robot.firmeware_version=default \
    --robot.speed_percent=10 \
    --robot.disable_torque_on_disconnect=False \
    --robot.cameras='{ front: {type: opencv, index_or_path: 6, width: 640, height: 480, fps: 30, fourcc: "MJPG"}, side: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30, fourcc: "MJPG"} }' \
    --policy.path=$HOME/models/smolvla_base \
    --dataset.repo_id=yuhang/nero_smolvla_test \
    --dataset.num_episodes=1 \
    --dataset.single_task="pick up the object" \
    --dataset.push_to_hub=false \
    --display_data=true
