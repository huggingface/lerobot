#!/bin/bash

# Example: ./record_xlerobot.sh your_hf_username
# or export HF_USER=your_hf_username and run without arguments.

if [ -n "$1" ]; then
    HF_USER="$1"
elif [ -z "$HF_USER" ]; then
    echo "Error: HF_USER not provided. Set it via 'export HF_USER=...' or pass as the first argument."
    exit 1
fi
echo "Using HF_USER=${HF_USER}"

lerobot-record \
  --robot.type=xlerobot \
  --robot.arms='{
      "left_arm_port": "/dev/ttyACM0",
      "right_arm_port": "/dev/ttyACM1",
      "id": "follower"
  }' \
  --robot.base='{
      "port": "/dev/ttyACM4",
      "wheel_radius_m": 0.05,
      "base_radius_m": 0.125
  }' \
  --robot.mount='{}' \
  --teleop.type=xlerobot_leader_gamepad \
  --teleop.arms='{
      "left_arm_port": "/dev/ttyACM2",
      "right_arm_port": "/dev/ttyACM3",
      "id": "leader"
  }' \
  --teleop.base='{
      "joystick_index": 0,
      "max_speed_mps": 0.8,
      "deadzone": 0.15,
      "yaw_speed_deg": 45
  }' \
  --teleop.mount='{}' \
  --display_data=false \
  --dataset.repo_id="${HF_USER}/xlerobot-dataset" \
  --dataset.num_episodes=1 \
  --dataset.single_task="xlerobot run" \
  --display_data=true \

echo "Recording complete."
