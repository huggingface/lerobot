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
      "left_arm_port": "/dev/ttyACM2",
      "right_arm_port": "/dev/ttyACM3",
      "id": "follower"
  }' \
  --robot.base='{
      "port": "/dev/ttyACM4",
      "wheel_radius_m": 0.05,
      "base_radius_m": 0.125
  }' \
  --robot.mount='{}' \
  --robot.cameras='{
      "left":  {"type": "opencv", "index_or_path": 8, "width": 640, "height": 480, "fps": 15},
      "right": {"type": "opencv", "index_or_path": 6, "width": 640, "height": 480, "fps": 15},
      "top":   {"type": "opencv", "index_or_path": 4, "width": 640, "height": 480, "fps": 15}
  }' \
  --teleop.type=xlerobot_leader_gamepad \
  --teleop.arms='{
      "left_arm_port": "/dev/ttyACM0",
      "right_arm_port": "/dev/ttyACM1",
      "id": "leader"
  }' \
  --teleop.base='{
      "joystick_index": 0,
      "max_speed_mps": 0.8,
      "deadzone": 0.15,
      "yaw_speed_deg": 45
  }' \
  --teleop.mount='{}' \
  --dataset.video=true \
  --dataset.repo_id="${HF_USER}/xlerobot-get-altereco" \
  --dataset.num_episodes=20 \
  --dataset.single_task="Use the right arm to pick up the altereco candy and place it in the paper box." \
  --dataset.episode_time_s=300 \
  --dataset.reset_time_s=300 \
  --dataset.push_to_hub=true\
  --dataset.root="/home/yihao/.cache/huggingface/lerobot/yihao-brain-bot/xlerobot-get-altereco/" \
  --display_data=true
echo "Recording complete."
