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
      "left_arm_port": "/dev/tty.usbmodem5A7C1184421",
      "right_arm_port": "/dev/tty.usbmodem5A7C1235321",
      "id": "follower"
  }' \
  --robot.base='{
      "port": "/dev/tty.usbmodem58FA0957301",
      "wheel_radius_m": 0.05,
      "base_radius_m": 0.125,
      "base_motor_ids": [1, 2, 3]
  }' \
 --robot.mount='{
      "port": "/dev/ttyACM5",
      "pan_motor_id": 0,
      "tilt_motor_id": 1
  }' \
  --robot.cameras='{
      "left":  {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30},
      "right": {"type": "opencv", "index_or_path": 2, "width": 640, "height": 480, "fps": 30},
      "top":   {"type": "opencv", "index_or_path": 1, "width": 640, "height": 480, "fps": 30}
  }' \
  --teleop.type=xlerobot_leader_gamepad \
  --teleop.arms='{
      "left_arm_port": "/dev/tty.usbmodem5AAF2703161",
      "right_arm_port": "/dev/tty.usbmodem5AAF2703681",
      "id": "leader"
  }' \
  --teleop.base='{
      "joystick_index": 0,
      "max_speed_mps": 0.8,
      "deadzone": 0.15,
      "yaw_speed_deg": 45
  }' \
 --teleop.mount='{
      "joystick_index": 0,
      "max_pan_speed_dps": 60.0,
      "max_tilt_speed_dps": 45.0,
      "deadzone": 0.15
  }' \
  --dataset.video=true \
  --dataset.repo_id="${HF_USER}/xlerobot-test" \
  --dataset.num_episodes=20 \
  --dataset.single_task="Use the right arm to pick up the candy and place it in the paper box." \
  --dataset.episode_time_s=300 \
  --dataset.reset_time_s=300 \
  --dataset.push_to_hub=true\
  --dataset.root="/Users/yifengwu/workspace/xlerobot-data/xlerobot-test/" \
  --display_data=true
echo "Recording complete."
