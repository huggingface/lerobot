#!/bin/bash

# Run the LeRobot teleoperation command with the XLERobot follower/leader setup
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
      "left":  {"type": "opencv", "index_or_path": 6, "width": 640, "height": 480, "fps": 15},
      "right": {"type": "opencv", "index_or_path": 8, "width": 640, "height": 480, "fps": 15},
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
  --display_data=true

echo "XLERobot teleoperation session ended!"
