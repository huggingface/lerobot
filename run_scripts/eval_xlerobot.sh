#!/bin/bash

set -euo pipefail

# Usage: ./eval_xlerobot.sh [additional eval_lerobot.py flags]
# Customize via environment variables if needed:
#   POLICY_HOST, POLICY_PORT, LANG_INSTRUCTION, ROBOT_ID

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

POLICY_HOST=${POLICY_HOST:-"0.0.0.0"}
POLICY_PORT=${POLICY_PORT:-5555}
LANG_INSTRUCTION=${LANG_INSTRUCTION:-"Move towards the table, align the left arm with the drink in the robot's basket, grab the drink, and place it on the table."}
ROBOT_ID=${ROBOT_ID:-"xlerobot_eval"}

python "${PROJECT_ROOT}/Isaac-GR00T/examples/SO-100/eval_lerobot.py" \
  --robot.type=xlerobot \
  --robot.id="${ROBOT_ID}" \
  --robot.arms='{
      "left_arm_port": "/dev/ttyACM0",
      "right_arm_port": "/dev/ttyACM1",
      "id": "follower"
  }' \
  --robot.base='{
      "port": "/dev/ttyACM2",
      "wheel_radius_m": 0.05,
      "base_radius_m": 0.125
  }' \
  --robot.mount='{}' \
  --robot.cameras='{
      "left":  {"type": "opencv", "index_or_path": 6, "width": 640, "height": 480, "fps": 15},
      "right": {"type": "opencv", "index_or_path": 8, "width": 640, "height": 480, "fps": 15},
      "top":   {"type": "opencv", "index_or_path": 4, "width": 640, "height": 480, "fps": 15}
  }' \
  --policy_host="${POLICY_HOST}" \
  --policy_port="${POLICY_PORT}" \
  --lang_instruction="${LANG_INSTRUCTION}" \
  --modality_config_path="${PROJECT_ROOT}/Isaac-GR00T/examples/SO-100/xlerobot_modality.json" \
  --action_horizon=20 \
  "$@"
