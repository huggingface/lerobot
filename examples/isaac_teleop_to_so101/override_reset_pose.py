#!/usr/bin/env python

# Copyright 2026 NVIDIA Corporation and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Save the current SO-101 joint positions as the reset-origin pose (override).

Move the arm to the desired reset pose by hand (torque off), then run this script to write
those joints to a per-arm file in the LeRobot cache. ``teleoperate.py`` / ``record.py`` load
it on startup (matched by ``--robot.id``) as the reset target instead of the defaults.

Usage::

    # 1. Move arm to desired reset pose by hand
    python -m examples.isaac_teleop_to_so101.override_reset_pose [--port /dev/ttyACM0] [--id so101_follower_arm]

    # 2. Launch teleop with the SAME --robot.id — it will now reset to this pose on startup
    python -m examples.isaac_teleop_to_so101.teleoperate --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=so101_follower_arm --teleop.type=xr_controller
"""

import argparse
import json
from pathlib import Path

from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig

from .common import RESET_POSE_FILE


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--port", type=str, default="/dev/ttyACM0")
    parser.add_argument("--id", type=str, default="so101_follower_arm")
    return parser.parse_args()


def main():
    args = parse_args()
    robot = SO100Follower(SO100FollowerConfig(port=args.port, id=args.id, use_degrees=True))
    robot.connect()
    # Always disconnect the follower so a failure never leaks the serial connection.
    try:
        obs = robot.get_observation()
        motor_names = list(robot.bus.motors.keys())
        pose = {name: float(obs[f"{name}.pos"]) for name in motor_names}
    finally:
        robot.disconnect()

    print("Current joint positions:")
    for name, val in pose.items():
        print(f"  {name:20s}: {val:.2f}")

    reset_pose_file = Path(RESET_POSE_FILE.format(robot_name=robot.name, robot_id=robot.id))
    reset_pose_file.parent.mkdir(parents=True, exist_ok=True)
    reset_pose_file.write_text(json.dumps(pose, indent=2))
    print(f"\nSaved to {reset_pose_file}")


if __name__ == "__main__":
    main()
