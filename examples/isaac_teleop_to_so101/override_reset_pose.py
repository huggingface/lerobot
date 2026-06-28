# !/usr/bin/env python

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

Move the arm to the desired reset/home position by hand (torque off), then run
this script.  It reads the current joint positions and writes them to
``reset_pose.json`` next to this file.  ``teleoperate.py`` / ``record.py`` load that file
on startup and use it as the reset target instead of the hardcoded defaults.

Usage::

    # 1. Move arm to desired reset pose by hand
    python override_reset_pose.py [--port /dev/ttyACM0] [--id so101_follower_arm]

    # 2. Inspect the saved values
    cat reset_pose.json

    # 3. Launch teleop — it will now reset to this pose on startup
    python teleoperate.py --robot.type=so101_follower --robot.port=/dev/ttyACM0 --teleop.type=xr_controller
"""

import argparse
import json
from pathlib import Path

from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig

RESET_POSE_FILE = Path(__file__).parent / "reset_pose.json"


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

    obs = robot.get_observation()
    motor_names = list(robot.bus.motors.keys())
    pose = {name: float(obs[f"{name}.pos"]) for name in motor_names}

    robot.disconnect()

    print("Current joint positions:")
    for name, val in pose.items():
        print(f"  {name:20s}: {val:.2f}")

    RESET_POSE_FILE.write_text(json.dumps(pose, indent=2))
    print(f"\nSaved to {RESET_POSE_FILE}")


if __name__ == "__main__":
    main()
