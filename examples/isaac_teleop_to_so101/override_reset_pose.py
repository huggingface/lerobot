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

"""Save the current joint positions as the reset-origin pose (override).

Move the arm to the desired reset pose by hand (torque off), then run this script to write
those joints to a per-arm file in the LeRobot cache. ``teleoperate.py`` / ``record.py`` load
it on startup (matched by ``--robot.id``) as the reset target instead of the profile's.

Takes the same ``--robot.*`` arguments as ``teleoperate.py``, so it works for any follower
with a ``RobotProfile``::

    # 1. Move the arm to the desired reset pose by hand
    python -m examples.isaac_teleop_to_so101.override_reset_pose \
        --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=so101_follower_arm

    python -m examples.isaac_teleop_to_so101.override_reset_pose \
        --robot.type=rebot_b601_follower --robot.motor_family=rs --robot.port=can0 --robot.id=rebot_arm

    # 2. Launch teleop with the SAME --robot.id — it will now reset to this pose on startup
"""

import json
from dataclasses import dataclass
from pathlib import Path

from lerobot.configs import parser
from lerobot.robots import RobotConfig, make_robot_from_config

from .common import RESET_POSE_FILE


@dataclass
class OverrideResetPoseConfig:
    # FOLLOWER arm, same arguments as teleoperate.py (--robot.type=... --robot.port=... --robot.id=...).
    robot: RobotConfig


@parser.wrap()
def override_reset_pose(cfg: OverrideResetPoseConfig):
    robot = make_robot_from_config(cfg.robot)
    # calibrate=False is load-bearing, not a shortcut: on an arm that re-zeroes at calibration
    # time (the reBot B601-RS) the default connect(calibrate=True) would silently redefine the
    # arm's zero to whatever pose is being captured here.
    robot.connect(calibrate=False)
    # Always disconnect the follower so a failure never leaks the connection.
    try:
        obs = robot.get_observation()
        motor_names = [key.removesuffix(".pos") for key in robot.action_features if key.endswith(".pos")]
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


def main():
    override_reset_pose()


if __name__ == "__main__":
    main()
