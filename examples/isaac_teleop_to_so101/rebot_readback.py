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

"""Validate a robot profile against the physical arm WITHOUT commanding it.

Reads the measured joints and FKs them. ``send_action`` is never called. Two things fall
out:

* **Joint signs.** Back-drive each joint through its travel. A reading that leaves the URDF's
  own limit for that joint means the follower's ``joint_directions`` disagrees with the URDF
  — which under torque would command a mirrored pose.
* **Zero agreement.** With the arm parked, the printed EE should match where the gripper
  physically is. If it does not, the arm's calibrated zero is not the URDF's zero.

``robot.connect()`` runs ``configure()``, which ends in ``bus.enable_all()`` — the motors are
energized for the moment before this disables torque again. No MIT command is sent in that
window, but stand clear and keep power reachable.

    python -m examples.isaac_teleop_to_so101.rebot_readback --port can0 --id rebot_arm
"""

import argparse
import time

import numpy as np

from lerobot.model.kinematics import RobotKinematics
from lerobot.robots.rebot_b601_follower import RebotB601Follower, RebotB601FollowerRobotConfig
from lerobot.utils.robot_utils import precise_sleep

from .common import ROBOT_PROFILES, _robot_profile_key

PROFILE_KEY = _robot_profile_key("rebot_b601_follower", motor_family="rs")


def _urdf_joint_limits_deg(kinematics: RobotKinematics) -> dict[str, tuple[float, float]]:
    """Per-joint (lower, upper) in degrees, read back off the loaded URDF."""
    return {
        name: tuple(np.degrees(kinematics.robot.get_joint_limits(name))) for name in kinematics.joint_names
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", default="can0", help="SocketCAN channel (or Damiao serial port)")
    parser.add_argument("--can-adapter", default="socketcan", choices=["socketcan", "damiao"])
    parser.add_argument("--id", default="rebot_arm")
    parser.add_argument("--hz", type=float, default=10.0)
    args = parser.parse_args()

    profile = ROBOT_PROFILES[PROFILE_KEY]
    kinematics = RobotKinematics(profile.urdf(), profile.ee_frame, profile.urdf_joint_names)
    urdf_limits = _urdf_joint_limits_deg(kinematics)

    robot = RebotB601Follower(
        RebotB601FollowerRobotConfig(
            port=args.port, can_adapter=args.can_adapter, id=args.id, motor_family="rs"
        )
    )
    print("Connecting (motors energize briefly, then torque is disabled) …")
    robot.connect(calibrate=False)
    robot.disable_torque()
    print("Torque disabled — the arm is back-drivable. Move each joint through its travel.\n")

    motor_names = [key.removesuffix(".pos") for key in robot.action_features if key.endswith(".pos")]
    arm_motors = [name for name in motor_names if name != "gripper"]
    seen: dict[str, list[float]] = {name: [] for name in arm_motors}
    out_of_urdf_range: set[str] = set()

    try:
        while True:
            t0 = time.perf_counter()
            observation = robot.get_observation()

            for motor_name, urdf_name in zip(arm_motors, profile.urdf_joint_names, strict=True):
                mapped = float(observation[f"{motor_name}.pos"])
                seen[motor_name].append(mapped)
                low, high = urdf_limits[urdf_name]
                if not (low <= mapped <= high):
                    out_of_urdf_range.add(motor_name)

            q = np.array([float(observation[f"{name}.pos"]) for name in motor_names])
            ee = kinematics.forward_kinematics(q)[:3, 3]
            motors_deg = [f"{float(observation[f'{name}.pos']):7.1f}" for name in motor_names]
            print(f"joints[deg] {' '.join(motors_deg)}  ->  EE {np.round(ee, 3)}", end="\r")
            precise_sleep(max(0.0, 1.0 / args.hz - (time.perf_counter() - t0)))
    except KeyboardInterrupt:
        print("\n\nTravel seen, against the URDF's own limits:")
        for motor_name, urdf_name in zip(arm_motors, profile.urdf_joint_names, strict=True):
            values = seen[motor_name]
            if not values:
                continue
            low, high = urdf_limits[urdf_name]
            flag = "  <-- LEAVES URDF RANGE: sign/offset is wrong" if motor_name in out_of_urdf_range else ""
            print(
                f"  {motor_name:14} {min(values):8.1f} .. {max(values):7.1f}   "
                f"urdf {urdf_name} allows {low:7.1f} .. {high:7.1f}{flag}"
            )
        verdict = "SIGNS LOOK WRONG" if out_of_urdf_range else "signs consistent with the URDF"
        print(f"\n{verdict} for the travel exercised.")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
