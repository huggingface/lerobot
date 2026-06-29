#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""
Script to find joint limits and end-effector bounds via teleoperation.

Example:

```shell
lerobot-find-joint-limits \
  --robot.type=so100_follower \
  --robot.port=/dev/tty.usbmodem58760432981 \
  --robot.id=black \
  --teleop.type=so100_leader \
  --teleop.port=/dev/tty.usbmodem58760434471 \
  --teleop.id=blue \
  --urdf_path=<user>/SO-ARM100-main/Simulation/SO101/so101_new_calib.urdf \
  --target_frame_name=gripper \
  --teleop_time_s=30 \
  --warmup_time_s=5 \
  --control_loop_fps=30
```
"""

import logging
import time
from dataclasses import dataclass

import draccus
import numpy as np

from lerobot.model import RobotKinematics
from lerobot.robots import (  # noqa: F401
    RobotConfig,
    bi_openarm_follower,
    bi_rebot_b601_follower,
    bi_so_follower,
    koch_follower,
    make_robot_from_config,
    omx_follower,
    openarm_follower,
    rebot_b601_follower,
    so_follower,
)
from lerobot.teleoperators import (  # noqa: F401
    TeleoperatorConfig,
    bi_openarm_leader,
    bi_openarm_mini,
    bi_rebot_102_leader,
    bi_so_leader,
    gamepad,
    koch_leader,
    make_teleoperator_from_config,
    omx_leader,
    openarm_leader,
    openarm_mini,
    rebot_102_leader,
    so_leader,
)
from lerobot.utils.robot_utils import precise_sleep

logger = logging.getLogger(__name__)


@dataclass
class FindJointLimitsConfig:
    teleop: TeleoperatorConfig
    robot: RobotConfig

    # Path to URDF file for kinematics
    # NOTE: It is highly recommended to use the urdf in the SO-ARM100 repo:
    # https://github.com/TheRobotStudio/SO-ARM100/blob/main/Simulation/SO101/so101_new_calib.urdf
    urdf_path: str
    target_frame_name: str = "gripper"

    # Duration of the recording phase in seconds
    teleop_time_s: float = 30
    # Duration of the warmup phase in seconds
    warmup_time_s: float = 5
    # Control loop frequency
    control_loop_fps: int = 30


@draccus.wrap()
def find_joint_and_ee_bounds(cfg: FindJointLimitsConfig):
    teleop = make_teleoperator_from_config(cfg.teleop)
    robot = make_robot_from_config(cfg.robot)

    logger.info(f"Connecting to robot: {cfg.robot.type}...")
    teleop.connect()
    robot.connect()
    logger.info("Devices connected.")

    # Initialize Kinematics
    try:
        kinematics = RobotKinematics(cfg.urdf_path, cfg.target_frame_name)
    except Exception as e:
        logger.error(f"Error initializing kinematics: {e}")
        logger.error("Ensure URDF path and target frame name are correct.")
        robot.disconnect()
        teleop.disconnect()
        return

    # Initialize variables
    max_pos = None
    min_pos = None
    max_ee = None
    min_ee = None

    start_t = time.perf_counter()
    warmup_done = False

    logger.info("\n" + "=" * 40)
    logger.info(f"  WARMUP PHASE ({cfg.warmup_time_s}s)")
    logger.info("  Move the robot freely to ensure control works.")
    logger.info("  Data is NOT being recorded yet.")
    logger.info("=" * 40 + "\n")

    try:
        while True:
            t0 = time.perf_counter()

            # 1. Teleoperation Control Loop
            action = teleop.get_action()
            robot.send_action(action)

            # 2. Read Observations
            observation = robot.get_observation()
            joint_positions = np.array([observation[f"{key}.pos"] for key in robot.bus.motors])

            # 3. Calculate Kinematics
            # Forward kinematics to get (x, y, z) translation
            ee_pos = kinematics.forward_kinematics(joint_positions)[:3, 3]

            current_time = time.perf_counter()
            elapsed = current_time - start_t

            # 4. Handle Phases
            if elapsed < cfg.warmup_time_s:
                # Still in warmup
                pass

            else:
                # Phase Transition: Warmup -> Recording
                if not warmup_done:
                    logger.info("\n" + "=" * 40)
                    logger.info("  RECORDING STARTED")
                    logger.info("  Move robot to ALL joint limits.")
                    logger.info("  Press Ctrl+C to stop early and save results.")
                    logger.info("=" * 40 + "\n")

                    # Initialize limits with current position at start of recording
                    max_pos = joint_positions.copy()
                    min_pos = joint_positions.copy()
                    max_ee = ee_pos.copy()
                    min_ee = ee_pos.copy()
                    warmup_done = True

                # Update Limits
                max_ee = np.maximum(max_ee, ee_pos)
                min_ee = np.minimum(min_ee, ee_pos)
                max_pos = np.maximum(max_pos, joint_positions)
                min_pos = np.minimum(min_pos, joint_positions)

                # Time check
                recording_time = elapsed - cfg.warmup_time_s
                remaining = cfg.teleop_time_s - recording_time

                # Simple throttle for print statements (every ~1 sec)
                if int(recording_time * 100) % 100 == 0:
                    logger.info(f"Time remaining: {remaining:.1f}s")

                if recording_time > cfg.teleop_time_s:
                    logger.info("Time limit reached.")
                    break

            precise_sleep(max(1.0 / cfg.control_loop_fps - (time.perf_counter() - t0), 0.0))

    except KeyboardInterrupt:
        logger.info("Interrupted by user. Stopping safely...")

    finally:
        # Safety: Disconnect devices
        logger.info("Disconnecting devices...")
        robot.disconnect()
        teleop.disconnect()

    # Results Output
    if max_pos is not None:
        logger.info("\n" + "=" * 40)
        logger.info("FINAL RESULTS")
        logger.info("=" * 40)

        # Rounding for readability
        r_max_ee = np.round(max_ee, 4).tolist()
        r_min_ee = np.round(min_ee, 4).tolist()
        r_max_pos = np.round(max_pos, 4).tolist()
        r_min_pos = np.round(min_pos, 4).tolist()

        logger.info("\n# End Effector Bounds (x, y, z):")
        logger.info(f"max_ee = {r_max_ee}")
        logger.info(f"min_ee = {r_min_ee}")

        logger.info("\n# Joint Position Limits (radians):")
        logger.info(f"max_pos = {r_max_pos}")
        logger.info(f"min_pos = {r_min_pos}")

    else:
        logger.warning("No data recorded (exited during warmup).")


def main():
    find_joint_and_ee_bounds()


if __name__ == "__main__":
    main()
