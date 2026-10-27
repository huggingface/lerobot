# !/usr/bin/env python

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
# See the License for the specif

"""Teleoperate an SO-100/SO-101 follower arm with a phone.

Needs the SO-101 URDF next to this script (see the phone teleop guide):

    uv run python teleoperate.py --port /dev/tty.usbmodem5A460814411
"""

import argparse
import time

from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import (
    RobotProcessorPipeline,
    robot_action_observation_to_transition,
    transition_to_robot_action,
)
from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig
from lerobot.robots.so_follower.robot_kinematic_processor import (
    EEBoundsAndSafety,
    EEReferenceAndDelta,
    GripperVelocityToJoint,
    InverseKinematicsEEToJoints,
)
from lerobot.teleoperators.phone import Phone, PhoneConfig
from lerobot.teleoperators.phone.config_phone import PhoneOS
from lerobot.teleoperators.phone.phone_processor import MapPhoneActionToRobotAction
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

FPS = 30

# Seconds spent interpolating to the calibration pose, on startup and on Ctrl+C.
HOMING_DURATION_S = 3.0
HOMING_FPS = 50


def homing_pose_targets(robot) -> dict[str, int]:
    """Raw encoder targets for the homing pose recorded during calibration.

    `set_half_turn_homings()` writes a homing offset so that the pose held during calibration —
    the one drawn in the calibration diagram — reads exactly one half turn on every motor, so
    commanding that value brings the arm back to it. Note this is not the same as commanding 0 in
    the robot's normalized units, which is the middle of each joint's *recorded range of motion*.

    The gripper is left wherever it currently is, so homing never closes it on whatever it holds.
    """
    raw_positions = robot.bus.sync_read("Present_Position", normalize=False)
    targets = {}
    for motor in robot.bus.motors:
        if motor == "gripper":
            targets[motor] = raw_positions[motor]
        else:
            resolution = robot.bus.model_resolution_table[robot.bus.motors[motor].model]
            targets[motor] = int((resolution - 1) / 2)
    return targets


def move_to_pose(robot, targets: dict[str, int], duration_s: float = HOMING_DURATION_S) -> None:
    """Smoothly interpolate the robot from its current joints to `targets` (raw encoder units)."""
    start = robot.bus.sync_read("Present_Position", normalize=False)
    steps = max(int(duration_s * HOMING_FPS), 1)
    for step in range(1, steps + 1):
        t = step / steps
        robot.bus.sync_write(
            "Goal_Position",
            {motor: round(start[motor] * (1.0 - t) + target * t) for motor, target in targets.items()},
            normalize=False,
        )
        precise_sleep(1.0 / HOMING_FPS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--port",
        default="/dev/tty.usbmodem5A460814411",
        help="serial port of the follower arm (find it with `lerobot-find-port`)",
    )
    parser.add_argument("--id", default="my_awesome_follower_arm", help="calibration id of the arm")
    parser.add_argument(
        "--urdf",
        default="./SO101/so101_new_calib.urdf",
        help="URDF used for kinematics, see the SO-ARM100 repo",
    )
    parser.add_argument(
        "--os",
        choices=["ios", "android"],
        default="ios",
        help="ios uses the HEBI Mobile I/O app; android serves a WebXR page, no app needed",
    )
    args = parser.parse_args()

    # Initialize the robot and teleoperator
    robot_config = SO100FollowerConfig(
        port=args.port,
        id=args.id,
        use_degrees=True,
        disable_torque_on_disconnect=False,
    )
    teleop_config = PhoneConfig(
        phone_os=PhoneOS.IOS if args.os == "ios" else PhoneOS.ANDROID,
    )

    # Initialize the robot and teleoperator
    robot = SO100Follower(robot_config)
    teleop_device = Phone(teleop_config)

    # NOTE: It is highly recommended to use the urdf in the SO-ARM100 repo: https://github.com/TheRobotStudio/SO-ARM100/blob/main/Simulation/SO101/so101_new_calib.urdf
    kinematics_solver = RobotKinematics(
        urdf_path=args.urdf,
        target_frame_name="gripper_frame_link",
        joint_names=list(robot.bus.motors.keys()),
    )

    # Build pipeline to convert phone action to ee pose action to joint action
    phone_to_robot_joints_processor = RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ](
        steps=[
            MapPhoneActionToRobotAction(platform=teleop_config.phone_os),
            EEReferenceAndDelta(
                kinematics=kinematics_solver,
                end_effector_step_sizes={"x": 0.5, "y": 0.5, "z": 0.5},
                motor_names=list(robot.bus.motors.keys()),
                use_latched_reference=True,
            ),
            EEBoundsAndSafety(
                end_effector_bounds={"min": [-1.0, -1.0, -1.0], "max": [1.0, 1.0, 1.0]},
                max_ee_step_m=0.10,
            ),
            GripperVelocityToJoint(
                speed_factor=20.0,
            ),
            InverseKinematicsEEToJoints(
                kinematics=kinematics_solver,
                motor_names=list(robot.bus.motors.keys()),
                initial_guess_current_joints=True,
            ),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )

    robot.connect()
    if not robot.is_connected:
        raise ValueError("Robot is not connected!")

    # Pose the arm was found in, before anything moves it: Ctrl+C returns it here.
    start_targets = robot.bus.sync_read("Present_Position", normalize=False)
    home_targets = homing_pose_targets(robot)

    try:
        print("Homing the arm...")
        move_to_pose(robot, home_targets)

        # Start the viewer before the phone is calibrated
        init_rerun(session_name="phone_so100_teleop")

        teleop_device.connect()
        if not teleop_device.is_connected:
            raise ValueError("Teleoperator is not connected!")

        enable_button = "B1" if teleop_config.phone_os == PhoneOS.IOS else "Move"
        print(f"Teleop running. Hold {enable_button} to move the robot. Ctrl+C to stop.")
        while True:
            t0 = time.perf_counter()

            robot_obs = robot.get_observation()

            phone_obs = teleop_device.get_action()
            if not phone_obs:
                precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))
                continue

            # Phone -> EE pose -> Joints transition
            joint_action = phone_to_robot_joints_processor((phone_obs, robot_obs))

            # Send action to robot
            _ = robot.send_action(joint_action)

            # Visualize
            log_rerun_data(observation=phone_obs, action=joint_action)

            precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))
    except KeyboardInterrupt:
        print("\nStopping, returning the arm to where it started...")
        move_to_pose(robot, start_targets)
    finally:
        if robot.is_connected:
            robot.disconnect()
        if teleop_device.is_connected:
            teleop_device.disconnect()


if __name__ == "__main__":
    main()
