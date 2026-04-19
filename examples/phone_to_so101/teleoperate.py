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
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Phone-to-SO101 teleoperation example.

Adapted from phone_to_so100 for the SO-101 follower arm.
Uses phone 6-DoF pose → IK pipeline → joint commands.

Prerequisites:
  - pip install "lerobot[kinematics]"
  - iOS: Install HEBI Mobile I/O app + pip install hebi-py
  - Android: pip install teleop
"""

import time
from pathlib import Path

import numpy as np
import rerun as rr

from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import RobotAction, RobotObservation, RobotProcessorPipeline
from lerobot.processor.converters import (
    robot_action_observation_to_transition,
    transition_to_robot_action,
)
from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
from lerobot.robots.so_follower.so_follower import SO101Follower
from lerobot.robots.so_follower.robot_kinematic_processor import (
    EEBoundsAndSafety,
    EEReferenceAndDelta,
    GripperVelocityToJoint,
    InverseKinematicsEEToJoints,
)
from lerobot.teleoperators.phone.config_phone import PhoneConfig, PhoneOS
from lerobot.teleoperators.phone.phone_processor import MapPhoneActionToRobotAction
from lerobot.teleoperators.phone.teleop_phone import Phone
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

FPS = 30

# --- Configuration (update these for your setup) ---
ROBOT_PORT = "/dev/tty.usbmodem5AAF2879361"
ROBOT_ID = "myfollower"
PHONE_OS = PhoneOS.ANDROID  # or PhoneOS.IOS

# URDF path (relative to repo root)
URDF_PATH = str(Path(__file__).resolve().parents[2] / "SO-ARM100" / "Simulation" / "SO101" / "so101_new_calib.urdf")

# Initialize the robot and teleoperator
robot_config = SO101FollowerConfig(
    port=ROBOT_PORT, id=ROBOT_ID, use_degrees=True
)
teleop_config = PhoneConfig(phone_os=PHONE_OS)

robot = SO101Follower(robot_config)
teleop_device = Phone(teleop_config)

# Initialize kinematics solver
kinematics_solver = RobotKinematics(
    urdf_path=URDF_PATH,
    target_frame_name="gripper_frame_link",
    joint_names=list(robot.bus.motors.keys()),
)

# Build pipeline: phone action → EE pose → joint commands
phone_to_robot_joints_processor = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
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
            max_ee_step_m=0.30,
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

# Connect robot (retry once on transient serial errors)
for attempt in range(3):
    try:
        robot.connect()
        break
    except ConnectionError as e:
        print(f"Connection attempt {attempt + 1} failed: {e}")
        if attempt < 2:
            print("Retrying in 2 seconds...")
            time.sleep(2)
        else:
            raise

teleop_device.connect()

# Init rerun viewer
init_rerun(session_name="phone_so101_teleop")

if not robot.is_connected or not teleop_device.is_connected:
    raise ValueError("Robot or teleop is not connected!")

print("Starting teleop loop. Move your phone to teleoperate the robot...")
try:
    while True:
        t0 = time.perf_counter()

        # Get robot observation
        robot_obs = robot.get_observation()

        # Get teleop action
        phone_obs = teleop_device.get_action()

        # Phone → EE pose → Joints
        joint_action = phone_to_robot_joints_processor((phone_obs, robot_obs))

        # Send action to robot
        _ = robot.send_action(joint_action)

        # Compute current and target EE poses for visualization
        q_curr = np.array([float(v) for k, v in robot_obs.items() if k.endswith(".pos")], dtype=float)
        t_curr = kinematics_solver.forward_kinematics(q_curr)
        q_target = np.array([float(v) for k, v in joint_action.items() if k.endswith(".pos")], dtype=float)
        t_target = kinematics_solver.forward_kinematics(q_target)

        # Log EE positions to Rerun
        rr.log("ee/current", rr.Points3D([t_curr[:3, 3]], colors=[[0, 255, 0]], radii=[0.01]))
        rr.log("ee/target", rr.Points3D([t_target[:3, 3]], colors=[[255, 0, 0]], radii=[0.01]))
        rr.log("ee/current_pos_z", rr.Scalars(t_curr[2, 3]))
        rr.log("ee/target_pos_z", rr.Scalars(t_target[2, 3]))

        # Log IK solution (joint angles) vs actual joint positions
        motor_names = list(robot.bus.motors.keys())
        for i, name in enumerate(motor_names):
            actual = float(robot_obs.get(f"{name}.pos", 0.0))
            commanded = float(joint_action.get(f"{name}.pos", 0.0))
            rr.log(f"joints/{name}/actual", rr.Scalars(actual))
            rr.log(f"joints/{name}/ik_solution", rr.Scalars(commanded))

        # Visualize
        log_rerun_data(observation=phone_obs, action=joint_action)

        precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))
except KeyboardInterrupt:
    print("\nTeleop stopped.")
finally:
    teleop_device.disconnect()
    robot.disconnect()
    print("Disconnected.")
