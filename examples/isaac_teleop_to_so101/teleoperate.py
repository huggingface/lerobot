# !/usr/bin/env python

# Copyright 2024 NVIDIA Corporation and The HuggingFace Inc. team. All rights reserved.
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

"""Teleoperate an SO-101 follower arm with an XR (VR) controller via Isaac Teleop.

This mirrors ``examples/phone_to_so100/teleoperate.py`` but swaps the phone for
an XR controller. The pipeline is::

    XRController.get_action()                       # clutch EE pose + wrist roll + closedness + clutch
      -> MapXRControllerActionToRobotAction         # engage-home delta, ee.gripper_pos, wrist_roll [rad]
      -> EEReferenceAndDelta                         # latch FK reference on enable rising edge
      -> EEBoundsAndSafety                           # workspace + per-step clamps
      -> InverseKinematicsEEToJoints                 # closed-loop Placo IK (passes ee.gripper_pos -> gripper.pos)
      -> OverwriteWristRollFromAngle                 # wrist_roll [rad] -> wrist_roll.pos [deg]

Squeeze (and hold) the controller grip past ``clutch_threshold`` to engage; on
each engage the engage-home re-arms in lock-step with the FK reference latch so
the arm does not jump. The analog gripper closedness is emitted as an absolute
``ee.gripper_pos`` joint target (so there is no gripper-velocity step), and the
wrist roll is written directly onto the joint after IK.

Requires the ``isaac-teleop`` extra (``isaacteleop``) and an OpenXR runtime.
"""

import time
from pathlib import Path

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
    InverseKinematicsEEToJoints,
)
from lerobot.teleoperators.isaac_teleop import (
    MapXRControllerActionToRobotAction,
    OverwriteWristRollFromAngle,
    XRController,
    XRControllerConfig,
)
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

FPS = 30

# The XR processor emits target_* as real metres in the robot base frame.
# EEReferenceAndDelta multiplies target_* by end_effector_step_sizes, so 1.0
# keeps the controller motion 1:1 with the commanded EE delta (tune to taste).
EE_STEP_SIZES = {"x": 1.0, "y": 1.0, "z": 1.0}

# EEBoundsAndSafety hard-raises if the commanded EE position changes by more than
# MAX_EE_STEP_M between two consecutive frames. The XR processor emits target_*
# relative to the engage origin, so the per-frame commanded delta = step_size *
# (per-frame controller motion). With step_size 1.0 at FPS=30, that is the raw
# controller displacement between frames, so the guard trips only if the
# controller moves faster than MAX_EE_STEP_M * FPS ~= 3 m/s, which normal
# teleoperation never reaches. Note the per-engagement delta itself can exceed
# MAX_EE_STEP_M (it accumulates from the origin); only the *per-frame* change is
# bounded, so a large but smooth reach is fine.
MAX_EE_STEP_M = 0.10

# CloudXR device-profile env file passed to the launcher (see webxr.env next to
# this script). Resolved absolutely so it loads regardless of the working dir.
CLOUDXR_ENV_FILE = str(Path(__file__).parent / "webxr.env")


def main():
    robot_config = SO100FollowerConfig(port="/dev/ttyACM0", id="jiwenc_follower_arm", use_degrees=True)
    teleop_config = XRControllerConfig(
        hand_side="right",
        clutch_threshold=0.5,
        cloudxr_env_file=CLOUDXR_ENV_FILE,
    )

    # SO100Follower is the shared SO-100/SO-101 follower class: so_follower
    # registers the same class under both "so100_follower" and "so101_follower".
    # Here it is configured for SO-101 (see the so101_new_calib.urdf below).
    robot = SO100Follower(robot_config)
    teleop_device = XRController(teleop_config)

    # Loads ./SO101/so101_new_calib.urdf relative to this folder. Run
    # `python download_assets.py` from this directory first to fetch the URDF and
    # its meshes from the SO-ARM100 repo:
    # https://github.com/TheRobotStudio/SO-ARM100/tree/main/Simulation/SO101
    kinematics_solver = RobotKinematics(
        urdf_path="./SO101/so101_new_calib.urdf",
        target_frame_name="gripper_frame_link",
        joint_names=list(robot.bus.motors.keys()),
    )

    # Build pipeline: XR action -> EE pose action -> joint action.
    xr_to_robot_joints_processor = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[
            # Run with LEROBOT_XR_DEBUG=1 to trace clutch/closedness/roll/deltas.
            MapXRControllerActionToRobotAction(),
            # use_latched_reference=True co-arms the no-teleport FK latch with the
            # bridge's engage-home: both re-latch on the squeeze rising edge so the
            # commanded delta is zero at engage and the arm does not jump. (This step
            # also re-emits ee.gripper_vel, which is now a benign orphan: nobody
            # consumes it downstream since GripperVelocityToJoint was removed, and
            # robot.send_action keeps only *.pos keys, so it is silently dropped.)
            EEReferenceAndDelta(
                kinematics=kinematics_solver,
                end_effector_step_sizes=EE_STEP_SIZES,
                motor_names=list(robot.bus.motors.keys()),
                use_latched_reference=True,
            ),
            EEBoundsAndSafety(
                end_effector_bounds={"min": [-1.0, -1.0, -1.0], "max": [1.0, 1.0, 1.0]},
                max_ee_step_m=MAX_EE_STEP_M,
            ),
            # No GripperVelocityToJoint: the bridge emits an absolute ee.gripper_pos
            # (motor units [0,100]) that the IK step passes straight to gripper.pos.
            # initial_guess_current_joints=True is REQUIRED here: it re-seeds the IK
            # from the measured joints every frame, so OverwriteWristRollFromAngle
            # writing wrist_roll.pos after IK does not drift the IK seed.
            InverseKinematicsEEToJoints(
                kinematics=kinematics_solver,
                motor_names=list(robot.bus.motors.keys()),
                initial_guess_current_joints=True,
            ),
            # Post-IK: write the operator's wrist roll [rad] onto wrist_roll.pos [deg],
            # overriding the under-determined IK roll on the 5-DOF arm.
            OverwriteWristRollFromAngle(),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )

    robot.connect()
    teleop_device.connect()

    init_rerun(session_name="xr_so101_teleop")

    if not robot.is_connected or not teleop_device.is_connected:
        raise ValueError("Robot or teleop is not connected!")

    print("Starting teleop loop. Squeeze and move the controller to teleoperate the robot...")
    while True:
        t0 = time.perf_counter()

        robot_obs = robot.get_observation()
        xr_action = teleop_device.get_action()

        joint_action = xr_to_robot_joints_processor((xr_action, robot_obs))

        _ = robot.send_action(joint_action)

        log_rerun_data(observation=xr_action, action=joint_action)

        precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))


if __name__ == "__main__":
    main()
