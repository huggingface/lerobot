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

"""Teleoperate an SO100 follower arm in end-effector space with only a keyboard.

No leader arm is needed: keyboard key presses are mapped to end-effector deltas,
converted to joint commands through the same kinematics processor steps used by
the HIL-SERL gym environment.

The end-effector reference pose is computed from the previously *commanded* joints
(via the IK solution carried across ticks), not the measured ones. Measured joints
sag under gravity and jitter around the command, so feeding them back into the
target makes the end-effector drift and oscillate. Commanding relative to the last
commanded pose also bounds how far the target can lead the robot, so motion
reverses instantly at workspace limits instead of having to unwind an accumulated
overshoot.

The SO100 has 5 joints, so on top of x/y/z the end-effector has exactly two
controllable orientation degrees of freedom: pitch (where the nose points, e.g.
pointing down at a piece on the table) and roll about the gripper axis (which acts
as yaw of the jaws when the nose points down). Both are enabled here via
``use_orientation=True``. True world-yaw is coupled to the base pan and follows
from the commanded x/y position.

Key mapping (see `KeyboardEndEffectorTeleop`):
    arrow keys   -> x / y translation
    shift        -> z down          right shift -> z up
    i / k        -> pitch nose up / down
    j / l        -> roll about the gripper axis
    left ctrl    -> close gripper   right ctrl  -> open gripper
"""

import logging
import time

from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import (
    MapDeltaActionToRobotActionStep,
    RobotProcessorPipeline,
    TransitionKey,
    robot_action_observation_to_transition,
    transition_to_robot_action,
)
from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig
from lerobot.robots.so_follower.robot_kinematic_processor import (
    EEBoundsAndSafety,
    EEReferenceAndDelta,
    GripperVelocityToJoint,
    InverseKinematicsRLStep,
)
from lerobot.teleoperators.keyboard import KeyboardEndEffectorTeleop, KeyboardEndEffectorTeleopConfig
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

FPS = 30

# Meters moved per control tick per unit keyboard delta (~0.12 m/s at 30 FPS).
EE_STEP_SIZE_M = 0.004

# Workspace box (meters, in the robot base frame) the end-effector target is clipped to.
# Adjust to your mounting: the SO100 reaches roughly 0.45 m.
EE_BOUNDS = {"min": [-0.45, -0.45, 0.02], "max": [0.45, 0.45, 0.45]}

# Gripper position change per tick for the discrete open/close keys. Depending on how
# your gripper was assembled and calibrated, open/close may be swapped: flip the sign.
GRIPPER_SPEED_FACTOR = 0.05

# Radians rotated per control tick per unit keyboard delta (~15 deg/s at 30 FPS).
EE_ROT_STEP_RAD = 0.0087


def main():
    # Initialize the robot and teleoperator
    robot_config = SO100FollowerConfig(
        port="/dev/tty.usbmodem5A460814411", id="my_awesome_follower_arm", use_degrees=True
    )
    teleop_config = KeyboardEndEffectorTeleopConfig(use_gripper=True, use_orientation=True)

    # Initialize the robot and teleoperator
    robot = SO100Follower(robot_config)
    teleop_device = KeyboardEndEffectorTeleop(teleop_config)

    # NOTE: It is highly recommended to use the urdf in the SO-ARM100 repo: https://github.com/TheRobotStudio/SO-ARM100/blob/main/Simulation/SO101/so101_new_calib.urdf
    kinematics_solver = RobotKinematics(
        urdf_path="./SO101/so101_new_calib.urdf",
        target_frame_name="gripper_frame_link",
        joint_names=list(robot.bus.motors.keys()),
    )

    # Carry complementary data (the previous IK solution) across control ticks so
    # EEReferenceAndDelta can reference the previously commanded joints.
    pipeline_memory: dict = {}

    def to_transition_with_memory(action_observation: tuple[RobotAction, RobotObservation]):
        transition = robot_action_observation_to_transition(action_observation)
        transition[TransitionKey.COMPLEMENTARY_DATA] = pipeline_memory
        return transition

    # Build pipeline to convert keyboard deltas to ee pose action to joint action
    keyboard_to_robot_joints_processor = RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ](
        steps=[
            # {delta_x/y/z, delta_pitch, delta_roll, gripper} -> {enabled, target_*, gripper_vel}
            MapDeltaActionToRobotActionStep(rotation_scale=EE_ROT_STEP_RAD),
            # Velocity-style control: per-tick deltas are applied relative to the
            # previously commanded pose (the IK solution from the last tick).
            EEReferenceAndDelta(
                kinematics=kinematics_solver,
                end_effector_step_sizes={"x": EE_STEP_SIZE_M, "y": EE_STEP_SIZE_M, "z": EE_STEP_SIZE_M},
                motor_names=list(robot.bus.motors.keys()),
                use_latched_reference=False,
                use_ik_solution=True,
            ),
            EEBoundsAndSafety(
                end_effector_bounds=EE_BOUNDS,
                max_ee_step_m=0.05,
            ),
            # Keyboard gripper commands are discrete: {0=close, 1=stay, 2=open}
            GripperVelocityToJoint(
                speed_factor=GRIPPER_SPEED_FACTOR,
                discrete_gripper=True,
            ),
            # Seed IK from its previous solution (not the measured joints) so the commanded
            # joints are deterministic for a stable target and cannot oscillate with the
            # motors. This step also publishes its solution to the complementary data.
            InverseKinematicsRLStep(
                kinematics=kinematics_solver,
                motor_names=list(robot.bus.motors.keys()),
                initial_guess_current_joints=False,
            ),
        ],
        to_transition=to_transition_with_memory,
        to_output=transition_to_robot_action,
    )

    # Connect to the robot and teleoperator
    robot.connect()
    teleop_device.connect()

    # Init rerun viewer (optional: the example runs fine without the viz extra)
    visualize = True
    try:
        init_rerun(session_name="keyboard_so100_teleop")
    except (ImportError, RuntimeError) as e:
        logging.warning(f"Rerun visualization unavailable ({e}), running without visualization.")
        visualize = False

    if not robot.is_connected or not teleop_device.is_connected:
        raise ValueError("Robot or teleop is not connected!")

    print("Starting teleop loop. Use the keyboard to teleoperate the robot...")
    try:
        while True:
            t0 = time.perf_counter()

            # Get robot observation
            robot_obs = robot.get_observation()

            # Get teleop action
            keyboard_action = teleop_device.get_action()

            # Keyboard deltas -> EE pose -> Joints transition
            joint_action = keyboard_to_robot_joints_processor((keyboard_action, robot_obs))

            # Send action to robot
            _ = robot.send_action(joint_action)

            # Visualize
            if visualize:
                log_rerun_data(observation=keyboard_action, action=joint_action)

            precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))
    except KeyboardInterrupt:
        print("Teleoperation stopped.")
    finally:
        robot.disconnect()
        teleop_device.disconnect()


if __name__ == "__main__":
    main()
