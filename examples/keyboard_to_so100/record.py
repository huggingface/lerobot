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

"""Record LeRobot datasets by teleoperating an SO100 in end-effector space with a keyboard.

Actions are recorded in end-effector space (``ee.*``), so the resulting dataset has the
same schema as one recorded with ``examples/phone_to_so100/record.py`` and can be used
with the same ``replay.py`` / ``evaluate.py`` / ``rollout.py`` scripts.

Teleoperation keys (see `KeyboardEndEffectorTeleop`):
    arrow keys   -> x / y translation
    shift        -> z down          right shift -> z up
    left ctrl    -> close gripper   right ctrl  -> open gripper

Recording controls use letters only, because the usual arrow/Esc bindings of
``init_keyboard_listener`` would collide with the teleoperation keys above
(arrows drive the arm, Esc disconnects the teleoperator):
    n -> end the episode early
    r -> re-record the current episode
    q -> stop recording
"""

import logging

from pynput import keyboard as pynput_keyboard

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.configs import FeatureType, PolicyFeature
from lerobot.datasets import LeRobotDataset, aggregate_pipeline_dataset_features, create_initial_features
from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import (
    MapDeltaActionToRobotActionStep,
    RobotProcessorPipeline,
    TransitionKey,
    observation_to_transition,
    robot_action_observation_to_transition,
    transition_to_observation,
    transition_to_robot_action,
)
from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig
from lerobot.robots.so_follower.robot_kinematic_processor import (
    EEBoundsAndSafety,
    EEReferenceAndDelta,
    ForwardKinematicsJointsToEE,
    GripperVelocityToJoint,
    InverseKinematicsRLStep,
)
from lerobot.scripts.lerobot_record import record_loop
from lerobot.teleoperators.keyboard import KeyboardEndEffectorTeleop, KeyboardEndEffectorTeleopConfig
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.feature_utils import combine_feature_dicts
from lerobot.utils.keyboard_input import apply_recording_control
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun

NUM_EPISODES = 2
FPS = 30
EPISODE_TIME_SEC = 60
RESET_TIME_SEC = 30
TASK_DESCRIPTION = "My task description"
HF_REPO_ID = "<hf_username>/<dataset_repo_id>"

# Meters moved per control tick per unit keyboard delta (~0.12 m/s at 30 FPS).
EE_STEP_SIZE_M = 0.004

# Workspace box (meters, in the robot base frame) the end-effector target is clipped to.
# Adjust to your mounting: the SO100 reaches roughly 0.45 m.
EE_BOUNDS = {"min": [-0.45, -0.45, 0.02], "max": [0.45, 0.45, 0.45]}

# Gripper position change per tick for the discrete open/close keys.
GRIPPER_SPEED_FACTOR = 0.05


def init_letter_keyboard_listener():
    """Recording controls on letter keys only: the teleoperator owns the arrows and Esc."""
    events = {"exit_early": False, "rerecord_episode": False, "stop_recording": False}
    letter_to_control = {"n": "right", "r": "left", "q": "esc"}

    def on_press(key):
        char = getattr(key, "char", None)
        if char and char.lower() in letter_to_control:
            apply_recording_control(letter_to_control[char.lower()], events)

    listener = pynput_keyboard.Listener(on_press=on_press)
    listener.start()
    return listener, events


def main():
    # Create the robot and teleoperator configurations
    camera_config = {"front": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=FPS)}
    robot_config = SO100FollowerConfig(
        port="/dev/tty.usbmodem5A460814411",
        id="my_awesome_follower_arm",
        cameras=camera_config,
        use_degrees=True,
    )
    teleop_config = KeyboardEndEffectorTeleopConfig(use_gripper=True)

    # Initialize the robot and teleoperator
    robot = SO100Follower(robot_config)
    teleop_device = KeyboardEndEffectorTeleop(teleop_config)

    # NOTE: It is highly recommended to use the urdf in the SO-ARM100 repo:
    #   https://github.com/TheRobotStudio/SO-ARM100/blob/main/Simulation/SO101/so101_new_calib.urdf
    kinematics_solver = RobotKinematics(
        urdf_path="./SO101/so101_new_calib.urdf",
        target_frame_name="gripper_frame_link",
        joint_names=list(robot.bus.motors.keys()),
    )

    # Carry complementary data (the previous IK solution) across control ticks and across
    # the two action pipelines below: InverseKinematicsRLStep publishes its solution into
    # this dict, and EEReferenceAndDelta reads it back on the next tick so the reference
    # pose follows the previously *commanded* joints instead of the measured ones
    # (see examples/keyboard_to_so100/teleoperate.py for why).
    pipeline_memory: dict = {}

    def to_transition_with_memory(action_observation: tuple[RobotAction, RobotObservation]):
        transition = robot_action_observation_to_transition(action_observation)
        transition[TransitionKey.COMPLEMENTARY_DATA] = pipeline_memory
        return transition

    # Build pipeline to convert keyboard deltas to EE action (recorded in the dataset).
    keyboard_to_robot_ee_pose_processor = RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ](
        steps=[
            # {delta_x, delta_y, delta_z, gripper} -> {enabled, target_*, gripper_vel}
            MapDeltaActionToRobotActionStep(),
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
        ],
        to_transition=to_transition_with_memory,
        to_output=transition_to_robot_action,
    )

    # Build pipeline to convert EE action to joints action (IK, sent to the robot).
    robot_ee_to_joints_processor = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[
            # Seed IK from its previous solution (not the measured joints) so the commanded
            # joints are deterministic for a stable target and cannot oscillate with the
            # motors. This step also publishes its solution to the shared pipeline memory.
            InverseKinematicsRLStep(
                kinematics=kinematics_solver,
                motor_names=list(robot.bus.motors.keys()),
                initial_guess_current_joints=False,
            ),
        ],
        to_transition=to_transition_with_memory,
        to_output=transition_to_robot_action,
    )

    # Build pipeline to convert joint observation to EE observation (FK).
    robot_joints_to_ee_pose = RobotProcessorPipeline[RobotObservation, RobotObservation](
        steps=[
            ForwardKinematicsJointsToEE(
                kinematics=kinematics_solver, motor_names=list(robot.bus.motors.keys())
            )
        ],
        to_transition=observation_to_transition,
        to_output=transition_to_observation,
    )

    # Create the dataset, deriving features from the pipelines so the on-disk schema
    # matches exactly what the pipelines produce at runtime. The initial action features
    # are the keyboard deltas consumed by MapDeltaActionToRobotActionStep.
    dataset = LeRobotDataset.create(
        repo_id=HF_REPO_ID,
        fps=FPS,
        features=combine_feature_dicts(
            aggregate_pipeline_dataset_features(
                pipeline=keyboard_to_robot_ee_pose_processor,
                initial_features=create_initial_features(
                    action={
                        f"delta_{axis}": PolicyFeature(type=FeatureType.ACTION, shape=(1,))
                        for axis in ["x", "y", "z"]
                    }
                ),
                use_videos=True,
            ),
            aggregate_pipeline_dataset_features(
                pipeline=robot_joints_to_ee_pose,
                initial_features=create_initial_features(observation=robot.observation_features),
                use_videos=True,
            ),
        ),
        robot_type=robot.name,
        use_videos=True,
        image_writer_threads=4,
    )

    # Connect the robot and teleoperator
    robot.connect()
    teleop_device.connect()

    # Initialize the letter-key recording controls and rerun visualization
    listener, events = init_letter_keyboard_listener()

    # Optional: the example records fine without the viz extra
    visualize = True
    try:
        init_rerun(session_name="keyboard_so100_record")
    except (ImportError, RuntimeError) as e:
        logging.warning(f"Rerun visualization unavailable ({e}), recording without visualization.")
        visualize = False

    try:
        if not robot.is_connected or not teleop_device.is_connected:
            raise ValueError("Robot or teleop is not connected!")

        print("Starting record loop. Use the keyboard to teleoperate the robot...")
        print("Recording controls: n = next episode, r = re-record, q = stop recording.")
        episode_idx = 0
        while episode_idx < NUM_EPISODES and not events["stop_recording"]:
            log_say(f"Recording episode {episode_idx + 1} of {NUM_EPISODES}")

            # Main record loop
            record_loop(
                robot=robot,
                events=events,
                fps=FPS,
                teleop_action_processor=keyboard_to_robot_ee_pose_processor,
                robot_action_processor=robot_ee_to_joints_processor,
                robot_observation_processor=robot_joints_to_ee_pose,
                teleop=teleop_device,
                dataset=dataset,
                control_time_s=EPISODE_TIME_SEC,
                single_task=TASK_DESCRIPTION,
                display_data=visualize,
            )

            # Reset the environment if not stopping or re-recording
            if not events["stop_recording"] and (
                episode_idx < NUM_EPISODES - 1 or events["rerecord_episode"]
            ):
                log_say("Reset the environment")
                record_loop(
                    robot=robot,
                    events=events,
                    fps=FPS,
                    teleop_action_processor=keyboard_to_robot_ee_pose_processor,
                    robot_action_processor=robot_ee_to_joints_processor,
                    robot_observation_processor=robot_joints_to_ee_pose,
                    teleop=teleop_device,
                    control_time_s=RESET_TIME_SEC,
                    single_task=TASK_DESCRIPTION,
                    display_data=visualize,
                )

            if events["rerecord_episode"]:
                log_say("Re-recording episode")
                events["rerecord_episode"] = False
                events["exit_early"] = False
                dataset.clear_episode_buffer()
                continue

            # Save episode
            dataset.save_episode()
            episode_idx += 1
    finally:
        # Clean up
        log_say("Stop recording")
        robot.disconnect()
        teleop_device.disconnect()
        listener.stop()

        dataset.finalize()
        dataset.push_to_hub()


if __name__ == "__main__":
    main()
