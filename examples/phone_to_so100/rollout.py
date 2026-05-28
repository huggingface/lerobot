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

"""Run a trained EE-space policy on SO100 (phone-trained) without recording.

Mirrors ``examples/so100_to_so100_EE/rollout.py`` — the model was trained
with phone teleoperation in EE space, so at deployment we only need the
joint↔EE conversion on the robot side; the phone is not used.

Uses :class:`BaseStrategy` (no recording) + :class:`SyncInferenceConfig`
(inline policy call).  For recording during rollout, switch to Sentry,
Highlight, or DAgger via ``lerobot-rollout --strategy.type=...``.
"""

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.configs import PreTrainedConfig
from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import (
    RobotProcessorPipeline,
    observation_to_transition,
    robot_action_observation_to_transition,
    transition_to_observation,
    transition_to_robot_action,
)
from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig
from lerobot.robots.so_follower.robot_kinematic_processor import (
    ForwardKinematicsJointsToEE,
    InverseKinematicsEEToJoints,
)
from lerobot.rollout import BaseStrategyConfig, RolloutConfig, build_rollout_context
from lerobot.rollout.inference import SyncInferenceConfig
from lerobot.rollout.strategies import BaseStrategy
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.process import ProcessSignalHandler
from lerobot.utils.utils import init_logging

FPS = 30
DURATION_SEC = 60
TASK_DESCRIPTION = "My task description"
HF_MODEL_ID = "<hf_username>/<model_repo_id>"


def main():
    init_logging()

    camera_config = {"front": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=FPS)}
    robot_config = SO100FollowerConfig(
        port="/dev/tty.usbmodem58760434471",
        id="my_awesome_follower_arm",
        cameras=camera_config,
        use_degrees=True,
    )

    # Peek at motor names once to build the kinematic solver.
    temp_robot = SO100Follower(robot_config)
    motor_names = list(temp_robot.bus.motors.keys())

    kinematics_solver = RobotKinematics(
        urdf_path="./SO101/so101_new_calib.urdf",
        target_frame_name="gripper_frame_link",
        joint_names=motor_names,
    )

    robot_joints_to_ee_pose_processor = RobotProcessorPipeline[RobotObservation, RobotObservation](
        steps=[ForwardKinematicsJointsToEE(kinematics=kinematics_solver, motor_names=motor_names)],
        to_transition=observation_to_transition,
        to_output=transition_to_observation,
    )

    robot_ee_to_joints_processor = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[
            InverseKinematicsEEToJoints(
                kinematics=kinematics_solver,
                motor_names=motor_names,
                initial_guess_current_joints=True,
            ),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )

    policy_config = PreTrainedConfig.from_pretrained(HF_MODEL_ID)
    policy_config.pretrained_path = HF_MODEL_ID

    cfg = RolloutConfig(
        robot=robot_config,
        policy=policy_config,
        strategy=BaseStrategyConfig(),
        inference=SyncInferenceConfig(),
        fps=FPS,
        duration=DURATION_SEC,
        task=TASK_DESCRIPTION,
    )

    signal_handler = ProcessSignalHandler(use_threads=True)

    ctx = build_rollout_context(
        cfg,
        signal_handler.shutdown_event,
        robot_action_processor=robot_ee_to_joints_processor,
        robot_observation_processor=robot_joints_to_ee_pose_processor,
    )

    strategy = BaseStrategy(cfg.strategy)
    try:
        strategy.setup(ctx)
        strategy.run(ctx)
    finally:
        strategy.teardown(ctx)


if __name__ == "__main__":
    main()
