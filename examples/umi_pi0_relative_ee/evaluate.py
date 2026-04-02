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
Inference script for a pi0 model trained with **relative EE actions** on an OpenArm robot.

Single right OpenArm follower with one wrist camera.

This uses the built-in ``DeriveStateFromActionStep`` (no-op at inference),
``RelativeActionsProcessorStep``, ``AbsoluteActionsProcessorStep``, and
``RelativeStateProcessorStep`` that are already wired into pi0's processor
pipeline.

The inference loop:
  1. Reads joint positions from the robot (7-DOF arm + gripper).
  2. Converts to EE pose via forward kinematics (FK).
     This produces ``observation.state`` with the current EE pose.
  3. The pi0 preprocessor:
     a) ``DeriveStateFromActionStep`` — no-op (state comes from robot).
     b) ``RelativeActionsProcessorStep`` caches the raw state.
     c) ``RelativeStateProcessorStep`` buffers prev state, stacks, subtracts.
     d) ``NormalizerProcessorStep`` normalizes state and actions.
  4. pi0 predicts relative action chunk.
  5. The pi0 postprocessor:
     a) ``UnnormalizerProcessorStep`` unnormalizes.
     b) ``AbsoluteActionsProcessorStep`` adds cached state → absolute EE.
  6. IK converts absolute EE → joint targets → robot.

Usage:
    python evaluate.py
"""

from __future__ import annotations

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.datasets.feature_utils import combine_feature_dicts
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.model.kinematics import RobotKinematics
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.processor import (
    RelativeStateProcessorStep,
    RobotProcessorPipeline,
    make_default_teleop_action_processor,
)
from lerobot.processor.converters import (
    observation_to_transition,
    robot_action_observation_to_transition,
    transition_to_observation,
    transition_to_robot_action,
)
from lerobot.robots.openarm_follower import OpenArmFollower, OpenArmFollowerConfig
from lerobot.robots.so_follower.robot_kinematic_processor import (
    ForwardKinematicsJointsToEE,
    InverseKinematicsEEToJoints,
)
from lerobot.scripts.lerobot_record import record_loop
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun

FPS = 30
EPISODE_TIME_SEC = 60
TASK_DESCRIPTION = "red cube"

HF_MODEL_ID = "pepijn223/grabette-umi-pi0"

# Latency compensation: skip this many steps from the start of each predicted
# action chunk. Formula: ceil(total_latency_ms / (1000 / FPS)).
# E.g. at 10Hz with ~200ms total system latency: ceil(200 / 100) = 2.
LATENCY_SKIP_STEPS = 0

# EE feature keys produced by ForwardKinematicsJointsToEE (arm pose only).
# Gripper joints use absolute position control, not EE-relative.
EE_KEYS = ["x", "y", "z", "wx", "wy", "wz"]

URDF_PATH = "src/lerobot/robots/openarm_follower/urdf/openarm_bimanual_pybullet.urdf"
URDF_EE_FRAME = "openarm_right_link7"


def main():
    camera_config = {"cam0": OpenCVCameraConfig(index_or_path=0, width=960, height=720, fps=FPS)}
    robot_config = OpenArmFollowerConfig(
        port="can0",
        id="right_openarm",
        side="right",
        cameras=camera_config,
        max_relative_target=8.0,
        gripper_port="/dev/ttyUSB0",
    )
    robot = OpenArmFollower(robot_config)

    policy = PI0Policy.from_pretrained(HF_MODEL_ID)
    policy.config.latency_skip_steps = LATENCY_SKIP_STEPS

    arm_motor_names = list(robot.bus.motors.keys())
    gripper_motor_names = list(robot.gripper_bus.motors.keys())

    kinematics_solver = RobotKinematics(
        urdf_path=URDF_PATH,
        target_frame_name=URDF_EE_FRAME,
        joint_names=arm_motor_names,
    )

    # The policy starts from the robot's current EE pose (via FK below).
    # Relative actions are predicted as deltas from that pose, so no manual
    # re-centering is needed — the starting point is always the live EE tip.

    # FK: joint observation → EE observation (produces observation.state).
    # gripper_names=[] means proximal/distal pass through as absolute positions.
    robot_joints_to_ee_processor = RobotProcessorPipeline[RobotObservation, RobotObservation](
        steps=[
            ForwardKinematicsJointsToEE(
                kinematics=kinematics_solver,
                motor_names=arm_motor_names,
                gripper_names=[],
            )
        ],
        to_transition=observation_to_transition,
        to_output=transition_to_observation,
    )

    # IK: EE action → joint targets. Gripper actions are absolute and pass through.
    robot_ee_to_joints_processor = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[
            InverseKinematicsEEToJoints(
                kinematics=kinematics_solver,
                motor_names=arm_motor_names,
                gripper_names=[],
                initial_guess_current_joints=True,
            ),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )

    # OpenArm observations include .vel and .torque per motor; the EE policy
    # pipeline only needs .pos (converted to EE by FK) and camera features.
    obs_features = {
        k: v
        for k, v in robot.observation_features.items()
        if not (k.endswith(".vel") or k.endswith(".torque"))
    }

    # A dataset object is needed for its .features and .meta.stats even when
    # not recording — record_loop uses them for building observation/action frames.
    dataset = LeRobotDataset.create(
        repo_id="tmp/openarm_eval_scratch",
        fps=FPS,
        features=combine_feature_dicts(
            aggregate_pipeline_dataset_features(
                pipeline=robot_joints_to_ee_processor,
                initial_features=create_initial_features(observation=obs_features),
                use_videos=True,
            ),
            aggregate_pipeline_dataset_features(
                pipeline=make_default_teleop_action_processor(),
                initial_features=create_initial_features(
                    action={
                        **{f"ee.{k}": PolicyFeature(type=FeatureType.ACTION, shape=(1,)) for k in EE_KEYS},
                        **{
                            f"{g}.pos": PolicyFeature(type=FeatureType.ACTION, shape=(1,))
                            for g in gripper_motor_names
                        },
                    }
                ),
                use_videos=True,
            ),
        ),
        robot_type=robot.name,
        use_videos=True,
        image_writer_threads=4,
    )

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy,
        pretrained_path=HF_MODEL_ID,
        dataset_stats=dataset.meta.stats,
        preprocessor_overrides={"device_processor": {"device": str(policy.config.device)}},
    )

    _relative_state_steps = [s for s in preprocessor.steps if isinstance(s, RelativeStateProcessorStep)]

    robot.connect()

    listener, events = init_keyboard_listener()
    init_rerun(session_name="openarm_umi_pi0_relative_ee_evaluate")

    try:
        if not robot.is_connected:
            raise ValueError("Robot is not connected!")

        log_say("Starting policy execution")
        for step in _relative_state_steps:
            step.reset()

        record_loop(
            robot=robot,
            events=events,
            fps=FPS,
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            dataset=dataset,
            control_time_s=EPISODE_TIME_SEC,
            single_task=TASK_DESCRIPTION,
            display_data=True,
            teleop_action_processor=make_default_teleop_action_processor(),
            robot_action_processor=robot_ee_to_joints_processor,
            robot_observation_processor=robot_joints_to_ee_processor,
        )
    finally:
        robot.disconnect()
        listener.stop()


if __name__ == "__main__":
    main()
