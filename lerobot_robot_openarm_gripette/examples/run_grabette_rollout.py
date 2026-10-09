#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""Run a Cartesian-relative Grabette policy through LeRobot's rollout engine."""

import argparse
from pathlib import Path
from typing import cast

import numpy as np
from gripette.grasp_projection import GraspProjection
from huggingface_hub import snapshot_download
from openarm_gripette_model import OPENARM_RIGHT_DIR

from lerobot.configs import PreTrainedConfig
from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import (
    RobotProcessorPipeline,
    observation_to_transition,
    robot_action_observation_to_transition,
    transition_to_observation,
    transition_to_robot_action,
)
from lerobot.robots.openarm_follower import OpenArmFollowerConfigBase
from lerobot.rollout import BaseStrategyConfig, RolloutConfig, build_rollout_context
from lerobot.rollout.inference import SyncInferenceConfig
from lerobot.rollout.robot_wrapper import ThreadSafeRobot
from lerobot.rollout.strategies import BaseStrategy
from lerobot.utils.device_utils import auto_select_torch_device
from lerobot.utils.process import ProcessSignalHandler
from lerobot.utils.utils import init_logging
from lerobot.utils.visualization_utils import init_visualization, shutdown_visualization
from lerobot_robot_openarm_gripette import OpenArmGripetteFollower, OpenArmGripetteFollowerConfig
from lerobot_robot_openarm_gripette.grabette_processors import (
    ROBOT_JOINTS,
    GrabetteRelativePoseToJoints,
    GripettePolicyObservation,
)
from lerobot_robot_openarm_gripette.manual_start import prepare_manual_start

URDF_JOINTS = [
    "r_arm_pitch",
    "r_arm_roll",
    "r_arm_yaw",
    "r_elbow",
    "r_wrist_yaw",
    "r_wrist_roll",
    "r_wrist_pitch",
]


class DryRunRobotWrapper(ThreadSafeRobot):
    def send_action(self, action):
        return action


def resolve_policy(args: argparse.Namespace) -> PreTrainedConfig:
    path = Path(args.policy)
    if not path.is_dir():
        subfolder = f"checkpoints/{args.checkpoint}/pretrained_model" if args.checkpoint else ""
        root = snapshot_download(
            args.policy, revision=args.revision, allow_patterns=f"{subfolder}/*" if subfolder else None
        )
        path = Path(root) / subfolder
    config = PreTrainedConfig.from_pretrained(path)
    if (
        not config.use_relative_actions
        or config.relative_action_mode != "pose"
        or config.relative_reference_key is not None
        or config.relative_pose_rotation_format != "rot6d"
    ):
        raise ValueError("Policy must use chunk-relative Cartesian poses with rot6d rotations")
    config.pretrained_path = str(path)
    config.n_action_steps = args.n_action_steps
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default="CarolinePascal/grabette-pick-and-place-texture-act")
    parser.add_argument("--checkpoint", default="070000")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--port", default="can0")
    parser.add_argument("--gripette", default="172.18.7.107:50051")
    parser.add_argument("--id", default="my_follower")
    parser.add_argument("--device", default=None)
    parser.add_argument("--task", default="")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--n_action_steps", type=int, default=10)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--joint_signs", type=float, nargs=7, default=[1.0] * 7)
    parser.add_argument("--max_joint_step", type=float, default=5.0)
    parser.add_argument("--initial_gripper_strategy", type=float, default=None)
    parser.add_argument("--initial_gripper_closure", type=float, default=None)
    parser.add_argument("--display_data", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    for name in ("initial_gripper_strategy", "initial_gripper_closure"):
        value = getattr(args, name)
        if value is not None and not 0.0 <= value <= 1.0:
            parser.error(f"--{name} must be between 0 and 1")

    init_logging()
    policy_config = resolve_policy(args)
    (image_key, image_feature), *others = policy_config.image_features.items()
    if others:
        raise ValueError(f"Expected a single camera input, got {list(policy_config.image_features)}")
    camera_name = image_key.removeprefix("observation.images.")

    kinematics = RobotKinematics(str(OPENARM_RIGHT_DIR), target_frame_name="oak_l", joint_names=URDF_JOINTS)
    for dof in ("proximal", "distal", "r_wrist_roll_mimic"):
        kinematics.solver.mask_dof(dof)
    kinematics.solver.enable_joint_limits(True)
    kinematics.solver.add_regularization_task(1e-4)

    observation_processor = RobotProcessorPipeline[RobotObservation, RobotObservation](
        steps=[GripettePolicyObservation()],
        to_transition=observation_to_transition,
        to_output=transition_to_observation,
    )
    action_step = GrabetteRelativePoseToJoints(
        kinematics=kinematics,
        joint_signs=tuple(args.joint_signs),
        max_joint_step=args.max_joint_step,
        resync_on_chunk_start=args.dry_run,
    )
    action_processor = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[action_step],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )

    robot_config = OpenArmGripetteFollowerConfig(
        id=args.id,
        arm_config=OpenArmFollowerConfigBase(port=args.port, side="right", max_relative_target=8.0),
        gripette_addr=args.gripette,
        gripper_camera_name=camera_name,
        gripper_camera_height=image_feature.shape[1],
        gripper_camera_width=image_feature.shape[2],
    )
    rollout_config = RolloutConfig(
        robot=robot_config,
        policy=policy_config,
        strategy=BaseStrategyConfig(),
        inference=SyncInferenceConfig(),
        fps=args.fps,
        duration=args.duration,
        device=args.device or auto_select_torch_device().type,
        task=args.task,
        display_data=args.display_data,
        return_to_initial_position=False,
    )
    signal_handler = ProcessSignalHandler(use_threads=True)
    context = build_rollout_context(
        rollout_config,
        signal_handler.shutdown_event,
        robot_action_processor=action_processor,
        robot_observation_processor=observation_processor,
    )
    action_step.bind_action_queue(context.policy.policy.count_queued_actions, args.n_action_steps)

    robot = cast(OpenArmGripetteFollower, context.hardware.robot_wrapper.inner)
    signs = np.asarray(args.joint_signs)
    q_cmd = prepare_manual_start(robot, signs)
    held = robot.arm.get_observation()
    context.hardware.initial_position = {
        f"{joint}.pos": float(held[f"{joint}.pos"]) for joint in ROBOT_JOINTS
    }
    print("Held joints (URDF deg):", np.round(q_cmd, 1))
    print("Held oak_l position (m):", np.round(kinematics.forward_kinematics(q_cmd)[:3, 3], 3))

    if args.initial_gripper_strategy is not None or args.initial_gripper_closure is not None:
        observation = robot.get_observation()
        projection = GraspProjection()
        measured_strategy, measured_closure = projection.encode(
            observation["proximal.pos"], observation["distal.pos"]
        )
        strategy = (
            measured_strategy if args.initial_gripper_strategy is None else args.initial_gripper_strategy
        )
        closure = measured_closure if args.initial_gripper_closure is None else args.initial_gripper_closure
        proximal, distal = projection.decode(strategy, closure)
        print(
            f"{'DRY RUN: ' if args.dry_run else ''}Initial gripper: "
            f"strategy={strategy:.2f} closure={closure:.2f}"
        )
        if not args.dry_run:
            robot.send_action(
                {
                    **{
                        f"{joint}.pos": sign * target
                        for joint, sign, target in zip(ROBOT_JOINTS, signs, q_cmd, strict=True)
                    },
                    "proximal.pos": proximal,
                    "distal.pos": distal,
                }
            )

    if args.dry_run:
        context.hardware.robot_wrapper = DryRunRobotWrapper(robot)
    input(f"{'DRY RUN: the robot will not move. ' if args.dry_run else ''}Press ENTER to start...")

    strategy_runner = BaseStrategy(rollout_config.strategy)
    if rollout_config.display_data:
        init_visualization(
            rollout_config.display_mode,
            session_name="openarm_gripette_rollout",
            ip=rollout_config.display_ip,
            port=rollout_config.display_port,
        )
    try:
        strategy_runner.setup(context)
        strategy_runner.run(context)
    finally:
        try:
            input("Support the arm, then press ENTER to disable torque and disconnect...")
            strategy_runner.teardown(context)
        finally:
            if rollout_config.display_data:
                shutdown_visualization(rollout_config.display_mode)


if __name__ == "__main__":
    main()
