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

from dataclasses import dataclass

import numpy as np
import torch

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.model.kinematics import RobotKinematics
from lerobot.processor.pipeline import EnvTransition, ProcessorStepRegistry, TransitionKey
from lerobot.robots import Robot
from lerobot.teleoperators import Teleoperator
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.utils.rotation import Rotation

from .pipeline import ProcessorStep


@ProcessorStepRegistry.register("leader_follower_processor")
@dataclass
class LeaderFollowerProcessor(ProcessorStep):
    """
    Processor for leader-follower teleoperation mode.

    This processor:
    1. Sends follower positions to leader arm when not intervening
    2. Computes EE delta actions from leader when intervening
    3. Handles teleop events from the leader device
    """

    leader_device: Teleoperator
    motor_names: list[str]
    robot: Robot
    kinematics: RobotKinematics
    end_effector_step_sizes: np.ndarray | None = None
    use_gripper: bool = True
    prev_leader_gripper: float | None = None
    max_gripper_pos: float = 100.0

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """Process transition with leader-follower logic."""
        # Get current follower position from complementary data
        # raw_joint_pos = transition.get(TransitionKey.COMPLEMENTARY_DATA, {}).get("raw_joint_positions")
        raw_joint_pos = transition.get(TransitionKey.OBSERVATION)
        if raw_joint_pos is not None:
            # Send follower position to leader (for follow mode)
            # follower_action = {
            #     f"{motor}.pos": float(raw_joint_pos[motor])
            #     for motor in self.motor_names
            # }
            self.leader_device.send_action(raw_joint_pos)

        # Only compute EE action if intervention is active
        # (AddTeleopEventsAsInfo already added IS_INTERVENTION to info)
        info = transition.get(TransitionKey.INFO, {})
        if info.get(TeleopEvents.IS_INTERVENTION, False):
            # Get leader joint positions from teleop_action
            # (AddTeleopActionAsComplimentaryData already got the action)
            complementary = transition.get(TransitionKey.COMPLEMENTARY_DATA, {})
            teleop_action = complementary.get("teleop_action", {})

            if isinstance(teleop_action, dict) and raw_joint_pos is not None:
                leader_pos = np.array([teleop_action[f"{motor}.pos"] for motor in self.motor_names])
                follower_pos = np.array([raw_joint_pos[f"{motor}.pos"] for motor in self.motor_names])

                leader_ee = self.kinematics.forward_kinematics(leader_pos)
                follower_ee = self.kinematics.forward_kinematics(follower_pos)
                follower_gripper_pos = raw_joint_pos["gripper.pos"]

                leader_ee_pos = leader_ee[:3, 3]
                leader_ee_rvec = Rotation.from_matrix(leader_ee[:3, :3]).as_rotvec()
                leader_gripper_pos = teleop_action["gripper.pos"]

                follower_ee_pos = follower_ee[:3, 3]
                # follower_ee_rvec = Rotation.from_matrix(follower_ee[:3, :3]).as_rotvec()

                delta_pos = leader_ee_pos - follower_ee_pos

                # For rotation: compute relative rotation from follower to leader
                # R_leader = R_follower * R_delta  =>  R_delta = R_follower^T * R_leader
                r_delta = follower_ee[:3, :3].T @ leader_ee[:3, :3]
                delta_rvec = Rotation.from_matrix(r_delta).as_rotvec()

                delta_gripper = leader_gripper_pos - follower_gripper_pos

                desired = np.eye(4, dtype=float)
                desired[:3, :3] = follower_ee[:3, :3] @ r_delta
                desired[:3, 3] = follower_ee[:3, 3] + delta_pos

                pos = desired[:3, 3]
                tw = Rotation.from_matrix(desired[:3, :3]).as_rotvec()

                assert np.allclose(pos, leader_ee_pos), "Position delta computation error"
                assert np.allclose(tw, leader_ee_rvec), "Orientation delta computation error"
                assert np.isclose(follower_gripper_pos + delta_gripper, leader_gripper_pos), (
                    "Gripper delta computation error"
                )

                intervention_action = np.array(
                    [
                        delta_pos[0] / self.end_effector_step_sizes["x"],
                        delta_pos[1] / self.end_effector_step_sizes["y"],
                        delta_pos[2] / self.end_effector_step_sizes["z"],
                        delta_rvec[0] / self.end_effector_step_sizes["wx"],
                        delta_rvec[1] / self.end_effector_step_sizes["wy"],
                        delta_rvec[2] / self.end_effector_step_sizes["wz"],
                        np.clip(delta_gripper, -self.max_gripper_pos, self.max_gripper_pos)
                        / self.max_gripper_pos,
                    ],
                    dtype=float,
                )

                #         # Extract leader positions from teleop action dict
                #         # leader_pos = np.array([teleop_action.get(f"{motor}.pos", 0) for motor in self.motor_names])
                #         # follower_pos = np.array([raw_joint_pos[f"{motor}.pos"] for motor in self.motor_names])

                #         teleop_action = self.leader_device.bus.sync_read("Present_Position")
                #         raw_joint_pos = self.robot.bus.sync_read("Present_Position")
                #         leader_pos = np.array([teleop_action.get(f"{motor}", 0) for motor in self.motor_names])
                #         follower_pos = np.array([raw_joint_pos[f"{motor}"] for motor in self.motor_names])

                #         # Compute EE positions
                #         leader_ee_fi = self.kinematics.forward_kinematics(leader_pos)
                #         leader_ee_pos = leader_ee_fi[:3, 3]
                #         # leader_ee_rot = Rotation.from_matrix(leader_ee_fi[:3, :3]).as_rotvec()
                #         leader_ee = np.concat([leader_ee_pos, [0,0,0]])

                #         if "IK_solution" in transition.get(TransitionKey.COMPLEMENTARY_DATA):
                #             follower_ee = transition.get(TransitionKey.COMPLEMENTARY_DATA)["IK_solution"]
                #         else:
                #             follower_pos = np.array([raw_joint_pos[f"{motor}.pos"] for motor in self.motor_names])
                #             follower_ee_fi = self.kinematics.forward_kinematics(follower_pos)
                #             follower_ee_pos = follower_ee_fi[:3, 3]
                #             # follower_ee_rot = Rotation.from_matrix(follower_ee_fi[:3, :3]).as_rotvec()
                #             follower_ee = np.concat([follower_ee_pos, [0,0,0]])

                #         # Compute normalized EE delta
                #         if self.end_effector_step_sizes is not None:
                #             ee_delta = np.clip(
                #                 leader_ee - follower_ee,
                #                 -self.end_effector_step_sizes,
                #                 self.end_effector_step_sizes
                #             )
                #             ee_delta_normalized = ee_delta / self.end_effector_step_sizes
                #         else:
                #             ee_delta_normalized = leader_ee - follower_ee

                #         # Handle gripper
                #         if self.use_gripper and len(leader_pos) > 3:
                #             if self.prev_leader_gripper is None:
                #                 self.prev_leader_gripper = np.clip(
                #                     leader_pos[-1], 0, self.max_gripper_pos
                #                 )

                #             leader_gripper = leader_pos[-1]
                #             gripper_delta = leader_gripper - self.prev_leader_gripper
                #             normalized_delta = gripper_delta / self.max_gripper_pos

                #             # Quantize gripper action
                #             if normalized_delta >= 0.3:
                #                 gripper_action = 2
                #             elif normalized_delta <= -0.1:
                #                 gripper_action = 0
                #             else:
                #                 gripper_action = 1

                #             self.prev_leader_gripper = leader_gripper

                #             # Create intervention action
                #             intervention_action = np.append(ee_delta_normalized, gripper_action)
                #         else:
                #             intervention_action = ee_delta_normalized

                #         # Override teleop_action with computed EE action
                complementary["teleop_action"] = torch.from_numpy(intervention_action).float()
                transition[TransitionKey.COMPLEMENTARY_DATA] = complementary  # type: ignore[misc]

        return transition

    def reset(self) -> None:
        """Reset leader-follower state."""
        self.prev_leader_gripper = None
        if hasattr(self.leader_device, "reset"):
            self.leader_device.reset()

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features
