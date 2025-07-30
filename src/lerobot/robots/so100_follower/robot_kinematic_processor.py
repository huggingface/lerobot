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

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation

from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import EnvTransition, ProcessorStepRegistry, TransitionKey


@ProcessorStepRegistry.register("ee_reference_and_delta")
@dataclass
class EEReferenceAndDelta:
    kinematics: RobotKinematics
    end_effector_step_sizes: dict
    motor_names: list[str]

    reference_ee_pose: np.ndarray | None = field(default=None, init=False, repr=False)
    last_ee_pose: np.ndarray | None = field(default=None, init=False, repr=False)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        observation = transition.get(TransitionKey.OBSERVATION)
        action = transition.get(TransitionKey.ACTION)

        current_joint_pose = np.array([observation[f"{name}.pos"] for name in self.motor_names], dtype=float)
        current_ee_pose = self.kinematics.forward_kinematics(current_joint_pose)

        enabled = bool(action.get("enabled", 0))
        new_action = dict(action)

        if enabled:
            if self.reference_ee_pose is None:
                self.reference_ee_pose = current_ee_pose.copy()
                self.last_ee_pose = current_ee_pose.copy()

            ref_pose = self.reference_ee_pose
            desired_ee_pose = np.eye(4)

            delta = np.array(
                [
                    action.get("target_x", 0.0) * self.end_effector_step_sizes["x"],
                    action.get("target_y", 0.0) * self.end_effector_step_sizes["y"],
                    action.get("target_z", 0.0) * self.end_effector_step_sizes["z"],
                ]
            )
            desired_position = ref_pose[:3, 3] + delta

            q = [
                action.get(k, d)
                for k, d in zip(
                    ["target_qx", "target_qy", "target_qz", "target_qw"], [0.0, 0.0, 0.0, 1.0], strict=False
                )
            ]
            rot = Rotation.from_quat(q)

            desired_ee_pose[:3, 3] = desired_position
            desired_ee_pose[:3, :3] = ref_pose[:3, :3] @ rot.as_matrix()

            new_action["desired_ee_pose"] = desired_ee_pose
        else:
            self.reference_ee_pose = None
            self.last_ee_pose = None

        transition[TransitionKey.ACTION] = new_action
        return transition

    def reset(self):
        self.reference_ee_pose = None
        self.last_ee_pose = None


@ProcessorStepRegistry.register("ee_bounds_and_safety")
@dataclass
class EEBoundsAndSafety:
    end_effector_bounds: dict
    max_ee_step_m: float = 0.05

    last_position: np.ndarray | None = field(default=None, init=False, repr=False)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        act = transition.get(TransitionKey.ACTION)
        if "desired_ee_pose" not in act:
            return transition

        pose = act["desired_ee_pose"].copy()
        pos = pose[:3, 3]
        pos_clamped = np.clip(pos, self.end_effector_bounds["min"], self.end_effector_bounds["max"])

        if self.last_position is not None:
            dist = np.linalg.norm(pos_clamped - self.last_position)
            if dist > self.max_ee_step_m:
                raise ValueError(
                    f"EE target jump of {dist:.3f}m exceeds safety limit of {self.max_ee_step_m}m."
                )

        pose[:3, 3] = pos_clamped
        self.last_position = pos_clamped

        new_action = dict(act)
        new_action["desired_ee_pose"] = pose
        transition[TransitionKey.ACTION] = new_action
        return transition

    def reset(self):
        self.last_position = None


@ProcessorStepRegistry.register("inverse_kinematics")
@dataclass
class InverseKinematics:
    """
    Converts a desired end-effector pose into joint angles using IK.
    Also processes the gripper command.
    """

    kinematics: RobotKinematics
    motor_names: list[str]
    gripper_speed_factor: float = 5.0

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        observation = transition.get(TransitionKey.OBSERVATION)
        action = transition.get(TransitionKey.ACTION)

        current_joint_pose = np.array([observation[f"{name}.pos"] for name in self.motor_names])

        if "desired_ee_pose" in action:
            target_joints = self.kinematics.inverse_kinematics(current_joint_pose, action["desired_ee_pose"])
            for i, name in enumerate(self.motor_names):
                if name != "gripper":
                    current_joint_pose[i] = target_joints[i]

        # Always process the gripper
        idx = self.motor_names.index("gripper")
        delta = float(action.get("gripper", 0.0)) * self.gripper_speed_factor
        current_gripper_pos = current_joint_pose[idx]
        current_joint_pose[idx] = np.clip(current_gripper_pos + delta, 0, 100)

        joint_action = {f"{name}.pos": current_joint_pose[i] for i, name in enumerate(self.motor_names)}

        transition[TransitionKey.ACTION] = joint_action
        return transition
