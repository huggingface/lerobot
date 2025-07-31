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

from lerobot.configs.types import PolicyFeature
from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import EnvTransition, ProcessorStepRegistry, TransitionKey


@ProcessorStepRegistry.register("ee_reference_and_delta")
@dataclass
class EEReferenceAndDelta:
    """
    Maintains a reference end-effector pose and converts relative delta targets
    into an absolute desired EE transform. When enabled, the first incoming pose
    becomes the reference; subsequent (x,y,z) deltas and optional orientation
    offsets (quaternion) are applied to steer the EE. Also passes through the
    gripper and enable commands, and resets reference when disabled.
    """

    kinematics: RobotKinematics
    end_effector_step_sizes: dict
    motor_names: list[str]

    reference_ee_pose: np.ndarray | None = field(default=None, init=False, repr=False)
    last_ee_pose: np.ndarray | None = field(default=None, init=False, repr=False)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        obs = transition.get(TransitionKey.OBSERVATION)
        act = transition.get(TransitionKey.ACTION)

        current_joint_pose = np.array([obs[f"{name}.pos"] for name in self.motor_names], dtype=float)
        current_ee_pose = self.kinematics.forward_kinematics(current_joint_pose)

        enabled = act.pop("enabled", 0)  # pop enabled from action because only needed for this step
        new_action = dict(act)

        if enabled:
            if self.reference_ee_pose is None:
                self.reference_ee_pose = current_ee_pose.copy()
                self.last_ee_pose = current_ee_pose.copy()

            ref_pose = self.reference_ee_pose
            desired_ee_pose = np.eye(4)

            delta = np.array(
                [
                    act.pop("target_x", 0.0)
                    * self.end_effector_step_sizes[
                        "x"
                    ],  # pop x,y,z from action because only needed for this step
                    act.pop("target_y", 0.0) * self.end_effector_step_sizes["y"],
                    act.pop("target_z", 0.0) * self.end_effector_step_sizes["z"],
                ]
            )

            desired_position = ref_pose[:3, 3] + delta

            q = np.array(
                [
                    act.pop(
                        "target_qx", 0.0
                    ),  # pop qx,qy,qz,qw from action because only needed for this step
                    act.pop("target_qy", 0.0),
                    act.pop("target_qz", 0.0),
                    act.pop("target_qw", 1.0),
                ]
            )

            rot = Rotation.from_quat(q)

            desired_ee_pose[:3, 3] = desired_position
            desired_ee_pose[:3, :3] = ref_pose[:3, :3] @ rot.as_matrix()

            new_action["desired_ee_pose"] = desired_ee_pose
        else:
            self.reference_ee_pose = None
            self.last_ee_pose = None

            new_action["desired_ee_pose"] = None

        transition[TransitionKey.ACTION] = new_action
        return transition

    def reset(self):
        self.reference_ee_pose = None
        self.last_ee_pose = None

    def feature_contract(self, features: dict[str, PolicyFeature]) -> dict[str, PolicyFeature]:
        return {"desired_ee_pose": np.ndarray}


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

    def feature_contract(self, features: dict[str, PolicyFeature]) -> dict[str, PolicyFeature]:
        # Only clamps the EE pose
        return features


@ProcessorStepRegistry.register("inverse_kinematics_ee_to_joints")
@dataclass
class InverseKinematicsEEToJoints:
    """
    Converts a desired end-effector pose into joint angles using Inverse Kinematics to joints.
    """

    kinematics: RobotKinematics
    motor_names: list[str]

    _arm_joint_names: list[str] = field(init=False, repr=False)

    def __post_init__(self):
        self._arm_joint_names = [n for n in self.motor_names if n != "gripper"]

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        obs = transition.get(TransitionKey.OBSERVATION) or {}
        act = transition.get(TransitionKey.ACTION) or {}

        if "desired_ee_pose" not in act:
            return transition

        q_curr = np.array([obs[f"{n}.pos"] for n in self.motor_names], dtype=float)

        # Solve Inverse Kinematics; assume the kinematics returns all joints in the same order as motor_names
        q_target = self.kinematics.inverse_kinematics(q_curr, act["desired_ee_pose"])
        act.pop("desired_ee_pose", None)  # Not needed anymore

        new_action = dict(act)
        for i, name in enumerate(self.motor_names):
            if name == "gripper":
                continue
            new_action[f"{name}.pos"] = float(q_target[i])

        transition[TransitionKey.ACTION] = new_action
        return transition

    def feature_contract(self, features: dict[str, PolicyFeature]) -> dict[str, PolicyFeature]:
        # Add new features and remove desired_ee_pose from features
        features = {**features, **{f"{name}.pos": float for name in self._arm_joint_names}}
        features.pop("desired_ee_pose", None)
        return features


@ProcessorStepRegistry.register("gripper_velocity_to_joint")
@dataclass
class GripperVelocityToJoint:
    """
    Converts a scalar velocity command in action['gripper'] into absolute gripper position 'gripper.pos',
    based on the current observation. Clips to [0, 100].
    """

    motor_names: list[str]
    speed_factor: float = 5.0
    clip_min: float = 0.0
    clip_max: float = 100.0

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        obs = transition.get(TransitionKey.OBSERVATION)
        act = transition.get(TransitionKey.ACTION)

        if "gripper" not in act:
            return transition

        if "gripper" not in self.motor_names:
            new_action = dict(act)
            new_action.pop("gripper", None)  # Remove gripper from action features
            transition[TransitionKey.ACTION] = new_action
            return transition

        # Current absolute position from observation
        curr_pos = float(obs.get("gripper.pos", 0.0))

        # Apply velocity command
        delta = float(act.get("gripper", 0.0)) * float(self.speed_factor)
        target = float(np.clip(curr_pos + delta, self.clip_min, self.clip_max))

        new_action = dict(act)
        new_action["gripper.pos"] = target
        new_action.pop("gripper", None)

        transition[TransitionKey.ACTION] = new_action
        return transition

    def feature_contract(self, features: dict[str, PolicyFeature]) -> dict[str, PolicyFeature]:
        features["gripper.pos"] = float  # Add gripper.pos to action features
        features.pop("gripper", None)  # Remove gripper from action features
        return features


@ProcessorStepRegistry.register("forward_kinematics_joints_to_ee")
@dataclass
class ForwardKinematicsJointsToEE:
    """Transforms the current joint positions to the EE pose in the observation using forward kinematics."""

    kinematics: RobotKinematics
    motor_names: list[str]

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        observation = transition.get(TransitionKey.OBSERVATION)
        # Safe-guard: only proceed if we have all joints
        if not all(f"{n}.pos" in observation for n in self.motor_names):
            return transition
        q = np.array([observation[f"{n}.pos"] for n in self.motor_names], dtype=float)
        ee_t = self.kinematics.forward_kinematics(q)
        pos = ee_t[:3, 3]
        new_obs = dict(observation)
        new_obs["ee.x"], new_obs["ee.y"], new_obs["ee.z"] = float(pos[0]), float(pos[1]), float(pos[2])
        new_obs["ee.qx"], new_obs["ee.qy"], new_obs["ee.qz"], new_obs["ee.qw"] = Rotation.from_matrix(
            ee_t[:3, :3]
        ).as_quat()
        new_obs["gripper"] = observation["gripper.pos"]
        transition[TransitionKey.OBSERVATION] = new_obs
        return transition

    def feature_contract(self, features: dict[str, PolicyFeature]) -> dict[str, PolicyFeature]:
        # This step augments OBSERVATION with measured EE pose + gripper (scalar 0..100).
        return {
            "ee.x": float,
            "ee.y": float,
            "ee.z": float,
            "ee.qx": float,
            "ee.qy": float,
            "ee.qz": float,
            "ee.qw": float,
            "gripper": float,
        }
