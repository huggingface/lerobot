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
import torch
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

    Expected input ACTION keys:
    {
        "enabled": bool,
        "target_x": float,
        "target_y": float,
        "target_z": float,
        "target_qx": float,
        "target_qy": float,
        "target_qz": float,
        "target_qw": float,
    }

    Output transformed ACTION keys:
    {
        "desired_ee_pose": np.ndarray,
    }
    """

    kinematics: RobotKinematics
    end_effector_step_sizes: dict
    motor_names: list[str]

    reference_ee_pose: np.ndarray | None = field(default=None, init=False, repr=False)
    last_ee_pose: np.ndarray | None = field(default=None, init=False, repr=False)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        obs = transition.get(TransitionKey.OBSERVATION) or {}
        act = transition.get(TransitionKey.ACTION) or {}

        # Current joints from observation.state
        q_curr = np.array(
            [
                float((obs.get(f"observation.state.{name}.pos") or torch.tensor(0.0)).item())
                for name in self.motor_names
            ],
            dtype=float,
        )
        current_ee_pose = self.kinematics.forward_kinematics(q_curr)

        enabled = bool(act.pop("enabled", 0))  # consumed here
        new_action = dict(act)

        if enabled:
            if self.reference_ee_pose is None:
                self.reference_ee_pose = current_ee_pose.copy()
                self.last_ee_pose = current_ee_pose.copy()

            ref_pose = self.reference_ee_pose
            desired_ee_pose = np.eye(4)

            delta = np.array(
                [
                    float(act.pop("target_x", 0.0)) * float(self.end_effector_step_sizes["x"]),
                    float(act.pop("target_y", 0.0)) * float(self.end_effector_step_sizes["y"]),
                    float(act.pop("target_z", 0.0)) * float(self.end_effector_step_sizes["z"]),
                ],
                dtype=float,
            )

            desired_position = ref_pose[:3, 3] + delta

            q = np.array(
                [
                    float(act.pop("target_qx", 0.0)),
                    float(act.pop("target_qy", 0.0)),
                    float(act.pop("target_qz", 0.0)),
                    float(act.pop("target_qw", 1.0)),
                ],
                dtype=float,
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
        # Accept both scoped/unscoped inputs; emit scoped outputs
        for k in (
            "enabled",
            "target_x",
            "target_y",
            "target_z",
            "target_qx",
            "target_qy",
            "target_qz",
            "target_qw",
        ):
            features.pop(f"action.{k}", None)
            features.pop(k, None)
        features["action.desired_ee_pose"] = np.ndarray
        return features


@ProcessorStepRegistry.register("ee_bounds_and_safety")
@dataclass
class EEBoundsAndSafety:
    """
    Clamps the EE pose to the bounds and checks for safety.

    Expected input ACTION keys:
    {
        "desired_ee_pose": np.ndarray,
    }

    Output transformed ACTION keys:
    {
        "desired_ee_pose": np.ndarray,
    }
    """

    end_effector_bounds: dict
    max_ee_step_m: float = 0.05

    last_position: np.ndarray | None = field(default=None, init=False, repr=False)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        act = transition.get(TransitionKey.ACTION) or {}
        if "desired_ee_pose" not in act or act["desired_ee_pose"] is None:
            return transition

        pose = act["desired_ee_pose"].copy()
        pos = pose[:3, 3]  # Extract position from pose
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

    Expected input ACTION keys:
    {
        "desired_ee_pose": np.ndarray,
    }

    Output transformed ACTION keys:
    {
        "motor_name_1.pos": float,
        "motor_name_2.pos": float,
        ...
        "motor_name_n.pos": float,
    }
    """

    kinematics: RobotKinematics
    motor_names: list[str]

    _arm_joint_names: list[str] = field(init=False, repr=False)

    def __post_init__(self):
        self._arm_joint_names = [n for n in self.motor_names if n != "gripper"]

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        obs = transition.get(TransitionKey.OBSERVATION) or {}
        act = transition.get(TransitionKey.ACTION) or {}

        desired = act.pop("desired_ee_pose", None)
        if desired is None:
            return transition

        q_curr = np.array(
            [
                float((obs.get(f"observation.state.{n}.pos") or torch.tensor(0.0)).item())
                for n in self.motor_names
            ],
            dtype=float,
        )
        q_target = self.kinematics.inverse_kinematics(q_curr, desired)

        new_action = dict(act)
        for i, name in enumerate(self.motor_names):
            if name == "gripper":
                continue
            new_action[f"{name}.pos"] = float(q_target[i])

        transition[TransitionKey.ACTION] = new_action
        return transition

    def feature_contract(self, features: dict[str, PolicyFeature]) -> dict[str, PolicyFeature]:
        # Emit scoped keys; remove desired_ee_pose
        for name in self._arm_joint_names:
            features[f"action.{name}.pos"] = float
        features.pop("action.desired_ee_pose", None)
        features.pop("desired_ee_pose", None)


@ProcessorStepRegistry.register("gripper_velocity_to_joint")
@dataclass
class GripperVelocityToJoint:
    """
    Converts a scalar velocity command in action['gripper'] into absolute gripper position 'gripper.pos',
    based on the current observation. Clips to [0, 100].

    Expected input ACTION keys:
    {
        "gripper": float,
    }

    Output transformed ACTION keys:
    {
        "gripper.pos": float,
    }
    """

    motor_names: list[str]
    speed_factor: float = 5.0
    clip_min: float = 0.0
    clip_max: float = 100.0

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        obs = transition.get(TransitionKey.OBSERVATION) or {}
        act = transition.get(TransitionKey.ACTION) or {}

        if "gripper" not in act:
            return transition

        if "gripper" not in self.motor_names:
            new_action = dict(act)
            new_action.pop("gripper", None)
            transition[TransitionKey.ACTION] = new_action
            return transition

        curr_pos = float((obs.get("observation.state.gripper.pos") or torch.tensor(0.0)).item())
        delta = float(act.get("gripper", 0.0)) * float(self.speed_factor)
        target = float(np.clip(curr_pos + delta, self.clip_min, self.clip_max))

        new_action = dict(act)
        new_action["gripper.pos"] = target
        new_action.pop("gripper", None)
        transition[TransitionKey.ACTION] = new_action
        return transition

    def feature_contract(self, features: dict[str, PolicyFeature]) -> dict[str, PolicyFeature]:
        features["action.gripper.pos"] = float
        features.pop("action.gripper", None)
        features.pop("gripper", None)
        return features


@ProcessorStepRegistry.register("forward_kinematics_joints_to_ee")
@dataclass
class ForwardKinematicsJointsToEE:
    """Compute EE pose from joints; writes to observation.state.ee.{x,y,z,qx,qy,qz,qw}.

    Expected input OBSERVATION keys:
    {
        "observation.state.motor_name_1.pos": float,
        "observation.state.motor_name_2.pos": float,
        ...
        "observation.state.motor_name_n.pos": float,
    }

    Output transformed OBSERVATION keys:
    {
        "observation.state.ee.x": float,
        "observation.state.ee.y": float,
        "observation.state.ee.z": float,
        "observation.state.ee.qx": float,
        "observation.state.ee.qy": float,
        "observation.state.ee.qz": float,
        "observation.state.ee.qw": float,
    }
    """

    kinematics: RobotKinematics
    motor_names: list[str]

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        observation = transition.get(TransitionKey.OBSERVATION) or {}

        # Require all joints
        if not all(f"observation.state.{n}.pos" in observation for n in self.motor_names):
            return transition

        q = np.array(
            [
                float((observation[f"observation.state.{n}.pos"]).item())
                if isinstance(observation[f"observation.state.{n}.pos"], torch.Tensor)
                else float(observation[f"observation.state.{n}.pos"])
                for n in self.motor_names
            ],
            dtype=float,
        )
        ee_t = self.kinematics.forward_kinematics(q)
        pos = ee_t[:3, 3]
        quat_xyzw = Rotation.from_matrix(ee_t[:3, :3]).as_quat()  # x,y,z,w

        new_obs = dict(observation)
        # Store as tensors for downstream steps
        new_obs["observation.state.ee.x"] = torch.tensor(float(pos[0]), dtype=torch.float32)
        new_obs["observation.state.ee.y"] = torch.tensor(float(pos[1]), dtype=torch.float32)
        new_obs["observation.state.ee.z"] = torch.tensor(float(pos[2]), dtype=torch.float32)
        new_obs["observation.state.ee.qx"] = torch.tensor(float(quat_xyzw[0]), dtype=torch.float32)
        new_obs["observation.state.ee.qy"] = torch.tensor(float(quat_xyzw[1]), dtype=torch.float32)
        new_obs["observation.state.ee.qz"] = torch.tensor(float(quat_xyzw[2]), dtype=torch.float32)
        new_obs["observation.state.ee.qw"] = torch.tensor(float(quat_xyzw[3]), dtype=torch.float32)

        transition[TransitionKey.OBSERVATION] = new_obs
        return transition

    def feature_contract(self, features: dict[str, PolicyFeature]) -> dict[str, PolicyFeature]:
        # Augment observation contract with EE fields
        features.update(
            {
                "observation.state.ee.x": float,
                "observation.state.ee.y": float,
                "observation.state.ee.z": float,
                "observation.state.ee.qx": float,
                "observation.state.ee.qy": float,
                "observation.state.ee.qz": float,
                "observation.state.ee.qw": float,
            }
        )
        return features
