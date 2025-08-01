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
from lerobot.processor.pipeline import EnvTransition, ProcessorStepRegistry, TransitionKey
from lerobot.robots.robot import Robot


@ProcessorStepRegistry.register("ee_reference_and_delta")
@dataclass
class EEReferenceAndDelta:
    """
    Compute the desired end-effector pose from the target pose and the current pose.

    Input ACTION keys:
    {
        "action.enabled": bool,
        "action.target_x": float,
        "action.target_y": float,
        "action.target_z": float,
        "action.target_qx": float,
        "action.target_qy": float,
        "action.target_qz": float,
        "action.target_qw": float,
    }

    Output ACTION keys:
    {
        "action.ee.desired_T": np.ndarray,
    }
    """

    kinematics: RobotKinematics
    end_effector_step_sizes: dict
    motor_names: list[str]
    reference_ee_pose: np.ndarray | None = field(default=None, init=False, repr=False)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        obs = transition.get(TransitionKey.OBSERVATION) or {}
        act = transition.get(TransitionKey.ACTION) or {}

        # current pose from FK on measured joints
        q = np.array([obs[f"observation.state.{n}.pos"] for n in self.motor_names], dtype=float)
        t_curr = self.kinematics.forward_kinematics(q)

        enabled = bool(act.pop("action.enabled", 0))
        tx = float(act.pop("action.target_x", 0.0))
        ty = float(act.pop("action.target_y", 0.0))
        tz = float(act.pop("action.target_z", 0.0))
        qx = float(act.pop("action.target_qx", 0.0))
        qy = float(act.pop("action.target_qy", 0.0))
        qz = float(act.pop("action.target_qz", 0.0))
        qw = float(act.pop("action.target_qw", 1.0))

        new_act = dict(act)
        if enabled:
            if self.reference_ee_pose is None:
                self.reference_ee_pose = t_curr.copy()

            t_ref = self.reference_ee_pose
            delta_p = np.array(
                [
                    tx * self.end_effector_step_sizes["x"],
                    ty * self.end_effector_step_sizes["y"],
                    tz * self.end_effector_step_sizes["z"],
                ]
            )
            r_delta = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()

            t_des = np.eye(4)
            t_des[:3, :3] = t_ref[:3, :3] @ r_delta
            t_des[:3, 3] = t_ref[:3, 3] + delta_p
            new_act["action.ee.desired_T"] = t_des
        else:
            self.reference_ee_pose = None
            new_act["action.ee.desired_T"] = None

        transition[TransitionKey.ACTION] = new_act
        return transition

    def reset(self):
        self.reference_ee_pose = None

    def feature_contract(self, features: dict[str, PolicyFeature]) -> dict[str, PolicyFeature]:
        # remove consumed fields; add desired pose
        for k in [
            "action.enabled",
            "action.target_x",
            "action.target_y",
            "action.target_z",
            "action.target_qx",
            "action.target_qy",
            "action.target_qz",
            "action.target_qw",
        ]:
            features.pop(k, None)
        features["action.ee.desired_T"] = np.ndarray
        return features


@ProcessorStepRegistry.register("ee_bounds_and_safety")
@dataclass
class EEBoundsAndSafety:
    """
    Clip the end-effector pose to the bounds and check for jumps.

    Input ACTION keys:
    {
        "action.ee.desired_T": np.ndarray,
    }

    Output ACTION keys:
    {
        "action.ee.desired_T": np.ndarray,
    }
    """

    end_effector_bounds: dict
    max_ee_step_m: float = 0.05
    _last_pos: np.ndarray | None = field(default=None, init=False, repr=False)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        act = transition.get(TransitionKey.ACTION) or {}
        t = act.get("action.ee.desired_T", None)
        if t is None:
            return transition

        pos = t[:3, 3]
        pos = np.clip(pos, self.end_effector_bounds["min"], self.end_effector_bounds["max"])
        if self._last_pos is not None:
            jump = np.linalg.norm(pos - self._last_pos)
            if jump > self.max_ee_step_m:
                raise ValueError(f"EE jump {jump:.3f}m > {self.max_ee_step_m}m")
        t[:3, 3] = pos
        self._last_pos = pos

        new_act = dict(act)
        new_act["action.ee.desired_T"] = t
        transition[TransitionKey.ACTION] = new_act
        return transition

    def reset(self):
        self._last_pos = None

    def feature_contract(self, f: dict[str, PolicyFeature]) -> dict[str, PolicyFeature]:
        return f


@ProcessorStepRegistry.register("inverse_kinematics_ee_to_joints")
@dataclass
class InverseKinematicsEEToJoints:
    """
    Compute the desired joint positions from the desired end-effector pose.

    Input ACTION keys:
    {
        "action.ee.desired_T": np.ndarray,
    }

    Output ACTION keys:
    {
        "action.joint_name_1.pos": float,
        "action.joint_name_2.pos": float,
        ...
        "action.joint_name_n.pos": float,
    }
    """

    kinematics: RobotKinematics
    motor_names: list[str]

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        obs = transition.get(TransitionKey.OBSERVATION) or {}
        act = transition.get(TransitionKey.ACTION) or {}

        t = act.pop("action.ee.desired_T", None)
        if t is None:
            return transition

        q_curr = np.array([obs[f"observation.state.{n}.pos"] for n in self.motor_names], dtype=float)
        q_target = self.kinematics.inverse_kinematics(q_curr, t)

        new_act = dict(act)
        for i, name in enumerate(self.motor_names):
            new_act[f"action.{name}.pos"] = float(q_target[i])
        transition[TransitionKey.ACTION] = new_act
        return transition

    def feature_contract(self, features: dict[str, PolicyFeature]) -> dict[str, PolicyFeature]:
        features.pop("action.ee.desired_T", None)
        for n in self.motor_names:
            features[f"action.{n}.pos"] = float
        return features


@ProcessorStepRegistry.register("gripper_velocity_to_joint")
@dataclass
class GripperVelocityToJoint:
    """
    Convert the gripper velocity to a joint velocity.

    Input ACTION keys:
    {
        "action.gripper": float,
    }

    Output ACTION keys:
    {
        "action.gripper.pos": float,
    }
    """

    motor_names: list[str]
    speed_factor: float = 5.0
    clip_min: float = 0.0
    clip_max: float = 100.0

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        obs = transition.get(TransitionKey.OBSERVATION) or {}
        act = transition.get(TransitionKey.ACTION) or {}

        if "action.gripper" not in act:
            return transition

        if "gripper" not in self.motor_names:
            new_act = dict(act)
            new_act.pop("action.gripper", None)
            transition[TransitionKey.ACTION] = new_act
            return transition

        curr = float(obs.get("observation.state.gripper.pos", 0.0))
        delta = float(act.get("action.gripper", 0.0)) * float(self.speed_factor)
        tgt = float(np.clip(curr + delta, self.clip_min, self.clip_max))

        new_act = dict(act)
        new_act["action.gripper.pos"] = tgt
        new_act.pop("action.gripper", None)
        transition[TransitionKey.ACTION] = new_act
        return transition

    def feature_contract(self, features: dict[str, PolicyFeature]) -> dict[str, PolicyFeature]:
        features["action.gripper.pos"] = float
        features.pop("action.gripper", None)
        return features


@ProcessorStepRegistry.register("forward_kinematics_joints_to_ee")
@dataclass
class ForwardKinematicsJointsToEE:
    """
    Compute the end-effector pose from the joint positions.

    Input OBSERVATION keys:
    {
        "observation.state.joint_name_1.pos": float,
    }

    Output OBSERVATION keys:
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
        obs = transition.get(TransitionKey.OBSERVATION) or {}
        if not all(f"observation.state.{n}.pos" in obs for n in self.motor_names):
            return transition

        q = np.array([obs[f"observation.state.{n}.pos"] for n in self.motor_names], dtype=float)
        t = self.kinematics.forward_kinematics(q)
        pos = t[:3, 3]
        quat = Rotation.from_matrix(t[:3, :3]).as_quat()  # x,y,z,w

        new_obs = dict(obs)
        new_obs["observation.state.ee.x"] = float(pos[0])
        new_obs["observation.state.ee.y"] = float(pos[1])
        new_obs["observation.state.ee.z"] = float(pos[2])
        new_obs["observation.state.ee.qx"] = float(quat[0])
        new_obs["observation.state.ee.qy"] = float(quat[1])
        new_obs["observation.state.ee.qz"] = float(quat[2])
        new_obs["observation.state.ee.qw"] = float(quat[3])
        transition[TransitionKey.OBSERVATION] = new_obs
        return transition

    def feature_contract(self, features: dict[str, PolicyFeature]) -> dict[str, PolicyFeature]:
        # add EE pose under observation.state.ee.*
        for k in ["x", "y", "z", "qx", "qy", "qz", "qw"]:
            features[f"observation.state.ee.{k}"] = float
        return features


@ProcessorStepRegistry.register("add_robot_observation")
@dataclass
class AddRobotObservation:
    """
    Read the robot's current observation and insert it into the transition.

    - Joint positions are added under:  observation.state.<motor>.pos
    - If include_images=True, camera frames are added under: observation.images.<camera_key>

    This makes the current state available to downstream steps (e.g., IK and FK)
    without the outer loop needing to inject or merge observations.
    """

    robot: Robot
    include_images: bool = False

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        obs_dict = transition.get(TransitionKey.OBSERVATION) or {}
        device_obs = self.robot.get_observation()

        for key, val in device_obs.items():
            if isinstance(val, np.ndarray) and val.dtype == np.uint8 and val.ndim == 3:
                if self.include_images:
                    obs_dict[f"observation.images.{key}"] = val
            else:
                obs_dict[f"observation.state.{key}"] = float(val) if np.isscalar(val) else val

        transition[TransitionKey.OBSERVATION] = obs_dict
        return transition

    def feature_contract(self, features: dict[str, PolicyFeature]) -> dict[str, PolicyFeature]:
        for key, ftype in self.robot.observation_features.items():
            if isinstance(ftype, tuple) and len(ftype) == 3:
                if self.include_images:
                    features[f"observation.images.{key}"] = ftype
            else:
                features[f"observation.state.{key}"] = ftype
        return features
