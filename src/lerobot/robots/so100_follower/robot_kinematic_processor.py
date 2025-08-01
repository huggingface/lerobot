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
        "action.ee.{x,y,z,wx,wy,wz}" : float
        "complementary_data.raw_joint_positions": dict,
    }

    Output ACTION keys:
    {
        "action.ee.{x,y,z,wx,wy,wz}" : float
    }
    """

    kinematics: RobotKinematics
    end_effector_step_sizes: dict
    motor_names: list[str]

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        act = transition.get(TransitionKey.ACTION) or {}
        comp = transition.get(TransitionKey.COMPLEMENTARY_DATA) or {}

        # get joint positions from complimentary data
        raw = comp.get("raw_joint_positions", None)
        if raw is None:
            raise ValueError(
                "raw_joint_positions is not in complementary data and is required for EEReferenceAndDelta"
            )

        q = np.array([float(raw[n]) for n in self.motor_names], dtype=float)

        # current pose from FK on measured joints
        t_curr = self.kinematics.forward_kinematics(q)

        enabled = bool(act.pop("action.enabled", 0))
        tx = float(act.pop("action.target_x", 0.0))
        ty = float(act.pop("action.target_y", 0.0))
        tz = float(act.pop("action.target_z", 0.0))
        wx = float(act.pop("action.target_wx", 0.0))
        wy = float(act.pop("action.target_wy", 0.0))
        wz = float(act.pop("action.target_wz", 0.0))

        new_act = dict(act)
        if enabled:
            delta_p = np.array(
                [
                    tx * self.end_effector_step_sizes["x"],
                    ty * self.end_effector_step_sizes["y"],
                    tz * self.end_effector_step_sizes["z"],
                ]
            )
            r_delta = Rotation.from_rotvec([wx, wy, wz]).as_matrix()

            t_des = np.eye(4)
            t_des[:3, :3] = t_curr[:3, :3] @ r_delta
            t_des[:3, 3] = t_curr[:3, 3] + delta_p

            # Add as absolute desired pose as scalars (pos + twist)
            pos = t_des[:3, 3]
            tw = Rotation.from_matrix(t_des[:3, :3]).as_rotvec()
            new_act["action.ee.x"] = float(pos[0])
            new_act["action.ee.y"] = float(pos[1])
            new_act["action.ee.z"] = float(pos[2])
            new_act["action.ee.wx"] = float(tw[0])
            new_act["action.ee.wy"] = float(tw[1])
            new_act["action.ee.wz"] = float(tw[2])
        else:
            new_act["action.ee.x"] = None
            new_act["action.ee.y"] = None
            new_act["action.ee.z"] = None
            new_act["action.ee.wx"] = None
            new_act["action.ee.wy"] = None
            new_act["action.ee.wz"] = None

        transition[TransitionKey.ACTION] = new_act
        return transition

    def feature_contract(self, features: dict[str, PolicyFeature]) -> dict[str, PolicyFeature]:
        # remove consumed fields; add desired pose
        for k in [
            "action.enabled",
            "action.target_x",
            "action.target_y",
            "action.target_z",
            "action.target_wx",
            "action.target_wy",
            "action.target_wz",
        ]:
            features.pop(k, None)
        features["action.ee.x"] = float
        features["action.ee.y"] = float
        features["action.ee.z"] = float
        features["action.ee.wx"] = float
        features["action.ee.wy"] = float
        features["action.ee.wz"] = float
        return features


@ProcessorStepRegistry.register("ee_bounds_and_safety")
@dataclass
class EEBoundsAndSafety:
    """
    Clip the end-effector pose to the bounds and check for jumps.

    Input ACTION keys:
    {
        "action.ee.{x,y,z,wx,wy,wz}" : float
    }

    Output ACTION keys:
    {
        "action.ee.{x,y,z,wx,wy,wz}" : float
    }
    """

    end_effector_bounds: dict
    max_ee_step_m: float = 0.05
    max_ee_twist_step_rad: float = 0.20
    _last_pos: np.ndarray | None = field(default=None, init=False, repr=False)
    _last_twist: np.ndarray | None = field(default=None, init=False, repr=False)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        act = transition.get(TransitionKey.ACTION) or {}
        x = act.pop("action.ee.x", None)
        y = act.pop("action.ee.y", None)
        z = act.pop("action.ee.z", None)
        wx = act.pop("action.ee.wx", None)
        wy = act.pop("action.ee.wy", None)
        wz = act.pop("action.ee.wz", None)

        if None in (x, y, z, wx, wy, wz):
            return transition

        pos = np.array([x, y, z], dtype=float)
        twist = np.array([wx, wy, wz], dtype=float)

        # clip position
        pos = np.clip(pos, self.end_effector_bounds["min"], self.end_effector_bounds["max"])

        # check for jumps in position
        if self._last_pos is not None:
            dpos = pos - self._last_pos
            n = float(np.linalg.norm(dpos))
            if n > self.max_ee_step_m and n > 0:
                pos = self._last_pos + dpos * (self.max_ee_step_m / n)
                raise ValueError(f"EE jump {n:.3f}m > {self.max_ee_step_m}m")

        # check for jumps in twist
        if self._last_twist is not None:
            dtw = twist - self._last_twist
            n = float(np.linalg.norm(dtw))
            if n > self.max_ee_twist_step_rad and n > 0:
                twist = self._last_twist + dtw * (self.max_ee_twist_step_rad / n)
                raise ValueError(f"EE twist jump {n:.3f}rad > {self.max_ee_twist_step_rad}rad")

        self._last_pos = pos
        self._last_twist = twist

        new_act = dict(act)
        new_act["action.ee.x"] = float(pos[0])
        new_act["action.ee.y"] = float(pos[1])
        new_act["action.ee.z"] = float(pos[2])
        new_act["action.ee.wx"] = float(twist[0])
        new_act["action.ee.wy"] = float(twist[1])
        new_act["action.ee.wz"] = float(twist[2])
        transition[TransitionKey.ACTION] = new_act
        return transition

    def reset(self):
        self._last_pos = None
        self._last_twist = None

    def feature_contract(self, features: dict[str, PolicyFeature]) -> dict[str, PolicyFeature]:
        return features


@ProcessorStepRegistry.register("inverse_kinematics_ee_to_joints")
@dataclass
class InverseKinematicsEEToJoints:
    """
    Compute the desired joint positions from the desired end-effector pose.

    Input ACTION keys:
    {
        "action.ee.{x,y,z,wx,wy,wz}" : float
        "complementary_data.raw_joint_positions": dict,
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
        act = transition.get(TransitionKey.ACTION) or {}
        comp = transition.get(TransitionKey.COMPLEMENTARY_DATA) or {}

        x = act.pop("action.ee.x", None)
        y = act.pop("action.ee.y", None)
        z = act.pop("action.ee.z", None)
        wx = act.pop("action.ee.wx", None)
        wy = act.pop("action.ee.wy", None)
        wz = act.pop("action.ee.wz", None)

        if None in (x, y, z, wx, wy, wz):
            # Nothing to do; restore what we popped and return
            act.update(
                {
                    "action.ee.x": x,
                    "action.ee.y": y,
                    "action.ee.z": z,
                    "action.ee.wx": wx,
                    "action.ee.wy": wy,
                    "action.ee.wz": wz,
                }
            )
            transition[TransitionKey.ACTION] = act
            return transition

        # Get joint positions from complimentary data
        raw = comp.get("raw_joint_positions", None)
        if raw is None:
            raise ValueError(
                "raw_joint_positions is not in complementary data and is required for EEReferenceAndDelta"
            )

        q_curr = np.array([float(raw[n]) for n in self.motor_names], dtype=float)

        # Build desired 4x4 transform from pos + rotvec (twist)
        t_des = np.eye(4, dtype=float)
        t_des[:3, :3] = Rotation.from_rotvec([wx, wy, wz]).as_matrix()
        t_des[:3, 3] = [x, y, z]

        # Compute inverse kinematics
        q_target = self.kinematics.inverse_kinematics(q_curr, t_des)

        new_act = dict(act)
        for i, name in enumerate(self.motor_names):
            new_act[f"action.{name}.pos"] = float(q_target[i])
        transition[TransitionKey.ACTION] = new_act
        return transition

    def feature_contract(self, features: dict[str, PolicyFeature]) -> dict[str, PolicyFeature]:
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
        "observation.state.{joint_name_1,joint_name_2,...,joint_name_n}.pos": float,
    }

    Output OBSERVATION keys:
    {
        "observation.state.ee.{x,y,z,wx,wy,wz}" : float
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
        tw = Rotation.from_matrix(t[:3, :3]).as_rotvec()

        new_obs = dict(obs)
        new_obs["observation.state.ee.x"] = float(pos[0])
        new_obs["observation.state.ee.y"] = float(pos[1])
        new_obs["observation.state.ee.z"] = float(pos[2])
        new_obs["observation.state.ee.wx"] = float(tw[0])
        new_obs["observation.state.ee.wy"] = float(tw[1])
        new_obs["observation.state.ee.wz"] = float(tw[2])
        transition[TransitionKey.OBSERVATION] = new_obs
        return transition

    def feature_contract(self, features: dict[str, PolicyFeature]) -> dict[str, PolicyFeature]:
        # add EE pose under observation.state.ee.*
        for k in ["x", "y", "z", "wx", "wy", "wz"]:
            features[f"observation.state.ee.{k}"] = float
        return features


@ProcessorStepRegistry.register("add_robot_observation")
@dataclass
class AddRobotObservationAsComplimentaryData:
    """
    Read the robot's current observation and insert it into the transition as complementary data.

    - Joint positions are added under complementary_data["raw_joint_positions"] as a dict:
        { "<motor_name>": <float position>, ... }
    """

    robot: Robot

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        obs = self.robot.get_observation()

        raw_joint_positions = {
            k.removesuffix(".pos"): float(v)
            for k, v in obs.items()
            if isinstance(k, str) and k.endswith(".pos")
        }

        comp = transition.get(TransitionKey.COMPLEMENTARY_DATA) or {}
        new_comp = dict(comp)
        new_comp["raw_joint_positions"] = raw_joint_positions
        transition[TransitionKey.COMPLEMENTARY_DATA] = new_comp
        return transition

    def feature_contract(self, features: dict[str, PolicyFeature]) -> dict[str, PolicyFeature]:
        return features
