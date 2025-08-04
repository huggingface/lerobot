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
from lerobot.processor.pipeline import (
    ActionProcessor,
    ComplementaryDataProcessor,
    EnvTransition,
    ObservationProcessor,
    ProcessorStepRegistry,
    TransitionKey,
)
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

    reference_ee_pose: np.ndarray | None = field(default=None, init=False, repr=False)
    _prev_enabled: bool = field(default=False, init=False, repr=False)
    _command_when_disabled: np.ndarray | None = field(default=None, init=False, repr=False)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        act = transition.get(TransitionKey.ACTION) or {}
        comp = transition.get(TransitionKey.COMPLEMENTARY_DATA) or {}

        # Get joint positions from complimentary data
        raw = comp.get("raw_joint_positions", None)
        if raw is None:
            raise ValueError(
                "raw_joint_positions is not in complementary data and is required for EEReferenceAndDelta"
            )

        q = np.array([float(raw[n]) for n in self.motor_names], dtype=float)

        # Current pose from FK on measured joints
        t_curr = self.kinematics.forward_kinematics(q)

        enabled = bool(act.pop("action.enabled", 0))
        tx = float(act.pop("action.target_x", 0.0))
        ty = float(act.pop("action.target_y", 0.0))
        tz = float(act.pop("action.target_z", 0.0))
        wx = float(act.pop("action.target_wx", 0.0))
        wy = float(act.pop("action.target_wy", 0.0))
        wz = float(act.pop("action.target_wz", 0.0))

        desired = None

        if enabled:
            # Latch a reference at the rising edge; also be defensive if None
            if not self._prev_enabled or self.reference_ee_pose is None:
                self.reference_ee_pose = t_curr.copy()

            ref = self.reference_ee_pose if self.reference_ee_pose is not None else t_curr

            delta_p = np.array(
                [
                    tx * self.end_effector_step_sizes["x"],
                    ty * self.end_effector_step_sizes["y"],
                    tz * self.end_effector_step_sizes["z"],
                ],
                dtype=float,
            )
            r_abs = Rotation.from_rotvec([wx, wy, wz]).as_matrix()

            desired = np.eye(4, dtype=float)
            desired[:3, :3] = ref[:3, :3] @ r_abs
            desired[:3, 3] = ref[:3, 3] + delta_p

            self._command_when_disabled = desired.copy()
        else:
            # While disabled, keep sending the same command to avoid drift.
            if self._command_when_disabled is None:
                # If we've never had an enabled command yet, freeze current FK pose once.
                self._command_when_disabled = t_curr.copy()
            desired = self._command_when_disabled.copy()

        # Write action fields
        pos = desired[:3, 3]
        tw = Rotation.from_matrix(desired[:3, :3]).as_rotvec()
        act.update(
            {
                "action.ee.x": float(pos[0]),
                "action.ee.y": float(pos[1]),
                "action.ee.z": float(pos[2]),
                "action.ee.wx": float(tw[0]),
                "action.ee.wy": float(tw[1]),
                "action.ee.wz": float(tw[2]),
            }
        )

        self._prev_enabled = enabled
        transition[TransitionKey.ACTION] = act
        return transition


@ProcessorStepRegistry.register("ee_bounds_and_safety")
@dataclass
class EEBoundsAndSafety(ActionProcessor):
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

    def action(self, act: dict | None) -> dict:
        x = act.pop("action.ee.x", None)
        y = act.pop("action.ee.y", None)
        z = act.pop("action.ee.z", None)
        wx = act.pop("action.ee.wx", None)
        wy = act.pop("action.ee.wy", None)
        wz = act.pop("action.ee.wz", None)

        if None in (x, y, z, wx, wy, wz):
            return act

        pos = np.array([x, y, z], dtype=float)
        twist = np.array([wx, wy, wz], dtype=float)

        # clip position
        pos = np.clip(pos, self.end_effector_bounds["min"], self.end_effector_bounds["max"])

        # Check for jumps in position
        if self._last_pos is not None:
            dpos = pos - self._last_pos
            n = float(np.linalg.norm(dpos))
            if n > self.max_ee_step_m and n > 0:
                pos = self._last_pos + dpos * (self.max_ee_step_m / n)
                raise ValueError(f"EE jump {n:.3f}m > {self.max_ee_step_m}m")

        self._last_pos = pos
        self._last_twist = twist

        act.update(
            {
                "action.ee.x": float(pos[0]),
                "action.ee.y": float(pos[1]),
                "action.ee.z": float(pos[2]),
                "action.ee.wx": float(twist[0]),
                "action.ee.wy": float(twist[1]),
                "action.ee.wz": float(twist[2]),
            }
        )
        return act

    def reset(self):
        self._last_pos = None

    def dataset_features(self, features: dict[str, PolicyFeature]) -> dict[str, PolicyFeature]:
        # Because this is last step we specify the dataset features of this step that we want to be stored in the dataset
        features["action.ee.x"] = float
        features["action.ee.y"] = float
        features["action.ee.z"] = float
        features["action.ee.wx"] = float
        features["action.ee.wy"] = float
        features["action.ee.wz"] = float
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

        x = act.get("action.ee.x", None)
        y = act.get("action.ee.y", None)
        z = act.get("action.ee.z", None)
        wx = act.get("action.ee.wx", None)
        wy = act.get("action.ee.wy", None)
        wz = act.get("action.ee.wz", None)

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

    def dataset_features(self, features: dict[str, PolicyFeature]) -> dict[str, PolicyFeature]:
        # We specify the dataset features of this step that we want to be stored in the dataset
        features["action.ee.x"] = float
        features["action.ee.y"] = float
        features["action.ee.z"] = float
        features["action.ee.wx"] = float
        features["action.ee.wy"] = float
        features["action.ee.wz"] = float
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
    speed_factor: float = 20.0
    clip_min: float = 0.0
    clip_max: float = 100.0

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        obs = transition.get(TransitionKey.OBSERVATION) or {}
        act = transition.get(TransitionKey.ACTION) or {}
        comp = transition.get(TransitionKey.COMPLEMENTARY_DATA) or {}

        if "action.gripper" not in act:
            return transition

        if "gripper" not in self.motor_names:
            new_act = dict(act)
            new_act.pop("action.gripper", None)
            transition[TransitionKey.ACTION] = new_act
            return transition

        # Get current gripper position from complementary data
        raw = comp.get("raw_joint_positions") or {}
        curr_pos = float(raw.get("gripper"))

        # Compute desired gripper velocity
        u = float(act.get("action.gripper", 0.0))
        delta = u * float(self.speed_factor)
        gripper_pos = float(np.clip(curr_pos + delta, self.clip_min, self.clip_max))

        new_act = dict(act)
        new_act["action.gripper.pos"] = gripper_pos
        new_act.pop("action.gripper", None)
        transition[TransitionKey.ACTION] = new_act

        obs.update({"observation.state.gripper.pos": curr_pos})
        transition[TransitionKey.OBSERVATION] = obs
        return transition

    def dataset_features(self, features: dict[str, PolicyFeature]) -> dict[str, PolicyFeature]:
        # We specify the dataset features of this step that we want to be stored in the dataset
        features["observation.state.gripper.pos"] = float
        features["action.gripper.pos"] = float
        return features


@ProcessorStepRegistry.register("forward_kinematics_joints_to_ee")
@dataclass
class ForwardKinematicsJointsToEE(ObservationProcessor):
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

    def observation(self, obs: dict | None) -> dict:
        if not all(f"observation.state.{n}.pos" in obs for n in self.motor_names):
            return obs

        q = np.array([obs[f"observation.state.{n}.pos"] for n in self.motor_names], dtype=float)
        t = self.kinematics.forward_kinematics(q)
        pos = t[:3, 3]
        tw = Rotation.from_matrix(t[:3, :3]).as_rotvec()

        obs.update(
            {
                "observation.state.ee.x": float(pos[0]),
                "observation.state.ee.y": float(pos[1]),
                "observation.state.ee.z": float(pos[2]),
                "observation.state.ee.wx": float(tw[0]),
                "observation.state.ee.wy": float(tw[1]),
                "observation.state.ee.wz": float(tw[2]),
            }
        )
        return obs

    def dataset_features(self, features: dict[str, PolicyFeature]) -> dict[str, PolicyFeature]:
        # We specify the dataset features of this step that we want to be stored in the dataset
        for k in ["x", "y", "z", "wx", "wy", "wz"]:
            features[f"observation.state.ee.{k}"] = float
        return features


@ProcessorStepRegistry.register("add_robot_observation")
@dataclass
class AddRobotObservationAsComplimentaryData(ComplementaryDataProcessor):
    """
    Read the robot's current observation and insert it into the transition as complementary data.

    - Joint positions are added under complementary_data["raw_joint_positions"] as a dict:
        { "<motor_name>": <float position>, ... }
    """

    robot: Robot

    def complementary_data(self, comp: dict | None) -> dict:
        comp = {} if comp is None else dict(comp)
        obs = self.robot.get_observation()

        comp["raw_joint_positions"] = {
            k.removesuffix(".pos"): float(v)
            for k, v in obs.items()
            if isinstance(k, str) and k.endswith(".pos")
        }
        return comp
