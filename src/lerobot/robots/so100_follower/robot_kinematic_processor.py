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

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.constants import ACTION, OBS_STATE
from lerobot.model.kinematics import RobotKinematics
from lerobot.processor.pipeline import (
    ActionProcessor,
    ComplementaryDataProcessor,
    EnvTransition,
    ObservationProcessor,
    ProcessorStep,
    ProcessorStepRegistry,
    TransitionKey,
)
from lerobot.robots.robot import Robot


@ProcessorStepRegistry.register("ee_reference_and_delta")
@dataclass
class EEReferenceAndDelta(ActionProcessor):
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
    use_latched_reference: bool = (
        True  # If True, latch reference on enable; if False, always use current pose
    )

    reference_ee_pose: np.ndarray | None = field(default=None, init=False, repr=False)
    _prev_enabled: bool = field(default=False, init=False, repr=False)
    _command_when_disabled: np.ndarray | None = field(default=None, init=False, repr=False)

    def action(self, action):
        new_action = action.copy()
        comp = self.transition.get(TransitionKey.COMPLEMENTARY_DATA)

        # Get joint positions from complimentary data
        raw = comp.get("raw_joint_positions", None)
        if raw is None:
            raise ValueError(
                "raw_joint_positions is not in complementary data and is required for EEReferenceAndDelta"
            )

        if "reference_joint_positions" in comp:
            q = comp["reference_joint_positions"]
        else:
            q = np.array([float(raw[n]) for n in self.motor_names], dtype=float)

        # Current pose from FK on measured joints
        t_curr = self.kinematics.forward_kinematics(q)

        enabled = bool(new_action.pop(f"{ACTION}.enabled", 0))
        tx = float(new_action.pop(f"{ACTION}.target_x", 0.0))
        ty = float(new_action.pop(f"{ACTION}.target_y", 0.0))
        tz = float(new_action.pop(f"{ACTION}.target_z", 0.0))
        wx = float(new_action.pop(f"{ACTION}.target_wx", 0.0))
        wy = float(new_action.pop(f"{ACTION}.target_wy", 0.0))
        wz = float(new_action.pop(f"{ACTION}.target_wz", 0.0))

        desired = None

        if enabled:
            ref = t_curr
            if self.use_latched_reference:
                # Latched reference mode: latch reference at the rising edge
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
        new_action[f"{ACTION}.ee.x"] = float(pos[0])
        new_action[f"{ACTION}.ee.y"] = float(pos[1])
        new_action[f"{ACTION}.ee.z"] = float(pos[2])
        new_action[f"{ACTION}.ee.wx"] = float(tw[0])
        new_action[f"{ACTION}.ee.wy"] = float(tw[1])
        new_action[f"{ACTION}.ee.wz"] = float(tw[2])

        self._prev_enabled = enabled
        return new_action

    def reset(self):
        self._prev_enabled = False
        self.reference_ee_pose = None
        self._command_when_disabled = None

    def transform_features(self, features: dict[str, PolicyFeature]) -> dict[str, PolicyFeature]:
        # Create a copy to avoid modifying the input dictionary
        new_features = features.copy()

        new_features.pop(f"{ACTION}.enabled", None)
        new_features.pop(f"{ACTION}.target_x", None)
        new_features.pop(f"{ACTION}.target_y", None)
        new_features.pop(f"{ACTION}.target_z", None)
        new_features.pop(f"{ACTION}.target_wx", None)
        new_features.pop(f"{ACTION}.target_wy", None)
        new_features.pop(f"{ACTION}.target_wz", None)

        new_features["action.ee.x"] = (PolicyFeature(type=FeatureType.ACTION, shape=(1,)),)
        new_features["action.ee.y"] = (PolicyFeature(type=FeatureType.ACTION, shape=(1,)),)
        new_features["action.ee.z"] = (PolicyFeature(type=FeatureType.ACTION, shape=(1,)),)
        new_features["action.ee.wx"] = (PolicyFeature(type=FeatureType.ACTION, shape=(1,)),)
        new_features["action.ee.wy"] = (PolicyFeature(type=FeatureType.ACTION, shape=(1,)),)
        new_features["action.ee.wz"] = (PolicyFeature(type=FeatureType.ACTION, shape=(1,)),)
        return new_features


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
    _last_twist: np.ndarray | None = field(default=None, init=False, repr=False)

    def action(self, act: dict) -> dict:
        x = act.get(f"{ACTION}.ee.x", None)
        y = act.get(f"{ACTION}.ee.y", None)
        z = act.get(f"{ACTION}.ee.z", None)
        wx = act.get(f"{ACTION}.ee.wx", None)
        wy = act.get(f"{ACTION}.ee.wy", None)
        wz = act.get(f"{ACTION}.ee.wz", None)

        if None in (x, y, z, wx, wy, wz):
            return act

        # Create a copy to avoid modifying the input dictionary
        new_act = act.copy()

        pos = np.array([x, y, z], dtype=float)
        twist = np.array([wx, wy, wz], dtype=float)

        # Clip position
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

        new_act[f"{ACTION}.ee.x"] = float(pos[0])
        new_act[f"{ACTION}.ee.y"] = float(pos[1])
        new_act[f"{ACTION}.ee.z"] = float(pos[2])
        new_act[f"{ACTION}.ee.wx"] = float(twist[0])
        new_act[f"{ACTION}.ee.wy"] = float(twist[1])
        new_act[f"{ACTION}.ee.wz"] = float(twist[2])
        return new_act

    def reset(self):
        self._last_pos = None
        self._last_twist = None


@ProcessorStepRegistry.register("inverse_kinematics_ee_to_joints")
@dataclass
class InverseKinematicsEEToJoints(ProcessorStep):
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
    q_curr: np.ndarray | None = field(default=None, init=False, repr=False)
    initial_guess_current_joints: bool = True

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        act = transition.get(TransitionKey.ACTION) or {}
        comp = transition.get(TransitionKey.COMPLEMENTARY_DATA) or {}

        x = act.get(f"{ACTION}.ee.x", None)
        y = act.get(f"{ACTION}.ee.y", None)
        z = act.get(f"{ACTION}.ee.z", None)
        wx = act.get(f"{ACTION}.ee.wx", None)
        wy = act.get(f"{ACTION}.ee.wy", None)
        wz = act.get(f"{ACTION}.ee.wz", None)

        if None in (x, y, z, wx, wy, wz):
            return transition

        # Get joint positions from complimentary data
        raw = comp.get("raw_joint_positions", None)
        if raw is None:
            raise ValueError(
                "raw_joint_positions is not in complementary data and is required for EEReferenceAndDelta"
            )

        if self.initial_guess_current_joints:  # Use current joints as initial guess
            self.q_curr = np.array([float(raw[n]) for n in self.motor_names], dtype=float)
        else:  # Use previous ik solution as initial guess
            if self.q_curr is None:
                self.q_curr = np.array([float(raw[n]) for n in self.motor_names], dtype=float)

        # Build desired 4x4 transform from pos + rotvec (twist)
        t_des = np.eye(4, dtype=float)
        t_des[:3, :3] = Rotation.from_rotvec([wx, wy, wz]).as_matrix()
        t_des[:3, 3] = [x, y, z]

        # Compute inverse kinematics
        q_target = self.kinematics.inverse_kinematics(self.q_curr, t_des)
        self.q_curr = q_target

        # Create a copy of the transition to avoid modifying the input
        new_transition = transition.copy()

        new_act = dict(act)
        for i, name in enumerate(self.motor_names):
            if name == "gripper":
                new_act[f"{OBS_STATE}.gripper.pos"] = float(raw["gripper"])
            else:
                new_act[f"{ACTION}.{name}.pos"] = float(q_target[i])
        new_transition[TransitionKey.ACTION] = new_act

        if not self.initial_guess_current_joints:
            new_comp = comp.copy()
            new_comp["reference_joint_positions"] = q_target
            new_transition[TransitionKey.COMPLEMENTARY_DATA] = new_comp

        return new_transition

    def transform_features(self, features: dict[str, PolicyFeature]) -> dict[str, PolicyFeature]:
        # Create a copy to avoid modifying the input dictionary
        new_features = features.copy()

        new_features[f"{OBS_STATE}.gripper.pos"] = (PolicyFeature(type=FeatureType.ACTION, shape=(1,)),)
        new_features[f"{ACTION}.gripper.pos"] = (PolicyFeature(type=FeatureType.ACTION, shape=(1,)),)
        for name in self.motor_names:
            new_features[f"{ACTION}.{name}.pos"] = (PolicyFeature(type=FeatureType.ACTION, shape=(1,)),)

        return new_features

    def reset(self):
        self.q_curr = None


@ProcessorStepRegistry.register("gripper_velocity_to_joint")
@dataclass
class GripperVelocityToJoint(ProcessorStep):
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
    discrete_gripper: bool = False

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        obs = transition.get(TransitionKey.OBSERVATION) or {}
        act = transition.get(TransitionKey.ACTION) or {}
        comp = transition.get(TransitionKey.COMPLEMENTARY_DATA) or {}

        if f"{ACTION}.gripper" not in act:
            return transition

        if "gripper" not in self.motor_names:
            new_transition = transition.copy()
            new_act = dict(act)
            new_act.pop(f"{ACTION}.gripper", None)
            new_transition[TransitionKey.ACTION] = new_act
            return new_transition

        # Create a copy of the transition to avoid modifying the input
        new_transition = transition.copy()
        new_act = dict(act)

        if self.discrete_gripper:
            # Discrete gripper actions are in [0, 1, 2]
            # 0: open, 1: close, 2: stay
            # We need to shift them to [-1, 0, 1] and then scale them to clip_max
            gripper_action = new_act.get(f"{ACTION}.gripper", 1.0)
            gripper_action = gripper_action - 1.0
            gripper_action *= self.clip_max
            new_act[f"{ACTION}.gripper"] = gripper_action

        # Get current gripper position from complementary data
        raw = comp.get("raw_joint_positions") or {}
        curr_pos = float(raw.get("gripper"))

        # Compute desired gripper velocity
        u = float(new_act.get(f"{ACTION}.gripper", 0.0))
        delta = u * float(self.speed_factor)
        gripper_pos = float(np.clip(curr_pos + delta, self.clip_min, self.clip_max))

        new_act[f"{ACTION}.gripper.pos"] = gripper_pos
        new_act.pop(f"{ACTION}.gripper", None)
        new_transition[TransitionKey.ACTION] = new_act

        new_obs = obs.copy()
        new_obs[f"{OBS_STATE}.gripper.pos"] = curr_pos
        new_transition[TransitionKey.OBSERVATION] = new_obs
        return new_transition

    def transform_features(self, features: dict[str, PolicyFeature]) -> dict[str, PolicyFeature]:
        # Create a copy to avoid modifying the input dictionary
        new_features = features.copy()

        new_features.pop(f"{ACTION}.gripper", None)
        new_features[f"{ACTION}.gripper.pos"] = (PolicyFeature(type=FeatureType.ACTION, shape=(1,)),)
        return new_features


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

    def observation(self, obs: dict) -> dict:
        if not all(f"{OBS_STATE}.{n}.pos" in obs for n in self.motor_names):
            return obs

        # Create a copy to avoid modifying the input dictionary
        new_obs = obs.copy()

        q = np.array([obs[f"{OBS_STATE}.{n}.pos"] for n in self.motor_names], dtype=float)
        t = self.kinematics.forward_kinematics(q)
        pos = t[:3, 3]
        tw = Rotation.from_matrix(t[:3, :3]).as_rotvec()

        new_obs[f"{OBS_STATE}.ee.x"] = float(pos[0])
        new_obs[f"{OBS_STATE}.ee.y"] = float(pos[1])
        new_obs[f"{OBS_STATE}.ee.z"] = float(pos[2])
        new_obs[f"{OBS_STATE}.ee.wx"] = float(tw[0])
        new_obs[f"{OBS_STATE}.ee.wy"] = float(tw[1])
        new_obs[f"{OBS_STATE}.ee.wz"] = float(tw[2])
        return new_obs

    def transform_features(self, features: dict[str, PolicyFeature]) -> dict[str, PolicyFeature]:
        # Create a copy to avoid modifying the input dictionary
        new_features = features.copy()

        # We specify the dataset features of this step that we want to be stored in the dataset
        for k in ["x", "y", "z", "wx", "wy", "wz"]:
            new_features[f"{OBS_STATE}.ee.{k}"] = (PolicyFeature(type=FeatureType.ACTION, shape=(1,)),)
        return new_features


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
        # Handle None case properly - create copy or empty dict
        new_comp = comp.copy() if comp is not None else {}
        obs = (
            self.robot.get_observation()
        )  # todo(steven): why not self.trtansition.get(TransitionKey.OBSERVATION)?

        new_comp["raw_joint_positions"] = {
            k.removesuffix(".pos"): float(v)
            for k, v in obs.items()
            if isinstance(k, str) and k.endswith(".pos")
        }
        return new_comp
