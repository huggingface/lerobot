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

from dataclasses import dataclass, field

import numpy as np

from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.constants import OBS_STATE
from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import (
    ComplementaryDataProcessorStep,
    EnvTransition,
    ObservationProcessorStep,
    ProcessorStep,
    ProcessorStepRegistry,
    RobotActionProcessorStep,
    TransitionKey,
)
from lerobot.processor.core import RobotAction
from lerobot.robots.robot import Robot
from lerobot.utils.rotation import Rotation


@ProcessorStepRegistry.register("ee_reference_and_delta")
@dataclass
class EEReferenceAndDelta(RobotActionProcessorStep):
    """
    Computes a target end-effector pose from a relative delta command.

    This step takes a desired change in position and orientation (`target_*`) and applies it to a
    reference end-effector pose to calculate an absolute target pose. The reference pose is derived
    from the current robot joint positions using forward kinematics.

    The processor can operate in two modes:
    1.  `use_latched_reference=True`: The reference pose is "latched" or saved at the moment the action
        is first enabled. Subsequent commands are relative to this fixed reference.
    2.  `use_latched_reference=False`: The reference pose is updated to the robot's current pose at
        every step.

    Attributes:
        kinematics: The robot's kinematic model for forward kinematics.
        end_effector_step_sizes: A dictionary scaling the input delta commands.
        motor_names: A list of motor names required for forward kinematics.
        use_latched_reference: If True, latch the reference pose on enable; otherwise, always use the
            current pose as the reference.
        reference_ee_pose: Internal state storing the latched reference pose.
        _prev_enabled: Internal state to detect the rising edge of the enable signal.
        _command_when_disabled: Internal state to hold the last command while disabled.
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

    def action(self, action: RobotAction) -> RobotAction:
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

        enabled = bool(new_action.pop("enabled", 0))
        tx = float(new_action.pop("target_x", 0.0))
        ty = float(new_action.pop("target_y", 0.0))
        tz = float(new_action.pop("target_z", 0.0))
        wx = float(new_action.pop("target_wx", 0.0))
        wy = float(new_action.pop("target_wy", 0.0))
        wz = float(new_action.pop("target_wz", 0.0))

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
        new_action["ee.x"] = float(pos[0])
        new_action["ee.y"] = float(pos[1])
        new_action["ee.z"] = float(pos[2])
        new_action["ee.wx"] = float(tw[0])
        new_action["ee.wy"] = float(tw[1])
        new_action["ee.wz"] = float(tw[2])

        self._prev_enabled = enabled
        return new_action

    def reset(self):
        """Resets the internal state of the processor."""
        self._prev_enabled = False
        self.reference_ee_pose = None
        self._command_when_disabled = None

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        features[PipelineFeatureType.ACTION].pop("enabled", None)
        features[PipelineFeatureType.ACTION].pop("target_x", None)
        features[PipelineFeatureType.ACTION].pop("target_y", None)
        features[PipelineFeatureType.ACTION].pop("target_z", None)
        features[PipelineFeatureType.ACTION].pop("target_wx", None)
        features[PipelineFeatureType.ACTION].pop("target_wy", None)
        features[PipelineFeatureType.ACTION].pop("target_wz", None)

        features[PipelineFeatureType.ACTION]["ee.x"] = PolicyFeature(type=FeatureType.ACTION, shape=(1,))
        features[PipelineFeatureType.ACTION]["ee.y"] = PolicyFeature(type=FeatureType.ACTION, shape=(1,))
        features[PipelineFeatureType.ACTION]["ee.z"] = PolicyFeature(type=FeatureType.ACTION, shape=(1,))
        features[PipelineFeatureType.ACTION]["ee.wx"] = PolicyFeature(type=FeatureType.ACTION, shape=(1,))
        features[PipelineFeatureType.ACTION]["ee.wy"] = PolicyFeature(type=FeatureType.ACTION, shape=(1,))
        features[PipelineFeatureType.ACTION]["ee.wz"] = PolicyFeature(type=FeatureType.ACTION, shape=(1,))
        return features


@ProcessorStepRegistry.register("ee_bounds_and_safety")
@dataclass
class EEBoundsAndSafety(RobotActionProcessorStep):
    """
    Clips the end-effector pose to predefined bounds and checks for unsafe jumps.

    This step ensures that the target end-effector pose remains within a safe operational workspace.
    It also moderates the command to prevent large, sudden movements between consecutive steps.

    Attributes:
        end_effector_bounds: A dictionary with "min" and "max" keys for position clipping.
        max_ee_step_m: The maximum allowed change in position (in meters) between steps.
        max_ee_twist_step_rad: The maximum allowed change in orientation (in radians) between steps.
        _last_pos: Internal state storing the last commanded position.
        _last_twist: Internal state storing the last commanded orientation.
    """

    end_effector_bounds: dict
    max_ee_step_m: float = 0.05
    max_ee_twist_step_rad: float = 0.20
    _last_pos: np.ndarray | None = field(default=None, init=False, repr=False)
    _last_twist: np.ndarray | None = field(default=None, init=False, repr=False)

    def action(self, act: RobotAction) -> RobotAction:
        x = act.get("ee.x", None)
        y = act.get("ee.y", None)
        z = act.get("ee.z", None)
        wx = act.get("ee.wx", None)
        wy = act.get("ee.wy", None)
        wz = act.get("ee.wz", None)

        if None in (x, y, z, wx, wy, wz):
            raise ValueError(
                "Missing required end-effector pose components: x, y, z, wx, wy, wz must all be present in action"
            )

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

        act["ee.x"] = float(pos[0])
        act["ee.y"] = float(pos[1])
        act["ee.z"] = float(pos[2])
        act["ee.wx"] = float(twist[0])
        act["ee.wy"] = float(twist[1])
        act["ee.wz"] = float(twist[2])
        return act

    def reset(self):
        """Resets the last known position and orientation."""
        self._last_pos = None
        self._last_twist = None

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register("inverse_kinematics_ee_to_joints")
@dataclass
class InverseKinematicsEEToJoints(ProcessorStep):
    """
    Computes desired joint positions from a target end-effector pose using inverse kinematics (IK).

    This step translates a Cartesian command (position and orientation of the end-effector) into
    the corresponding joint-space commands for each motor.

    Attributes:
        kinematics: The robot's kinematic model for inverse kinematics.
        motor_names: A list of motor names for which to compute joint positions.
        q_curr: Internal state storing the last joint positions, used as an initial guess for the IK solver.
        initial_guess_current_joints: If True, use the robot's current joint state as the IK guess.
            If False, use the solution from the previous step.
    """

    kinematics: RobotKinematics
    motor_names: list[str]
    q_curr: np.ndarray | None = field(default=None, init=False, repr=False)
    initial_guess_current_joints: bool = True

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        new_transition = transition.copy()
        act = new_transition.get(TransitionKey.ACTION) or {}

        if not isinstance(act, dict):
            raise ValueError(f"Action should be a RobotAction type got {type(act)}")

        comp = new_transition.get(TransitionKey.COMPLEMENTARY_DATA) or {}

        x = act.get("ee.x", None)
        y = act.get("ee.y", None)
        z = act.get("ee.z", None)
        wx = act.get("ee.wx", None)
        wy = act.get("ee.wy", None)
        wz = act.get("ee.wz", None)

        if None in (x, y, z, wx, wy, wz):
            return new_transition

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

        new_act = dict(act)
        for i, name in enumerate(self.motor_names):
            if name == "gripper":
                # TODO(pepijn): Investigate if this is correct
                # Do we want an observation key in the action field?
                new_act["gripper.pos"] = float(raw["gripper"])
            else:
                new_act[f"{name}.pos"] = float(q_target[i])
        new_transition[TransitionKey.ACTION] = new_act
        if not self.initial_guess_current_joints:
            new_transition[TransitionKey.COMPLEMENTARY_DATA]["reference_joint_positions"] = q_target
        return new_transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        features[PipelineFeatureType.ACTION]["gripper.pos"] = PolicyFeature(
            type=FeatureType.ACTION, shape=(1,)
        )
        for name in self.motor_names:
            features[PipelineFeatureType.ACTION][f"{name}.pos"] = PolicyFeature(
                type=FeatureType.ACTION, shape=(1,)
            )

        return features

    def reset(self):
        """Resets the initial guess for the IK solver."""
        self.q_curr = None


@ProcessorStepRegistry.register("gripper_velocity_to_joint")
@dataclass
class GripperVelocityToJoint(ProcessorStep):
    """
    Converts a gripper velocity command into a target gripper joint position.

    This step integrates a normalized velocity command over time to produce a position command,
    taking the current gripper position as a starting point. It also supports a discrete mode
    where integer actions map to open, close, or no-op.

    Attributes:
        motor_names: A list of motor names, which must include 'gripper'.
        speed_factor: A scaling factor to convert the normalized velocity command to a position change.
        clip_min: The minimum allowed gripper joint position.
        clip_max: The maximum allowed gripper joint position.
        discrete_gripper: If True, treat the input action as discrete (0: open, 1: close, 2: stay).
    """

    motor_names: list[str]
    speed_factor: float = 20.0
    clip_min: float = 0.0
    clip_max: float = 100.0
    discrete_gripper: bool = False

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        new_transition = transition.copy()
        obs = new_transition.get(TransitionKey.OBSERVATION) or {}
        act = new_transition.get(TransitionKey.ACTION) or {}
        comp = new_transition.get(TransitionKey.COMPLEMENTARY_DATA) or {}

        if not isinstance(act, dict):
            raise ValueError(f"Action should be a RobotAction type got {type(act)}")

        if "gripper" not in act:
            raise ValueError("Required action key 'gripper' not found in transition")

        if "gripper" not in self.motor_names:
            raise ValueError(
                f"Required motor name 'gripper' not found in self.motor_names={self.motor_names}"
            )

        if self.discrete_gripper:
            # Discrete gripper actions are in [0, 1, 2]
            # 0: open, 1: close, 2: stay
            # We need to shift them to [-1, 0, 1] and then scale them to clip_max
            gripper_action = act.get("gripper", 1.0)
            gripper_action = gripper_action - 1.0
            gripper_action *= self.clip_max
            act["gripper"] = gripper_action

        # Get current gripper position from complementary data
        raw = comp.get("raw_joint_positions") or {}
        curr_pos = float(raw.get("gripper"))

        # Compute desired gripper velocity
        u = float(act.get("gripper", 0.0))
        delta = u * float(self.speed_factor)
        gripper_pos = float(np.clip(curr_pos + delta, self.clip_min, self.clip_max))

        new_act = dict(act)
        new_act["gripper.pos"] = gripper_pos
        new_act.pop("gripper", None)
        new_transition[TransitionKey.ACTION] = new_act

        obs[f"{OBS_STATE}.gripper.pos"] = curr_pos
        new_transition[TransitionKey.OBSERVATION] = obs
        return new_transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        features[PipelineFeatureType.ACTION].pop("gripper", None)
        features[PipelineFeatureType.ACTION]["gripper.pos"] = PolicyFeature(
            type=FeatureType.ACTION, shape=(1,)
        )
        features[PipelineFeatureType.OBSERVATION][f"{OBS_STATE}.gripper.pos"] = PolicyFeature(
            type=FeatureType.STATE, shape=(1,)
        )

        return features


@ProcessorStepRegistry.register("forward_kinematics_joints_to_ee")
@dataclass
class ForwardKinematicsJointsToEE(ObservationProcessorStep):
    """
    Computes the end-effector pose from joint positions using forward kinematics (FK).

    This step is typically used to add the robot's Cartesian pose to the observation space,
    which can be useful for visualization or as an input to a policy.

    Attributes:
        kinematics: The robot's kinematic model.
        motor_names: A list of motor names whose joint positions are used for FK.
    """

    kinematics: RobotKinematics
    motor_names: list[str]

    def observation(self, obs: dict) -> dict:
        if not all(f"{OBS_STATE}.{n}.pos" in obs for n in self.motor_names):
            raise ValueError(f"Missing required joint positions for motors: {self.motor_names}")

        q = np.array([obs[f"{OBS_STATE}.{n}.pos"] for n in self.motor_names], dtype=float)
        t = self.kinematics.forward_kinematics(q)
        pos = t[:3, 3]
        tw = Rotation.from_matrix(t[:3, :3]).as_rotvec()

        obs[f"{OBS_STATE}.ee.x"] = float(pos[0])
        obs[f"{OBS_STATE}.ee.y"] = float(pos[1])
        obs[f"{OBS_STATE}.ee.z"] = float(pos[2])
        obs[f"{OBS_STATE}.ee.wx"] = float(tw[0])
        obs[f"{OBS_STATE}.ee.wy"] = float(tw[1])
        obs[f"{OBS_STATE}.ee.wz"] = float(tw[2])
        return obs

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        # We specify the dataset features of this step that we want to be stored in the dataset
        for k in ["x", "y", "z", "wx", "wy", "wz"]:
            features[PipelineFeatureType.OBSERVATION][f"{OBS_STATE}.ee.{k}"] = PolicyFeature(
                type=FeatureType.STATE, shape=(1,)
            )
        return features


@ProcessorStepRegistry.register("add_robot_observation")
@dataclass
class AddRobotObservationAsComplimentaryData(ComplementaryDataProcessorStep):
    """
    Reads the robot's current observation and adds it to the transition's complementary data.

    This step acts as a bridge to the physical robot, injecting its real-time sensor readings
    (like raw joint positions) into the data processing pipeline. This data is then available
    for other processing steps.

    Attributes:
        robot: An instance of a `Robot` class used to get observations from hardware.
    """

    robot: Robot

    def complementary_data(self, comp: dict | None) -> dict:
        new_comp = dict(comp)
        obs = (
            self.robot.get_observation()
        )  # todo(steven): why not self.trtansition.get(TransitionKey.OBSERVATION)?

        new_comp["raw_joint_positions"] = {
            k.removesuffix(".pos"): float(v)
            for k, v in obs.items()
            if isinstance(k, str) and k.endswith(".pos")
        }
        return new_comp

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features
