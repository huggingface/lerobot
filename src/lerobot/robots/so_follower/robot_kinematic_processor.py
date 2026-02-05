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
from typing import Any

import numpy as np
import time

from lerobot.utils.visualization_utils import visualize_robot, parse_urdf_graph

from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import (
    EnvTransition,
    ObservationProcessorStep,
    ProcessorStep,
    ProcessorStepRegistry,
    RobotAction,
    RobotObservation,
    RobotActionProcessorStep,
    RobotObservation,
    TransitionKey,
)
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
    use_ik_solution: bool = False

    reference_ee_pose: np.ndarray | None = field(default=None, init=False, repr=False)
    _prev_enabled: bool = field(default=False, init=False, repr=False)
    _command_when_disabled: np.ndarray | None = field(default=None, init=False, repr=False)

    def action(self, action: RobotAction) -> RobotAction:
        observation = self.transition.get(TransitionKey.OBSERVATION).copy()

        if observation is None:
            raise ValueError("Joints observation is require for computing robot kinematics")

        if self.use_ik_solution and "IK_solution" in self.transition.get(TransitionKey.COMPLEMENTARY_DATA):
            q_raw = self.transition.get(TransitionKey.COMPLEMENTARY_DATA)["IK_solution"]
        else:
            q_raw = np.array(
                [
                    float(v)
                    for k, v in observation.items()
                    if isinstance(k, str)
                    and k.endswith(".pos")
                    and k.removesuffix(".pos") in self.motor_names
                ],
                dtype=float,
            )

        if q_raw is None:
            raise ValueError("Joints observation is require for computing robot kinematics")

        # Current pose from FK on measured joints
        t_curr = self.kinematics.forward_kinematics(q_raw)

        enabled = bool(action.pop("enabled"))
        tx = float(action.pop("target_x"))
        ty = float(action.pop("target_y"))
        tz = float(action.pop("target_z"))
        wx = float(action.pop("target_wx"))
        wy = float(action.pop("target_wy"))
        wz = float(action.pop("target_wz"))
        gripper_vel = float(action.pop("gripper_vel"))

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
        action["ee.x"] = float(pos[0])
        action["ee.y"] = float(pos[1])
        action["ee.z"] = float(pos[2])
        action["ee.wx"] = float(tw[0])
        action["ee.wy"] = float(tw[1])
        action["ee.wz"] = float(tw[2])
        action["ee.gripper_vel"] = gripper_vel

        self._prev_enabled = enabled
        return action

    def reset(self):
        """Resets the internal state of the processor."""
        self._prev_enabled = False
        self.reference_ee_pose = None
        self._command_when_disabled = None

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        for feat in [
            "enabled",
            "target_x",
            "target_y",
            "target_z",
            "target_wx",
            "target_wy",
            "target_wz",
            "gripper_vel",
        ]:
            features[PipelineFeatureType.ACTION].pop(f"{feat}", None)

        for feat in ["x", "y", "z", "wx", "wy", "wz", "gripper_vel"]:
            features[PipelineFeatureType.ACTION][f"ee.{feat}"] = PolicyFeature(
                type=FeatureType.ACTION, shape=(1,)
            )

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
        _last_pos: Internal state storing the last commanded position.
    """

    end_effector_bounds: dict
    max_ee_step_m: float = 0.05
    _last_pos: np.ndarray | None = field(default=None, init=False, repr=False)
    _last_twist: np.ndarray | None = field(default=None, init=False, repr=False)
    prefix: str = ""

    def action(self, action: RobotAction) -> RobotAction:
        x = action[f"{self.prefix}ee.x"]
        y = action[f"{self.prefix}ee.y"]
        z = action[f"{self.prefix}ee.z"]
        wx = action[f"{self.prefix}ee.wx"]
        wy = action[f"{self.prefix}ee.wy"]
        wz = action[f"{self.prefix}ee.wz"]
        # TODO(Steven): ee.gripper_vel does not need to be bounded

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

        action[f"{self.prefix}ee.x"] = float(pos[0])
        action[f"{self.prefix}ee.y"] = float(pos[1])
        action[f"{self.prefix}ee.z"] = float(pos[2])
        action[f"{self.prefix}ee.wx"] = float(twist[0])
        action[f"{self.prefix}ee.wy"] = float(twist[1])
        action[f"{self.prefix}ee.wz"] = float(twist[2])
        return action

    def reset(self):
        """Resets the last known position and orientation."""
        self._last_pos = None

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


def angular_diff(a, b):
    diff = (a - b + 180) % 360 - 180
    return abs(diff)

@ProcessorStepRegistry.register("inverse_kinematics_ee_to_joints")
@dataclass
class InverseKinematicsEEToJoints(RobotActionProcessorStep):
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
        display_data: If True, visualize the robot in rerun.
        entity_path_prefix: Prefix for the rerun entity path.
        offset: Y-axis offset for visualization.
    """

    kinematics: RobotKinematics
    motor_names: list[str]
    q_curr: np.ndarray | None = field(default=None, init=False, repr=False)
    initial_guess_current_joints: bool = True
    prefix: str = ""
    threshold_deg: float = 90.0
    display_data: bool = False
    entity_path_prefix: str = "follower"
    offset: float = 0.0
    _first_solve: bool = field(default=True, init=False, repr=False)
    _rerun_initialized: bool = field(default=False, init=False, repr=False)
    _urdf_graph: dict | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        """Initialize rerun logging if display_data is enabled."""
        if self.display_data and not self._rerun_initialized:
            import rerun as rr
            rr.log_file_from_path(
                self.kinematics.urdf_path,
                entity_path_prefix=self.entity_path_prefix,
                static=True
            )
            # Parse URDF graph for visualization
            self._urdf_graph = parse_urdf_graph(self.kinematics.urdf_path)
            self._rerun_initialized = True

    def action(self, action: RobotAction) -> RobotAction:
        x = action.pop(f"{self.prefix}ee.x")
        y = action.pop(f"{self.prefix}ee.y")
        z = action.pop(f"{self.prefix}ee.z")
        wx = action.pop(f"{self.prefix}ee.wx")
        wy = action.pop(f"{self.prefix}ee.wy")
        wz = action.pop(f"{self.prefix}ee.wz")
        gripper_pos = action.pop(f"{self.prefix}ee.gripper_pos")
        
        print(f"x: {x}, y: {y}, z: {z}, wx: {wx}, wy: {wy}, wz: {wz}, gripper_pos: {gripper_pos}")

        if None in (x, y, z, wx, wy, wz, gripper_pos):
            raise ValueError(
                "Missing required end-effector pose components: ee.x, ee.y, ee.z, ee.wx, ee.wy, ee.wz, ee.gripper_pos must all be present in action"
            )

        observation = self.transition.get(TransitionKey.OBSERVATION).copy()
        if observation is None:
            raise ValueError("Joints observation is require for computing robot kinematics")

        q_raw = np.array(
            [float(v) for k, v in observation.items() if isinstance(k, str) and k.endswith(".pos") and k.startswith(f"{self.prefix}")],
            dtype=float,
        )
        if q_raw is None or len(q_raw) == 0:
            raise ValueError("Joints observation is require for computing robot kinematics")

        if self.initial_guess_current_joints:  # Use current joints as initial guess
            self.q_curr = q_raw
        else:  # Use previous ik solution as initial guess
            if self.q_curr is None:
                self.q_curr = q_raw

        # Build desired 4x4 transform from pos + rotvec (twist)
        t_des = np.eye(4, dtype=float)
        t_des[:3, :3] = Rotation.from_rotvec([wx, wy, wz]).as_matrix()
        t_des[:3, 3] = [x, y, z]

        # Compute inverse kinematics
        num_tries = 5 if self._first_solve else 3
        if self._first_solve:
            print("IK first solve, trying multiple times to get an optimal initial solution")
            self.q_curr = q_raw
        for attempt_no in range(num_tries): # Multiple first attempts to get a solution
            if attempt_no > 0:
                print(f"IK retry attempt {attempt_no} to get an optimal initial solution")
                self.q_curr = q_raw # Reset to current joints for subsequent tries
            # Try more for the first time
            if self._first_solve:
                for _ in range(num_tries):
                    q_target = self.kinematics.inverse_kinematics(self.q_curr, t_des)
                    self.q_curr = q_target
            else:
                q_target = self.kinematics.inverse_kinematics(self.q_curr, t_des)
                self.q_curr = q_target
            # TODO: This is sentitive to order of motor_names = q_target mapping
            for i, name in enumerate(self.motor_names):
                if "gripper" not in name:
                    action[f"{name}.pos"] = float(q_target[i])
                else:
                    action[f"{name}.pos"] = float(gripper_pos)

            if self.verify_solution_within_joint_limits(action, observation):
                break

        # Visualize the robot if enabled
        if self.display_data and self._urdf_graph is not None:
            offset = np.eye(4)
            offset[1, 3] = self.offset
            visualize_robot(
                self.kinematics.robot,
                step=int(time.time()),
                urdf_prefix=f"{self.entity_path_prefix}/robot",
                urdf_graph=self._urdf_graph,
                offset=offset,
            )

        self._first_solve = False
        if not self.verify_solution_within_joint_limits(action, observation):
            raise ValueError("Inverse kinematics failed to find a valid solution within joint safety limits.")
        return action

    def verify_solution_within_joint_limits(self, action: RobotAction, observation: RobotObservation) -> bool:
        for motor_name in self.motor_names[:-3]: # exclude gripper and wrist roll and wrist flex
            full_motor_name = f"{motor_name}.pos"
            target_pos = action[full_motor_name]
            current_pos = observation[full_motor_name]
            if angular_diff(target_pos, current_pos) > self.threshold_deg:
                print(f"JointSafety: commanded {full_motor_name} is too large, likely issue with your IK solution: target:{target_pos}, current:{current_pos}, threshold: {self.threshold_deg}")
                return False
        return True


    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        for feat in ["x", "y", "z", "wx", "wy", "wz", "gripper_pos"]:
            features[PipelineFeatureType.ACTION].pop(f"{self.prefix}ee.{feat}", None)

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
class GripperVelocityToJoint(RobotActionProcessorStep):
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

    speed_factor: float = 20.0
    clip_min: float = 0.0
    clip_max: float = 100.0
    discrete_gripper: bool = False

    def action(self, action: RobotAction) -> RobotAction:
        observation = self.transition.get(TransitionKey.OBSERVATION).copy()

        gripper_vel = action.pop("ee.gripper_vel")

        if observation is None:
            raise ValueError("Joints observation is require for computing robot kinematics")

        q_raw = np.array(
            [float(v) for k, v in observation.items() if isinstance(k, str) and k.endswith(".pos")],
            dtype=float,
        )
        if q_raw is None:
            raise ValueError("Joints observation is require for computing robot kinematics")

        if self.discrete_gripper:
            # Discrete gripper actions are in [0, 1, 2]
            # 0: open, 1: close, 2: stay
            # We need to shift them to [-1, 0, 1] and then scale them to clip_max
            gripper_vel = (gripper_vel - 1) * self.clip_max

        # Compute desired gripper position
        delta = gripper_vel * float(self.speed_factor)
        # TODO: This assumes gripper is the last specified joint in the robot
        gripper_pos = float(np.clip(q_raw[-1] + delta, self.clip_min, self.clip_max))
        action["ee.gripper_pos"] = gripper_pos

        return action

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        features[PipelineFeatureType.ACTION].pop("ee.gripper_vel", None)
        features[PipelineFeatureType.ACTION]["ee.gripper_pos"] = PolicyFeature(
            type=FeatureType.ACTION, shape=(1,)
        )

        return features


def compute_forward_kinematics_joints_to_ee(
    joints: dict[str, Any], kinematics: RobotKinematics, motor_names: list[str], gripper_name: str
) -> dict[str, Any]:
    motor_joint_values = [joints[f"{n}.pos"] for n in motor_names]

    q = np.array(motor_joint_values, dtype=float)
    t = kinematics.forward_kinematics(q)
    pos = t[:3, 3]
    tw = Rotation.from_matrix(t[:3, :3]).as_rotvec()
    gripper_pos = joints[f"{gripper_name}.pos"]
    for n in motor_names:
        joints.pop(f"{n}.pos")
    if "left_" in motor_names[0]:
        prefix = "left_"
    elif "right_" in motor_names[0]:
        prefix = "right_"
    else:
        prefix = ""
    joints[f"{prefix}ee.x"] = float(pos[0])
    joints[f"{prefix}ee.y"] = float(pos[1])
    joints[f"{prefix}ee.z"] = float(pos[2])
    joints[f"{prefix}ee.wx"] = float(tw[0])
    joints[f"{prefix}ee.wy"] = float(tw[1])
    joints[f"{prefix}ee.wz"] = float(tw[2])
    joints[f"{prefix}ee.gripper_pos"] = float(gripper_pos)
    return joints


@ProcessorStepRegistry.register("forward_kinematics_joints_to_ee_observation")
@dataclass
class ForwardKinematicsJointsToEEObservation(ObservationProcessorStep):
    """
    Computes the end-effector pose from joint positions using forward kinematics (FK).

    This step is typically used to add the robot's Cartesian pose to the observation space,
    which can be useful for visualization or as an input to a policy.

    Attributes:
        kinematics: The robot's kinematic model.
    """

    kinematics: RobotKinematics
    motor_names: list[str]
    gripper_name: str

    def observation(self, observation: RobotObservation) -> RobotObservation:
        return compute_forward_kinematics_joints_to_ee(observation, self.kinematics, self.motor_names, self.gripper_name)

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        # We only use the ee pose in the dataset, so we don't need the joint positions
        for n in self.motor_names:
            features[PipelineFeatureType.OBSERVATION].pop(f"{n}.pos", None)
        # Preserve the prefix
        if "left_" in self.motor_names[0]:
            prefix = "left_"
        elif "right_" in self.motor_names[0]:
            prefix = "right_"
        else:
            prefix = ""
        # We specify the dataset features of this step that we want to be stored in the dataset
        for k in ["x", "y", "z", "wx", "wy", "wz", "gripper_pos"]:
            features[PipelineFeatureType.OBSERVATION][f"{prefix}ee.{k}"] = PolicyFeature(
                type=FeatureType.STATE, shape=(1,)
            )
        return features


@ProcessorStepRegistry.register("forward_kinematics_joints_to_ee_action")
@dataclass
class ForwardKinematicsJointsToEEAction(RobotActionProcessorStep):
    """
    Computes the end-effector pose from joint positions using forward kinematics (FK).

    This step is typically used to add the robot's Cartesian pose to the observation space,
    which can be useful for visualization or as an input to a policy.

    Attributes:
        kinematics: The robot's kinematic model.
        motor_names: A list of motor names for which to compute joint positions.
        gripper_name: Name of the gripper motor.
        display_data: If True, visualize the robot in rerun.
        entity_path_prefix: Prefix for the rerun entity path.
        offset: Y-axis offset for visualization.
    """

    kinematics: RobotKinematics
    motor_names: list[str]
    gripper_name: str
    display_data: bool = False
    entity_path_prefix: str = "follower"
    offset: float = 0.0
    _rerun_initialized: bool = field(default=False, init=False, repr=False)
    _urdf_graph: dict | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        """Initialize rerun logging if display_data is enabled."""
        if self.display_data and not self._rerun_initialized:
            import rerun as rr
            rr.log_file_from_path(
                self.kinematics.urdf_path,
                entity_path_prefix=self.entity_path_prefix,
                static=True
            )
            # Parse URDF graph for visualization
            self._urdf_graph = parse_urdf_graph(self.kinematics.urdf_path)
            self._rerun_initialized = True

    def action(self, action: RobotAction) -> RobotAction:
        result = compute_forward_kinematics_joints_to_ee(action, self.kinematics, self.motor_names, self.gripper_name)

        # Visualize the robot if enabled
        if self.display_data and self._urdf_graph is not None:
            offset_matrix = np.eye(4)
            offset_matrix[1, 3] = self.offset
            visualize_robot(
                self.kinematics.robot,
                step=int(time.time()),
                urdf_prefix=f"{self.entity_path_prefix}/robot",
                urdf_graph=self._urdf_graph,
                offset=offset_matrix,
            )

        return result

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        # We only use the ee pose in the dataset, so we don't need the joint positions
        for n in self.motor_names:
            features[PipelineFeatureType.ACTION].pop(f"{n}.pos", None)
        # Preserve the prefix
        if "left_" in self.motor_names[0]:
            prefix = "left_"
        elif "right_" in self.motor_names[0]:
            prefix = "right_"
        else:
            prefix = ""
        # We specify the dataset features of this step that we want to be stored in the dataset
        for k in ["x", "y", "z", "wx", "wy", "wz", "gripper_pos"]:
            features[PipelineFeatureType.ACTION][f"{prefix}ee.{k}"] = PolicyFeature(
                type=FeatureType.STATE, shape=(1,)
            )
        return features


@ProcessorStepRegistry.register(name="forward_kinematics_joints_to_ee")
@dataclass
class ForwardKinematicsJointsToEE(ProcessorStep):
    kinematics: RobotKinematics
    motor_names: list[str]
    gripper_name: str
    display_data: bool = False
    entity_path_prefix: str = "follower"
    offset: float = 0.0

    def __post_init__(self):
        self.joints_to_ee_action_processor = ForwardKinematicsJointsToEEAction(
            kinematics=self.kinematics,
            motor_names=self.motor_names,
            gripper_name=self.gripper_name,
            display_data=self.display_data,
            entity_path_prefix=self.entity_path_prefix,
            offset=self.offset,
        )
        # self.joints_to_ee_observation_processor = ForwardKinematicsJointsToEEObservation(
        #     kinematics=self.kinematics, motor_names=self.motor_names, gripper_name=self.gripper_name
        # )

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        # if transition.get(TransitionKey.ACTION) is not None and len(transition.get(TransitionKey.ACTION)) > 0:
        transition = self.joints_to_ee_action_processor(transition)
        # if transition.get(TransitionKey.OBSERVATION) is not None and len(transition.get(TransitionKey.OBSERVATION)) > 0:
        #     transition = self.joints_to_ee_observation_processor(transition)
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        # if features[PipelineFeatureType.ACTION] is not None and len(features[PipelineFeatureType.ACTION]) > 0:
        features = self.joints_to_ee_action_processor.transform_features(features)
        # if features[PipelineFeatureType.OBSERVATION] is not None and len(features[PipelineFeatureType.OBSERVATION]) > 0:
        #     features = self.joints_to_ee_observation_processor.transform_features(features)
        return features


@ProcessorStepRegistry.register("inverse_kinematics_rl_step")
@dataclass
class InverseKinematicsRLStep(ProcessorStep):
    """
    Computes desired joint positions from a target end-effector pose using inverse kinematics (IK).

    This is modified from the InverseKinematicsEEToJoints step to be used in the RL pipeline.
    """

    kinematics: RobotKinematics
    motor_names: list[str]
    q_curr: np.ndarray | None = field(default=None, init=False, repr=False)
    initial_guess_current_joints: bool = True

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        new_transition = dict(transition)
        action = new_transition.get(TransitionKey.ACTION)
        if action is None:
            raise ValueError("Action is required for InverseKinematicsEEToJoints")
        action = dict(action)

        x = action.pop("ee.x")
        y = action.pop("ee.y")
        z = action.pop("ee.z")
        wx = action.pop("ee.wx")
        wy = action.pop("ee.wy")
        wz = action.pop("ee.wz")
        gripper_pos = action.pop("ee.gripper_pos")

        if None in (x, y, z, wx, wy, wz, gripper_pos):
            raise ValueError(
                "Missing required end-effector pose components: ee.x, ee.y, ee.z, ee.wx, ee.wy, ee.wz, ee.gripper_pos must all be present in action"
            )

        observation = new_transition.get(TransitionKey.OBSERVATION).copy()
        if observation is None:
            raise ValueError("Joints observation is require for computing robot kinematics")

        q_raw = np.array(
            [float(v) for k, v in observation.items() if isinstance(k, str) and k.endswith(".pos")],
            dtype=float,
        )
        if q_raw is None:
            raise ValueError("Joints observation is require for computing robot kinematics")

        if self.initial_guess_current_joints:  # Use current joints as initial guess
            self.q_curr = q_raw
        else:  # Use previous ik solution as initial guess
            if self.q_curr is None:
                self.q_curr = q_raw

        # Build desired 4x4 transform from pos + rotvec (twist)
        t_des = np.eye(4, dtype=float)
        t_des[:3, :3] = Rotation.from_rotvec([wx, wy, wz]).as_matrix()
        t_des[:3, 3] = [x, y, z]

        # Compute inverse kinematics
        q_target = self.kinematics.inverse_kinematics(self.q_curr, t_des)
        self.q_curr = q_target

        # TODO: This is sentitive to order of motor_names = q_target mapping
        for i, name in enumerate(self.motor_names):
            if name != "gripper":
                action[f"{name}.pos"] = float(q_target[i])
            else:
                action["gripper.pos"] = float(gripper_pos)

        new_transition[TransitionKey.ACTION] = action
        complementary_data = new_transition.get(TransitionKey.COMPLEMENTARY_DATA, {})
        complementary_data["IK_solution"] = q_target
        new_transition[TransitionKey.COMPLEMENTARY_DATA] = complementary_data
        return new_transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        for feat in ["x", "y", "z", "wx", "wy", "wz", "gripper_pos"]:
            features[PipelineFeatureType.ACTION].pop(f"ee.{feat}", None)

        for name in self.motor_names:
            features[PipelineFeatureType.ACTION][f"{name}.pos"] = PolicyFeature(
                type=FeatureType.ACTION, shape=(1,)
            )

        return features

    def reset(self):
        """Resets the initial guess for the IK solver."""
        self.q_curr = None
