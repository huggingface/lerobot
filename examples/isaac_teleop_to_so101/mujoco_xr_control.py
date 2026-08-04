#!/usr/bin/env python

"""Hardware-free XR clutch + existing LeRobot Cartesian IK control core."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import (
    RobotProcessorPipeline,
    robot_action_observation_to_transition,
    transition_to_robot_action,
)
from lerobot.robots.so_follower.robot_kinematic_processor import (
    EEBoundsAndSafety,
    InverseKinematicsEEToJoints,
)
from lerobot.types import RobotAction, RobotObservation

from .common import IK_ORIENTATION_WEIGHT, MAX_EE_STEP_M, _ensure_so101_urdf
from .isaac_teleop import Clutch, MapXRControllerActionToRobotAction
from .mujoco_sink import MOTOR_NAMES

_IK_REGULARIZATION_WEIGHT = 0.001
_IK_ENGAGE_MAX_ITERATIONS = 100
_IK_ENGAGE_CONVERGENCE_DEG = 1e-6


@dataclass
class ControlMetrics:
    frames: int = 0
    ik_total_s: float = 0.0
    ik_max_s: float = 0.0
    reclutches: int = 0
    max_reclutch_ee_jump_m: float = 0.0
    max_reclutch_joint_jump_deg: float = 0.0

    @property
    def ik_mean_ms(self) -> float:
        return self.ik_total_s / self.frames * 1000.0 if self.frames else 0.0


def build_kinematics() -> RobotKinematics:
    kinematics = RobotKinematics(
        urdf_path=_ensure_so101_urdf(),
        target_frame_name="gripper_frame_link",
        joint_names=list(MOTOR_NAMES),
    )
    kinematics.solver.add_regularization_task(_IK_REGULARIZATION_WEIGHT)
    return kinematics


def _initialize_ik_engage(
    processor,
    ee_action: RobotAction,
    observation: RobotObservation,
) -> RobotAction:
    """Converge the hidden same-pose IK seed before latching its joint baseline."""
    previous: np.ndarray | None = None
    output: RobotAction | None = None
    arm_names = MOTOR_NAMES[:-1]
    for _ in range(_IK_ENGAGE_MAX_ITERATIONS):
        action_input = {
            "ee_pose": np.asarray(ee_action["ee_pose"], dtype=np.float32).copy(),
            "closedness": float(ee_action["closedness"]),
        }
        output = processor((action_input, observation))
        current = np.array([float(output[f"{name}.pos"]) for name in arm_names])
        if previous is not None and float(np.max(np.abs(current - previous))) <= _IK_ENGAGE_CONVERGENCE_DEG:
            break
        previous = current
    if output is None:
        raise RuntimeError("IK engage initialization produced no action")
    return output


class _IKJointRebase:
    """Apply solver-space arm deltas on the measured MuJoCo joint baseline."""

    def __init__(self) -> None:
        self._arm_names = MOTOR_NAMES[:-1]
        self._offsets = dict.fromkeys(self._arm_names, 0.0)

    def engage(self, ik_action: RobotAction, observation: RobotObservation) -> RobotAction:
        self._offsets = {
            name: float(observation[f"{name}.pos"]) - float(ik_action[f"{name}.pos"])
            for name in self._arm_names
        }
        action = self.apply(ik_action)
        for name in self._arm_names:
            action[f"{name}.pos"] = float(observation[f"{name}.pos"])
        return action

    def apply(self, ik_action: RobotAction) -> RobotAction:
        action = dict(ik_action)
        for name in self._arm_names:
            action[f"{name}.pos"] = float(action[f"{name}.pos"]) + self._offsets[name]
        return action


def build_xr_to_joints_processor(kinematics: RobotKinematics):
    """Construct the same three processor steps used by the real follower path."""
    return RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[
            MapXRControllerActionToRobotAction(),
            EEBoundsAndSafety(
                end_effector_bounds={"min": [-1.0, -1.0, 0.0], "max": [1.0, 1.0, 1.0]},
                max_ee_step_m=MAX_EE_STEP_M,
                raise_on_jump=False,
            ),
            InverseKinematicsEEToJoints(
                kinematics=kinematics,
                motor_names=list(MOTOR_NAMES),
                initial_guess_current_joints=False,
                orientation_weight=IK_ORIENTATION_WEIGHT,
            ),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )


class XRToSO101Controller:
    """Stateful clutch and IK mapping shared by synthetic and live XR loops."""

    def __init__(self, initial_observation: RobotObservation, clutch_threshold: float = 0.5):
        self.kinematics = build_kinematics()
        self.processor = build_xr_to_joints_processor(self.kinematics)
        self.clutch_threshold = clutch_threshold
        self.metrics = ControlMetrics()
        self._enabled = False
        self._clutch = Clutch(self.forward_kinematics(initial_observation))
        self._joint_rebase = _IKJointRebase()

    @staticmethod
    def joint_vector(observation: RobotObservation) -> np.ndarray:
        return np.array([float(observation[f"{name}.pos"]) for name in MOTOR_NAMES], dtype=float)

    def forward_kinematics(self, observation: RobotObservation) -> np.ndarray:
        return self.kinematics.forward_kinematics(self.joint_vector(observation))

    def compute(self, xr_action: RobotAction, observation: RobotObservation) -> RobotAction | None:
        grip_pos = np.asarray(xr_action["grip_pos"], dtype=float)
        grip_quat = np.asarray(xr_action["grip_quat"], dtype=float)
        enabled = float(xr_action["squeeze"]) > self.clutch_threshold
        engage = enabled and not self._enabled

        measured_pose = None
        if engage:
            measured_pose = self.forward_kinematics(observation)
            # Reconstructing from measured FK uses the baseline Clutch public API while
            # re-anchoring both position and orientation for this hardware-free path.
            self._clutch = Clutch(measured_pose)
            self._clutch.engage(grip_pos, grip_quat)
            self.processor.reset()
        self._enabled = enabled
        if not enabled:
            return None

        ee_pos, ee_quat = self._clutch.rebase(grip_pos, grip_quat)
        started = time.perf_counter()
        ee_action = {
            "ee_pose": np.concatenate([ee_pos, ee_quat]).astype(np.float32),
            "closedness": float(xr_action["trigger"]),
        }
        ik_action = (
            _initialize_ik_engage(self.processor, ee_action, observation)
            if engage
            else self.processor((ee_action, observation))
        )
        elapsed = time.perf_counter() - started
        self.metrics.frames += 1
        self.metrics.ik_total_s += elapsed
        self.metrics.ik_max_s = max(self.metrics.ik_max_s, elapsed)

        action = (
            self._joint_rebase.engage(ik_action, observation)
            if engage
            else self._joint_rebase.apply(ik_action)
        )

        if engage and measured_pose is not None:
            ee_jump = float(np.linalg.norm(ee_pos - measured_pose[:3, 3]))
            target = np.array([float(action[f"{name}.pos"]) for name in MOTOR_NAMES[:5]])
            measured = self.joint_vector(observation)[:5]
            joint_jump = float(np.max(np.abs(target - measured)))
            self.metrics.reclutches += 1
            self.metrics.max_reclutch_ee_jump_m = max(self.metrics.max_reclutch_ee_jump_m, ee_jump)
            self.metrics.max_reclutch_joint_jump_deg = max(
                self.metrics.max_reclutch_joint_jump_deg, joint_jump
            )
        return action
