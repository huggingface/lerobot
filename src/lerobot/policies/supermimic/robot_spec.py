#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""Robot dynamics specification shared by SuperMimic deployment and training.

A :class:`RobotDynamicsSpec` fixes everything that links actor outputs to joint commands
and the simulated plant to the trained policy: joint order, default pose, PD gains,
per-joint action scale, effort/velocity limits, rotor armature, joint ranges and the
control/physics periods. Deployment checks an export's manifest against it; training
builds its simulator and action mapping from it.

The G1 values follow Uni-SuperMimic's ``robot.py`` and ``supermimic.xml``
(https://github.com/Motphys/Uni-SuperMimic, Apache-2.0): gains come from each actuator's
armature at a 10 Hz natural frequency and damping ratio 2, and the action scale is a
quarter of the effort limit divided by kp.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

G1_JOINT_NAMES: tuple[str, ...] = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

# Bodies whose poses enter the tracking reward, pelvis first.
G1_TRACKED_BODIES: tuple[str, ...] = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
)

_ARMATURE_5020 = 0.003609725
_ARMATURE_7520_14 = 0.010177520
_ARMATURE_7520_22 = 0.025101925
_ARMATURE_4010 = 0.00425
_NATURAL_FREQ = 10 * 2.0 * 3.1415926535
_DAMPING_RATIO = 2.0

# (effort limit [Nm], velocity limit [rad/s], reflected rotor armature [kg m^2])
_M5020 = (25.0, 37.0, _ARMATURE_5020)
_M4010 = (5.0, 22.0, _ARMATURE_4010)
_G1_JOINT_MOTORS: dict[str, tuple[float, float, float]] = {
    "hip_pitch": (88.0, 32.0, _ARMATURE_7520_14),
    "hip_roll": (139.0, 20.0, _ARMATURE_7520_22),
    "hip_yaw": (88.0, 32.0, _ARMATURE_7520_14),
    "knee": (139.0, 20.0, _ARMATURE_7520_22),
    "ankle_pitch": (50.0, 37.0, 2.0 * _ARMATURE_5020),
    "ankle_roll": (50.0, 37.0, 2.0 * _ARMATURE_5020),
    "waist_yaw": (88.0, 32.0, _ARMATURE_7520_14),
    "waist_roll": (50.0, 37.0, 2.0 * _ARMATURE_5020),
    "waist_pitch": (50.0, 37.0, 2.0 * _ARMATURE_5020),
    "shoulder_pitch": _M5020,
    "shoulder_roll": _M5020,
    "shoulder_yaw": _M5020,
    "elbow": _M5020,
    "wrist_roll": _M5020,
    "wrist_pitch": _M4010,
    "wrist_yaw": _M4010,
}

_G1_DEFAULT_JOINT_POS = {
    ".*_hip_pitch_joint": -0.312,
    ".*_knee_joint": 0.669,
    ".*_ankle_pitch_joint": -0.363,
    ".*_elbow_joint": 0.6,
    "left_shoulder_roll_joint": 0.2,
    "left_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "right_shoulder_pitch_joint": 0.2,
}

_G1_JOINT_RANGE = {
    "hip_pitch": (-2.5307, 2.8798),
    "left_hip_roll": (-0.5236, 2.9671),
    "right_hip_roll": (-2.9671, 0.5236),
    "hip_yaw": (-2.7576, 2.7576),
    "knee": (-0.087267, 2.8798),
    "ankle_pitch": (-0.87267, 0.5236),
    "ankle_roll": (-0.2618, 0.2618),
    "waist_yaw": (-2.618, 2.618),
    "waist_roll": (-0.52, 0.52),
    "waist_pitch": (-0.52, 0.52),
    "shoulder_pitch": (-3.0892, 2.6704),
    "left_shoulder_roll": (-1.5882, 2.2515),
    "right_shoulder_roll": (-2.2515, 1.5882),
    "shoulder_yaw": (-2.618, 2.618),
    "elbow": (-1.0472, 2.0944),
    "wrist_roll": (-1.972222054, 1.972222054),
    "wrist_pitch": (-1.614429558, 1.614429558),
    "wrist_yaw": (-1.614429558, 1.614429558),
}


def _motor_key(joint_name: str) -> str:
    return joint_name.replace("left_", "").replace("right_", "").replace("_joint", "")


def _joint_range(joint_name: str) -> tuple[float, float]:
    sided = joint_name.replace("_joint", "")
    return _G1_JOINT_RANGE.get(sided) or _G1_JOINT_RANGE[_motor_key(joint_name)]


@dataclass(frozen=True)
class RobotDynamicsSpec:
    """Per-joint constants in the actor's joint order (tuples so the spec stays hashable)."""

    name: str
    joint_names: tuple[str, ...]
    default_q: tuple[float, ...]
    kp: tuple[float, ...]
    kd: tuple[float, ...]
    action_scale: tuple[float, ...]
    effort_limit: tuple[float, ...]
    velocity_limit: tuple[float, ...]
    armature: tuple[float, ...]
    joint_lower: tuple[float, ...]
    joint_upper: tuple[float, ...]
    control_dt: float
    physics_dt: float

    def __post_init__(self) -> None:
        num_joints = len(self.joint_names)
        for field_name in (
            "default_q",
            "kp",
            "kd",
            "action_scale",
            "effort_limit",
            "velocity_limit",
            "armature",
            "joint_lower",
            "joint_upper",
        ):
            values = getattr(self, field_name)
            if len(values) != num_joints:
                raise ValueError(f"{field_name} has {len(values)} entries for {num_joints} joints")
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"{field_name} must be finite")
        if self.control_dt <= 0 or self.physics_dt <= 0:
            raise ValueError("control_dt and physics_dt must be positive")
        decimation = self.control_dt / self.physics_dt
        if abs(decimation - round(decimation)) > 1e-9:
            raise ValueError("control_dt must be an integer multiple of physics_dt")

    @property
    def num_joints(self) -> int:
        return len(self.joint_names)

    @property
    def decimation(self) -> int:
        return round(self.control_dt / self.physics_dt)

    def array(self, field_name: str) -> np.ndarray:
        return np.asarray(getattr(self, field_name), dtype=np.float32)

    def action_to_target(self, action: np.ndarray, joint_offset: np.ndarray | None = None) -> np.ndarray:
        """``q_target = default_q + joint_offset + action_scale * action`` (no clipping)."""
        target = self.array("default_q") + self.array("action_scale") * np.asarray(action, np.float32)
        return target if joint_offset is None else target + np.asarray(joint_offset, np.float32)

    def to_dict(self) -> dict[str, Any]:
        return {
            key: list(value) if isinstance(value, tuple) else value for key, value in asdict(self).items()
        }

    def check_manifest(self, manifest: Mapping[str, Any], *, atol: float = 1e-6) -> None:
        """Reject an export whose joint order, pose, gains or periods disagree with this spec."""
        robot = manifest["robot"]
        if tuple(robot["joint_names"]) != self.joint_names:
            raise ValueError("Export joint order differs from the robot dynamics spec")
        pairs = {
            "default_q": robot["default_q_rad"],
            "kp": robot["kp"],
            "kd": robot["kd"],
            "action_scale": robot["action_scale_rad"],
        }
        for field_name, exported in pairs.items():
            if not np.allclose(
                np.asarray(exported, np.float64), getattr(self, field_name), atol=atol, rtol=0
            ):
                raise ValueError(f"Export {field_name} differs from the robot dynamics spec")
        for field_name in ("control_dt", "physics_dt"):
            if not math.isclose(float(robot[f"{field_name}_s"]), getattr(self, field_name), abs_tol=1e-9):
                raise ValueError(f"Export {field_name} differs from the robot dynamics spec")


def g1_29dof_spec(joint_names: Sequence[str] = G1_JOINT_NAMES) -> RobotDynamicsSpec:
    """The Unitree G1 29-DoF plant the SuperMimic G1 actors were trained on."""
    kp, kd, scale, effort, velocity, armature, default, lower, upper = ([] for _ in range(9))
    for name in joint_names:
        effort_limit, velocity_limit, rotor = _G1_JOINT_MOTORS[_motor_key(name)]
        joint_kp = rotor * _NATURAL_FREQ**2
        kp.append(joint_kp)
        kd.append(2.0 * _DAMPING_RATIO * rotor * _NATURAL_FREQ)
        scale.append(0.25 * effort_limit / joint_kp)
        effort.append(effort_limit)
        velocity.append(velocity_limit)
        armature.append(rotor)
        position = 0.0
        for pattern, value in _G1_DEFAULT_JOINT_POS.items():
            if re.fullmatch(pattern, name):
                position = value
        default.append(position)
        low, high = _joint_range(name)
        lower.append(low)
        upper.append(high)
    return RobotDynamicsSpec(
        name="unitree_g1_29dof",
        joint_names=tuple(joint_names),
        default_q=tuple(default),
        kp=tuple(kp),
        kd=tuple(kd),
        action_scale=tuple(scale),
        effort_limit=tuple(effort),
        velocity_limit=tuple(velocity),
        armature=tuple(armature),
        joint_lower=tuple(lower),
        joint_upper=tuple(upper),
        control_dt=0.02,
        physics_dt=0.005,
    )
