# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Local FK/IK for bounded corrections, independent of any motor transport.

The executor interpolates joints, not a Cartesian straight line. Sample that path
before accepting it. This is a kinematics check, not a collision checker.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from lerobot.utils.import_utils import _mujoco_available, _scipy_available, require_package

if TYPE_CHECKING or _scipy_available:
    from scipy.spatial.transform import Rotation

if TYPE_CHECKING or _mujoco_available:
    import mujoco


def vector(value, size, name):
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ValueError(f"{name} must have {size} numbers")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in value):
        raise ValueError(f"{name} must contain numbers")
    result = np.asarray(value, dtype=float)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def parse_pose(value):
    require_package("scipy", extra="hybrid-ik")
    if not isinstance(value, dict) or set(value) != {"position_m", "quaternion_wxyz"}:
        raise ValueError("An end-effector pose needs position_m and quaternion_wxyz")
    position = vector(value["position_m"], 3, "position_m")
    quaternion = vector(value["quaternion_wxyz"], 4, "quaternion_wxyz")
    if abs(np.linalg.norm(quaternion) - 1) > 1e-3:
        raise ValueError("quaternion_wxyz must be unit length")
    return position, Rotation.from_quat(quaternion[[1, 2, 3, 0]])


@dataclass
class EndEffectorConfig:
    model_path: str
    site: str
    joint_names: list[str]
    action_keys: list[str]
    frame_description: str
    max_translation_m: float = 0.03
    max_rotation_rad: float = 0.15
    max_linear_speed_m_s: float = 0.03
    max_angular_speed_rad_s: float = 0.15
    position_tolerance_m: float = 0.001
    rotation_tolerance_rad: float = 0.01

    def __post_init__(self):
        if not self.model_path or not self.site or not self.frame_description.strip():
            raise ValueError("End-effector model, site and frame description are required")
        if not self.action_keys or len(self.action_keys) != len(self.joint_names):
            raise ValueError("End-effector joint names and action keys must match")
        if len(set(self.action_keys)) != len(self.action_keys) or len(set(self.joint_names)) != len(
            self.joint_names
        ):
            raise ValueError("End-effector joint mappings must be unique")
        for value in (
            self.max_translation_m,
            self.max_rotation_rad,
            self.max_linear_speed_m_s,
            self.max_angular_speed_rad_s,
            self.position_tolerance_m,
            self.rotation_tolerance_rad,
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("End-effector limits must be finite and positive")


class EndEffectorKinematics:
    def __init__(self, config: EndEffectorConfig):
        require_package("mujoco", extra="hybrid-ik")
        require_package("scipy", extra="hybrid-ik")
        self.config = config
        self.model = mujoco.MjModel.from_xml_path(config.model_path)
        self.data = mujoco.MjData(self.model)
        self.site_id = self.model.site(config.site).id
        joints = [self.model.joint(name).id for name in config.joint_names]
        if any(self.model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE for j in joints):
            raise ValueError("End-effector action keys must map to hinge joints in radians")
        if any(not self.model.jnt_limited[j] for j in joints):
            raise ValueError("IK requires explicit model joint limits")
        if np.any(self.model.jnt_type == mujoco.mjtJoint.mjJNT_FREE):
            raise ValueError("IK requires a fixed-base model")
        self.qpos_ids = self.model.jnt_qposadr[joints]
        self.dof_ids = self.model.jnt_dofadr[joints]
        self.bounds = self.model.jnt_range[joints].copy()

    def _forward(self, q):
        self.data.qpos[self.qpos_ids] = q
        mujoco.mj_forward(self.model, self.data)
        return self.data.site_xpos[self.site_id].copy(), Rotation.from_matrix(
            self.data.site_xmat[self.site_id].reshape(3, 3)
        )

    def forward(self, pose):
        q = np.asarray([pose[key] for key in self.config.action_keys], dtype=float)
        if not np.isfinite(q).all():
            raise ValueError("FK requires finite measured joint positions")
        position, rotation = self._forward(q)
        return {"position_m": position.tolist(), "quaternion_wxyz": rotation.as_quat()[[3, 0, 1, 2]].tolist()}

    def reached(self, requested, pose):
        desired_p, desired_r = parse_pose(requested)
        measured_p, measured_r = parse_pose(self.forward(pose))
        return (
            np.linalg.norm(desired_p - measured_p) <= self.config.position_tolerance_m
            and (desired_r * measured_r.inv()).magnitude() <= self.config.rotation_tolerance_rad
        )

    def solve(self, requested, pose, limits, duration):
        """Seed from measured joints; reject unreachable, excessive or fast paths."""
        cfg = self.config
        target_p, target_r = parse_pose(requested)
        initial = np.asarray([pose[k] for k in cfg.action_keys], dtype=float)
        if not np.isfinite(initial).all() or not math.isfinite(duration) or duration <= 0:
            raise ValueError("IK needs a finite pose and positive duration")
        start_p, start_r = self._forward(initial)
        if np.linalg.norm(target_p - start_p) > cfg.max_translation_m:
            raise ValueError("End-effector translation limit exceeded")
        if (target_r * start_r.inv()).magnitude() > cfg.max_rotation_rad:
            raise ValueError("End-effector rotation limit exceeded")
        low = np.maximum(
            self.bounds[:, 0],
            [max(limits[k].minimum, pose[k] - limits[k].max_delta) for k in cfg.action_keys],
        )
        high = np.minimum(
            self.bounds[:, 1],
            [min(limits[k].maximum, pose[k] + limits[k].max_delta) for k in cfg.action_keys],
        )
        if np.any(low > high) or np.any(initial < low) or np.any(initial > high):
            raise ValueError("Measured pose is outside IK joint limits")
        q = initial.copy()
        for _ in range(100):
            p, r = self._forward(q)
            dp, dr = target_p - p, (target_r * r.inv()).as_rotvec()
            # Solve more tightly than the acceptance tolerance to avoid swallowing small corrections.
            if (
                np.linalg.norm(dp) <= cfg.position_tolerance_m / 4
                and np.linalg.norm(dr) <= cfg.rotation_tolerance_rad / 4
            ):
                break
            jp, jr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, self.data, jp, jr, self.site_id)
            unused = np.setdiff1d(np.arange(self.model.nv), self.dof_ids)
            if np.linalg.norm(jp[:, unused]) + np.linalg.norm(jr[:, unused]) > 1e-8:
                raise ValueError("End-effector joint mapping omits a contributing joint")
            jac = np.vstack((jp[:, self.dof_ids], 0.2 * jr[:, self.dof_ids]))
            error = np.r_[dp, 0.2 * dr]
            dq = jac.T @ np.linalg.solve(jac @ jac.T + 1e-4 * np.eye(6), error)
            dq *= min(1.0, 0.04 / max(np.max(np.abs(dq)), 1e-12))
            q = np.clip(q + dq, low, high)
        p, r = self._forward(q)
        if (
            np.linalg.norm(target_p - p) > cfg.position_tolerance_m
            or (target_r * r.inv()).magnitude() > cfg.rotation_tolerance_rad
        ):
            raise ValueError("IK did not converge within the correction limits")
        # Check the actual joint-interpolated path, including intermediate TCP speed
        # and excursions; an endpoint-only check misses curved paths near singularities.
        count = max(41, int(np.ceil(np.max(np.abs(q - initial)) / 0.002)) + 1)
        previous_p, previous_r = start_p, start_r
        dt = duration / (count - 1)
        for fraction in np.linspace(0, 1, count)[1:]:
            p, r = self._forward(initial + fraction * (q - initial))
            if (
                np.linalg.norm(p - start_p) > cfg.max_translation_m
                or (r * start_r.inv()).magnitude() > cfg.max_rotation_rad
            ):
                raise ValueError("IK path exceeds Cartesian correction limits")
            if (
                np.linalg.norm(p - previous_p) / dt > cfg.max_linear_speed_m_s
                or (r * previous_r.inv()).magnitude() / dt > cfg.max_angular_speed_rad_s
            ):
                raise ValueError("IK path exceeds Cartesian speed limits")
            previous_p, previous_r = p, r
        return dict(zip(cfg.action_keys, q.tolist(), strict=True))
