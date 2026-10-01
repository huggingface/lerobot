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

"""Local Cartesian waypoint IK. No hardware I/O and no assumed visual calibration."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from lerobot.utils.import_utils import _mujoco_available, _scipy_available, require_package

from .configuration import CartesianArm, PositionAxis

if TYPE_CHECKING or _mujoco_available:
    import mujoco
if TYPE_CHECKING or _scipy_available:
    from scipy.spatial.transform import Rotation


class CartesianKinematics:
    def __init__(self, config: CartesianArm, axes: dict[str, PositionAxis]):
        require_package("mujoco", extra="agent-ik")
        require_package("scipy", extra="agent-ik")
        self.config = config
        self.keys = list(config.joints)
        self.scale = np.array([config.joint_scale.get(k, 1.0) for k in self.keys])
        self.measurement_tolerance = np.array([axes[k].measurement_tolerance for k in self.keys]) * np.abs(
            self.scale
        )
        self.offset = np.array([config.joint_offset.get(k, 0.0) for k in self.keys])
        self.model = mujoco.MjModel.from_xml_path(config.model_path)
        self.data = mujoco.MjData(self.model)
        self.site = self.model.site(config.site).id
        joints = [self.model.joint(name).id for name in config.joints.values()]
        if np.any(self.model.jnt_type == mujoco.mjtJoint.mjJNT_FREE):
            raise ValueError("Cartesian control requires a fixed-base model")
        if any(
            self.model.jnt_type[j] not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)
            or not self.model.jnt_limited[j]
            for j in joints
        ):
            raise ValueError("IK needs explicitly limited hinge or slide joints")
        self.qids = self.model.jnt_qposadr[joints]
        self.dids = self.model.jnt_dofadr[joints]
        self.unused = np.setdiff1d(np.arange(self.model.nv), self.dids)
        native = np.array([[axes[k].minimum, axes[k].maximum] for k in self.keys])
        mapped = native * self.scale[:, None] + self.offset[:, None]
        self.low = np.maximum(mapped.min(axis=1), self.model.jnt_range[joints, 0])
        self.high = np.minimum(mapped.max(axis=1), self.model.jnt_range[joints, 1])
        if np.any(self.low > self.high):
            raise ValueError("Robot and model joint limits do not overlap")
        self.reference = Rotation.identity()

    def _q(self, pose: dict) -> np.ndarray:
        q = np.array([pose[k] for k in self.keys]) * self.scale + self.offset
        if (
            not np.isfinite(q).all()
            or np.any(q < self.low - self.measurement_tolerance)
            or np.any(q > self.high + self.measurement_tolerance)
        ):
            raise ValueError("Measured pose is outside model joint limits")
        return np.clip(q, self.low, self.high)

    def _forward(self, q: np.ndarray):
        self.data.qpos[self.qids] = q
        mujoco.mj_forward(self.model, self.data)
        return self.data.site_xpos[self.site].copy(), Rotation.from_matrix(
            self.data.site_xmat[self.site].reshape(3, 3)
        )

    def reset(self, pose: dict):
        _, self.reference = self._forward(self._q(pose))

    def observe(self, pose: dict) -> list[float]:
        position, rotation = self._forward(self._q(pose))
        # inspect-robots-yam: positive pitch is about NEGATIVE base Y.
        yaw, negative_pitch, roll = (rotation * self.reference.inv()).as_euler("ZYX")
        return [*position.tolist(), float(yaw), float(-negative_pitch), float(roll)]

    def solve(self, target: np.ndarray, seed: dict) -> dict[str, float]:
        cfg = self.config
        target_p = target[:3]
        target_r = Rotation.from_euler("ZYX", [target[3], -target[4], target[5]]) * self.reference
        q = self._q(seed)
        for _ in range(150):
            p, r = self._forward(q)
            dp, dr = target_p - p, (target_r * r.inv()).as_rotvec()
            if (
                np.linalg.norm(dp) < cfg.position_tolerance / 4
                and np.linalg.norm(dr) < cfg.rotation_tolerance / 4
            ):
                break
            jp, jr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, self.data, jp, jr, self.site)
            if np.linalg.norm(jp[:, self.unused]) + np.linalg.norm(jr[:, self.unused]) > 1e-8:
                raise ValueError("Cartesian joint mapping omits a contributing joint")
            jac = np.vstack((jp[:, self.dids], 0.2 * jr[:, self.dids]))
            dq = jac.T @ np.linalg.solve(jac @ jac.T + 1e-6 * np.eye(6), np.r_[dp, 0.2 * dr])
            dq *= min(1, 0.04 / max(np.max(np.abs(dq)), 1e-12))
            q = np.clip(q + dq, self.low, self.high)
        p, r = self._forward(q)
        if (
            np.linalg.norm(target_p - p) > cfg.position_tolerance
            or (target_r * r.inv()).magnitude() > cfg.rotation_tolerance
        ):
            raise ValueError("Cartesian waypoint is unreachable within configured IK tolerances")
        return dict(zip(self.keys, ((q - self.offset) / self.scale).tolist(), strict=True))
