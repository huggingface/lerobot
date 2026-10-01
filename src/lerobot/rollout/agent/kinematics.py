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

"""URDF validation and native-unit adapter around LeRobot's existing FK/IK."""

from __future__ import annotations

from typing import TYPE_CHECKING
from xml.etree import ElementTree  # nosec B405: operator-supplied local robot model

import numpy as np

from lerobot.model import RobotKinematics
from lerobot.utils.import_utils import _scipy_available, require_package

from .configuration import CartesianArm, PositionAxis

if TYPE_CHECKING or _scipy_available:
    from scipy.spatial.transform import Rotation


class CartesianKinematics:
    def __init__(self, config: CartesianArm, axes: dict[str, PositionAxis]):
        require_package("scipy", extra="agent-ik")
        self.config = config
        self.keys = list(config.joints)
        self.scale = np.array([config.joint_scale.get(k, 1.0) for k in self.keys])
        self.offset = np.array([config.joint_offset.get(k, 0.0) for k in self.keys])
        self.measurement_tolerance = np.array([axes[k].measurement_tolerance for k in self.keys]) * np.abs(
            self.scale
        )
        # Validate the tool chain before creating a solver. RobotKinematics uses
        # degrees for every mapped joint, so prismatic joints are not supported.
        root = ElementTree.parse(config.urdf_path).getroot()  # nosec B314: local URDF
        links = {link.attrib["name"] for link in root.findall("link")}
        if config.target_frame_name not in links:
            raise ValueError("target_frame_name must name a URDF tool link")
        parents = {}
        for joint in root.findall("joint"):
            child = joint.find("child")
            if child is None or "link" not in child.attrib:
                raise ValueError("URDF joint is missing its child link")
            parents[child.attrib["link"]] = joint
        chain = {}
        link = config.target_frame_name
        visited = set()
        while link in parents:
            if link in visited:
                raise ValueError("URDF tool chain contains a cycle")
            visited.add(link)
            joint = parents[link]
            if joint.attrib["type"] != "fixed":
                if joint.attrib["type"] != "revolute" or joint.find("mimic") is not None:
                    raise ValueError(
                        "Cartesian control requires independent, limited revolute tool-chain joints"
                    )
                chain[joint.attrib["name"]] = joint
            parent = joint.find("parent")
            if parent is None or "link" not in parent.attrib:
                raise ValueError("URDF joint is missing its parent link")
            link = parent.attrib["link"]
        names = list(config.joints.values())
        if set(names) != set(chain):
            raise ValueError("Cartesian joint mapping must cover exactly the moving joints of the tool chain")
        ranges = []
        for name in names:
            limit = chain[name].find("limit")
            if limit is None or "lower" not in limit.attrib or "upper" not in limit.attrib:
                raise ValueError(f"URDF joint {name} needs explicit limits")
            ranges.append([float(limit.attrib["lower"]), float(limit.attrib["upper"])])
        ranges = np.array(ranges)
        if not np.isfinite(ranges).all() or np.any(ranges[:, 0] >= ranges[:, 1]):
            raise ValueError("URDF joint limits must be finite ordered ranges")
        native = np.array([[axes[k].minimum, axes[k].maximum] for k in self.keys])
        mapped = native * self.scale[:, None] + self.offset[:, None]
        self.low = np.maximum(mapped.min(axis=1), ranges[:, 0])
        self.high = np.minimum(mapped.max(axis=1), ranges[:, 1])
        if np.any(self.low > self.high):
            raise ValueError("Robot and model joint limits do not overlap")
        self.kinematics = RobotKinematics(config.urdf_path, config.target_frame_name, names)
        self.kinematics.solver.enable_joint_limits(True)
        for name, low, high in zip(names, self.low, self.high, strict=True):
            self.kinematics.robot.set_joint_limits(name, float(low), float(high))
        # Extra branches (e.g. fingers) must not be recruited to solve an arm pose.
        for name in set(self.kinematics.robot.joint_names()) - set(names):
            self.kinematics.solver.mask_dof(name)
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
        transform = self.kinematics.forward_kinematics(np.rad2deg(q))
        return transform[:3, 3].copy(), Rotation.from_matrix(transform[:3, :3])

    def reset(self, pose: dict):
        _, self.reference = self._forward(self._q(pose))

    def observe(self, pose: dict) -> list[float]:
        position, rotation = self._forward(self._q(pose))
        # inspect-robots-yam: positive pitch is about NEGATIVE base Y.
        yaw, negative_pitch, roll = (rotation * self.reference.inv()).as_euler("ZYX")
        return [*position.tolist(), float(yaw), float(-negative_pitch), float(roll)]

    def solve(self, target: np.ndarray, seed: dict) -> dict[str, float]:
        cfg = self.config
        target_r = Rotation.from_euler("ZYX", [target[3], -target[4], target[5]]) * self.reference
        transform = np.eye(4)
        transform[:3, :3] = target_r.as_matrix()
        transform[:3, 3] = target[:3]
        try:
            degrees = self.kinematics.inverse_kinematics(
                np.rad2deg(self._q(seed)), transform, orientation_weight=1.0, max_iters=cfg.ik_max_iters
            )
        except RuntimeError as exc:
            raise ValueError("Cartesian waypoint IK failed") from exc
        q = np.deg2rad(degrees)
        if q.shape != self.low.shape or not np.isfinite(q).all():
            raise ValueError("IK returned invalid joint positions")
        if np.any(q < self.low - 1e-7) or np.any(q > self.high + 1e-7):
            raise ValueError("IK solution exceeds configured joint limits")
        q = np.clip(q, self.low, self.high)
        p, r = self._forward(q)
        if (
            np.linalg.norm(target[:3] - p) > cfg.position_tolerance
            or (target_r * r.inv()).magnitude() > cfg.rotation_tolerance
        ):
            raise ValueError("Cartesian waypoint is unreachable within configured IK tolerances")
        return dict(zip(self.keys, ((q - self.offset) / self.scale).tolist(), strict=True))
