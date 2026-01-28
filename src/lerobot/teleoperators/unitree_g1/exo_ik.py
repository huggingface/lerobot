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

"""
IK helper for exoskeleton-to-G1 teleoperation. We map Exoskeleton joint angles to end-effector pose in world frame,
visualizing the result in meshcat after calibration.
"""

import logging
import os
from dataclasses import dataclass

import numpy as np

from lerobot.robots.unitree_g1.g1_utils import G1_29_JointArmIndex
from lerobot.robots.unitree_g1.robot_kinematic_processor import G1_29_ArmIK

from .exo_calib import JOINTS

logger = logging.getLogger(__name__)


def _frame_id(model, name: str) -> int | None:
    try:
        fid = model.getFrameId(name)
        return fid if 0 <= fid < model.nframes else None
    except Exception:
        return None


@dataclass
class ArmCfg:
    side: str  # "left" | "right"
    urdf: str  # exo_left.urdf / exo_right.urdf
    root: str  # "exo_left" / "exo_right"
    g1_ee: str  # "l_ee" / "r_ee"
    offset: np.ndarray  # world offset for viz + target
    marker_prefix: str  # "left" / "right"


class Markers:
    """Creates meshcat visualization primitives, showing end-effector frames of exoskeleton and G1"""

    def __init__(self, viewer):
        self.v = viewer

    def sphere(self, path: str, r: float, rgba: tuple[float, float, float, float]):
        import meshcat.geometry as mg

        c = (int(rgba[0] * 255) << 16) | (int(rgba[1] * 255) << 8) | int(rgba[2] * 255)
        self.v[path].set_object(
            mg.Sphere(r),
            mg.MeshPhongMaterial(color=c, opacity=rgba[3], transparent=rgba[3] < 1.0),
        )

    def axes(self, path: str, axis_len: float = 0.1, axis_w: int = 6):
        import meshcat.geometry as mg

        pts = np.array(
            [[0, 0, 0], [axis_len, 0, 0], [0, 0, 0], [0, axis_len, 0], [0, 0, 0], [0, 0, axis_len]],
            dtype=np.float32,
        ).T
        cols = np.array(
            [[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]],
            dtype=np.float32,
        ).T
        self.v[path].set_object(
            mg.LineSegments(
                mg.PointsGeometry(position=pts, color=cols),
                mg.LineBasicMaterial(linewidth=axis_w, vertexColors=True),
            )
        )

    def tf(self, path: str, mat: np.ndarray):
        self.v[path].set_transform(mat)


class ExoskeletonIKHelper:
    """
    - Loads G1 robot and exoskeleton URDF models via Pinocchio
    - Computes forward kinematics on exoskeleton to get end-effector poses
    - Solves inverse kinematics on G1 to match those poses
    - Provides meshcat visualization showing both robots and targets

    Args:
        frozen_joints: List of G1 joint names to exclude from IK (kept at neutral).
    """

    def __init__(self, frozen_joints: list[str] | None = None):
        try:
            import pinocchio as pin
        except ImportError as e:
            raise ImportError("ik mode needs pinocchio: pip install pin") from e

        self.pin = pin
        self.frozen_joints = frozen_joints or []

        self.g1_ik = G1_29_ArmIK()
        self.robot_g1 = self.g1_ik.reduced_robot
        self.robot_g1.data = self.robot_g1.model.createData()
        self.q_g1 = pin.neutral(self.robot_g1.model)

        assets_dir = os.path.join(self.g1_ik.repo_path, "assets")

        self.frozen_idx = self._frozen_joint_indices()

        self.arms = [
            ArmCfg(
                side="left",
                urdf=os.path.join(assets_dir, "exo_left.urdf"),
                root="exo_left",
                g1_ee="L_ee",
                offset=np.array([0.6, 0.3, 0.0]),
                marker_prefix="left",
            ),
            ArmCfg(
                side="right",
                urdf=os.path.join(assets_dir, "exo_right.urdf"),
                root="exo_right",
                g1_ee="R_ee",
                offset=np.array([0.6, -0.3, 0.0]),
                marker_prefix="right",
            ),
        ]

        self.exo = {}  # side -> pin.RobotWrapper
        self.q_exo = {}  # side -> q
        self.ee_id_exo = {}  # side -> frame id
        self.qmap = {}  # side -> {joint_name: q_idx}
        self.ee_id_g1 = {}  # side -> frame id

        self._load_exo_models(assets_dir)
        for a in self.arms:
            self.ee_id_g1[a.side] = _frame_id(self.robot_g1.model, a.g1_ee)

        self.viewer = None
        self.markers: Markers | None = None
        self.viz_g1 = None
        self.viz_exo = {}  # side -> viz

    def _frozen_joint_indices(self) -> dict[str, int]:
        out = {}
        m = self.robot_g1.model
        for name in self.frozen_joints:
            if name in m.names:
                jid = m.getJointId(name)
                out[name] = m.idx_qs[jid]
                logger.info(f"freezing joint: {name} (q_idx={out[name]})")
        return out

    def _find_exo_ee(self, model, ee_name: str = "ee") -> int:
        ee = _frame_id(model, ee_name)
        if ee is not None:
            return ee
        for fid in reversed(range(model.nframes)):
            if model.frames[fid].type == self.pin.FrameType.BODY:
                return fid
        return 0

    def _build_joint_map(self, robot) -> dict[str, int]:
        m = robot.model
        return {n: m.idx_qs[m.getJointId(n)] for n in JOINTS if n in m.names}

    def _load_exo_models(self, assets_dir: str):
        pin = self.pin
        for a in self.arms:
            if not os.path.exists(a.urdf):
                logger.warning(f"{a.side} exo urdf not found: {a.urdf}")
                continue
            r = pin.RobotWrapper.BuildFromURDF(a.urdf, assets_dir)
            self.exo[a.side] = r
            self.q_exo[a.side] = pin.neutral(r.model)
            self.ee_id_exo[a.side] = self._find_exo_ee(r.model)
            self.qmap[a.side] = self._build_joint_map(r)
            logger.info(f"loaded {a.side} exo urdf: {a.urdf}")

    def init_visualization(self):
        """
        Creates a browser-based visualization of exoskeleton and G1 robot,
        highlighting end-effector frames and target positions.
        """
        try:
            from pinocchio.visualize import MeshcatVisualizer
        except ImportError as e:
            logger.warning(f"meshcat viz unavailable: {e}")
            return

        # g1
        self.viz_g1 = MeshcatVisualizer(
            self.robot_g1.model, self.robot_g1.collision_model, self.robot_g1.visual_model
        )
        self.viz_g1.initViewer(open=True)
        self.viz_g1.loadViewerModel("g1")
        self.viz_g1.display(self.q_g1)

        self.viewer = self.viz_g1.viewer
        self.markers = Markers(self.viewer)

        # exos
        for a in self.arms:
            if a.side not in self.exo:
                continue
            r = self.exo[a.side]
            v = MeshcatVisualizer(r.model, r.collision_model, r.visual_model)
            v.initViewer(open=False)
            v.viewer = self.viewer
            v.loadViewerModel(a.root)
            offset_tf = np.eye(4)
            offset_tf[:3, 3] = a.offset
            self.viewer[a.root].set_transform(offset_tf)
            v.display(self.q_exo[a.side])
            self.viz_exo[a.side] = v

        # markers
        for a in self.arms:
            p = a.marker_prefix
            self.markers.sphere(f"markers/{p}_exo_ee", 0.012, (0.2, 1.0, 0.2, 0.9))
            self.markers.sphere(f"markers/{p}_g1_ee", 0.015, (1.0, 0.2, 0.2, 0.9))
            self.markers.sphere(f"markers/{p}_ik_target", 0.015, (0.1, 0.3, 1.0, 0.9))
            self.markers.axes(f"markers/{p}_exo_axes", 0.06)
            self.markers.axes(f"markers/{p}_g1_axes", 0.08)

        logger.info(f"meshcat viz initialized: {self.viewer.url()}")
        print(f"\nmeshcat url: {self.viewer.url()}\n")

    def _fk_target_world(self, side: str, angles: dict[str, float]) -> np.ndarray | None:
        """returns wrist frame target to be used for G1 IK in 4x4 homogeneous transform. Takes offset into account."""
        if side not in self.exo or not angles:
            return None

        pin = self.pin
        q = self.q_exo[side]
        qmap = self.qmap[side]

        for name, ang in angles.items():
            idx = qmap.get(name)
            if idx is not None:
                q[idx] = float(ang)

        r = self.exo[side]
        pin.forwardKinematics(r.model, r.data, q)
        pin.updateFramePlacements(r.model, r.data)

        ee = r.data.oMf[self.ee_id_exo[side]]
        target = np.eye(4)
        target[:3, :3] = ee.rotation
        # offset gets applied in world space
        cfg = next(a for a in self.arms if a.side == side)
        target[:3, 3] = cfg.offset + ee.translation
        return target

    def update_visualization(self):
        if self.viewer is None or self.markers is None:
            return

        pin = self.pin

        # g1
        if self.viz_g1 is not None:
            self.viz_g1.display(self.q_g1)
            pin.forwardKinematics(self.robot_g1.model, self.robot_g1.data, self.q_g1)
            pin.updateFramePlacements(self.robot_g1.model, self.robot_g1.data)

            for a in self.arms:
                fid = self.ee_id_g1.get(a.side)
                if fid is None:
                    continue
                ee_tf = self.robot_g1.data.oMf[fid].homogeneous
                p = a.marker_prefix
                self.markers.tf(f"markers/{p}_g1_ee", ee_tf)
                self.markers.tf(f"markers/{p}_g1_axes", ee_tf)

        # exos
        for a in self.arms:
            side = a.side
            v = self.viz_exo.get(side)
            if v is None:
                continue

            v.display(self.q_exo[side])
            r = self.exo[side]
            pin.forwardKinematics(r.model, r.data, self.q_exo[side])
            pin.updateFramePlacements(r.model, r.data)

            ee = r.data.oMf[self.ee_id_exo[side]]
            world_tf = (pin.SE3(np.eye(3), a.offset) * ee).homogeneous
            p = a.marker_prefix
            self.markers.tf(f"markers/{p}_exo_ee", world_tf)
            self.markers.tf(f"markers/{p}_exo_axes", world_tf)

            target_tf = np.eye(4)
            target_tf[:3, :3] = ee.rotation
            target_tf[:3, 3] = a.offset + ee.translation
            self.markers.tf(f"markers/{p}_ik_target", target_tf)

    def compute_g1_joints_from_exo(
        self,
        left_angles: dict[str, float],
        right_angles: dict[str, float],
    ) -> dict[str, float]:
        """
        Performs FK on exoskeleton to get end-effector poses in world frame,
        after which it solves IK on G1 to return joint angles matching those poses in G1 motor order.
        """
        pin = self.pin

        targets = {
            "left": self._fk_target_world("left", left_angles),
            "right": self._fk_target_world("right", right_angles),
        }

        # fallback to current g1 ee pose if missing target
        pin.forwardKinematics(self.robot_g1.model, self.robot_g1.data, self.q_g1)
        pin.updateFramePlacements(self.robot_g1.model, self.robot_g1.data)

        for a in self.arms:
            if targets[a.side] is not None:
                continue
            fid = self.ee_id_g1.get(a.side)
            if fid is not None:
                targets[a.side] = self.robot_g1.data.oMf[fid].homogeneous

        if targets["left"] is None or targets["right"] is None:
            logger.warning("missing ik targets, returning current pose")
            return {}

        frozen_vals = {n: self.q_g1[i] for n, i in self.frozen_idx.items()}

        self.q_g1, _ = self.g1_ik.solve_ik(
            targets["left"], targets["right"], current_lr_arm_motor_q=self.q_g1
        )

        for n, i in self.frozen_idx.items():
            self.q_g1[i] = frozen_vals[n]

        return {
            f"{j.name}.q": float(self.q_g1[i])
            for i, j in enumerate(G1_29_JointArmIndex)
            if i < len(self.q_g1)
        }
