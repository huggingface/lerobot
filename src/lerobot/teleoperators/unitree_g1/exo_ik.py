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
from typing import TYPE_CHECKING

import numpy as np

from lerobot.robots.unitree_g1.g1_kinematics import G1_29_ARM_JOINTS, G1ArmKinematics
from lerobot.robots.unitree_g1.g1_utils import G1_29_JointArmIndex

from .exo_calib import JOINTS

if TYPE_CHECKING:
    import meshcat
    from pinocchio.robot_wrapper import RobotWrapper
    from pinocchio.visualize import MeshcatVisualizer

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
    - Loads the exoskeleton URDF models via Pinocchio
    - Computes forward kinematics on exoskeleton to get end-effector poses
    - Solves inverse kinematics on G1 to match those poses, with `G1ArmKinematics`
    - Provides meshcat visualization showing both robots and targets

    Args:
        frozen_joints: List of G1 arm joint names to exclude from IK (kept at zero).
    """

    def __init__(self, frozen_joints: list[str] | None = None):
        try:
            import pinocchio as pin
        except ImportError as e:
            raise ImportError(
                "ik mode needs pinocchio, installed with placo: pip install 'lerobot[unitree_g1]'"
            ) from e

        self.pin = pin
        self.frozen_joints = frozen_joints or []
        for name in self.frozen_joints:
            logger.info(f"freezing joint: {name}")

        self.g1_ik = G1ArmKinematics(self.frozen_joints)
        self.q_g1 = np.zeros(len(G1_29_ARM_JOINTS))

        assets_dir = str(self.g1_ik.urdf_path.parent)

        self.arms = [
            ArmCfg(
                side="left",
                urdf=os.path.join(assets_dir, "exo_left.urdf"),
                root="exo_left",
                offset=np.array([0.6, 0.3, 0.0]),
                marker_prefix="left",
            ),
            ArmCfg(
                side="right",
                urdf=os.path.join(assets_dir, "exo_right.urdf"),
                root="exo_right",
                offset=np.array([0.6, -0.3, 0.0]),
                marker_prefix="right",
            ),
        ]

        self.exo: dict[str, RobotWrapper] = {}  # side -> exo robot model
        self.q_exo: dict[str, np.ndarray] = {}  # side -> q
        self.ee_id_exo: dict[str, int] = {}  # side -> frame id
        self.qmap: dict[str, dict[str, int]] = {}  # side -> {joint_name: q_idx}

        self._load_exo_models(assets_dir)

        self.viewer: meshcat.Visualizer | None = None
        self.markers: Markers | None = None
        self.robot_g1: RobotWrapper | None = None  # display model, built with the visualization
        self.viz_g1: MeshcatVisualizer | None = None
        self.viz_exo: dict[str, MeshcatVisualizer] = {}  # side -> viz

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

        # g1, with everything but the arms locked at zero
        pin = self.pin
        urdf = self.g1_ik.urdf_path
        full = pin.RobotWrapper.BuildFromURDF(str(urdf), str(urdf.parent))
        locked = [name for name in full.model.names[1:] if name not in G1_29_ARM_JOINTS]
        self.robot_g1 = full.buildReducedRobot(
            list_of_joints_to_lock=locked, reference_configuration=pin.neutral(full.model)
        )
        self.viz_g1 = MeshcatVisualizer(
            self.robot_g1.model, self.robot_g1.collision_model, self.robot_g1.visual_model
        )
        self.viz_g1.initViewer(open=True)
        self.viz_g1.loadViewerModel("g1")
        self.viz_g1.display(self._g1_display_q())

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
            self.viz_g1.display(self._g1_display_q())
            for a, ee_tf in zip(self.arms, self.g1_ik.forward_kinematics(self.q_g1), strict=True):
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
        left = self._fk_target_world("left", left_angles)
        right = self._fk_target_world("right", right_angles)
        if left is None or right is None:
            # An arm without a target holds its current end-effector pose.
            current_left, current_right = self.g1_ik.forward_kinematics(self.q_g1)
            left = current_left if left is None else left
            right = current_right if right is None else right

        self.q_g1 = self.g1_ik.solve_ik(left, right)
        return {f"{j.name}.q": float(self.q_g1[i]) for i, j in enumerate(G1_29_JointArmIndex)}

    def _g1_display_q(self) -> np.ndarray:
        """Arm joint positions in the display model's configuration order."""
        if self.robot_g1 is None:
            raise RuntimeError("Call init_visualization() before displaying the G1")
        model = self.robot_g1.model
        q = self.pin.neutral(model)
        for name, value in zip(G1_29_ARM_JOINTS, self.q_g1, strict=True):
            q[model.idx_qs[model.getJointId(name)]] = value
        return q
