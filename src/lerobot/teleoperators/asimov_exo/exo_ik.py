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
IK helper for exoskeleton-to-Asimov teleoperation. Exoskeleton FK gives each end-effector pose
relative to the exo shoulder (abduction joint); it is scaled to Asimov's reach, placed at
Asimov's shoulder and solved with Asimov arm IK. Meshcat shows both after calibration.
"""

import logging
import os
from typing import Any

import numpy as np

from lerobot.robots.asimov.asimov_kinematics import AsimovArmIK

from ..unitree_g1.exo_calib import JOINTS
from ..unitree_g1.exo_ik import Markers, _frame_id

logger = logging.getLogger(__name__)

SIDES = ("left", "right")
# Where the exo models are drawn in meshcat, in front of the robot.
VIZ_OFFSET = {"left": np.array([0.5, 0.3, 0.0]), "right": np.array([0.5, -0.3, 0.0])}
# Exo models are drawn scaled about their shoulder joint.
VIZ_SCALE = 0.8


class AsimovExoIKHelper:
    """
    - Loads the exoskeleton URDFs (shared with the G1 teleoperator) and the Asimov URDF
    - Maps exoskeleton end-effector poses onto Asimov's shoulders, scaled to its reach
    - Solves Asimov arm IK to match those poses
    - Provides meshcat visualization showing both and the targets

    The exo root frames are axis-aligned with the robot pelvis frame (as for the G1), and the
    exo ``ee`` frame and Asimov ``L_ee``/``R_ee`` frames share their convention, so
    orientations are used as-is.
    """

    def __init__(
        self,
        frozen_joints: list[str] | None = None,
        arm_scale: float | None = None,
        rotation_weight: float = 0.02,
        wrist_roll_offset_deg: float = 0.0,
        visualize: bool = False,
    ):
        try:
            import pinocchio as pin
        except ImportError as e:
            raise ImportError("ik mode needs pinocchio: pip install pin") from e
        from huggingface_hub import snapshot_download

        self.pin = pin
        self.frozen = frozenset(frozen_joints or [])
        # Asimov's ee frame has the forearm along local x, so roll is a local x rotation.
        roll = np.deg2rad(wrist_roll_offset_deg)
        self.wrist_roll = {
            "left": pin.utils.rpyToMatrix(roll, 0.0, 0.0),
            "right": pin.utils.rpyToMatrix(-roll, 0.0, 0.0),
        }
        self.ik = AsimovArmIK(with_meshes=visualize, rotation_weight=rotation_weight)

        assets_dir = os.path.join(snapshot_download("lerobot/unitree-g1-mujoco"), "assets")
        self.exo = {}  # side -> pin.RobotWrapper
        self.q_exo = {}  # side -> q
        self.ee_id_exo = {}  # side -> frame id
        self.qmap = {}  # side -> {joint_name: q_idx}
        self.shoulder_exo = {}  # side -> exo shoulder position in the exo root frame
        self.shoulder_asimov = {}  # side -> Asimov shoulder position in the pelvis frame
        self.scale = {}  # side -> exo-to-Asimov length scale
        for side in SIDES:
            urdf = os.path.join(assets_dir, f"exo_{side}.urdf")
            r = pin.RobotWrapper.BuildFromURDF(urdf, assets_dir)
            m = r.model
            self.exo[side] = r
            self.q_exo[side] = pin.neutral(m)
            self.ee_id_exo[side] = _frame_id(m, "ee")
            if self.ee_id_exo[side] is None:
                raise ValueError(f"{urdf} has no 'ee' frame")
            self.qmap[side] = {n: m.idx_qs[m.getJointId(n)] for n in JOINTS if n in m.names}

            pin.framesForwardKinematics(m, r.data, self.q_exo[side])
            # The exo's second joint ("shoulder_yaw") is the abduction joint, Asimov's shoulder roll.
            shoulder = r.data.oMi[m.getJointId("shoulder_yaw")].translation.copy()
            elbow = r.data.oMi[m.getJointId("elbow_flex")].translation
            ee = r.data.oMf[self.ee_id_exo[side]].translation
            exo_reach = float(np.linalg.norm(elbow - shoulder) + np.linalg.norm(ee - elbow))
            self.shoulder_exo[side] = shoulder
            self.shoulder_asimov[side] = self.ik.shoulder_position(side)
            self.scale[side] = arm_scale if arm_scale is not None else self.ik.arm_reach(side) / exo_reach
            logger.info(f"{side} exo -> Asimov scale: {self.scale[side]:.3f}")

        self.viewer = None
        self.markers: Markers | None = None
        self.viz_asimov = None
        self.viz_exo: dict[str, Any] = {}  # side -> viz
        self.targets: dict[str, np.ndarray | None] = dict.fromkeys(SIDES)

    def _exo_ee(self, side: str, angles: dict[str, float]):
        pin = self.pin
        q = self.q_exo[side]
        for name, ang in angles.items():
            idx = self.qmap[side].get(name)
            if idx is not None:
                q[idx] = float(ang)
        r = self.exo[side]
        pin.framesForwardKinematics(r.model, r.data, q)
        return r.data.oMf[self.ee_id_exo[side]]

    def _target(self, side: str, angles: dict[str, float]) -> np.ndarray | None:
        """Asimov end-effector target (4x4, pelvis frame) for the exo pose."""
        if not angles:
            return None
        ee = self._exo_ee(side, angles)
        target = np.eye(4)
        target[:3, :3] = ee.rotation @ self.wrist_roll[side]
        target[:3, 3] = self.shoulder_asimov[side] + self.scale[side] * (
            ee.translation - self.shoulder_exo[side]
        )
        return target

    def compute_asimov_joints_from_exo(
        self,
        left_angles: dict[str, float],
        right_angles: dict[str, float],
    ) -> dict[str, float]:
        """Exo FK -> Asimov end-effector targets -> Asimov arm joints, as ``<SDK joint>.pos``."""
        self.targets = {
            "left": self._target("left", left_angles),
            "right": self._target("right", right_angles),
        }
        if all(t is None for t in self.targets.values()):
            logger.warning("missing ik targets, returning no action")
            return {}
        self.ik.solve(self.targets, frozen=self.frozen)
        return {f"{name}.pos": value for name, value in self.ik.arm_positions().items()}

    def init_visualization(self):
        """Browser visualization of Asimov and both exoskeletons, with end-effector markers."""
        try:
            from pinocchio.visualize import MeshcatVisualizer
        except ImportError as e:
            logger.warning(f"meshcat viz unavailable: {e}")
            return
        if self.ik.robot is None:
            raise RuntimeError("Create AsimovExoIKHelper(visualize=True) to load Asimov meshes.")

        robot = self.ik.robot
        self.viz_asimov = MeshcatVisualizer(robot.model, robot.collision_model, robot.visual_model)
        self.viz_asimov.initViewer(open=True)
        self.viz_asimov.loadViewerModel("asimov")
        self.viz_asimov.display(self.ik.q)

        self.viewer = self.viz_asimov.viewer
        self.markers = Markers(self.viewer)

        for side in SIDES:
            r = self.exo[side]
            v = MeshcatVisualizer(r.model, r.collision_model, r.visual_model)
            v.initViewer(open=False)
            v.viewer = self.viewer
            v.loadViewerModel(f"exo_{side}")
            self.viewer[f"exo_{side}"].set_transform(self._exo_viz_tf(side))
            v.display(self.q_exo[side])
            self.viz_exo[side] = v

            self.markers.sphere(f"markers/{side}_exo_ee", 0.012, (0.2, 1.0, 0.2, 0.9))
            self.markers.sphere(f"markers/{side}_asimov_ee", 0.015, (1.0, 0.2, 0.2, 0.9))
            self.markers.sphere(f"markers/{side}_ik_target", 0.015, (0.1, 0.3, 1.0, 0.9))
            self.markers.axes(f"markers/{side}_exo_axes", 0.06)
            self.markers.axes(f"markers/{side}_asimov_axes", 0.08)

        logger.info(f"meshcat viz initialized: {self.viewer.url()}")
        print(f"\nmeshcat url: {self.viewer.url()}\n")

    def _exo_viz_tf(self, side: str) -> np.ndarray:
        """Meshcat transform of an exo model: offset, then scaled about its shoulder."""
        tf = np.eye(4)
        tf[:3, :3] *= VIZ_SCALE
        tf[:3, 3] = VIZ_OFFSET[side] + (1.0 - VIZ_SCALE) * self.shoulder_exo[side]
        return tf

    def update_visualization(self):
        if self.viewer is None or self.markers is None:
            return

        self.viz_asimov.display(self.ik.q)
        for side in SIDES:
            ee_tf = self.ik.ee_pose(side)
            self.markers.tf(f"markers/{side}_asimov_ee", ee_tf)
            self.markers.tf(f"markers/{side}_asimov_axes", ee_tf)

            self.viz_exo[side].display(self.q_exo[side])
            exo_tf = self._exo_ee(side, {}).homogeneous
            exo_tf[:3, 3] = (self._exo_viz_tf(side) @ np.append(exo_tf[:3, 3], 1.0))[:3]
            self.markers.tf(f"markers/{side}_exo_ee", exo_tf)
            self.markers.tf(f"markers/{side}_exo_axes", exo_tf)

            if self.targets[side] is not None:
                self.markers.tf(f"markers/{side}_ik_target", self.targets[side])
