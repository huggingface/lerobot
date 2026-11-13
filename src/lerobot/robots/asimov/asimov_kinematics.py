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

"""Asimov 1 arm kinematics on Menlo's open-source 25-DOF URDF.

Each 5-DOF arm (shoulder pitch/roll/yaw, elbow, wrist yaw) is solved independently with
damped least squares, so only stock Pinocchio is needed (no CasADi bindings).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from huggingface_hub import sync_bucket

from lerobot.utils.constants import HF_LEROBOT_HOME

logger = logging.getLogger(__name__)

# Copy of menloresearch/asimov-1 sim-model/asimov_1_25dof (CERN-OHL-S-2.0) at commit
# 96a6807c163e4e6cabdc124eefcc45b0ae2cb57a, in the public lerobot/robot-urdfs bucket.
ASIMOV_MODEL_BUCKET = "hf://buckets/lerobot/robot-urdfs/asimov"
ASIMOV_MODEL_DIR = "asimov_1_25dof"
ASIMOV_MESHES = (
    "pelvis_link",
    "waist_yaw_link",
    "neck_yaw_link",
    "neck_pitch_link",
    *(
        f"{side}_{link}_link"
        for side in ("left", "right")
        for link in (
            "hip_pitch",
            "hip_roll",
            "hip_yaw",
            "knee",
            "ankle_pitch",
            "ankle_roll",
            "shoulder_pitch",
            "shoulder_roll",
            "shoulder_yaw",
            "elbow",
            "wrist_yaw",
        )
    ),
)

# Arm joints in Menlo SDK names, in firmware order.
ASIMOV_ARM_JOINTS: dict[str, tuple[str, ...]] = {
    side: tuple(
        f"{side[0].upper()}_{joint}"
        for joint in ("Shoulder_Pitch", "Shoulder_Roll", "Shoulder_Yaw", "Elbow", "Wrist_Yaw")
    )
    for side in ("left", "right")
}

# The wrist_yaw_link mesh ends ~0.084 m along the forearm axis, where the hand mounts.
DEFAULT_EE_OFFSET = 0.08


def urdf_joint_name(sdk_name: str) -> str:
    """``L_Shoulder_Pitch`` -> ``left_shoulder_pitch_joint``. Not valid for the ankle motors."""
    side, _, joint = sdk_name.partition("_")
    prefix = {"L": "left_", "R": "right_"}.get(side)
    return f"{prefix}{joint.lower()}_joint" if prefix else f"{sdk_name.lower()}_joint"


def get_asimov_urdf(with_meshes: bool = False) -> Path:
    """Fetch the Asimov 1 URDF (and optionally its meshes) from the robot-urdfs bucket into the cache."""
    dest = HF_LEROBOT_HOME / "robot-urdfs" / "asimov"
    root = dest / ASIMOV_MODEL_DIR
    urdf = root / "urdf" / "asimov_1_25dof.urdf"
    needed = [urdf, *(root / "meshes" / f"{mesh}.STL" for mesh in ASIMOV_MESHES)] if with_meshes else [urdf]
    if not all(path.exists() for path in needed):
        include = [f"{ASIMOV_MODEL_DIR}/urdf/*"] + ([f"{ASIMOV_MODEL_DIR}/meshes/*"] if with_meshes else [])
        logger.info(f"Downloading the Asimov model from {ASIMOV_MODEL_BUCKET}")
        sync_bucket(ASIMOV_MODEL_BUCKET, str(dest), include=include, quiet=True)
        missing = [str(path) for path in needed if not path.exists()]
        if missing:
            raise FileNotFoundError(
                f"Asimov model files missing after download from {ASIMOV_MODEL_BUCKET}: {missing}"
            )
    return urdf


class AsimovArmIK:
    """Solves both Asimov arms for world-frame (pelvis) end-effector targets.

    Frames ``L_ee``/``R_ee`` sit ``ee_offset`` along the forearm axis of each wrist yaw link.
    At elbow = 90° (forearm forward) they are aligned with the pelvis frame, the same
    convention as the G1 ``L_ee``/``R_ee`` frames and the exoskeleton ``ee`` frame.
    """

    def __init__(
        self,
        urdf_path: str | Path | None = None,
        with_meshes: bool = False,
        ee_offset: float = DEFAULT_EE_OFFSET,
        position_weight: float = 1.0,
        rotation_weight: float = 0.02,
        damping: float = 1e-3,
        max_iters: int = 50,
        max_step: float = 0.2,
        tolerance: float = 1e-4,
        retry_threshold: float = 0.02,
    ) -> None:
        try:
            import pinocchio as pin
        except ImportError as e:
            raise ImportError("Asimov IK needs pinocchio: pip install pin") from e

        self.pin = pin
        self.urdf_path = Path(urdf_path) if urdf_path else get_asimov_urdf(with_meshes=with_meshes)
        self.mesh_dir = self.urdf_path.parent.parent / "meshes"
        if with_meshes:
            self.robot = pin.RobotWrapper.BuildFromURDF(str(self.urdf_path), str(self.mesh_dir))
            self.model = self.robot.model
        else:
            self.robot = None
            self.model = pin.buildModelFromUrdf(str(self.urdf_path))

        self.ee_frame: dict[str, int] = {}
        for side, name in (("left", "L_ee"), ("right", "R_ee")):
            parent = self.model.getJointId(f"{side}_wrist_yaw_joint")
            self.ee_frame[side] = self.model.addFrame(
                pin.Frame(
                    name, parent, pin.SE3(np.eye(3), np.array([ee_offset, 0.0, 0.0])), pin.FrameType.OP_FRAME
                )
            )
        if self.robot is not None:
            self.robot.data = self.model.createData()
            self.data = self.robot.data
        else:
            self.data = self.model.createData()

        self.q = pin.neutral(self.model)
        self.lower = self.model.lowerPositionLimit.copy()
        self.upper = self.model.upperPositionLimit.copy()

        # side -> (q indices, v indices) of the five arm joints, in firmware order.
        self.arm_idx: dict[str, tuple[list[int], list[int]]] = {}
        for side, names in ASIMOV_ARM_JOINTS.items():
            jids = [self.model.getJointId(urdf_joint_name(n)) for n in names]
            self.arm_idx[side] = (
                [self.model.idx_qs[j] for j in jids],
                [self.model.idx_vs[j] for j in jids],
            )

        self.weights = np.array([position_weight] * 3 + [rotation_weight] * 3)
        self.damping = damping
        self.max_iters = max_iters
        self.max_step = max_step
        self.tolerance = tolerance
        self.retry_threshold = retry_threshold

    def shoulder_position(self, side: str) -> np.ndarray:
        """Shoulder roll (abduction) joint origin in the pelvis frame; the arm hangs below it."""
        q0 = self.pin.neutral(self.model)
        self.pin.forwardKinematics(self.model, self.data, q0)
        shoulder = self.data.oMi[self.model.getJointId(f"{side}_shoulder_roll_joint")].translation.copy()
        self.pin.framesForwardKinematics(self.model, self.data, self.q)
        return shoulder

    def arm_reach(self, side: str) -> float:
        """Shoulder -> elbow -> end-effector length."""
        q0 = self.pin.neutral(self.model)
        self.pin.framesForwardKinematics(self.model, self.data, q0)
        shoulder = self.data.oMi[self.model.getJointId(f"{side}_shoulder_roll_joint")].translation
        elbow = self.data.oMi[self.model.getJointId(f"{side}_elbow_joint")].translation
        ee = self.data.oMf[self.ee_frame[side]].translation
        reach = float(np.linalg.norm(elbow - shoulder) + np.linalg.norm(ee - elbow))
        self.pin.framesForwardKinematics(self.model, self.data, self.q)
        return reach

    def ee_pose(self, side: str) -> np.ndarray:
        self.pin.framesForwardKinematics(self.model, self.data, self.q)
        return self.data.oMf[self.ee_frame[side]].homogeneous.copy()

    def set_arm_positions(self, positions: dict[str, float]) -> None:
        """Seed the solver from SDK-named arm positions (e.g. the measured robot pose)."""
        for side, names in ASIMOV_ARM_JOINTS.items():
            for name, qi in zip(names, self.arm_idx[side][0], strict=True):
                if name in positions:
                    self.q[qi] = np.clip(positions[name], self.lower[qi], self.upper[qi])

    def arm_positions(self) -> dict[str, float]:
        return {
            name: float(self.q[qi])
            for side, names in ASIMOV_ARM_JOINTS.items()
            for name, qi in zip(names, self.arm_idx[side][0], strict=True)
        }

    def solve(self, targets: dict[str, Any], frozen: set[str] | frozenset[str] = frozenset()) -> np.ndarray:
        """Move each arm towards its 4x4 target (None keeps it); frozen SDK joint names stay put.

        Warm-starts from the previous solution. When that lands far from the target (a local
        minimum after a large jump), it retries from the middle of the joint range.
        """
        pin = self.pin
        for side, target in targets.items():
            if target is None:
                continue
            goal = pin.SE3(np.asarray(target, dtype=float))
            q_idx, v_idx = self.arm_idx[side]
            active = [k for k, name in enumerate(ASIMOV_ARM_JOINTS[side]) if name not in frozen]
            if not active:
                continue
            if self._solve_arm(goal, side, active) > self.retry_threshold:
                warm = self.q[q_idx].copy()
                warm_err = self._pose_error(goal, side)
                for k in active:
                    qi = q_idx[k]
                    self.q[qi] = 0.5 * (self.lower[qi] + self.upper[qi])
                if self._solve_arm(goal, side, active) > warm_err:
                    self.q[q_idx] = warm
        pin.framesForwardKinematics(self.model, self.data, self.q)
        return self.q

    def _pose_error(self, goal: Any, side: str) -> float:
        self.pin.framesForwardKinematics(self.model, self.data, self.q)
        cur = self.data.oMf[self.ee_frame[side]]
        rot = self.pin.log3(goal.rotation @ cur.rotation.T)
        err = np.concatenate([goal.translation - cur.translation, rot])
        return float(np.sqrt(np.sum(self.weights * err**2)))

    def _solve_arm(self, goal: Any, side: str, active: list[int]) -> float:
        pin = self.pin
        q_idx, v_idx = self.arm_idx[side]
        fid = self.ee_frame[side]
        for _ in range(self.max_iters):
            pin.framesForwardKinematics(self.model, self.data, self.q)
            cur = self.data.oMf[fid]
            err = np.concatenate(
                [goal.translation - cur.translation, pin.log3(goal.rotation @ cur.rotation.T)]
            )
            jac = pin.computeFrameJacobian(
                self.model, self.data, self.q, fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
            )[:, [v_idx[k] for k in active]]
            jtw = jac.T * self.weights
            dq = np.linalg.solve(jtw @ jac + self.damping * np.eye(len(active)), jtw @ err)
            norm = np.linalg.norm(dq)
            if norm > self.max_step:
                dq *= self.max_step / norm
            for k, step in zip(active, dq, strict=True):
                qi = q_idx[k]
                self.q[qi] = np.clip(self.q[qi] + step, self.lower[qi], self.upper[qi])
            if norm < self.tolerance:
                break
        return self._pose_error(goal, side)
