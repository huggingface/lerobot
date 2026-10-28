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

"""The simulated SO-101: MuJoCo Menagerie's model, unit conversions and inverse kinematics.

Joint values outside MuJoCo use the units of LeRobot's ``so_follower`` with ``use_degrees=True``:
degrees for the five arm joints and 0-100 for the gripper opening.
"""

import os
import shutil
import subprocess  # nosec B404
import tempfile
from pathlib import Path

import mujoco
import numpy as np

MENAGERIE_REPO = "https://github.com/google-deepmind/mujoco_menagerie.git"
MENAGERIE_SHA = "0059d4335f8156206f63a35662313385f7ad6d74"
MENAGERIE_MODEL = "robotstudio_so101"

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
JOINT_NAMES = [f"{joint}.pos" for joint in JOINTS]
ARM_JOINTS = 5
WRIST_ROLL = 4
GRIPPER = 5
# Raised and folded back, clear of objects up to 7 cm tall in front of the base, gripper closed.
REST = np.array([0.0, -98.0, 60.0, 75.0, 0.0, 5.0])
# Folded down with the jaws just above the table: where real SO-101 recordings start (the median first
# frame of community datasets), shoulder_lift clipped to the simulated limit.
PARKED = np.array([-3.0, -100.0, 94.0, 61.0, 0.0, 2.0])
TCP_SITE = "gripperframe"
# Menagerie's joint ranges in degrees, the gripper's in degrees of its jaw (0-100 spans it).
JOINT_LIMITS_DEG = (
    (-110.0, 110.0),
    (-100.0, 100.0),
    (-96.8, 96.8),
    (-95.0, 95.0),
    (-157.2, 162.8),
    (-10.0, 100.0),
)
GRIPPER_RANGE_DEG = JOINT_LIMITS_DEG[5]


def menagerie_dir() -> Path:
    """A Menagerie checkout holding the SO-101, fetched at the pinned commit into the cache if needed.

    ``LEROBOT_MENAGERIE_DIR`` points at an existing checkout instead.
    """
    if os.environ.get("LEROBOT_MENAGERIE_DIR"):
        root = Path(os.environ["LEROBOT_MENAGERIE_DIR"]).expanduser()
        if not (root / MENAGERIE_MODEL / "so101.xml").is_file():
            raise FileNotFoundError(f"{root} has no {MENAGERIE_MODEL}/so101.xml")
        return root
    root = Path.home() / ".cache" / "lerobot" / "mujoco_menagerie" / MENAGERIE_SHA
    if not (root / MENAGERIE_MODEL / "so101.xml").is_file():
        root.parent.mkdir(parents=True, exist_ok=True)
        # Fetched beside the target and renamed into place, so an interrupted or concurrent fetch never
        # leaves a half-made checkout at ``root``.
        with tempfile.TemporaryDirectory(dir=root.parent) as tmp:
            # Only the SO-101 directory, at a pinned commit so an upstream change can't alter the data.
            for args in (
                ("init", "-q"),
                ("remote", "add", "origin", MENAGERIE_REPO),
                ("sparse-checkout", "set", MENAGERIE_MODEL),
                ("fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", MENAGERIE_SHA),
                ("checkout", "-q", "FETCH_HEAD"),
            ):
                subprocess.run(("git", "-C", tmp, *args), check=True)  # nosec B603 B607
            shutil.rmtree(root, ignore_errors=True)
            try:
                os.rename(tmp, root)
            except OSError:
                # Another process got there first.
                if not (root / MENAGERIE_MODEL / "so101.xml").is_file():
                    raise
    return root


def arm_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(menagerie_dir() / MENAGERIE_MODEL / "so101.xml"))


class Arm:
    """Indices of the SO-101 in a compiled model, unit conversions and kinematics."""

    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.qadr = np.array([model.joint(name).qposadr[0] for name in JOINTS])
        self.dofadr = np.array([model.joint(name).dofadr[0] for name in JOINTS])
        self.actuators = np.array([model.actuator(name).id for name in JOINTS])
        self.lo, self.hi = model.actuator_ctrlrange[self.actuators].T.copy()
        self.tcp = model.site(TCP_SITE).id
        self.base = model.body("base").id
        self.bodies = set(range(self.base, model.body("moving_jaw_so101_v1").id + 1))
        self._scratch = mujoco.MjData(model)
        self._measure_jaws()

    def _measure_jaws(self) -> None:
        """Jaw gap as a function of the gripper opening, and where the fixed jaw sits in the site frame."""
        data = self._scratch
        fixed = self.model.geom("fixed_jaw_sph_tip1").id
        moving = self.model.geom("moving_jaw_sph_tip1").id
        openings = np.linspace(0, 100, 21)
        gaps = []
        for opening in openings:
            data.qpos[self.qadr] = self.to_ctrl(np.array([0.0, 0.0, 0.0, 90.0, 0.0, opening]))
            mujoco.mj_kinematics(self.model, data)
            gaps.append(np.linalg.norm(data.geom_xpos[moving] - data.geom_xpos[fixed]))
        self._openings, self._gaps = openings, np.asarray(gaps)
        rot = data.site_xmat[self.tcp].reshape(3, 3)
        # Offset of the fixed jaw from the site along the jaw axis (site z): about -2 cm.
        self.fixed_jaw_offset = float((rot.T @ (data.geom_xpos[fixed] - data.site_xpos[self.tcp]))[2])

    def opening_for_width(self, width: float) -> float:
        """Gripper opening (0-100) whose jaw gap is ``width`` meters."""
        return float(np.interp(width, self._gaps, self._openings))

    def to_ctrl(self, values: np.ndarray) -> np.ndarray:
        """Degrees and the 0-100 gripper opening to actuator radians."""
        ctrl = np.deg2rad(np.asarray(values, dtype=np.float64))
        ctrl[GRIPPER] = self.lo[GRIPPER] + values[GRIPPER] / 100 * (self.hi[GRIPPER] - self.lo[GRIPPER])
        return np.clip(ctrl, self.lo, self.hi)

    def from_qpos(self, qpos: np.ndarray) -> np.ndarray:
        """Joint positions of a full ``qpos`` in the units of ``to_ctrl``."""
        q = qpos[self.qadr]
        values = np.rad2deg(q)
        values[GRIPPER] = (q[GRIPPER] - self.lo[GRIPPER]) / (self.hi[GRIPPER] - self.lo[GRIPPER]) * 100
        return values

    def tcp_pose(self, values: np.ndarray, qpos: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Position and rotation matrix of the gripper site for joint ``values``."""
        data = self._scratch
        if qpos is not None:
            data.qpos[:] = qpos
        data.qpos[self.qadr] = self.to_ctrl(values)
        mujoco.mj_kinematics(self.model, data)
        return data.site_xpos[self.tcp].copy(), data.site_xmat[self.tcp].reshape(3, 3).copy()

    def ik(
        self,
        pos: np.ndarray,
        approach: np.ndarray,
        jaw: np.ndarray | None,
        init: np.ndarray,
        iters: int = 60,
        tol: float = 5e-4,
        roll: float | None = None,
    ) -> tuple[np.ndarray, float]:
        """Damped least squares on the five arm joints.

        Reaches ``pos`` with the gripper's approach axis (site x) along ``approach`` and, when given,
        its jaw axis (site z, from the fixed towards the moving jaw) along ``jaw``. ``roll`` (degrees)
        weakly pulls the wrist roll there instead, at the cost of a slightly looser approach: position
        and approach alone leave only a few solutions, some with the wrist half a turn away. Starts from
        ``init`` (degrees, gripper kept) and returns the solution with its position error in meters.
        """
        model, data = self.model, self._scratch
        q = self.to_ctrl(init)
        cols = self.dofadr[:ARM_JOINTS]
        jp = np.zeros((3, model.nv))
        jr = np.zeros((3, model.nv))
        approach = approach / np.linalg.norm(approach)
        err = np.inf
        for _ in range(iters):
            data.qpos[self.qadr] = q
            mujoco.mj_kinematics(model, data)
            mujoco.mj_comPos(model, data)
            p = data.site_xpos[self.tcp]
            rot = data.site_xmat[self.tcp].reshape(3, 3)
            a, z = rot[:, 0], rot[:, 2]
            mujoco.mj_jacSite(model, data, jp, jr, self.tcp)
            w = jr[:, cols]
            # d(axis)/dq = omega x axis, written as -[axis]x @ omega.
            residuals = [pos - p, 0.25 * (approach - a)]
            jacobians = [jp[:, cols], 0.25 * -_skew(a) @ w]
            if jaw is not None:
                residuals.append(0.15 * (jaw - z))
                jacobians.append(0.15 * -_skew(z) @ w)
            if roll is not None:
                residuals.append(np.array([0.1 * (np.deg2rad(roll) - q[WRIST_ROLL])]))
                jacobians.append(0.1 * np.eye(ARM_JOINTS)[WRIST_ROLL][None])
            e = np.concatenate(residuals)
            jac = np.concatenate(jacobians)
            err = float(np.linalg.norm(pos - p))
            if err < tol and np.linalg.norm(e[3:6]) < 2e-3 and (jaw is None or np.linalg.norm(e[6:9]) < 2e-3):
                break
            dq = jac.T @ np.linalg.solve(jac @ jac.T + 1e-4 * np.eye(len(e)), e)
            q[:ARM_JOINTS] = np.clip(q[:ARM_JOINTS] + dq, self.lo[:ARM_JOINTS], self.hi[:ARM_JOINTS])
        out = np.array(init, dtype=np.float64)
        out[:ARM_JOINTS] = np.rad2deg(q[:ARM_JOINTS])
        return out, err


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
