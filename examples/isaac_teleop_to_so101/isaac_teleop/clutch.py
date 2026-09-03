#!/usr/bin/env python

# Copyright 2026 NVIDIA Corporation and The HuggingFace Inc. team. All rights reserved.
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

"""Engage-relative clutch for the XR -> SO-101 teleop loop.

Turns the raw controller grip pose into an absolute base-frame EE target, so the XR
device can stay a thin raw-pose reader. Pure numpy + the local ``Rotation`` helper (no
``isaacteleop``), so it is unit-testable without the XR runtime.
"""

from __future__ import annotations

import numpy as np

from lerobot.utils.rotation import Rotation


class Clutch:
    """Engage-relative clutch for both position AND orientation.

    Latch an origin on engage, then track the base-frame delta from it, applied
    independently to position and orientation. State:

    - ``_last_commanded_pos`` / ``_last_commanded_rot``: last commanded EE pose; held
      while disengaged so the arm freezes where it was left.
    - ``_home_pos`` / ``_home_rot``: latched on engage — the EE pose the delta applies to.
      Both come from the arm's MEASURED pose when the caller provides it, so the first IK
      target after re-clutch exactly matches the current FK pose.
    - ``_origin_pos`` / ``_origin_rot``: latched on engage — the controller pose the delta
      is measured against.

    Each engaged frame :meth:`rebase` returns::

        pos = home_pos + (grip_pos - origin_pos)  # 1:1 controller -> EE translation
        rot = (R_ctrl @ R_origin ^ -1) @ R_home  # base-frame delta, left-composed

    On the engage edge the output is exactly the home pose (no teleport). The orientation
    delta is left-composed (base frame), so hand rotation about base Z maps to EE rotation
    about base Z. A re-clutch latches a fresh home/origin.

    Re-anchoring measured orientation matters on this 5-DOF arm: carrying a stale soft
    orientation target across a processor reset can select a very different IK solution even
    when Cartesian position is continuous.
    """

    def __init__(self, home_base_T_ee: np.ndarray):  # noqa: N803
        # Seed the held pose from the arm's measured startup EE pose so the first
        # engage latches home there (no jump on the first squeeze).
        home = np.asarray(home_base_T_ee, dtype=float)
        self._last_commanded_pos = home[:3, 3].copy()
        self._last_commanded_rot = Rotation.from_matrix(home[:3, :3])
        self._home_pos = self._last_commanded_pos.copy()
        self._home_rot = self._last_commanded_rot
        self._origin_pos = np.zeros(3, dtype=float)
        self._origin_rot = Rotation.from_quat(np.array([0.0, 0.0, 0.0, 1.0]))

    def engage(
        self,
        grip_pos: np.ndarray,
        grip_quat: np.ndarray,
        measured_base_T_ee: np.ndarray | None = None,  # noqa: N803
    ) -> None:
        """Latch the engage home (where the arm is now) and controller origin.

        Pass ``measured_base_T_ee`` (FK of the measured joints) so the complete home pose is
        where the arm physically is — if the arm moved while disengaged (gravity sag,
        external contact), latching the stale last-commanded position would make the
        first engaged frame command a jump back to it.
        """
        if measured_base_T_ee is not None:
            measured = np.asarray(measured_base_T_ee, dtype=float)
            self._home_pos = measured[:3, 3].copy()
            self._home_rot = Rotation.from_matrix(measured[:3, :3])
        else:
            self._home_pos = self._last_commanded_pos.copy()
            self._home_rot = self._last_commanded_rot
        self._origin_pos = np.asarray(grip_pos, dtype=float).copy()
        self._origin_rot = Rotation.from_quat(np.asarray(grip_quat, dtype=float))

    def rebase(self, grip_pos: np.ndarray, grip_quat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the absolute base-frame EE target ``(pos [m], quat [xyzw])`` for this frame."""
        pos = self._home_pos + (np.asarray(grip_pos, dtype=float) - self._origin_pos)
        rot_ctrl = Rotation.from_quat(np.asarray(grip_quat, dtype=float))
        rot = (rot_ctrl * self._origin_rot.inv()) * self._home_rot
        self._last_commanded_pos = pos.copy()
        self._last_commanded_rot = rot
        return pos, rot.as_quat()
