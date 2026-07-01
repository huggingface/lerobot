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

The clutch turns the raw controller grip pose into an absolute base-frame EE
target so the owning loop's XR device can stay a thin raw-pose reader (see
``examples/isaac_teleop_to_so101/common.py``). It is the pure-math counterpart to
Isaac Teleop's ``SO101ClutchRetargeter``, lifted here so it can be unit-tested
without the XR runtime.

Like ``xr_controller_processor``, this module does **not** import ``isaacteleop``
(only ``numpy`` + the local ``Rotation`` helper), so it can be unit-tested without
the XR runtime.
"""

from __future__ import annotations

import numpy as np

from lerobot.utils.rotation import Rotation


class Clutch:
    """Engage-relative clutch for both position AND orientation.

    Mirrors Isaac Teleop's ``SO101ClutchRetargeter`` but lives outside the device
    so it can stay a thin raw-pose reader. Clutching is the same idea for both
    channels — latch an origin on engage, then track the base-frame delta from it —
    applied independently to position and orientation. State:

    - ``_last_commanded_pos`` / ``_last_commanded_rot``: the EE pose the loop last
      commanded; held while disengaged so the arm freezes where it was left.
    - ``_home_pos`` / ``_home_rot``: latched on the engage edge — the EE pose the
      per-frame delta is applied to.
    - ``_origin_pos`` / ``_origin_rot``: latched on the engage edge — the controller
      pose the per-frame delta is measured against.

    Each engaged frame :meth:`rebase` returns::

        pos = home_pos + (grip_pos - origin_pos)  # 1:1 controller -> EE translation
        rot = (R_ctrl @ R_origin ^ -1) @ R_home  # base-frame delta, left-composed

    On the engage edge ``grip_pos == origin_pos`` and ``R_ctrl == R_origin``, so the
    output is exactly the home pose (== the last commanded pose), i.e. no teleport in
    position OR orientation. The orientation delta is expressed in the base frame
    (left multiply), so rotating the hand 30° about base Z rotates the EE 30° about
    base Z — matching the position convention the operator sees in the room. A
    mid-task re-clutch latches a fresh home/origin, so the EE resumes from where it
    was left and tracks the new delta.

    NOTE: ``_home_rot`` is the last *commanded* orientation, not the achieved one. On
    the 5-DOF SO-101 the arm cannot fully realize an arbitrary orientation, so the
    commanded and achieved wrist orientation differ — but the commanded signal is
    continuous across a re-clutch, so there is still no jump.
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

    def engage(self, grip_pos: np.ndarray, grip_quat: np.ndarray) -> None:
        """Latch the engage home (where the arm is now) and controller origin."""
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
