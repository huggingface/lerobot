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

"""Hold the lower body rigid so only the arms are under test.

The same division of labour as ``SonicLowerBodyController`` -- legs and waist here, arms
left to ``send_action`` -- with the balance policy removed. Nothing is decoded and nothing
reacts to the IMU: the legs and waist are pinned to a fixed pose by the joint PD and stay
there.

That makes it strictly a **harnessed-only** controller. It cannot balance, cannot step,
and will not catch the robot if it starts to fall; hung from a gantry it holds a clean,
repeatable posture, and on the floor it topples. What it buys is that arm behaviour is no
longer entangled with a balance policy shifting the torso underneath it -- the arms' base
frame, and the head and wrist cameras with it, stop moving for reasons the arms did not
ask for.

The hold pose is whatever the joints read on the first tick, so engaging it never yanks
the robot anywhere; put the legs where you want them, then start this.
"""

from __future__ import annotations

import logging

import numpy as np

from ..g1_utils import G1_29_JointArmIndex, G1_29_JointIndex
from ..unitree_g1 import RobotController

logger = logging.getLogger(__name__)

CONTROL_DT = 0.02  # 50 Hz is ample for a setpoint that never changes.


class StiffLowerBodyController(RobotController):
    """Pin legs and waist to their startup pose; leave motors 15-28 to ``send_action``."""

    control_dt = CONTROL_DT
    # Legs and waist: everything below the first arm joint. Same split as the SONIC variant,
    # so the arm side of the interface is identical and the laptop cannot tell them apart.
    controlled_joints = tuple(range(G1_29_JointArmIndex.kLeftShoulderPitch))
    # Arm joint targets, matching SonicLowerBodyController so both ends agree on the schema.
    action_ft = {f"{m.name}.q": float for m in G1_29_JointArmIndex}
    # No token echo: the retargeting layer needs the real arm angles to run FK, which is what
    # the robot advertises when this is None.
    observation_ft = None

    def __init__(self) -> None:
        self._hold: dict[str, float] | None = None
        self._owned = [m for m in G1_29_JointIndex if m.value in set(self.controlled_joints)]

    def run_step(self, action: dict, lowstate) -> dict:
        if self._hold is None:
            if lowstate is None:
                return {}
            self._hold = {f"{m.name}.q": float(lowstate.motor_state[m.value].q) for m in self._owned}
            held = np.rad2deg(list(self._hold.values()))
            logger.info(
                "[StiffLowerBody] holding %d joints at their startup pose "
                "(max %.1f deg from zero). Harness required: this cannot balance.",
                len(self._hold),
                np.abs(held).max(),
            )
        return dict(self._hold)
