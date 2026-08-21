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

The hold pose is a fixed nominal stance, not wherever the robot happened to be left, so
every session starts from the same geometry and an arm result from one run means the same
thing as an arm result from the next. Declaring it as ``default_angles`` is what makes
``connect()`` ease into it through the normal reset sweep instead of snapping.
"""

from __future__ import annotations

import logging

import numpy as np

from ..g1_utils import G1_29_JointArmIndex, G1_29_JointIndex
from ..unitree_g1 import RobotController

logger = logging.getLogger(__name__)

CONTROL_DT = 0.02  # 50 Hz is ample for a setpoint that never changes.

# The G1's nominal stance: legs in the usual slight crouch, waist square, arms a little
# forward of the body so they neither rest on the thighs nor start the IK from a
# singular straight-down pose. Same leg angles the Holosoma controller homes to, so a
# harnessed run and a standing one start from the same shape.
STAND_POSE = np.zeros(29, dtype=np.float32)
STAND_POSE[[0, 6]] = -0.312  # hip pitch
STAND_POSE[[3, 9]] = 0.669  # knee
STAND_POSE[[4, 10]] = -0.363  # ankle pitch
STAND_POSE[[15, 22]] = 0.2  # shoulder pitch
STAND_POSE[16] = 0.2  # left shoulder roll
STAND_POSE[23] = -0.2  # right shoulder roll
STAND_POSE[[18, 25]] = 0.6  # elbow


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

    def __init__(self, stand_pose: np.ndarray | None = None) -> None:
        # Read by the robot at connect() to home the whole body before the loop starts, and
        # so covers the arms too even though only the lower body is held here.
        self.default_angles = STAND_POSE if stand_pose is None else np.asarray(stand_pose, np.float32)
        owned = set(self.controlled_joints)
        self._hold = {
            f"{m.name}.q": float(self.default_angles[m.value]) for m in G1_29_JointIndex if m.value in owned
        }
        self._announced = False

    def run_step(self, action: dict, lowstate) -> dict:
        if not self._announced:
            self._announced = True
            logger.info(
                "[StiffLowerBody] holding %d lower-body joints at the nominal stance. "
                "Harness required: this cannot balance.",
                len(self._hold),
            )
        return dict(self._hold)
