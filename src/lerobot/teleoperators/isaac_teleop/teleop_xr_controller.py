#!/usr/bin/env python

# Copyright 2024 NVIDIA Corporation and The HuggingFace Inc. team. All rights reserved.
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

"""XR (VR) controller device for NVIDIA Isaac Teleop, exposed to LeRobot.

``XRController`` is the first concrete :class:`IsaacTeleopTeleoperator` device
(see :mod:`lerobot.teleoperators.isaac_teleop.base` for the multi-device
pattern). It wires an Isaac Teleop retargeting pipeline (the three SO-101
retargeters) that turns a single XR controller into a clutch-rebased
end-effector pose, a wrist-roll angle, an analog gripper closedness, and a
clutch (``enabled``) signal. The output dict from :meth:`XRController.get_action` is
designed to feed LeRobot's existing closed-loop IK pipeline (the same one phone
teleoperation uses) via :class:`MapXRControllerActionToRobotAction`.

The shared ``TeleopSession`` lifecycle and per-step health guard live on the
base class; this module only adds the controller-specific pipeline and action
unpacking. The ``isaacteleop`` package is an optional, separately distributed
NVIDIA dependency (the ``isaac-teleop`` extra); all imports of it are deferred
so this module can be imported — and the processor unit-tested — without it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from lerobot.types import RobotAction

from .base import IsaacTeleopTeleoperator
from .config_isaac_teleop import XRControllerConfig

if TYPE_CHECKING:
    from isaacteleop.retargeting_engine.interface import OutputCombiner


class XRController(IsaacTeleopTeleoperator):
    """XR controller -> EE pose + wrist roll + gripper + clutch teleoperator.

    Wraps Isaac Teleop's three SO-101 retargeters (``SO101ClutchRetargeter``,
    ``SO101RollRetargeter``, ``SO101GripperRetargeter``) plus the raw
    ``ControllersSource`` passthrough (for the squeeze clutch) into a single
    pipeline. Squeezing the controller grip past
    :attr:`XRControllerConfig.clutch_threshold` engages the clutch
    (``enabled``); releasing it freezes the robot, exactly like the phone
    teleoperator's hold-to-enable button.
    """

    config_class = XRControllerConfig
    name = "isaac_teleop_controller"

    def __init__(self, config: XRControllerConfig):
        super().__init__(config)
        self.config: XRControllerConfig = config

    # ------------------------------------------------------------------
    # Pipeline construction
    # ------------------------------------------------------------------

    def _build_pipeline(self) -> OutputCombiner:
        """Build the XR controller retargeting pipeline.

        Reuses the three Isaac Teleop SO-101 retargeters (5-DOF arm), one
        ``ControllersSource`` feeding all of them, plus a raw controller
        passthrough for the squeeze clutch::

            ControllersSource ─┬─ SO101ClutchRetargeter ──── ee_pose (7D, base frame)
                               ├─ SO101RollRetargeter ────── roll_command (rad)
                               ├─ SO101GripperRetargeter ─── gripper_command (closedness [0,1])
                               └─ (passthrough) ──────────── controller (raw inputs)

        The clutch retargeter latches its own controller origin on the connect
        frame; the downstream :class:`MapXRControllerActionToRobotAction`
        re-latches an engage-home on the squeeze rising edge so the emitted
        delta is zero at engage (no teleport). The raw ``controller_{side}``
        passthrough is read in :meth:`get_action` for ``SQUEEZE_VALUE`` (the
        sole clutch — see :class:`XRControllerConfig.clutch_threshold`).
        """
        from isaacteleop.retargeters import (
            SO101ClutchRetargeter,
            SO101GripperRetargeter,
            SO101RollRetargeter,
        )
        from isaacteleop.retargeting_engine.deviceio_source_nodes import ControllersSource
        from isaacteleop.retargeting_engine.interface import OutputCombiner

        side = self.config.hand_side
        controller_key = f"controller_{side}"
        controllers = ControllersSource(name="controllers")

        # Clutch-rebased absolute EE pose -> 7D ee_pose (output key "ee_pose").
        clutch = SO101ClutchRetargeter(name="ee_pose", input_device=controller_key)
        connected_clutch = clutch.connect({controller_key: controllers.output(controller_key)})

        # Absolute swing-twist wrist roll [rad] (output group "roll_command").
        roll = SO101RollRetargeter(name="roll", input_device=controller_key)
        connected_roll = roll.connect({controller_key: controllers.output(controller_key)})

        # Proportional jaw closedness [0,1] (output group "gripper_command").
        gripper = SO101GripperRetargeter(name="gripper", input_device=controller_key)
        connected_gripper = gripper.connect({controller_key: controllers.output(controller_key)})

        return OutputCombiner(
            {
                "ee_pose": connected_clutch.output("ee_pose"),
                "roll": connected_roll.output("roll_command"),
                "gripper": connected_gripper.output("gripper_command"),
                "controller": controllers.output(controller_key),
            }
        )

    # ------------------------------------------------------------------
    # Action features
    # ------------------------------------------------------------------

    @property
    def action_features(self) -> dict:
        return {
            "ee_pose": {
                "dtype": "float32",
                "shape": (7,),
                "names": {"x": 0, "y": 1, "z": 2, "qx": 3, "qy": 4, "qz": 5, "qw": 6},
            },
            # ``get_action`` returns scalars for these three, so the advertised
            # shape is () (0-d) to stay consistent with the returned values.
            "wrist_roll": {
                "dtype": "float32",
                "shape": (),
                "names": None,
            },
            "closedness": {
                "dtype": "float32",
                "shape": (),
                "names": None,
            },
            "enabled": {
                "dtype": "bool",
                "shape": (),
                "names": None,
            },
        }

    @property
    def feedback_features(self) -> dict:
        return {}

    # ------------------------------------------------------------------
    # Action extraction
    # ------------------------------------------------------------------

    def get_action(self) -> RobotAction:
        """Step the Isaac Teleop session and return EE pose + roll + gripper + clutch.

        Returns:
            A ``RobotAction`` dict with keys:

            - ``"ee_pose"``: ``np.ndarray`` shape ``(7,)`` — clutch-rebased
              absolute pose ``[x, y, z, qx, qy, qz, qw]`` in the robot base
              frame (position in metres).
            - ``"wrist_roll"``: ``float`` — absolute wrist-roll angle in
              **radians** (swing-twist about world Z).
            - ``"closedness"``: ``float`` — jaw closedness in ``[0, 1]``
              (``0`` = open, ``1`` = closed).
            - ``"enabled"``: ``bool`` — clutch state (squeeze held past
              ``clutch_threshold``).
        """
        # Steps the session and applies the shared staleness/worker-health
        # guard (see IsaacTeleopTeleoperator._step).
        result = self._step()

        # Retargeter outputs are batched (leading slot/batch dim); index [0]
        # selects the single tracked controller and drops that dim.
        # SO101ClutchRetargeter outputs a 7D array: [x, y, z, qx, qy, qz, qw].
        ee_pose = np.asarray(result["ee_pose"][0], dtype=np.float32)
        # SO101RollRetargeter emits a single absolute roll angle [rad].
        wrist_roll = float(result["roll"][0])
        # SO101GripperRetargeter emits a single closedness scalar in [0, 1].
        closedness = float(result["gripper"][0])

        # Raw controller passthrough -> clutch from the squeeze analog. When
        # the controller is not tracked the optional group is None; treat that
        # as "not engaged" so the robot freezes safely.
        from isaacteleop.retargeting_engine.tensor_types.indices import ControllerInputIndex

        controller = result["controller"]
        # Defensive: a passthrough controller group may not be tracked every frame
        # (untracked/odd frame, missing squeeze axis, unexpected wrapper shape). Any
        # failure to read the squeeze is treated as "not engaged" (squeeze = 0.0) so
        # the arm freezes safely instead of crashing the teleop loop.
        if getattr(controller, "is_none", False):
            squeeze = 0.0
        else:
            try:
                squeeze = float(controller[ControllerInputIndex.SQUEEZE_VALUE])
            except (IndexError, KeyError, TypeError, ValueError):
                squeeze = 0.0
        enabled = bool(squeeze > self.config.clutch_threshold)

        return {
            "ee_pose": ee_pose,
            "wrist_roll": wrist_roll,
            "closedness": closedness,
            "enabled": enabled,
        }
