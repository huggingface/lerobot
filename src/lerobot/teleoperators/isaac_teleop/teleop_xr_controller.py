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
pattern). It wires an Isaac Teleop retargeting pipeline that turns a single XR
controller into an absolute end-effector pose, a gripper command, and a clutch
(``enabled``) signal. The output dict from :meth:`XRController.get_action` is
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
    """XR controller -> EE pose + gripper + clutch teleoperator.

    Wraps Isaac Teleop's ``ControllersSource`` + ``Se3AbsRetargeter`` +
    ``GripperRetargeter`` into a single pipeline.  Squeezing the controller
    grip past :attr:`XRControllerConfig.clutch_threshold` engages the clutch
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

        Pipeline graph::

            ControllersSource ─┬─ Se3AbsRetargeter ──── ee_pose (7D)
                               │
                               ├─ GripperRetargeter ─── gripper_command (scalar)
            HandsSource ───────┘
                               │
            ControllersSource ─┴─ (passthrough) ─────── controller (raw inputs)

        ``HandsSource`` is required: ``GripperRetargeter.input_spec()``
        declares both ``controller_{side}`` and ``hand_{side}`` (it can fall
        back to hand pinch-distance when controller trigger data is absent).
        The raw ``controller_{side}`` output is also passed through so
        :meth:`get_action` can read ``SQUEEZE_VALUE`` for the clutch.
        """
        from isaacteleop.retargeters import (
            GripperRetargeter,
            GripperRetargeterConfig,
            Se3AbsRetargeter,
            Se3RetargeterConfig,
        )
        from isaacteleop.retargeting_engine.deviceio_source_nodes import (
            ControllersSource,
            HandsSource,
        )
        from isaacteleop.retargeting_engine.interface import OutputCombiner

        side = self.config.hand_side
        controllers = ControllersSource(name="controllers")
        hands = HandsSource(name="hands")

        # Se3 absolute pose retargeter: controller aim pose -> 7D EE pose.
        # zero_out_xy_rotation defaults to True, which matches intent for the
        # position-dominant 5-DOF SO-101 IK.
        se3_cfg = Se3RetargeterConfig(input_device=f"controller_{side}")
        se3 = Se3AbsRetargeter(se3_cfg, name="ee_pose")
        connected_se3 = se3.connect({f"controller_{side}": controllers.output(f"controller_{side}")})

        # Gripper retargeter: trigger/pinch -> scalar command.
        gripper_cfg = GripperRetargeterConfig(hand_side=side)
        gripper = GripperRetargeter(gripper_cfg, name="gripper")
        connected_gripper = gripper.connect(
            {
                f"controller_{side}": controllers.output(f"controller_{side}"),
                f"hand_{side}": hands.output(f"hand_{side}"),
            }
        )

        return OutputCombiner(
            {
                "ee_pose": connected_se3.output("ee_pose"),
                "gripper": connected_gripper.output("gripper_command"),
                "controller": controllers.output(f"controller_{side}"),
            }
        )

    # ------------------------------------------------------------------
    # Action features
    # ------------------------------------------------------------------

    @property
    def action_features(self) -> dict:
        return {
            "ee_pos": {
                "dtype": "float32",
                "shape": (3,),
                "names": {"x": 0, "y": 1, "z": 2},
            },
            "ee_quat": {
                "dtype": "float32",
                "shape": (4,),
                "names": {"qx": 0, "qy": 1, "qz": 2, "qw": 3},
            },
            # ``get_action`` returns scalars for these two, so the advertised
            # shape is () (0-d) to stay consistent with the returned values.
            "gripper": {
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
        """Step the Isaac Teleop session and return EE pose + gripper + clutch.

        Returns:
            A ``RobotAction`` dict with keys:

            - ``"ee_pos"``: ``np.ndarray`` shape ``(3,)`` — position in metres
              (absolute, OpenXR frame).
            - ``"ee_quat"``: ``np.ndarray`` shape ``(4,)`` — quaternion
              ``(x, y, z, w)`` (absolute, OpenXR frame).
            - ``"gripper"``: ``float`` — ``-1.0`` (closed) or ``1.0`` (open).
            - ``"enabled"``: ``bool`` — clutch state (squeeze held past
              ``clutch_threshold``).
        """
        # Steps the session and applies the shared staleness/worker-health
        # guard (see IsaacTeleopTeleoperator._step).
        result = self._step()

        # Retargeter outputs are batched (leading slot/batch dim); index [0]
        # selects the single tracked controller and drops that dim.
        # Se3AbsRetargeter outputs a 7D array: [x, y, z, qx, qy, qz, qw].
        ee_pose = np.asarray(result["ee_pose"][0], dtype=np.float32)
        # gripper is a binary command, not a velocity: -1.0 (closed) / +1.0 (open).
        gripper_val = float(result["gripper"][0])

        # Raw controller passthrough -> clutch from the squeeze analog. When
        # the controller is not tracked the optional group is None; treat that
        # as "not engaged" so the robot freezes safely.
        from isaacteleop.retargeting_engine.tensor_types.indices import ControllerInputIndex

        controller = result["controller"]
        # Defensive default: a passthrough controller group may not be tracked
        # every frame, and not all output wrappers guarantee an ``is_none``
        # attribute, so fall back to "tracked" rather than crash.
        if getattr(controller, "is_none", False):
            squeeze = 0.0
        else:
            squeeze = float(controller[ControllerInputIndex.SQUEEZE_VALUE])
        enabled = bool(squeeze > self.config.clutch_threshold)

        return {
            "ee_pos": ee_pose[:3],
            "ee_quat": ee_pose[3:],
            "gripper": gripper_val,
            "enabled": enabled,
        }
