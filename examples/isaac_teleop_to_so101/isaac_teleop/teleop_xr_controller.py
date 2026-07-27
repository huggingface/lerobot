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

"""XR (VR) controller device for NVIDIA Isaac Teleop, exposed to LeRobot.

A deliberately thin reader: exposes the raw controller grip pose off
``ControllersSource`` (statically rebased into the robot base frame by
``ControllerTransform``), plus squeeze and trigger. No retargeters and no clutch —
the clutch rebasing and gripper mapping live downstream in the owning loop, so this
device is stateless across frames.

``isaacteleop`` imports are guarded behind the availability flag so this module imports
without it (construction fails fast via the base class).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from lerobot.types import RobotAction

from .base import IsaacTeleopTeleoperator, _isaacteleop_available
from .config_isaac_teleop import XRControllerConfig

if TYPE_CHECKING or _isaacteleop_available:
    from isaacteleop.retargeting_engine.deviceio_source_nodes import ControllersSource
    from isaacteleop.retargeting_engine.interface import OutputCombiner, TensorGroup, ValueInput
    from isaacteleop.retargeting_engine.tensor_types import TransformMatrix
    from isaacteleop.retargeting_engine.tensor_types.indices import ControllerInputIndex
else:
    ControllersSource = None
    OutputCombiner = None
    TensorGroup = None
    ValueInput = None
    TransformMatrix = None
    ControllerInputIndex = None

# Source-node name for the static base_T_anchor rebase input fed via
# ``TeleopSession.step(external_inputs=...)`` each frame.
_BASE_T_ANCHOR_INPUT = "base_T_anchor"


class XRController(IsaacTeleopTeleoperator):
    """Raw XR controller grip-pose teleoperator (base-frame), no retargeters.

    Reads the raw grip pose + squeeze + trigger off a ``ControllersSource`` rebased into
    the robot base frame. :meth:`get_action` returns the absolute base-frame grip pose
    untouched; the owning loop owns the clutch and gripper mapping.
    """

    config_class = XRControllerConfig
    name = "isaac_teleop_controller"

    def __init__(self, config: XRControllerConfig):
        super().__init__(config)
        self.config: XRControllerConfig = config

        # Constant base_T_anchor input, built once in connect() (a TensorGroup is heavy and
        # isaacteleop-backed) and reused every step.
        self._external_inputs: dict[str, Any] | None = None
        # Whether the last get_action() read a tracked controller; the owning loop polls this
        # to wait for the operator to connect before driving the arm.
        self._is_tracking = False

    # ------------------------------------------------------------------
    # Pipeline construction
    # ------------------------------------------------------------------

    def _build_pipeline(self) -> OutputCombiner:
        """Build the raw-grip-pose pipeline: a ``ControllersSource`` rebased into the base
        frame by ``ControllerTransform``, exposed verbatim as ``"controller"``. No retargeters.
        """
        side = self.config.hand_side
        controller_key = f"controller_{side}"

        controllers = ControllersSource(name="controllers")
        # Static base_T_anchor rebase fed via external_inputs each step.
        xform = ValueInput(_BASE_T_ANCHOR_INPUT, TransformMatrix())
        transformed = controllers.transformed(xform.output("value"))
        ctrl = transformed.output(controller_key)

        return OutputCombiner({"controller": ctrl})

    def _build_external_inputs(self) -> dict[str, Any]:
        """Materialize the constant ``base_T_anchor`` external input (once, in connect)."""
        tg = TensorGroup(TransformMatrix())
        tg[0] = np.asarray(self.config.base_T_anchor, dtype=np.float32)
        return {_BASE_T_ANCHOR_INPUT: {"value": tg}}

    def connect(self, calibrate: bool = True) -> None:
        super().connect(calibrate=calibrate)
        try:
            self._external_inputs = self._build_external_inputs()
        except Exception:
            # Roll the session/runtime back so a failed connect() leaves no half-state
            # (a live session behind a raised connect would leak the CloudXR runtime).
            self.disconnect()
            raise

    # ------------------------------------------------------------------
    # Action features
    # ------------------------------------------------------------------

    @property
    def action_features(self) -> dict:
        return {
            "grip_pos": {
                "dtype": "float32",
                "shape": (3,),
                "names": {"x": 0, "y": 1, "z": 2},
            },
            "grip_quat": {
                "dtype": "float32",
                "shape": (4,),
                "names": {"qx": 0, "qy": 1, "qz": 2, "qw": 3},
            },
            # ``get_action`` returns scalars for these two, so the advertised
            # shape is () (0-d) to stay consistent with the returned values.
            "squeeze": {
                "dtype": "float32",
                "shape": (),
                "names": None,
            },
            "trigger": {
                "dtype": "float32",
                "shape": (),
                "names": None,
            },
        }

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_tracking(self) -> bool:
        """Whether the last :meth:`get_action` read a tracked controller. ``False`` until the
        headset is connected over CloudXR and its controllers are live; the owning loop polls
        it to wait for the operator before commanding the arm."""
        return self._is_tracking

    # ------------------------------------------------------------------
    # Action extraction
    # ------------------------------------------------------------------

    def get_action(self) -> RobotAction:
        """Step the session and return the raw base-frame grip pose.

        Reads the grip pose + squeeze + trigger off the transformed controller stream (with
        the constant ``base_T_anchor`` rebase). When the controller is not tracked, returns
        identity pose and squeeze/trigger = 0.0 so the owning loop freezes the arm.

        Returns:
            ``{"grip_pos": (3,) [m], "grip_quat": (4,) [qx,qy,qz,qw], "squeeze": float,
            "trigger": float}`` — pose in the robot base frame; squeeze/trigger in ``[0, 1]``.
        """
        result = self._step(execution_events=self._running_events(), external_inputs=self._external_inputs)

        # Optional controller group is None until the headset is connected and its controllers
        # are live; expose that as is_tracking so the loop can wait before driving the arm.
        controller = result["controller"]
        grip_pos = np.zeros(3, dtype=np.float32)
        grip_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        squeeze = 0.0
        trigger = 0.0
        self._is_tracking = not getattr(controller, "is_none", False)
        if self._is_tracking:
            # Read ALL four fields into locals before committing any of them: a failure on a
            # partially-populated frame must not mix live values with the safe defaults (a
            # live squeeze paired with a defaulted trigger=0.0 would keep the clutch engaged
            # while commanding the gripper fully open, dropping whatever is grasped). On
            # failure the defaults stand untouched and the frame reports not-tracked.
            try:
                pos = np.asarray(controller[ControllerInputIndex.GRIP_POSITION], dtype=np.float32)
                quat = np.asarray(controller[ControllerInputIndex.GRIP_ORIENTATION], dtype=np.float32)
                squeeze_val = float(controller[ControllerInputIndex.SQUEEZE_VALUE])
                trigger_val = float(controller[ControllerInputIndex.TRIGGER_VALUE])
            except (IndexError, KeyError, TypeError, ValueError):
                self._is_tracking = False
            else:
                grip_pos, grip_quat = pos, quat
                squeeze, trigger = squeeze_val, trigger_val

        return {
            "grip_pos": grip_pos,
            "grip_quat": grip_quat,
            "squeeze": squeeze,
            "trigger": trigger,
        }
