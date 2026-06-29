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

``XRController`` is the first concrete :class:`IsaacTeleopTeleoperator` device
(see :mod:`lerobot.teleoperators.isaac_teleop.base` for the multi-device
pattern). It is a deliberately thin reader: it exposes the **raw** XR controller
grip pose straight off Isaac Teleop's ``ControllersSource`` (statically rebased
into the robot base frame by ``ControllerTransform``), plus the squeeze and
trigger analog values. There are **no** retargeters and **no** clutch/roll/
gripper logic in this device — the clutch rebasing (latch the controller origin
on engage, drive the EE from the delta) and the gripper mapping live downstream
in the owning loop (see ``examples/isaac_teleop_to_so101/teleoperate.py``), so
this device carries no per-frame state of its own.

The shared ``TeleopSession`` lifecycle and per-step health guard live on the
base class; this module only adds the controller-specific pipeline and action
unpacking. The ``isaacteleop`` package is an optional, separately distributed
NVIDIA dependency (the ``isaac-teleop`` extra); all imports of it are deferred
so this module can be imported — and the processor unit-tested — without it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from lerobot.types import RobotAction

from .base import IsaacTeleopTeleoperator
from .config_isaac_teleop import XRControllerConfig

if TYPE_CHECKING:
    from isaacteleop.retargeting_engine.interface import OutputCombiner

# Source-node name for the static base_T_anchor rebase input fed via
# ``TeleopSession.step(external_inputs=...)`` each frame.
_BASE_T_ANCHOR_INPUT = "base_T_anchor"


class XRController(IsaacTeleopTeleoperator):
    """Raw XR controller grip-pose teleoperator (base-frame), no retargeters.

    Wraps a single Isaac Teleop ``ControllersSource`` statically rebased into the
    robot base frame (``base_T_anchor``) by Isaac Teleop's native
    ``ControllerTransform``, and reads the raw grip pose + squeeze + trigger off
    it each frame. There are no SO-101 retargeters and no clutch in this device:
    :meth:`get_action` returns the controller's absolute base-frame grip pose
    untouched. The owning loop owns the clutch (latch the engage origin, drive the
    EE from the delta) and the gripper mapping, so this device is stateless across
    frames.
    """

    config_class = XRControllerConfig
    name = "isaac_teleop_controller"

    def __init__(self, config: XRControllerConfig):
        super().__init__(config)
        self.config: XRControllerConfig = config

        # Build the constant base_T_anchor input ONCE (a TensorGroup is a heavy,
        # isaacteleop-backed object), then reuse it every step. Constructed lazily
        # in connect() so this module imports without isaacteleop installed.
        self._external_inputs: dict[str, Any] | None = None
        # Whether the most recent get_action() read a tracked controller (headset
        # connected over CloudXR + controllers live). The owning loop polls this to
        # wait for the operator to connect before driving the arm.
        self._is_tracking = False

    # ------------------------------------------------------------------
    # Pipeline construction
    # ------------------------------------------------------------------

    def _build_pipeline(self) -> OutputCombiner:
        """Build the raw-grip-pose pipeline (no retargeters).

        A single ``ControllersSource`` statically rebased into the robot base
        frame by ``ControllerTransform`` (``controllers.transformed(...)``); the
        transformed controller stream is exposed verbatim as ``"controller"``::

            ControllersSource ── .transformed(base_T_anchor) ── controller (base-frame grip pose + buttons/axes)

        :meth:`get_action` reads the grip pose, squeeze, and trigger straight off
        that stream (``ControllerTransform`` rotates the grip pose into the base
        frame and copies the buttons/axes through verbatim). The clutch and
        gripper mapping live in the owning loop, so no retargeters are wired here.
        """
        from isaacteleop.retargeting_engine.deviceio_source_nodes import ControllersSource
        from isaacteleop.retargeting_engine.interface import OutputCombiner, ValueInput
        from isaacteleop.retargeting_engine.tensor_types import TransformMatrix

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
        from isaacteleop.retargeting_engine.interface import TensorGroup
        from isaacteleop.retargeting_engine.tensor_types import TransformMatrix

        tg = TensorGroup(TransformMatrix())
        tg[0] = np.asarray(self.config.base_T_anchor, dtype=np.float32)
        return {_BASE_T_ANCHOR_INPUT: {"value": tg}}

    def connect(self, calibrate: bool = True) -> None:
        super().connect(calibrate=calibrate)
        # Built after a successful connect so a failed connect leaves no half-state.
        self._external_inputs = self._build_external_inputs()

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
        """Whether the last :meth:`get_action` read a tracked controller.

        ``False`` until the headset is connected over CloudXR and its controllers are
        live (the stream's optional controller group is present). Mirrors
        :attr:`~lerobot.teleoperators.isaac_teleop.teleop_so101_leader_arm.SO101LeaderArm.is_tracking`;
        the owning loop polls it to wait for the operator to connect before commanding the arm.
        """
        return self._is_tracking

    # ------------------------------------------------------------------
    # Action extraction
    # ------------------------------------------------------------------

    def get_action(self) -> RobotAction:
        """Step the Isaac Teleop session and return the raw base-frame grip pose.

        Steps the session ``RUNNING`` (there is no clutch lifecycle to gate) with
        the static ``base_T_anchor`` rebase supplied as a constant external input,
        then reads the grip pose + squeeze + trigger straight off the transformed
        controller stream. No clutch, no per-frame state: the owning loop latches
        the engage origin and rebases the delta onto the EE.

        When the controller is not tracked this frame, the squeeze/trigger are
        reported as ``0.0`` and the grip pose as the last-known zeros, so the
        owning loop sees "not engaged" and freezes the arm safely.

        Returns:
            A ``RobotAction`` dict with keys:

            - ``"grip_pos"``: ``np.ndarray`` shape ``(3,)`` — absolute controller
              grip position ``[x, y, z]`` [m] in the robot base frame.
            - ``"grip_quat"``: ``np.ndarray`` shape ``(4,)`` — controller grip
              orientation quaternion ``[qx, qy, qz, qw]`` in the robot base frame.
            - ``"squeeze"``: ``float`` — squeeze analog in ``[0, 1]`` (the engage
              clutch input; thresholded by the owning loop).
            - ``"trigger"``: ``float`` — trigger analog in ``[0, 1]`` (the gripper
              input; mapped to a jaw target by the owning loop).
        """
        # Steps the session and applies the shared staleness/worker-health guard
        # (see IsaacTeleopTeleoperator._step). The base_T_anchor rebase is a
        # constant external input.
        result = self._step(execution_events=self._running_events(), external_inputs=self._external_inputs)

        from isaacteleop.retargeting_engine.tensor_types.indices import ControllerInputIndex

        # Transformed controller stream (ControllerTransform rotates the grip pose
        # into the base frame and copies buttons/axes through verbatim). When the
        # controller is not tracked the optional group is None; treat that as "not
        # engaged" (squeeze/trigger = 0.0) so the owning loop freezes the arm.
        controller = result["controller"]
        grip_pos = np.zeros(3, dtype=np.float32)
        grip_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        squeeze = 0.0
        trigger = 0.0
        # The optional controller group is None until the headset is connected and its
        # controllers are live; expose that as is_tracking so the owning loop can wait
        # for the operator to connect over CloudXR before driving the arm.
        self._is_tracking = not getattr(controller, "is_none", False)
        if self._is_tracking:
            # Defensive: a controller group may not be fully populated every frame
            # (odd frame, missing axis, unexpected wrapper shape). Any read failure
            # leaves the safe defaults above and is reported as not-tracked, so the loop
            # freezes the arm instead of crashing or trusting a partial frame.
            try:
                grip_pos = np.asarray(controller[ControllerInputIndex.GRIP_POSITION], dtype=np.float32)
                grip_quat = np.asarray(controller[ControllerInputIndex.GRIP_ORIENTATION], dtype=np.float32)
                squeeze = float(controller[ControllerInputIndex.SQUEEZE_VALUE])
                trigger = float(controller[ControllerInputIndex.TRIGGER_VALUE])
            except (IndexError, KeyError, TypeError, ValueError):
                self._is_tracking = False

        return {
            "grip_pos": grip_pos,
            "grip_quat": grip_quat,
            "squeeze": squeeze,
            "trigger": trigger,
        }
