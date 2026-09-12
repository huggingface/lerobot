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

A clutched device: the controller grip pose is rebased into the robot base frame by
``ControllerTransform`` and then driven through an in-pipeline
``SO101ClutchRetargeter``, so :meth:`XRController.get_action` returns an absolute
base-frame EE pose rather than a raw controller pose. Unlike the other devices here this one
**holds state across frames** — the clutch's latched home and origin live in the retargeter — so
it must be stepped every frame with real ``ExecutionEvents``. The analog trigger is still passed
through raw; the gripper mapping stays in the owning loop.

``isaacteleop`` imports are guarded behind the availability flag so this module imports
without it (construction fails fast via the base class).
"""

from __future__ import annotations

import importlib.metadata
from typing import TYPE_CHECKING, Any

import numpy as np

from lerobot.lerobot_types import RobotAction

from .base import IsaacTeleopTeleoperator, _isaacteleop_available
from .config_isaac_teleop import XRControllerConfig

if TYPE_CHECKING or _isaacteleop_available:
    from isaacteleop.retargeting_engine.deviceio_source_nodes import ControllersSource
    from isaacteleop.retargeting_engine.interface import (
        ExecutionEvents,
        ExecutionState,
        OptionalTensorGroup,
        OutputCombiner,
        TensorGroup,
        ValueInput,
    )
    from isaacteleop.retargeting_engine.interface.tensor_group_type import OptionalType
    from isaacteleop.retargeting_engine.tensor_types import ControllerInput, TransformMatrix
    from isaacteleop.retargeting_engine.tensor_types.indices import ControllerInputIndex
else:
    ControllersSource = None
    ControllerInput = None
    ExecutionEvents = None
    ExecutionState = None
    OptionalTensorGroup = None
    OptionalType = None
    OutputCombiner = None
    TensorGroup = None
    ValueInput = None
    TransformMatrix = None
    ControllerInputIndex = None

# The engage-relative clutch retargeter landed in isaacteleop 1.5; the rest of this example works
# against older releases. Resolve it tolerantly here and fail with an actionable message from
# XRController's constructor (see _require_clutch_retargeter) -- a hard import error here would
# also break the SO-101 leader-arm device, which never touches XR.
SO101ClutchRetargeter = None
_CLUTCH_IMPORT_ERROR: Exception | None = None
if _isaacteleop_available:
    try:
        import isaacteleop.retargeters as _isaacteleop_retargeters

        SO101ClutchRetargeter = _isaacteleop_retargeters.SO101ClutchRetargeter
    except (ImportError, AttributeError) as exc:
        # Retained and chained below so a genuinely broken install is not misreported as
        # "upgrade isaacteleop".
        _CLUTCH_IMPORT_ERROR = exc

# Source-node name for the static base_T_anchor rebase fed via
# ``TeleopSession.step(external_inputs=...)`` each frame.
#
# There is deliberately no companion constant for the measured-EE key: the producer side reads
# ``SO101ClutchRetargeter.MEASURED_BASE_T_EE_INPUT`` directly, so the producer key here and the
# consumer key there cannot drift apart. Re-declaring the literal would defeat that.
_BASE_T_ANCHOR_INPUT = "base_T_anchor"

_MIN_ISAACTELEOP_VERSION = "1.4.0"


def _require_clutch_retargeter() -> None:
    """Fail when the installed isaacteleop cannot supply the engage-relative clutch retargeter.

    Called from :meth:`XRController.__init__` rather than ``_build_pipeline``: the latter runs
    inside ``connect()``, *after* ``_ensure_cloudxr_runtime()``, so a purely static version
    mismatch would otherwise cost a ~30 s runtime launch and possibly an interactive EULA prompt
    before being reported.

    The probe is a CAPABILITY check, not a name check, and that distinction is load-bearing:
    ``SO101ClutchRetargeter`` also exists in isaacteleop 1.4, as a *different* retargeter (clutches
    position only, applies a fixed orientation offset, ``home_base_T_ee`` optional). Probing the
    name alone would therefore pass against 1.4 and then drive the arm wrongly, with no error.
    ``MEASURED_BASE_T_EE_INPUT`` exists only on the engage-relative implementation this device
    needs, so it is the signal that actually discriminates.

    ``_MIN_ISAACTELEOP_VERSION`` is deliberately still ``1.4.0``: the engage-relative clutch has
    not shipped in a published wheel yet, so naming a version PyPI cannot resolve would be worse
    advice than the capability probe above. Bump it -- and the install pins in ``README.md`` and
    ``docs/source/isaac_teleop.mdx`` -- when that wheel ships.
    """
    if SO101ClutchRetargeter is not None and hasattr(SO101ClutchRetargeter, "MEASURED_BASE_T_EE_INPUT"):
        return
    try:
        installed = importlib.metadata.version("isaacteleop")
    except importlib.metadata.PackageNotFoundError:
        installed = "an unknown version"
    raise ImportError(
        "XRController requires an isaacteleop whose SO101ClutchRetargeter is the engage-relative "
        f"full-pose clutch (it must expose MEASURED_BASE_T_EE_INPUT), but {installed} is "
        f"installed (>= {_MIN_ISAACTELEOP_VERSION} is necessary but not sufficient). Upgrade "
        "with:\n"
        '  uv pip install -U "isaacteleop[cloudxr,retargeters-lite]"'
    ) from _CLUTCH_IMPORT_ERROR


# Placeholder home for the retargeter, which must be constructed in ``_build_pipeline()`` (inside
# ``connect()``) — long before the arm's real EE pose is known. Safe because the session holds
# ``STOPPED`` until :meth:`XRController.start` is called, and the clutch cannot latch while
# STOPPED, so the home value is irrelevant in that window. The owning loop supplies the real one
# via :meth:`XRController.set_home_base_T_ee` before the first RUNNING frame.
_PLACEHOLDER_HOME_BASE_T_EE = np.eye(4, dtype=np.float64)


class XRController(IsaacTeleopTeleoperator):
    """Clutched XR controller teleoperator emitting an absolute base-frame EE pose.

    Reads the grip pose + squeeze + trigger off a ``ControllersSource`` rebased into the robot
    base frame, and drives them through an in-pipeline ``SO101ClutchRetargeter``.
    :meth:`get_action` returns the clutch-rebased absolute EE pose, the raw analog trigger, and
    whether the clutch is engaged; the owning loop owns the gripper mapping and the safety gate.

    Lifecycle, which the owning loop must drive:

    1. :meth:`connect` builds the pipeline and opens the session. The session holds ``STOPPED``,
       so the clutch cannot latch.
    2. The loop waits for the headset and homes the arm, stepping this device throughout.
    3. The loop calls :meth:`set_home_base_T_ee` with the arm's measured EE pose, then
       :meth:`start` — which flips the session to ``RUNNING`` and allows the clutch to engage.

    Holding ``STOPPED`` for steps 1-2 is a readiness interlock, not a formality: the graph is
    stepped throughout the connect wait and the homing slew, and the operator is tracked during
    both. Without it a squeeze while donning the headset would latch a home the arm has not
    reached.
    """

    config_class = XRControllerConfig
    name = "isaac_teleop_controller"

    def __init__(self, config: XRControllerConfig):
        super().__init__(config)
        self.config: XRControllerConfig = config
        # Before connect(), so a static version mismatch is reported without first paying for the
        # CloudXR runtime launch (and a possible interactive EULA prompt).
        _require_clutch_retargeter()

        # Constant base_T_anchor input, built once in connect() (a TensorGroup is heavy and
        # isaacteleop-backed) and reused every step.
        self._external_inputs: dict[str, Any] | None = None
        # Whether the last get_action() read a tracked controller; the owning loop polls this
        # to wait for the operator to connect before driving the arm.
        self._is_tracking = False
        # The in-pipeline clutch, built in _build_pipeline() and retained so get_action() can read
        # its engagement state back after each step.
        self._retargeter: SO101ClutchRetargeter | None = None
        # Readiness interlock: STOPPED until start() is called. Never None on the wire — passing
        # None makes TeleopSession.step auto-fire RUNNING, which would defeat the interlock.
        # Safe to name the enum here: the base __init__ above calls _require_isaacteleop(), which
        # raises before returning when isaacteleop is absent.
        self._execution_state: ExecutionState = ExecutionState.STOPPED
        # The arm's measured base_T_ee for this frame. CONSUMED AND CLEARED by get_action().
        self._measured_base_T_ee: np.ndarray | None = None
        # Whether set_home_base_T_ee() has run. start() refuses without it: the placeholder home
        # is identity, and an unseeded ORIENTATION has no measured-input rescue path.
        self._home_seeded = False

    # ------------------------------------------------------------------
    # Pipeline construction
    # ------------------------------------------------------------------

    def _build_pipeline(self) -> OutputCombiner:
        """Build the clutch pipeline: ``ControllersSource`` -> base-frame rebase -> clutch.

        Publishes two outputs. ``ee_pose`` is the clutch-rebased absolute EE target. ``controller``
        is the rebased controller group passed through verbatim — :meth:`get_action` derives both
        ``is_tracking`` and the analog trigger from it, and both would be lost if only ``ee_pose``
        were published.
        """
        side = self.config.hand_side
        controller_key = f"controller_{side}"

        controllers = ControllersSource(name="controllers")
        # Static base_T_anchor rebase fed via external_inputs each step.
        xform = ValueInput(_BASE_T_ANCHOR_INPUT, TransformMatrix())
        transformed = controllers.transformed(xform.output("value"))
        ctrl = transformed.output(controller_key)

        # OptionalType is load-bearing, but it is NOT permission to omit the key: a plain
        # ValueInput leaf is a required GRAPH input, and OptionalType only makes its *contents*
        # optional. TeleopSession.step separately requires every external leaf NAME on every step
        # (see get_action, which therefore always sends this key). OptionalType is what lets that
        # key carry an absent group, degrading to the retargeter's last-commanded home fallback
        # instead of failing the graph.
        measured = ValueInput(
            SO101ClutchRetargeter.MEASURED_BASE_T_EE_INPUT,
            OptionalType(TransformMatrix()),
        )

        self._retargeter = SO101ClutchRetargeter(
            "so101_clutch",
            _PLACEHOLDER_HOME_BASE_T_EE,
            input_device=controller_key,
            position_scale=self.config.clutch_position_scale,
            squeeze_threshold=self.config.clutch_threshold,
        )
        clutched = self._retargeter.connect(
            {
                controller_key: ctrl,
                SO101ClutchRetargeter.MEASURED_BASE_T_EE_INPUT: measured.output("value"),
            }
        )

        return OutputCombiner({"ee_pose": clutched.output("ee_pose"), "controller": ctrl})

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

    def disconnect(self) -> None:
        self._execution_state = ExecutionState.STOPPED
        self._retargeter = None
        self._measured_base_T_ee = None
        self._home_seeded = False
        super().disconnect()

    # ------------------------------------------------------------------
    # Readiness interlock and per-frame inputs (driven by the owning loop)
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Flip the session to ``RUNNING``, allowing the clutch to engage on a squeeze.

        Call once the arm is at its home pose and :meth:`set_home_base_T_ee` has been given that
        pose. Before this, squeezing does nothing. Takes effect on the next :meth:`get_action`.

        Raises:
            RuntimeError: If :meth:`set_home_base_T_ee` has not been called. The placeholder home
                is the identity transform, and while the measured-EE input rescues the home
                *position* on engage, nothing rescues the home *orientation* -- it always comes
                from the last commanded rotation. An unseeded clutch would therefore snap the
                wrist to base-frame identity on the first squeeze: real arm motion, no exception,
                and it reads like an IK bug rather than a missing call.
        """
        if not self._home_seeded:
            raise RuntimeError(
                "set_home_base_T_ee() must be called before start(): the clutch would otherwise "
                "latch its home orientation from the identity placeholder and snap the wrist to "
                "base-frame identity on the first squeeze."
            )
        self._execution_state = ExecutionState.RUNNING

    def stop(self) -> None:
        """Return the session to ``STOPPED``, disengaging the clutch and re-arming its latch.

        Takes effect on the next :meth:`get_action`; the value reported by ``get_action`` still
        reflects the last computed frame until then.
        """
        self._execution_state = ExecutionState.STOPPED

    def set_home_base_T_ee(self, base_T_ee: np.ndarray) -> None:  # noqa: N802, N803  (frameA_T_frameB convention)
        """Seed the clutch's held pose from the arm's measured ``base_T_ee`` [m].

        The retargeting graph is built in :meth:`connect`, long before the arm has been homed, so
        the retargeter starts on an identity placeholder. **Call this while the clutch is not
        engaged** -- in practice before :meth:`start`, while the session still holds ``STOPPED``
        and latching is impossible, which makes the ordering unambiguous: the new home takes
        effect before the first ``RUNNING`` frame. Calling it later re-arms the clutch's pending
        latch rather than jumping the arm, but is not the intended use.

        Raises:
            RuntimeError: If not connected.
        """
        if not self.is_connected or self._retargeter is None:
            raise RuntimeError("Not connected. Call connect() first.")
        self._retargeter.set_home_base_T_ee(base_T_ee)
        self._home_seeded = True

    def set_measured_base_T_ee(self, base_T_ee: np.ndarray) -> None:  # noqa: N802, N803  (frameA_T_frameB convention)
        """Supply the arm's measured ``base_T_ee`` [m] for the NEXT :meth:`get_action` only.

        The clutch latches its home *position* from this on the engage frame, so an arm that
        sagged or was pushed while disengaged is not commanded back to a stale target. The home
        orientation is never taken from it.

        The value is **consumed and cleared** by :meth:`get_action`. That is deliberate: a value
        that persisted would silently feed a stale forward-kinematics result forever the day the
        loop stopped calling this, whereas consume-on-read makes "stale by one frame"
        unrepresentable — the pose is either this frame's or absent, and absent lands on the
        retargeter's documented last-commanded fallback.

        No timestamp travels with the pose, and none is checked. "This frame" therefore means the
        caller's frame, not the retargeting graph's: on a ``frame_deadline_miss``
        (see ``base.py``'s stale-frame warning) the clutch can latch its home off forward
        kinematics that is one or two frames — roughly 33–66 ms at 30 Hz — old. At clutch speeds
        that is a small position error on the engage frame, not a stability problem, so it is
        recorded rather than mechanised.
        """
        self._measured_base_T_ee = np.asarray(base_T_ee, dtype=np.float64)

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
            # ``get_action`` returns a scalar for this, so the advertised shape is () (0-d)
            # to stay consistent with the returned value.
            "trigger": {
                "dtype": "float32",
                "shape": (),
                "names": None,
            },
            "engaged": {
                "dtype": "bool",
                "shape": (),
                "names": None,
            },
            # Returned per-frame as well as via the :attr:`is_tracking` property, so that both
            # halves of the command gate travel in one object and omitting one is not possible.
            "is_tracking": {
                "dtype": "bool",
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
        """Step the session and return the clutch-rebased EE target for this frame.

        Reads the pipeline's ``ee_pose`` (the clutch output) and its passthrough ``controller``
        group (for tracking state and the analog trigger), and reads engagement back off the
        clutch retargeter.

        The session is stepped with an explicit ``ExecutionEvents`` every frame — never ``None``,
        which would make ``TeleopSession.step`` auto-fire ``RUNNING`` and defeat the readiness
        interlock. Any measured EE pose supplied via :meth:`set_measured_base_T_ee` is consumed
        and cleared here.

        Note the two halves of the owning loop's safety gate come from different places:
        ``is_tracking`` is derived from the **returned** frame, while ``engaged`` is read live off
        the retargeter instance. Under ``RetargetingExecutionMode.SYNC`` (the default) those are
        the same frame. Under ``PIPELINED`` the returned outputs can lag the retargeter's internal
        state by a frame, so the two signals would describe different instants.

        Returns:
            ``{"ee_pose": (7,), "trigger": float, "engaged": bool, "is_tracking": bool}``.

            - ``ee_pose`` -- ``[x, y, z, qx, qy, qz, qw]`` in the robot base frame.
            - ``trigger`` -- analog trigger in ``[0, 1]``. On a partially-populated frame this is
              ``0.0``, which is a **floor, not a reading**: the command gate below covers the
              gripper path, but anything else consuming this value should treat it as unknown when
              ``is_tracking`` is ``False``.
            - ``engaged`` -- means **the clutch is latched**, nothing more. It deliberately does
              **not** fold in tracking: forcing it ``False`` while the latch is still held would
              make the loop see a spurious rising edge on recovery. Command gating is the
              caller's.
            - ``is_tracking`` -- whether this frame carried a live, fully-read controller.

            **The command gate is the conjunction**:
            ``if not (action["engaged"] and action["is_tracking"]): hold``. Both halves are
            returned together so that omitting one is not possible; ``engaged`` alone would
            release a live grasp on a partially-populated frame (see the ``except`` below).
        """
        external_inputs = dict(self._external_inputs or {})
        measured = self._measured_base_T_ee
        # Consume: the value is valid for exactly this frame (see set_measured_base_T_ee).
        self._measured_base_T_ee = None
        measured_group = TensorGroup(TransformMatrix())
        if measured is not None:
            measured_group[0] = measured.astype(np.float32)
        else:
            # NOT redundant work -- do not "optimise" this branch away. The leaf key must be
            # present on EVERY step even when there is no pose to send: TeleopSession.step
            # validates that every external leaf NAME appears in external_inputs and raises
            # otherwise, independently of the leaf's OptionalType. Dropping the key here fails the
            # very first get_action() inside the connect-wait -- dead on arrival at startup. An
            # absent OptionalTensorGroup satisfies the check and reaches the retargeter as
            # is_none, landing on its documented last-commanded home fallback.
            measured_group = OptionalTensorGroup(TransformMatrix())
        external_inputs[SO101ClutchRetargeter.MEASURED_BASE_T_EE_INPUT] = {"value": measured_group}

        result = self._step(
            execution_events=ExecutionEvents(execution_state=self._execution_state, reset=False),
            external_inputs=external_inputs,
        )

        ee_pose = np.asarray(np.from_dlpack(result["ee_pose"][0]), dtype=np.float32).copy()

        # Optional controller group is None until the headset is connected and its controllers
        # are live; expose that as is_tracking so the loop can wait before driving the arm. This
        # derivation MUST survive: the loop's connect-wait polls is_tracking and never returns if
        # it is stuck False.
        controller = result["controller"]
        trigger = 0.0
        # Attribute access, not getattr-with-default: ``is_none`` is part of the
        # OptionalTensorGroup contract, and a default would silently turn a future rename into
        # is_tracking stuck True -- the one derivation the connect-wait depends on.
        self._is_tracking = not controller.is_none
        if self._is_tracking:
            try:
                trigger = float(controller[ControllerInputIndex.TRIGGER_VALUE])
            except (IndexError, KeyError, TypeError, ValueError):
                # A partially-populated frame yields trigger = 0.0 (jaw fully OPEN). Note this no
                # longer disengages the clutch: ``squeeze`` is now read inside the retargeting
                # graph, on a path this except cannot reach, so ``engaged`` stays true and a live
                # grasp would be released if the loop acted on it. Reporting not-tracked is what
                # covers it -- the owning loop must gate its command on ``is_tracking`` as well as
                # ``engaged``. ``engaged`` itself is deliberately left as the pure retargeter
                # signal so its rising edge stays exactly the clutch's latch frame.
                self._is_tracking = False

        engaged = bool(self._retargeter.is_engaged) if self._retargeter is not None else False

        return {
            "ee_pose": ee_pose,
            "trigger": trigger,
            "engaged": engaged,
            "is_tracking": self._is_tracking,
        }
